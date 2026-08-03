"""The durable analysis-job queue — enqueue, claim, complete — Gate 3 Step 6.

Business logic goes through these helpers, not raw sessions (repository pattern,
as with ``media_analysis``/``quotas``). Together they are the shared broker that
replaces ``background._inflight``: dedup, "is this asset pending?", and "which
job may this replica run next?" all become questions the DATABASE answers, so
they hold across replicas instead of within one process.

Two atomicity concerns, both closed the same way ``media_analysis`` and
``shared_state_db`` document — by letting the database evaluate the guard
together with the write, never check-then-act in Python:

* **Dedup** — ``enqueue`` is a single ``INSERT ... ON CONFLICT DO NOTHING``
  against the unique ``(project_id, media_id)`` key, so two replicas racing the
  same asset produce one row, not two, with no lock and no lost update.
* **Claim** — ``claim_next`` selects one claimable job and flips it to running
  in one transaction. On PostgreSQL the select takes ``FOR UPDATE SKIP LOCKED``
  so two workers never claim the same row (the loser skips it and takes the
  next); on SQLite there is no such clause and none is needed — its global
  writer serialisation means the losing transaction simply waits, then re-reads
  the now-running row and skips it. The dialect split mirrors
  ``media_analysis._insert_for_dialect`` exactly.

Time is epoch seconds (see ``app.db.job_models``): a plain float, injectable as
``now`` so a test drives staleness deterministically without sleeping, and
identical on both backends. The default clock is wall time — NOT monotonic —
because the rows are shared across processes that cannot compare monotonic
clocks (the same choice ``shared_state_db`` makes).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from sqlalchemy import and_, delete, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.db.job_models import AnalysisJobRow

# The two live statuses a stored row can hold. Terminal outcomes DELETE the row
# (see job_models) so its unique key frees for a retry/re-analysis; "done" and
# "failed" name outcomes in logs, never a persisted state.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

LIVE_STATUSES = (STATUS_PENDING, STATUS_RUNNING)

# A running claim whose last heartbeat is older than this is treated as a
# crashed replica's and may be reclaimed. Generous on purpose: a background
# analysis behind a URL import can spend minutes downloading before it decodes,
# and reclaiming a job that is merely slow would run it twice.
CLAIM_TTL_SECONDS = 300

# Dialects that get a ``FOR UPDATE SKIP LOCKED`` claim. SQLite is absent on
# purpose — it serialises writers globally, so the claim is already atomic.
_POSTGRES_DIALECTS = frozenset({"postgresql", "postgres"})


@dataclass(frozen=True)
class ClaimedJob:
    """One job this replica now owns, returned by ``claim_next``. Immutable — a
    claim's identity must not change between being handed out and being run."""

    id: int
    project_id: str
    media_id: str
    owner_id: int


def _wall(now: float | None) -> float:
    return time.time() if now is None else now


def _is_postgres(session: Session) -> bool:
    return session.get_bind().dialect.name in _POSTGRES_DIALECTS


def _insert_for(session: Session) -> Callable:
    """The ``INSERT`` construct whose ``ON CONFLICT`` clause the session's own
    backend understands — the same dialect split ``media_analysis`` uses."""
    return postgresql_insert if _is_postgres(session) else sqlite_insert


def enqueue(
    session: Session,
    *,
    project_id: str,
    media_id: str,
    owner_id: int,
    now: float | None = None,
) -> bool:
    """Record a pending job for this asset, or no-op if one is already live.

    Returns ``True`` when a NEW row was created, ``False`` when a live job for
    this ``(project_id, media_id)`` already existed (this replica's earlier
    schedule, or another replica's). The unique key makes this the cluster-wide
    de-duplication point — a single ``INSERT ... ON CONFLICT DO NOTHING`` with
    no read-then-write window.
    """
    result = session.execute(
        _insert_for(session)(AnalysisJobRow)
        .values(
            project_id=project_id,
            media_id=media_id,
            owner_id=owner_id,
            status=STATUS_PENDING,
            heartbeat_at=_wall(now),
        )
        .on_conflict_do_nothing(index_elements=["project_id", "media_id"])
    )
    session.flush()
    return result.rowcount == 1


def claim_next(
    session: Session,
    *,
    replica_id: str,
    now: float | None = None,
    stale_before: float | None = None,
) -> ClaimedJob | None:
    """Atomically take one claimable job and mark it running under ``replica_id``.

    Claimable is "pending, OR running but its claim has gone stale" — the latter
    being a job a crashed replica left behind. The oldest such job wins (FIFO by
    ``created_at``); fairness ACROSS owners is the scheduler's job once the job
    is dispatched, not the queue's. Returns the claimed job, or ``None`` when
    nothing is claimable right now.

    On PostgreSQL the select takes ``FOR UPDATE SKIP LOCKED`` so a job another
    worker is mid-claim on is skipped rather than double-claimed; on SQLite the
    global writer lock serialises the whole select-then-update.
    """
    moment = _wall(now)
    horizon = moment - CLAIM_TTL_SECONDS if stale_before is None else stale_before
    query = (
        select(AnalysisJobRow)
        .where(
            or_(
                AnalysisJobRow.status == STATUS_PENDING,
                and_(
                    AnalysisJobRow.status == STATUS_RUNNING,
                    AnalysisJobRow.heartbeat_at <= horizon,
                ),
            )
        )
        .order_by(AnalysisJobRow.created_at, AnalysisJobRow.id)
        .limit(1)
    )
    if _is_postgres(session):
        query = query.with_for_update(skip_locked=True)
    row = session.execute(query).scalars().first()
    if row is None:
        return None
    session.execute(
        update(AnalysisJobRow)
        .where(AnalysisJobRow.id == row.id)
        .values(status=STATUS_RUNNING, claimed_by=replica_id, heartbeat_at=moment)
    )
    session.flush()
    return ClaimedJob(
        id=row.id,
        project_id=row.project_id,
        media_id=row.media_id,
        owner_id=row.owner_id,
    )


def heartbeat(session: Session, *, job_id: int, now: float | None = None) -> None:
    """Refresh a running claim so a genuinely long job is not mistaken for a
    crashed replica's and reclaimed. The separable worker loop calls this; the
    in-process bridge's short jobs finish well inside one TTL and need not."""
    session.execute(
        update(AnalysisJobRow)
        .where(AnalysisJobRow.id == job_id)
        .values(heartbeat_at=_wall(now))
    )
    session.flush()


def complete(session: Session, *, project_id: str, media_id: str) -> None:
    """Remove the job for this asset, freeing its unique key.

    Called on success AND failure (see job_models): a deleted row lets the next
    ``add_clip`` re-schedule — the retry the failure contract promises, and the
    fast cache-hit no-op a re-analysis takes. Idempotent: a second completion,
    or completing a job another replica already finished, matches zero rows.
    """
    session.execute(
        delete(AnalysisJobRow).where(
            AnalysisJobRow.project_id == project_id,
            AnalysisJobRow.media_id == media_id,
        )
    )
    session.flush()


def is_live(session: Session, *, project_id: str, media_id: str) -> bool:
    """Whether a background analysis for this asset is queued or running
    anywhere in the cluster — the cross-replica answer ``background.is_pending``
    reports. A live row is the only kind that exists (terminal deletes), so this
    is simply "does a row for this key exist"."""
    return (
        session.execute(
            select(AnalysisJobRow.id)
            .where(
                AnalysisJobRow.project_id == project_id,
                AnalysisJobRow.media_id == media_id,
                AnalysisJobRow.status.in_(LIVE_STATUSES),
            )
            .limit(1)
        ).first()
        is not None
    )
