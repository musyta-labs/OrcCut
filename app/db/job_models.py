"""Durable analysis-job queue table — Gate 3 Step 6.

The shared broker that replaces ``background._inflight``. A background analysis
is today deduplicated and reported as "pending" from a per-PROCESS set, which
lies the moment a second replica exists: replica B cannot see the job replica A
is running, so it runs it a second time and its ``is_pending`` under-reports.
This table moves that accounting into ONE place every replica reads.

The broker is Postgres itself (operator decision — no Redis/RQ/Celery): a job
is claimed with ``SELECT ... FOR UPDATE SKIP LOCKED`` on PostgreSQL and a plain
serialised transaction on SQLite (the self-host path, where SQLite's global
writer lock already gives the same guarantee). See
``app.db.repositories.analysis_jobs`` for the claim/dedup logic.

**A row exists for exactly as long as the job is LIVE** (pending or running).
``enqueue`` inserts it, a claim flips it to running under a replica id, and
completion — success OR failure — DELETES it, exactly as ``_inflight.discard``
freed the key. Keeping a terminal row would pin the unique
``(project_id, media_id)`` key forever and block the retry the failure contract
promises ("nothing stored, the next add_clip retries") and the fast cache-hit
no-op a later re-analysis takes. ``done``/``failed`` therefore name outcomes in
logs, not persisted states. The mitigation that makes this safe is the same one
``media_analysis.save`` documents: a double run is wasted CPU, never corruption.

**The claim carries a heartbeat, not just an owner.** ``heartbeat_at`` (epoch
seconds, a plain float — the same clock discipline ``state_models`` documents,
identical on SQLite and PostgreSQL with no timezone round-trip) lets a claim
from a replica that crashed mid-job be reclaimed once it goes stale, so a dead
replica cannot pin a job forever. ``created_at`` stays a real timestamp purely
for a human reading the row and for FIFO-ish claim ordering.

Registered on ``Base.metadata`` as a side effect of importing ``app.db.models``
(see the import there), and kept in lockstep with its Alembic migration
(``*_analysis_job_queue``), exactly as the other tables are with theirs.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AnalysisJobRow(Base):
    """One live background-analysis job for one media asset.

    The unique ``(project_id, media_id)`` key is the dedup guarantee: at most
    one live job per asset across the WHOLE cluster, so a second ``enqueue`` of
    the same key — on this replica or another — is a no-op that returns the
    existing job rather than a second row.
    """

    __tablename__ = "analysis_jobs"

    __table_args__ = (
        # At most one live job per asset, cluster-wide. This is the cross-replica
        # de-duplication: ``enqueue`` is an INSERT ... ON CONFLICT DO NOTHING
        # against this constraint, so two replicas racing the same key produce
        # one row, not two, and only the winner's claim ever runs it.
        UniqueConstraint("project_id", "media_id", name="uq_analysis_jobs_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # The asset key. Both are opaque application ids (project uuid, media id),
    # stored as plain strings so this table couples to no other table's shape.
    project_id: Mapped[str] = mapped_column(String(64))
    media_id: Mapped[str] = mapped_column(String(64))
    # The tenant the job runs for — carried so a claiming worker can bound the
    # owner's concurrency (the FairScheduler ceiling) and resolve the asset.
    owner_id: Mapped[int] = mapped_column(Integer, index=True)
    # "pending" while queued, "running" once claimed. Terminal outcomes delete
    # the row (see module docstring), so a stored status is only ever one of
    # these two live values.
    status: Mapped[str] = mapped_column(String(16))
    # The replica id that holds the current claim, or NULL while pending. Purely
    # diagnostic — the claim's validity is decided by ``heartbeat_at``, not by
    # matching this — so a reclaim needs no coordination with the dead holder.
    claimed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Epoch seconds of the last claim/heartbeat. A running job whose heartbeat
    # is older than the claim TTL is treated as a crashed replica's and may be
    # reclaimed. Numeric, compared against an injectable ``now`` — see module
    # docstring.
    heartbeat_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
