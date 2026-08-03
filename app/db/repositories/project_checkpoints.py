"""Persistence for the server-side VERSION HISTORY (``ProjectCheckpointRow``).

A checkpoint is a full, compressed copy of one project at one MUTATION NUMBER
(``ProjectRow.version``). Two things live here and nowhere else:

1. **The blob codec.** ``_encode``/``_decode`` are the only code in the
   application that knows the stored bytes are zlib over UTF-8 JSON. Keeping
   the format behind two functions means changing it later is one edit, not a
   hunt through every reader.
2. **The dedup + race contract.** ``create_checkpoint`` is the ONLY writer, and
   it is idempotent per (project, version): asking twice for a checkpoint of an
   unchanged project yields one row and reports the second call as "already
   there", never an error. Both the explicit route and the idle sweep go
   through it, so neither can invent a second policy.

WHY NOT WRITE ONE PER MUTATION. An earlier design put the write inside
``save_project`` so it could not be bypassed. That is the right instinct for a
mutation journal and the wrong one for a version history: every op bumps
``ProjectRow.version``, so nudging a clip two pixels would have minted a
"version", and the list a person opens to find the three states they care about
would have been two hundred rows of noise. Checkpoints are therefore COARSE by
construction — minted on an explicit request or after the editing stops — and
``save_project`` is left exactly as it was.

Ownership is derived THROUGH the project with an EXISTS, never duplicated onto
these rows, exactly as ``projects.get_history`` does it for the journal: one
owner column that can drift out of agreement with the project's is worse than
no owner column at all.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import (
    CheckpointTrigger, OperationLogRow, ProjectCheckpointRow, ProjectRow,
)

# zlib's default. Level 9 buys a few percent on text this repetitive for
# noticeably more CPU, and these blobs are written at most a handful of times
# per project per hour — there is nothing here worth optimising past the
# default trade.
_COMPRESSION_LEVEL = 6


def _encode(data: dict) -> bytes:
    """Serialize a ``project_to_dict`` document to the stored bytes.

    ``ensure_ascii=False`` keeps non-ASCII text (Cyrillic captions, emoji) as
    real UTF-8 rather than ``\\uXXXX`` escapes — three times smaller before
    compression, and it round-trips identically either way. The compact
    separators drop the whitespace ``json.dumps`` would otherwise pad every
    document with.
    """
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return zlib.compress(payload.encode("utf-8"), _COMPRESSION_LEVEL)


def _decode(blob: bytes) -> dict:
    """Inverse of ``_encode``. Raises ``zlib.error``/``ValueError`` on a
    corrupt blob rather than returning a half-document — a checkpoint that
    cannot be read is a 500, not an empty project silently offered as history.
    """
    return json.loads(zlib.decompress(blob).decode("utf-8"))


@dataclass(frozen=True)
class CheckpointEntry:
    """One row of the version list. Carries no blob: the list screen shows
    version numbers, times and the op that produced them, and loading every
    snapshot to render it would read megabytes to display kilobytes."""

    version: int
    trigger: str
    created_at: datetime
    op_name: str | None
    op_args: dict


@dataclass(frozen=True)
class CheckpointResult:
    """Outcome of asking for a checkpoint. ``created`` is False both when the
    project was unchanged since the last checkpoint and when a concurrent
    writer won the race — from the caller's point of view those are the same
    fact ("the checkpoint for this state exists, and it is not mine"), and
    neither is an error."""

    version: int
    created_at: datetime
    trigger: str
    created: bool


def _owned(project_id: str, owner_id: int):
    """EXISTS predicate: ``project_id`` names a project of ``owner_id``'s.

    Another tenant's project (and an unknown one) makes this False, so every
    read below reports ABSENCE rather than refusal — the same posture the rest
    of the repositories take, so none of these endpoints can be used as an
    oracle for which project ids exist.
    """
    return (
        select(ProjectRow.id)
        .where(ProjectRow.project_id == project_id, ProjectRow.owner_id == owner_id)
        .exists()
    )


def create_checkpoint(
    session: Session,
    project_id: str,
    *,
    owner_id: int,
    trigger: CheckpointTrigger = CheckpointTrigger.MANUAL,
) -> CheckpointResult | None:
    """Freeze this owner's project AS IT IS NOW, or report that its current
    state is already frozen. ``None`` for an unknown/foreign project.

    DEDUPLICATION is by mutation number, not by content: if a checkpoint
    already exists for the project's current ``version``, nothing is written
    and ``created=False`` comes back. That makes a repeated explicit request on
    an untouched project harmless (the operator's "save" button is a no-op the
    second time) and makes the idle sweep safe to run as often as it likes.

    THE RACE is handled, not avoided. The dedup SELECT and the INSERT are two
    statements, and two API replicas — or a replica and the sweep — can both
    pass the SELECT for the same (project, version). The unique index on that
    pair means only one INSERT survives; the loser's ``IntegrityError`` is
    caught here, rolled back to a SAVEPOINT so the caller's transaction is
    still usable, and reported as ``created=False``. Letting the exception
    escape would turn a benign coincidence into a 500 on a button press.
    """
    row = session.execute(
        select(ProjectRow).where(
            ProjectRow.project_id == project_id, ProjectRow.owner_id == owner_id
        )
    ).scalar_one_or_none()
    if row is None:
        return None

    existing = _find(session, project_id, row.version)
    if existing is not None:
        return CheckpointResult(
            version=existing.version,
            created_at=existing.created_at,
            trigger=existing.trigger,
            created=False,
        )

    checkpoint = ProjectCheckpointRow(
        project_id=project_id,
        version=row.version,
        data_gz=_encode(row.data),
        trigger=trigger.value,
    )
    try:
        # A SAVEPOINT, so a lost race rolls back only this INSERT. Without it
        # the IntegrityError would poison the caller's whole transaction on
        # PostgreSQL ("current transaction is aborted"), and an explicit
        # checkpoint request would take the rest of the request down with it.
        with session.begin_nested():
            session.add(checkpoint)
    except IntegrityError:
        winner = _find(session, project_id, row.version)
        if winner is None:  # pragma: no cover - only if the index went missing
            raise
        return CheckpointResult(
            version=winner.version,
            created_at=winner.created_at,
            trigger=winner.trigger,
            created=False,
        )
    return CheckpointResult(
        version=checkpoint.version,
        created_at=checkpoint.created_at,
        trigger=checkpoint.trigger,
        created=True,
    )


def _find(
    session: Session, project_id: str, version: int
) -> ProjectCheckpointRow | None:
    return session.execute(
        select(ProjectCheckpointRow).where(
            ProjectCheckpointRow.project_id == project_id,
            ProjectCheckpointRow.version == version,
        )
    ).scalar_one_or_none()


def list_checkpoints(
    session: Session, project_id: str, *, owner_id: int
) -> list[CheckpointEntry]:
    """This owner's checkpoints for one project, NEWEST FIRST — empty for a
    project that is not theirs.

    Newest first, unlike ``projects.get_history``'s oldest-first journal, and
    the difference is deliberate. The journal reads as a narrative: "here is
    everything that was done, in order". The version list is a PICK LIST — the
    operator opens it to get back to something recent — so it is ordered the
    way every other pick list in this application is (``list_project_records``,
    ``media_library.list_files``: "this owner's X, newest first").

    Each entry is joined to the journal by ``version_after`` so the list can say
    WHICH op produced the state that was frozen. The join is done in Python
    over one extra query rather than in SQL because a project can, in principle,
    carry two journal rows with the same ``version_after``, and an outer join
    would then duplicate the checkpoint row; a dict keyed by version cannot.
    ``op_name`` is None for a checkpoint whose journal row predates this table
    or was never written.
    """
    rows = list(
        session.execute(
            select(ProjectCheckpointRow)
            .where(
                ProjectCheckpointRow.project_id == project_id,
                _owned(project_id, owner_id),
            )
            .order_by(ProjectCheckpointRow.version.desc())
        ).scalars()
    )
    if not rows:
        return []
    ops = {
        op.version_after: op
        for op in session.execute(
            select(OperationLogRow).where(
                OperationLogRow.project_id == project_id,
                OperationLogRow.version_after.in_([r.version for r in rows]),
            )
        ).scalars()
    }
    return [
        CheckpointEntry(
            version=row.version,
            trigger=row.trigger,
            created_at=row.created_at,
            op_name=ops[row.version].op_name if row.version in ops else None,
            op_args=dict(ops[row.version].op_args or {}) if row.version in ops else {},
        )
        for row in rows
    ]


def get_checkpoint(
    session: Session, project_id: str, version: int, *, owner_id: int
) -> tuple[CheckpointEntry, dict] | None:
    """One checkpoint's metadata AND its decoded project document, or ``None``
    when the project is not this owner's or carries no checkpoint at that
    version. Both misses are the same ``None`` on purpose — the route maps
    either to 404, so neither can confirm the other's existence."""
    row = session.execute(
        select(ProjectCheckpointRow).where(
            ProjectCheckpointRow.project_id == project_id,
            ProjectCheckpointRow.version == version,
            _owned(project_id, owner_id),
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    op = session.execute(
        select(OperationLogRow).where(
            OperationLogRow.project_id == project_id,
            OperationLogRow.version_after == row.version,
        )
    ).scalars().first()
    entry = CheckpointEntry(
        version=row.version,
        trigger=row.trigger,
        created_at=row.created_at,
        op_name=op.op_name if op else None,
        op_args=dict(op.op_args or {}) if op else {},
    )
    return entry, _decode(row.data_gz)


def find_idle_project_ids(
    session: Session, *, idle_minutes: int, limit: int
) -> list[str]:
    """Project ids whose last mutation is older than ``idle_minutes`` and whose
    CURRENT mutation number has no checkpoint yet — the background sweep's work
    list.

    "Idle" is read from ``ProjectRow.updated_at``, which ``save_project``'s
    ``onupdate`` moves on every mutation and nothing else touches, so it is
    exactly "when was this project last edited". A project still being edited
    fails the age test and is left alone; that is the whole point of the
    trigger — a checkpoint marks where the editing came to REST.

    The NOT EXISTS is what makes the sweep converge: once a project has been
    checkpointed at its current version it drops out of this list and stays out
    until the next mutation. Without it the sweep would re-visit every idle
    project forever, and ``create_checkpoint``'s dedup — which would still hold
    the line — would be doing that work one wasted query at a time.

    ``limit`` bounds one pass so a backlog (a fleet that was down for a day)
    cannot turn a single sweep into an unbounded transaction; the remainder is
    picked up on the next interval, which is the same "TTL plus at most one
    interval" slack the retention sweep already accepts.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=idle_minutes)
    already = (
        select(ProjectCheckpointRow.id)
        .where(
            ProjectCheckpointRow.project_id == ProjectRow.project_id,
            ProjectCheckpointRow.version == ProjectRow.version,
        )
        .exists()
    )
    rows = session.execute(
        select(ProjectRow.project_id)
        .where(ProjectRow.updated_at < cutoff, ~already)
        .order_by(ProjectRow.updated_at)
        .limit(limit)
    ).scalars()
    return list(rows)


def owner_of(session: Session, project_id: str) -> int | None:
    """The owner id of a project, for the background sweep — the ONE caller
    that has no request principal to derive it from. Every request-time path
    takes ``owner_id`` from ``get_principal()`` instead; this is deliberately a
    separate, narrow function rather than an ``owner_id=None`` escape hatch on
    ``create_checkpoint``, which would be one forgotten argument away from
    letting a request checkpoint someone else's project."""
    return session.execute(
        select(ProjectRow.owner_id).where(ProjectRow.project_id == project_id)
    ).scalar_one_or_none()


def delete_for_projects(session: Session, project_ids) -> int:
    """Remove every checkpoint of the given projects, returning the row count.

    By predicate, like every other delete in the account cascade, so a second
    run matches nothing instead of failing — the idempotency
    ``delete_account`` requires. Called from there; there is no other project
    deletion path in this codebase (see that module).
    """
    ids = list(project_ids)
    if not ids:
        return 0
    return session.execute(
        delete(ProjectCheckpointRow).where(
            ProjectCheckpointRow.project_id.in_(ids)
        )
    ).rowcount
