"""Persistence for clip event detection results, keyed by media content.

Business logic goes through these helpers, not raw sessions (repository
pattern, as with ``projects``/``annotations``). Every function is a thin,
synchronous SQLite operation — the expensive part (decode + ffmpeg) happens in
``app.analysis``, deliberately with NO session held.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Callable

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.db.models import AnalysisAccessRow, MediaAnalysisRow


def _insert_for_dialect(dialect_name: str) -> Callable:
    """The ``INSERT`` construct whose ``ON CONFLICT`` clause this backend
    understands.

    ``sqlalchemy.dialects.sqlite.insert`` and
    ``sqlalchemy.dialects.postgresql.insert`` are DIFFERENT constructs — the
    upserts below (``on_conflict_do_update`` / ``on_conflict_do_nothing``) only
    compile to correct SQL when the construct matches the connection's dialect.
    Anything that is not PostgreSQL falls back to the SQLite construct, which is
    the portable self-host default and keeps an unrecognised backend from
    crashing the save path outright."""
    if dialect_name == "postgresql":
        return postgresql_insert
    return sqlite_insert


def _insert_for(session: Session) -> Callable:
    """``_insert_for_dialect`` resolved against the session's OWN bind, so the
    same repository code upserts correctly on either backend."""
    return _insert_for_dialect(session.get_bind().dialect.name)


def get_current(session: Session, content_hash: str, *, version: int) -> MediaAnalysisRow | None:
    """The stored analysis for this media, but ONLY when it was computed under
    exactly ``version``.

    Mismatch in either direction returns ``None``: an older row predates a
    contract change (today's VFR decode fix and the v2 solo-channel rule each
    silently invalidate every v1 result), and a NEWER row means we have rolled
    back — reading a future contract would hand callers fields this build does
    not understand. Both cases mean "recompute", never "return anyway".
    """
    row = session.execute(
        select(MediaAnalysisRow).where(MediaAnalysisRow.content_hash == content_hash)
    ).scalar_one_or_none()
    if row is None or row.version != version:
        return None
    return row


def save(
    session: Session,
    content_hash: str,
    *,
    version: int,
    data: dict,
    analysis_id: str | None = None,
) -> MediaAnalysisRow:
    """Upsert the analysis for this media — one row per ``content_hash``,
    always. A re-analysis SUPERSEDES the previous result in place; it does not
    append a second row.

    Written as a real ``ON CONFLICT`` upsert rather than read-then-write.
    ``content_hash`` is UNIQUE, so two tenants analysing the same bytes
    concurrently would both see "no row" and both INSERT, and the loser would
    take an IntegrityError instead of merging. The in-process lock in
    ``app.analysis.service`` does not cover that: it is per-process, and
    Gate 1's whole point is that there is more than one caller.
    """
    values = {
        "content_hash": content_hash,
        "version": version,
        "data": data,
        "analysis_id": analysis_id,
    }
    insert = _insert_for(session)
    session.execute(
        insert(MediaAnalysisRow)
        .values(**values)
        .on_conflict_do_update(
            index_elements=[MediaAnalysisRow.content_hash],
            set_={
                "version": version,
                "data": data,
                "analysis_id": analysis_id,
                "updated_at": func.now(),
            },
        )
    )
    session.flush()
    # populate_existing: the UPDATE ran at Core level, but if this session had
    # already loaded the row (a re-analysis right after a read), its identity
    # map still holds the pre-update values. Force the ORM to overwrite them
    # from the row this statement just wrote, or callers reading back through
    # the same session get a stale ``analysis_id``/``data``.
    return session.execute(
        select(MediaAnalysisRow)
        .where(MediaAnalysisRow.content_hash == content_hash)
        .execution_options(populate_existing=True)
    ).scalar_one()


def grant_access(session: Session, *, user_id: int, content_hash: str) -> None:
    """Record that this tenant has legitimately seen these bytes.

    Idempotent by construction: called on EVERY analysis, cache hit included,
    so it runs far more often than it inserts. ``ON CONFLICT DO NOTHING``
    against the ``(user_id, content_hash)`` unique constraint makes the repeat
    call free instead of a caught exception, and makes two concurrent first
    calls safe.
    """
    insert = _insert_for(session)
    session.execute(
        insert(AnalysisAccessRow)
        .values(user_id=user_id, content_hash=content_hash)
        .on_conflict_do_nothing(index_elements=["user_id", "content_hash"])
    )


def can_access_artifact(session: Session, *, user_id: int, filename: str) -> bool:
    """May this tenant fetch the keyframe strip named ``filename``?

    Strips are named ``{analysis_id}_...``, and ``analysis_id`` is stored on
    the analysis row, so the file's own name identifies the media it came from
    — no payload parsing, no second name->hash table. The question then
    reduces to: is there an access row for this tenant against the analysis
    that minted this id?

    A name with no ``_``, an unknown id, or a row whose ``analysis_id`` was
    never recorded (written before Step 10 and not recoverable from its
    payload) all return False. Failing closed is right here: the caller maps
    it to 404, and the cost of a wrong False is one recomputed analysis.
    """
    analysis_id = filename.split("_", 1)[0]
    if not analysis_id or "_" not in filename:
        return False
    return session.execute(
        select(AnalysisAccessRow.id)
        .join(
            MediaAnalysisRow,
            MediaAnalysisRow.content_hash == AnalysisAccessRow.content_hash,
        )
        .where(
            AnalysisAccessRow.user_id == user_id,
            MediaAnalysisRow.analysis_id == analysis_id,
        )
        .limit(1)
    ).first() is not None


def count_all(session: Session) -> int:
    """Total stored analyses — used by tests and the maintenance CLI's report."""
    return session.execute(select(func.count(MediaAnalysisRow.id))).scalar_one()


def keyframe_name(entry: str) -> str:
    """One stored ``keyframes`` entry reduced to its basename.

    Reads BOTH shapes on purpose. Records written before Gate 1 Step 10 hold
    absolute ``/ui/media/analysis/x_overview.png`` URLs; new ones hold bare
    ``x_overview.png`` names, so that storage no longer hard-codes a URL
    layout the serving route is free to change. Normalising on read means the
    split needed no ``ANALYSIS_VERSION`` bump and therefore no forced
    recompute of every stored analysis.
    """
    return PurePosixPath(entry).name


def keyframe_names(data: dict | None) -> list[str]:
    """Basenames of the strips one payload names, in order.

    Tolerant by construction: a missing ``keyframes`` key, a null, a non-list,
    or a non-string entry contributes nothing rather than raising. Both
    callers — the cleanup scan and the read API — are better off skipping one
    malformed record than failing wholesale.
    """
    keyframes = (data or {}).get("keyframes")
    if not isinstance(keyframes, list):
        return []
    return [keyframe_name(e) for e in keyframes if isinstance(e, str) and e]


def referenced_keyframe_names(session: Session) -> set[str]:
    """Basenames of every keyframe strip that ANY tenant's stored analysis
    still points at.

    DELIBERATELY GLOBAL, and this is a correctness requirement rather than an
    oversight (Gate 1 Step 10). Its one caller is the operator's disk-cleanup
    CLI, which prunes a media directory SHARED by every tenant. Scoping this
    to the tenant running the cleanup would make the sweep delete strips that
    another tenant's live record still references — the exact broken-image
    failure the protection exists to prevent. The reads elsewhere in this
    module are tenant-scoped; this one must not be.

    See ``app.mcp.maintenance_cli``: that script is not an MCP tool and is not
    protected by Step 11's tool-dictionary filter. What bounds it is process
    control — who may execute it in the container — not a row predicate.
    """
    names: set[str] = set()
    for row in session.execute(select(MediaAnalysisRow)).scalars():
        names.update(keyframe_names(row.data))
    return names
