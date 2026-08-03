"""Custom SQLAlchemy column types shared across ``app.db.models``.

``AwareDateTime`` exists because SQLite has no native timezone-aware storage:
even a plain ``DateTime(timezone=True)`` column returns a NAIVE
``datetime`` after a round trip through SQLite, while every write in this
codebase uses ``datetime.now(timezone.utc)``. Comparing that naive value
against a fresh aware ``datetime.now(timezone.utc)`` raises ``TypeError`` on
stdlib. This used to be patched ad hoc, per call site, inside
``app.db.repositories.tokens`` (``_as_aware_utc``) — a Gate 1 Step 3 -> 4
security-review finding flagged that as a bug generator: every future column
doing its own expiry/ordering comparison would need to remember to copy the
same guard, and forgetting it fails silently (either a ``TypeError``, or —
written carelessly — a comparison that quietly treats an expired row as
still valid). Centralizing the fix on the column type itself means every
value read through it is correct by construction, with no per-call-site
opt-in required.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime
from sqlalchemy.types import TypeDecorator


class AwareDateTime(TypeDecorator):
    """Same wire format as ``DateTime(timezone=True)`` — ``impl`` is exactly
    that type, so this has NO effect on the emitted DDL/schema and needs no
    Alembic migration of its own — but guarantees the Python value read back
    out is timezone-aware (UTC), regardless of what the underlying
    DBAPI/dialect hands back on read."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        """Reject naive values on write.

        Without this, a naive ``datetime`` would be stored as-is and then
        read back unconditionally stamped as UTC by
        ``process_result_value`` — silently mislabelling whatever it
        actually was. Every write in this codebase already uses
        ``datetime.now(timezone.utc)``, so this raises rather than guesses:
        an accidental naive write becomes an immediate error at the write,
        not wrong data surfacing at some later read.
        """
        if value is None or value.tzinfo is not None:
            return value
        raise ValueError(
            "AwareDateTime refuses naive datetimes — pass an aware value "
            "(datetime.now(timezone.utc)); storing naive here would be read "
            "back mislabelled as UTC."
        )

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None or value.tzinfo is not None:
            return value
        return value.replace(tzinfo=timezone.utc)
