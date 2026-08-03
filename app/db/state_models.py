"""Shared-state accounting tables — Gate 3 Step 4.

Two tables that back the "database" shared-state backend
(``app.common.shared_state_db``) so the sliding-window, inc-with-limit and
TTL-lease primitives are shared across every replica instead of each process
enforcing its own fraction of a limit. They are SEPARATE from the domain rows in
``app.db.models`` on purpose: this is coordination state (hot, churning, purely
mechanical), not persisted user data, and its own reason to change.

Both tables are pure additions — a CREATE TABLE on a live database that never
touches an existing table (see the Step 4 migration) — so the operator decision
"new tables are safe on a live DB, no blocking ALTER" holds. They exist only to
be created; the "memory" backend (the default, and the whole SQLite self-host
path) never reads or writes them.

**Time is stored as epoch seconds (a plain float), not a timestamp.** Every
expiry comparison here is a relative deadline evaluated against an injectable
``now`` — the same monotonic-or-wall clock both implementations share — and a
numeric column compares identically on SQLite and PostgreSQL with no timezone
round-trip (the very mismatch ``app.db.types.AwareDateTime`` exists to paper
over for the audit columns). ``created_at`` stays a real timestamp purely for a
human reading the row; nothing computes against it.

Kept in lockstep with the Step 4 migration (``migrations/versions`` —
``*_shared_state_tables``), exactly as ``QuotaRow`` is with its own revision.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SlotLeaseRow(Base):
    """One held slot for one key — the row form of an ``acquire``.

    A row exists for exactly as long as its lease is held: ``acquire`` inserts
    it (only when the live count for the key is under the limit), ``release``
    deletes it by ``lease_token``, and a lease with a non-null ``expires_at``
    is additionally reaped once ``now`` passes it, so a replica that crashed
    holding a slot cannot pin it forever. ``expires_at`` is NULL for a lease
    with no TTL (a scheduler ceiling slot that is only ever freed by an explicit
    release); it carries an epoch-seconds deadline for a TTL lease (a heavy-op
    slot).

    The live count for a key is ``COUNT(*)`` of its rows whose ``expires_at`` is
    NULL or still in the future — that count, guarded against the limit, is what
    ``acquire`` evaluates in the database.
    """

    __tablename__ = "state_slot_leases"

    id: Mapped[int] = mapped_column(primary_key=True)
    # The owner/tenant/token whose slots are being counted (e.g. an owner_id
    # rendered as text). A plain string so one table serves every keyspace.
    slot_key: Mapped[str] = mapped_column(String(255), index=True)
    # The opaque handle returned to the caller and required to release this one
    # lease — unique so a release targets exactly the row it acquired and a
    # double-release (idempotent) matches nothing the second time.
    lease_token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # Epoch-seconds deadline, or NULL for "never expires". A relative deadline
    # compared numerically against the shared ``now`` — see module docstring.
    expires_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class WindowHitRow(Base):
    """One recorded hit in a sliding window for one key.

    A hit is a lease nobody releases: it is recorded with an ``expires_at`` of
    ``hit_time + window`` and simply ages out. The window's live count is
    ``COUNT(*)`` of a key's rows still in the future; a hit is admitted only
    when that count is under the cap, and the soonest a throttled caller could
    succeed is the MIN of those future deadlines — the value the limiter reports
    as ``retry_after_seconds``.

    Separate from ``SlotLeaseRow`` despite the shared shape: a window hit has no
    token and no release path (its whole lifecycle is "expire"), so conflating
    the two would put a never-used ``lease_token`` on every hit and blur two
    distinct primitives.
    """

    __tablename__ = "state_window_hits"

    id: Mapped[int] = mapped_column(primary_key=True)
    window_key: Mapped[str] = mapped_column(String(255), index=True)
    # Epoch-seconds instant this hit ages out of its window (hit_time + window).
    expires_at: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
