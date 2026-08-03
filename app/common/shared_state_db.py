"""Database-backed implementations of the shared-state primitives — Gate 3
Step 4. Selected by ``SHARED_STATE_BACKEND=database`` for a multi-replica
Postgres deployment; on SQLite it is exercised only by the contract tests.

The whole point is that the count lives in ONE place every replica reads, so a
per-owner ceiling or a rate limit is global rather than "replicas × limit". The
tables (``app.db.state_models``) are shared; each operation is a short
transaction against the application database.

**Atomicity.** Every guard here is evaluated by the database together with the
write, never check-then-act in Python — the same discipline
``app.db.repositories.quotas.charge_storage`` documents. But unlike a quota (one
counter row an ``UPDATE`` can row-lock), a slot ceiling with TTL needs one row
PER held lease, so "is the live count under the limit" is a ``COUNT(*)`` guarding
an ``INSERT``, not a single-row update — and a bare count-then-insert races on
PostgreSQL's default READ COMMITTED (two acquires each counting ``limit-1`` and
both inserting). It is closed per key:

* **PostgreSQL** — a transaction-scoped ``pg_advisory_xact_lock(hashtext(key))``
  serialises acquire/hit for THAT key only (cross-key traffic still runs in
  parallel), released automatically at commit. This is the same advisory-lock
  primitive ``app.db.base`` already uses to serialise startup migrations.
* **SQLite** — no advisory lock and none needed: SQLite serialises writers
  globally, so a losing acquire sees ``database is locked`` and the caller
  retries (exactly as ``tests/test_quota_concurrency`` does), re-reading the now
  committed count and correctly refusing. The task's "SQLite serialisation saves
  the contract" rests on this.

``release`` and the count reads need no lock: a ``DELETE`` by unique
``lease_token`` is atomic and idempotent on its own, and a read is a read.

**Time is epoch seconds** (see ``app.db.state_models``): a plain float compared
numerically, so ``now`` stays injectable and identical across SQLite/PostgreSQL.
The default clock is wall time (``time.time()``) — NOT monotonic like the
in-process store, because monotonic clocks are not comparable across the
processes that share these rows.
"""
from __future__ import annotations

import math
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import delete, func, insert, or_, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.common.shared_state import SlotLease, WindowUsage, WindowVerdict
from app.db.state_models import SlotLeaseRow, WindowHitRow

# A denied caller retries in at least this many seconds — never 0, which reads
# as "allowed" — matching the in-process store and SlidingWindowLimiter.
_MIN_RETRY_AFTER_SECONDS = 1

# Dialect names for which a per-key advisory lock is issued. SQLite is absent on
# purpose: it serialises writers globally, so the guard is already atomic there.
_POSTGRES_DIALECTS = frozenset({"postgresql", "postgres"})


def _wall(now: float | None) -> float:
    return time.time() if now is None else now


def _default_session_factory() -> sessionmaker[Session]:
    # Imported lazily so selecting the "memory" backend never constructs the
    # application engine — a SQLite self-host that never sets this backend must
    # not pay for a database handle it does not use.
    from app.db.base import get_session_factory

    return get_session_factory()


class _DatabaseStoreBase:
    """Shared session handling and the per-key advisory lock for both stores."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        # Injectable so the contract tests point the store at a throwaway SQLite
        # database instead of the process's real engine.
        self._session_factory = session_factory or _default_session_factory()

    @contextmanager
    def _session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _lock_key(self, session: Session, key: str) -> None:
        """Serialise acquire/hit for ``key`` on PostgreSQL; a no-op on SQLite.

        Transaction-scoped, so it is held from here until the enclosing
        ``_session`` commits — covering the count-then-insert as one atomic unit
        — and released at commit without an explicit unlock.
        """
        if session.bind.dialect.name in _POSTGRES_DIALECTS:
            session.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:k))"), {"k": key}
            )


class DatabaseSlidingWindowStore(_DatabaseStoreBase):
    """Sliding-log rate limiter backed by ``state_window_hits``."""

    def hit(
        self,
        key: str,
        *,
        max_hits: int,
        window_seconds: float,
        now: float | None = None,
    ) -> WindowVerdict:
        if max_hits < 1:
            raise ValueError("max_hits must be at least 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        moment = _wall(now)
        with self._session() as session:
            self._lock_key(session, key)
            # Age out hits older than the window, then read the survivors: after
            # this delete every remaining row for the key is still in-window.
            session.execute(
                delete(WindowHitRow).where(
                    WindowHitRow.window_key == key,
                    WindowHitRow.expires_at <= moment,
                )
            )
            live_expiries = (
                session.execute(
                    select(WindowHitRow.expires_at)
                    .where(WindowHitRow.window_key == key)
                    .order_by(WindowHitRow.expires_at)
                )
                .scalars()
                .all()
            )
            if len(live_expiries) >= max_hits:
                # The soonest deadline is the oldest hit's expiry — the earliest
                # a slot frees.
                retry_after = math.ceil(live_expiries[0] - moment)
                return WindowVerdict(
                    allowed=False,
                    retry_after_seconds=max(retry_after, _MIN_RETRY_AFTER_SECONDS),
                )
            session.execute(
                insert(WindowHitRow).values(
                    window_key=key, expires_at=moment + window_seconds
                )
            )
            return WindowVerdict(allowed=True, retry_after_seconds=0)

    def usage(
        self, key: str, *, window_seconds: float, now: float | None = None
    ) -> WindowUsage:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        moment = _wall(now)
        with self._session() as session:
            # No advisory lock and no DELETE: this is a pure read, so expired
            # rows are FILTERED OUT of the answer rather than reaped. Skipping
            # the lock is what keeps a meter refresh — which the SPA polls —
            # from serialising against the charges it reports on.
            live_expiries = (
                session.execute(
                    select(WindowHitRow.expires_at)
                    .where(
                        WindowHitRow.window_key == key,
                        WindowHitRow.expires_at > moment,
                    )
                    .order_by(WindowHitRow.expires_at)
                )
                .scalars()
                .all()
            )
        if not live_expiries:
            return WindowUsage(used=0, resets_in_seconds=0)
        return WindowUsage(
            used=len(live_expiries),
            resets_in_seconds=max(math.ceil(live_expiries[0] - moment), 0),
        )

    def reset(self) -> None:
        with self._session() as session:
            session.execute(delete(WindowHitRow))


class DatabaseSlotStore(_DatabaseStoreBase):
    """Inc-with-limit slot counter (optional per-lease TTL) backed by
    ``state_slot_leases``."""

    @staticmethod
    def _live_predicate(moment: float):
        """A lease is live when it has no TTL or its TTL is still in the
        future."""
        return or_(
            SlotLeaseRow.expires_at.is_(None),
            SlotLeaseRow.expires_at > moment,
        )

    def acquire(
        self,
        key: str,
        *,
        limit: int,
        ttl_seconds: float | None = None,
        now: float | None = None,
    ) -> SlotLease | None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        moment = _wall(now)
        with self._session() as session:
            self._lock_key(session, key)
            # Reap this key's expired leases so a crashed holder's slot frees.
            session.execute(
                delete(SlotLeaseRow).where(
                    SlotLeaseRow.slot_key == key,
                    SlotLeaseRow.expires_at.is_not(None),
                    SlotLeaseRow.expires_at <= moment,
                )
            )
            live = session.execute(
                select(func.count())
                .select_from(SlotLeaseRow)
                .where(SlotLeaseRow.slot_key == key, self._live_predicate(moment))
            ).scalar_one()
            if live >= limit:
                return None
            token = uuid.uuid4().hex
            expires_at = None if ttl_seconds is None else moment + ttl_seconds
            session.execute(
                insert(SlotLeaseRow).values(
                    slot_key=key, lease_token=token, expires_at=expires_at
                )
            )
            return SlotLease(key=key, token=token)

    def release(self, lease: SlotLease) -> None:
        # A DELETE by the unique token is atomic and idempotent on its own — no
        # lock, and a second release simply matches zero rows.
        with self._session() as session:
            session.execute(
                delete(SlotLeaseRow).where(SlotLeaseRow.lease_token == lease.token)
            )

    def active_count(self, key: str, *, now: float | None = None) -> int:
        moment = _wall(now)
        with self._session() as session:
            return session.execute(
                select(func.count())
                .select_from(SlotLeaseRow)
                .where(SlotLeaseRow.slot_key == key, self._live_predicate(moment))
            ).scalar_one()

    def reset(self) -> None:
        with self._session() as session:
            session.execute(delete(SlotLeaseRow))
