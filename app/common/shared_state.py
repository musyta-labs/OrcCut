"""Shared-state accounting primitives — Gate 3 Step 4 (the multi-replica
foundation Steps 5-7 build on).

Three coordination needs recur across this codebase, each today a per-process
counter that under-counts the moment a second replica exists (every one carries
a ``GATE 3 DEBT`` note saying so):

* a **sliding window** — "``max_hits`` per ``window_seconds`` per key" — behind
  ``app.common.ratelimit.SlidingWindowLimiter``'s six rate limiters;
* an **inc-with-limit slot** — "hold at most ``limit`` slots for this key,
  acquire/release" — behind ``app.analysis.scheduler.FairScheduler``'s per-owner
  ceiling and ``app.web.concurrency.HeavyOpLimiter``'s heavy-op ceiling;
* the **same slot with a TTL** — so a replica that crashes holding one does not
  pin it forever — which is what ``HeavyOpLimiter`` specifically wants across
  replicas.

This module is the seam. It defines two small protocols — ``SlidingWindowStore``
and ``SlotStore`` (the TTL slot is the plain slot with an optional
``ttl_seconds``) — and hands out ONE implementation of each, chosen from config.
The default is ``memory`` (``app.common.shared_state_memory``): the exact
in-process logic those consumers have today, so the SQLite self-host /
single-replica path is unchanged bit-for-bit. A Postgres multi-replica
deployment selects ``database`` (``app.common.shared_state_db``), which backs
the same contract with atomic SQL against shared tables.

This step is deliberately additive: it introduces the seam and both
implementations, and CHANGES NO consumer. Steps 5-7 migrate the limiters,
scheduler and heavy-op limiter onto these stores one at a time.

**The clock is injectable.** Every method takes ``now`` (epoch/monotonic
seconds); a consumer or test passes it to drive windows and expiries
deterministically instead of sleeping, mirroring ``SlidingWindowLimiter.check``.
When omitted, each implementation supplies its own default clock — the two
implementations therefore need not share a clock SOURCE, only its meaning
(seconds), because every call within one store reads that store's own clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from app.config import get_settings

# Backend identifiers (mirror app.config.SHARED_STATE_BACKENDS). Kept here too
# so the factory dispatch reads against named constants, not string literals.
MEMORY_BACKEND = "memory"
DATABASE_BACKEND = "database"


@dataclass(frozen=True)
class WindowVerdict:
    """The immutable verdict for one ``SlidingWindowStore.hit`` call.

    ``retry_after_seconds`` is 0 when allowed, and otherwise the whole number of
    seconds until the oldest in-window hit ages out — the soonest the caller
    could succeed, and the value for an HTTP ``Retry-After`` header. Structurally
    identical to ``ratelimit.RateLimitDecision`` (Step 5 maps one onto the
    other); defined here so this module carries no import back to a consumer.
    """

    allowed: bool
    retry_after_seconds: int


@dataclass(frozen=True)
class WindowUsage:
    """A READ-ONLY account of one key's window — the sliding-log twin of
    ``SlotStore.active_count``, and the reason it exists: a budget a user is
    shown (G1-01's daily quota meter) must be readable without spending a unit
    of it, which ``hit`` cannot do by construction.

    ``resets_in_seconds`` is when the OLDEST in-window hit ages out, i.e. when
    the next unit frees — 0 when nothing is recorded. Deliberately the same
    number ``hit`` would return as ``retry_after_seconds`` at the cap, so a
    refusal and the meter beside it can never disagree about the wait.
    """

    used: int
    resets_in_seconds: int


@dataclass(frozen=True)
class SlotLease:
    """A held slot. Returned by ``SlotStore.acquire`` on success and handed back
    to ``release`` to free exactly that slot. Opaque to the caller — the
    ``token`` is the store's business — and immutable so it cannot be mutated
    between acquire and release."""

    key: str
    token: str


class SlidingWindowStore(Protocol):
    """A sliding-log rate limiter: at most ``max_hits`` per ``window_seconds``
    per key, exact (no fixed-window boundary burst)."""

    def hit(
        self,
        key: str,
        *,
        max_hits: int,
        window_seconds: float,
        now: float | None = None,
    ) -> WindowVerdict:
        """Record-and-verdict for one hit on ``key``. TESTS and, when it
        allows, RECORDS the hit — call exactly once per request meant to count.
        A denied hit is NOT recorded, so a throttled caller cannot push its own
        window further out by retrying."""
        ...

    def usage(
        self, key: str, *, window_seconds: float, now: float | None = None
    ) -> WindowUsage:
        """How much of ``key``'s window is spent, WITHOUT recording anything.
        A pure read: calling it a hundred times leaves the budget where it
        was."""
        ...

    def reset(self) -> None:
        """Drop all recorded hits. Test isolation only."""
        ...


class SlotStore(Protocol):
    """An inc-with-limit slot counter per key, with optional per-lease TTL."""

    def acquire(
        self,
        key: str,
        *,
        limit: int,
        ttl_seconds: float | None = None,
        now: float | None = None,
    ) -> SlotLease | None:
        """Take one slot for ``key`` if fewer than ``limit`` are live, else
        refuse. Returns the ``SlotLease`` to release on success, or ``None`` when
        the key is already at ``limit``. The check and the take are ONE atomic
        step, so concurrent acquires at the boundary can never both succeed.
        ``ttl_seconds`` (when set) makes the slot self-expire after that many
        seconds so a holder that never releases — a crashed replica — cannot pin
        it forever; ``None`` means the slot is freed only by ``release``."""
        ...

    def release(self, lease: SlotLease) -> None:
        """Free the slot ``acquire`` returned. Idempotent: releasing a lease
        that was never held, or twice, is a no-op and never drives the count
        negative to hand out a phantom slot."""
        ...

    def active_count(self, key: str, *, now: float | None = None) -> int:
        """Live (unexpired, unreleased) slots held for ``key``. Introspection /
        tests."""
        ...

    def reset(self) -> None:
        """Drop all slots. Test isolation only."""
        ...


@lru_cache(maxsize=1)
def get_sliding_window_store() -> SlidingWindowStore:
    """The process-wide sliding-window store selected by
    ``Settings.shared_state_backend`` (``memory`` by default)."""
    return _build_sliding_window_store(get_settings().shared_state_backend)


@lru_cache(maxsize=1)
def get_slot_store() -> SlotStore:
    """The process-wide slot store selected by
    ``Settings.shared_state_backend`` (``memory`` by default)."""
    return _build_slot_store(get_settings().shared_state_backend)


def _build_sliding_window_store(backend: str) -> SlidingWindowStore:
    if backend == MEMORY_BACKEND:
        from app.common.shared_state_memory import InProcessSlidingWindowStore

        return InProcessSlidingWindowStore()
    from app.common.shared_state_db import DatabaseSlidingWindowStore

    return DatabaseSlidingWindowStore()


def _build_slot_store(backend: str) -> SlotStore:
    if backend == MEMORY_BACKEND:
        from app.common.shared_state_memory import InProcessSlotStore

        return InProcessSlotStore()
    from app.common.shared_state_db import DatabaseSlotStore

    return DatabaseSlotStore()
