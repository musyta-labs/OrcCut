"""In-process implementations of the shared-state primitives — Gate 3 Step 4.

The DEFAULT backend, and a faithful port of the logic the consumers carry today:
``InProcessSlidingWindowStore`` is ``ratelimit.SlidingWindowLimiter``'s
sliding-log algorithm behind the generic key/limit signature, and
``InProcessSlotStore`` is the ``dict[key, count]``-under-a-lock ceiling that
``FairScheduler`` and ``HeavyOpLimiter`` share — extended only with the optional
per-lease TTL the shared contract adds. On the single-replica SQLite path this
IS the truth: process-local counters are global counters, so behaviour is
unchanged bit-for-bit.

These are coordination primitives, not domain objects — they own mutable state
by design, all guarded by a single ``threading.Lock``; nothing they hand back
(``WindowVerdict``, ``SlotLease``) is mutable. The default clock is
``time.monotonic()`` (as ``SlidingWindowLimiter`` uses), which is immune to a
wall-clock step; callers/tests inject ``now`` for determinism.
"""
from __future__ import annotations

import math
import threading
import time
import uuid
from collections import deque

from app.common.shared_state import SlotLease, WindowUsage, WindowVerdict

# A denied caller is told to retry in at least this many seconds — never 0,
# which would read as "allowed" — matching SlidingWindowLimiter's floor.
_MIN_RETRY_AFTER_SECONDS = 1


def _monotonic(now: float | None) -> float:
    return time.monotonic() if now is None else now


class InProcessSlidingWindowStore:
    """Sliding-log rate limiter: ``max_hits`` per ``window_seconds`` per key,
    all state in this process's heap. Bit-for-bit with
    ``ratelimit.SlidingWindowLimiter.check``."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

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
        moment = _monotonic(now)
        cutoff = moment - window_seconds
        with self._lock:
            bucket = self._hits.get(key)
            if bucket is None:
                bucket = deque()
                self._hits[key] = bucket
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= max_hits:
                # The oldest surviving hit must age out before a slot frees;
                # that is the soonest a retry can succeed.
                retry_after = math.ceil(bucket[0] + window_seconds - moment)
                return WindowVerdict(
                    allowed=False,
                    retry_after_seconds=max(retry_after, _MIN_RETRY_AFTER_SECONDS),
                )
            bucket.append(moment)
            return WindowVerdict(allowed=True, retry_after_seconds=0)

    def usage(
        self, key: str, *, window_seconds: float, now: float | None = None
    ) -> WindowUsage:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        moment = _monotonic(now)
        cutoff = moment - window_seconds
        with self._lock:
            bucket = self._hits.get(key)
            # Read-only: expired timestamps are FILTERED for the answer, never
            # popped. Pruning here would make a meter refresh mutate the very
            # state it reports, which is the one thing this method promises not
            # to do (``hit`` prunes on the next real charge anyway).
            live = [stamp for stamp in bucket if stamp > cutoff] if bucket else []
        if not live:
            return WindowUsage(used=0, resets_in_seconds=0)
        return WindowUsage(
            used=len(live),
            resets_in_seconds=max(math.ceil(live[0] + window_seconds - moment), 0),
        )

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


class InProcessSlotStore:
    """Inc-with-limit slot counter per key, with optional per-lease TTL.

    The no-TTL path is ``HeavyOpLimiter``'s ``dict[key, count]`` ceiling exactly
    (acquire if ``count < limit``, release floored at zero). A TTL lease records
    its own deadline so it can self-expire, which needs the per-lease token the
    plain counter never had — so the store keeps, per key, a token->deadline map
    rather than a bare integer.
    """

    def __init__(self) -> None:
        # key -> {lease_token: expires_at_or_None}
        self._leases: dict[str, dict[str, float | None]] = {}
        self._lock = threading.Lock()

    def _prune_expired(self, key: str, moment: float) -> dict[str, float | None]:
        """Drop this key's leases whose TTL has passed and return the survivors.
        Caller holds the lock."""
        held = self._leases.get(key)
        if held is None:
            return {}
        live = {
            token: expires_at
            for token, expires_at in held.items()
            if expires_at is None or expires_at > moment
        }
        if live:
            self._leases[key] = live
        else:
            self._leases.pop(key, None)
        return live

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
        moment = _monotonic(now)
        with self._lock:
            live = self._prune_expired(key, moment)
            if len(live) >= limit:
                return None
            token = uuid.uuid4().hex
            expires_at = None if ttl_seconds is None else moment + ttl_seconds
            live[token] = expires_at
            self._leases[key] = live
            return SlotLease(key=key, token=token)

    def release(self, lease: SlotLease) -> None:
        with self._lock:
            held = self._leases.get(lease.key)
            if held is None:
                return
            held.pop(lease.token, None)  # idempotent: absent token is a no-op
            if not held:
                self._leases.pop(lease.key, None)

    def active_count(self, key: str, *, now: float | None = None) -> int:
        moment = _monotonic(now)
        with self._lock:
            return len(self._prune_expired(key, moment))

    def reset(self) -> None:
        with self._lock:
            self._leases.clear()
