"""In-process sliding-window rate limiting — Gate 2 Step 5 (extended by Steps 8 and 7).

Six request classes, six keys, six budgets:

* ``LOGIN_LIMITER`` — keyed by client IP, guards ``/auth/login`` BEFORE any
  identity exists (the only defence there today is argon2 cost + a
  constant-time verify; see ``app.web.auth``).
* ``OWNER_LIMITER`` — keyed by ``owner_id`` (the resolved ``Principal.user_id``),
  shared by ``/api`` and ``/ui`` where an identity already exists post-Gate-1.
* ``MCP_LIMITER`` — keyed by ``token_id``, NOT by ``owner_id``. An agent is the
  heaviest client and hammers in a loop; one runaway token must not exhaust the
  budget of another token of the SAME owner (Gate-1 finding "Р10"). Keying on
  the token is the whole point of this limiter's existence.
* ``PASSWORD_RESET_LIMITER`` — keyed by client IP, guards
  ``/auth/password-reset/request`` the same way ``LOGIN_LIMITER`` guards
  login: no identity exists yet, so IP is the only key available (see
  ``app.web.ratelimit``).
* ``REGISTER_LIMITER`` — keyed by client IP, guards ``/auth/register``
  (Gate 2 Step 7) exactly as ``PASSWORD_RESET_LIMITER`` guards a reset
  request: no identity exists yet, and every hit for a new address mints an
  account and places an outbound confirmation email, so IP-throttling is what
  stops the surface being used to mass-create accounts.
* ``EMAIL_LIMITER`` — keyed by RECIPIENT address, not by caller. It exists to
  cap how many emails one address can be sent within a window, so a caller
  who can trigger outbound mail (password reset today, Step 7's registration
  confirmation next) cannot use this server to bombard a third party's inbox
  — a concern orthogonal to, and not covered by, ``PASSWORD_RESET_LIMITER``
  (which throttles the CALLER, not the recipient: one caller retrying from
  many IPs, or many callers naming the same victim address, both bypass an
  IP-keyed limit but not this one).

**Sliding-log algorithm.** Each key holds the timestamps of its recent hits;
timestamps older than the window are dropped, and a hit is denied when the
surviving count has already reached the cap. This is exact (no fixed-window
burst doubling at the boundary) and memory is bounded by the cap per active key.

**Two-mode counting (Gate 3 Step 5).** The sliding-log state no longer lives in
this class; ``SlidingWindowLimiter`` is now a thin cap/window/key adapter over
the shared ``app.common.shared_state.SlidingWindowStore`` chosen by
``Settings.shared_state_backend``:

* ``memory`` (the default) keeps every counter in this process's heap, exactly
  as before — the single-replica SQLite self-host is unchanged bit-for-bit.
* ``database`` backs the same counters with atomic SQL against a shared table,
  so N replicas behind a load balancer enforce ONE global ``cap`` instead of
  the old ``replicas × cap``. The atomicity (a per-key advisory lock on
  Postgres, writer serialisation on SQLite) is guaranteed by the store layer;
  this class adds none of its own.

Each limiter namespaces its keys with its own ``name`` before handing them to
the one process-wide store, so the six request classes never collide on a value
they happen to share (a client IP keys LOGIN, PASSWORD_RESET and REGISTER
alike). The namespace is stable across replicas, which is what lets the
``database`` backend sum a key's hits across processes.

Statefulness note: a rate limiter accumulates state by definition, but that
state now lives in the store; everything ``SlidingWindowLimiter`` hands back
(``RateLimitDecision``) is frozen and immutable.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.common.shared_state import (
    SlidingWindowStore,
    WindowUsage,
    get_sliding_window_store,
)

# --- budgets (cap, window) per request class -------------------------------
# Windows are in seconds. Caps are per-key hits allowed within one window.
LOGIN_MAX_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 60

# /api + /ui share one per-owner budget: a human clicks, so this is generous
# next to the login cap while still bounding a stuck client that retries a GET
# in a tight loop.
OWNER_MAX_REQUESTS = 240
OWNER_WINDOW_SECONDS = 60

# MCP is the agent surface — a higher ceiling because an agent legitimately
# fires many tool calls in a burst, but still bounded PER TOKEN.
MCP_MAX_CALLS = 120
MCP_WINDOW_SECONDS = 60

# Password-reset requests, per client IP. Tighter than LOGIN's cap: a login
# attempt costs the caller nothing but a guess, while a reset request costs a
# real side effect (a minted token, an email) for every request that names a
# real account — the smaller budget reflects that a legitimate caller almost
# never needs more than one or two of these per minute.
PASSWORD_RESET_MAX_ATTEMPTS = 5
PASSWORD_RESET_WINDOW_SECONDS = 60

# Registration requests, per client IP. Guards ``/auth/register`` the same way
# ``PASSWORD_RESET_LIMITER`` guards a reset request: no identity exists yet, so
# IP is the only key available, and every hit for a new address costs a real
# side effect (a minted account, a quota row, an outbound confirmation email).
# Sized like the reset budget — a legitimate visitor registers once, not in a
# loop — so this surface cannot be used to mass-create accounts.
REGISTER_MAX_ATTEMPTS = 5
REGISTER_WINDOW_SECONDS = 60

# Outbound email, per RECIPIENT address (not per caller — see module
# docstring). Generous enough that a real user who fat-fingers "resend" a
# couple of times is never affected, small enough that this server cannot be
# turned into a mail bomb against one address.
EMAIL_MAX_PER_RECIPIENT = 3
EMAIL_WINDOW_SECONDS = 3600


@dataclass(frozen=True)
class RateLimitDecision:
    """The immutable verdict for one ``check`` call.

    ``retry_after_seconds`` is 0 when allowed, and otherwise the whole number
    of seconds until the oldest in-window hit ages out — i.e. the soonest the
    caller could succeed. It is the value for an HTTP ``Retry-After`` header.
    """

    allowed: bool
    retry_after_seconds: int


class SlidingWindowLimiter:
    """A sliding-log limiter: ``max_hits`` per ``window_seconds`` per key.

    ``check`` both TESTS and RECORDS a hit when it allows one — callers must
    invoke it exactly once per request they mean to count. A denied request is
    NOT recorded (a caller being throttled cannot push its own retry window
    further out just by retrying).

    The counting itself is delegated to a shared ``SlidingWindowStore`` (the
    process-wide one chosen by config, unless one is injected), so the same
    limiter enforces a process-local cap on the ``memory`` backend and a global
    cross-replica cap on the ``database`` backend. ``name`` namespaces this
    limiter's keys inside that shared store: pass a stable name for the
    module-level singletons so their namespace is identical on every replica;
    omit it (tests constructing throwaway limiters) to get a per-instance
    namespace that cannot collide with any other limiter's keys.
    """

    def __init__(
        self,
        *,
        max_hits: int,
        window_seconds: float,
        name: str | None = None,
        store: SlidingWindowStore | None = None,
    ) -> None:
        if max_hits < 1:
            raise ValueError("max_hits must be at least 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._max_hits = max_hits
        self._window = window_seconds
        # A stable name shares a namespace across replicas (the point of the
        # database backend); an anonymous limiter gets a unique one so two
        # throwaway instances on the shared store never bleed into each other.
        self._namespace = name if name is not None else uuid.uuid4().hex
        self._store = store if store is not None else get_sliding_window_store()

    def _namespaced(self, key: str) -> str:
        # ASCII Unit Separator (0x1f) splits namespace from key: no client IP,
        # owner id, email or token contains it, so distinct (namespace, key)
        # pairs never alias. NOT NUL (0x00): the database backend stores this
        # string in a PostgreSQL text column, and Postgres rejects NUL bytes
        # in text outright — the memory backend hid that until the first
        # SHARED_STATE_BACKEND=database deployment 500'd on every login.
        return f"{self._namespace}\x1f{key}"

    def check(self, key: str, *, now: float | None = None) -> RateLimitDecision:
        """Record-and-verdict for ``key``. ``now`` is injectable (seconds) so
        tests drive the window deterministically instead of sleeping; when
        omitted the backing store supplies its own clock (monotonic in-process,
        wall time on the shared database)."""
        verdict = self._store.hit(
            self._namespaced(key),
            max_hits=self._max_hits,
            window_seconds=self._window,
            now=now,
        )
        return RateLimitDecision(
            allowed=verdict.allowed,
            retry_after_seconds=verdict.retry_after_seconds,
        )

    def usage(self, key: str, *, now: float | None = None) -> WindowUsage:
        """How much of ``key``'s budget is spent, spending nothing.

        The read half of ``check``, added for G1-01: a DAILY budget is a number
        a user is shown and an agent is told, and neither can be served by a
        primitive that only speaks by consuming. Production callers of the
        minute-scale limiters have no use for it — nobody displays "217 of 240
        requests this minute" — so it stays a plain read rather than being
        folded into ``check``'s return.
        """
        return self._store.usage(
            self._namespaced(key), window_seconds=self._window, now=now
        )

    def reset(self) -> None:
        """Drop all recorded hits. Used by the test suite between cases so the
        module-level singletons do not leak counts across tests. Test-only: the
        shared store's ``reset`` clears every namespace, which is exactly what
        ``reset_all_limiters`` wants and what production never calls."""
        self._store.reset()


# Module-level singletons: ONE limiter per request class for the whole process.
# Consumers reference these by attribute at call time (``ratelimit.MCP_LIMITER``)
# so a test can swap in a small-cap instance without rebuilding the app.
LOGIN_LIMITER = SlidingWindowLimiter(
    max_hits=LOGIN_MAX_ATTEMPTS, window_seconds=LOGIN_WINDOW_SECONDS, name="login"
)
OWNER_LIMITER = SlidingWindowLimiter(
    max_hits=OWNER_MAX_REQUESTS, window_seconds=OWNER_WINDOW_SECONDS, name="owner"
)
MCP_LIMITER = SlidingWindowLimiter(
    max_hits=MCP_MAX_CALLS, window_seconds=MCP_WINDOW_SECONDS, name="mcp"
)
PASSWORD_RESET_LIMITER = SlidingWindowLimiter(
    max_hits=PASSWORD_RESET_MAX_ATTEMPTS,
    window_seconds=PASSWORD_RESET_WINDOW_SECONDS,
    name="password-reset",
)
REGISTER_LIMITER = SlidingWindowLimiter(
    max_hits=REGISTER_MAX_ATTEMPTS, window_seconds=REGISTER_WINDOW_SECONDS, name="register"
)
EMAIL_LIMITER = SlidingWindowLimiter(
    max_hits=EMAIL_MAX_PER_RECIPIENT, window_seconds=EMAIL_WINDOW_SECONDS, name="email"
)


def reset_all_limiters() -> None:
    """Reset every module-level limiter. For test isolation only — the process
    never resets its own counters in production."""
    LOGIN_LIMITER.reset()
    OWNER_LIMITER.reset()
    MCP_LIMITER.reset()
    PASSWORD_RESET_LIMITER.reset()
    REGISTER_LIMITER.reset()
    EMAIL_LIMITER.reset()
