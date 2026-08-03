"""The resolved identity of the caller making a request — the SAME type
whether resolved from an MCP bearer token (``app.auth.resolve.resolve_bearer``)
or a browser cookie session (``app.auth.resolve.resolve_session``). See
``app.auth.resolve`` for how a ``Principal`` gets constructed and
``app.auth.sessions`` for the cookie format.

Gate 1 Step 4: this module is purely additive. No request-handling code
calls ``get_principal()`` yet — that wiring is Steps 5 (MCP resource server)
and 6 (``/api`` auth).
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    """One resolved, authenticated caller.

    ``user_id`` is the only identity that matters for tenant isolation
    (Gate 1 Step 7's ``owner_id`` predicate) — resolved identically whether
    the request came in on a bearer token or a cookie session; see
    ``app.auth.resolve`` for both paths converging on the same value for the
    same human.

    ``token_id`` distinguishes HOW this ``Principal`` was resolved, without
    a second type: a bearer-token ``Principal`` carries the token's row id
    (so a future revoke-endpoint can scope a revocation to it — Step 6+); a
    cookie-session ``Principal`` has ``token_id=None``, since a browser
    session is not a row in ``api_tokens`` at all.

    ``scopes`` is only meaningful when ``token_id`` is not ``None``: it is
    whatever the caller narrowed the token to at mint time
    (``app.db.repositories.tokens.mint_token``), and Step 11's trust tiers
    gate the MCP bearer entry point on it. A session-resolved ``Principal``
    (``token_id is None``) is the full, unrestricted account owner acting
    through the browser UI — its ``scopes`` is always ``()``, and that empty
    tuple must NOT be read as "no access"; callers branch on ``token_id``
    first, exactly like this codebase's own repositories already branch on
    "is this id None" rather than overloading a single field's meaning.
    """

    user_id: int
    token_id: int | None
    scopes: tuple[str, ...]


current_principal: ContextVar["Principal | None"] = ContextVar("current_principal", default=None)


class PrincipalNotBoundError(RuntimeError):
    """Raised by ``get_principal()`` when nothing has bound a ``Principal``
    to the current execution context — i.e. request-handling code (Steps
    5/6) reached business logic without first calling ``bind_principal``.
    This is a bug in the entry path, not "an unauthenticated caller": an
    unauthenticated request is rejected before it would ever reach code that
    calls ``get_principal()``."""


def get_principal() -> Principal:
    """The ``Principal`` bound to the current context, or raise.

    There is deliberately no "anonymous" ``Principal`` value — Steps 5/6
    bind exactly one ``Principal`` per request immediately after
    ``resolve_bearer``/``resolve_session`` succeeds, and reject the request
    before that point otherwise. Reaching this function with nothing bound
    means the entry path itself is broken.
    """
    principal = current_principal.get()
    if principal is None:
        raise PrincipalNotBoundError(
            "No Principal is bound to the current context — the request "
            "entry path must call bind_principal() before any code that "
            "calls get_principal()."
        )
    return principal


@contextmanager
def bind_principal(principal: Principal) -> Iterator[None]:
    """Bind ``principal`` to the current context for the duration of the
    ``with`` block, restoring whatever was bound before (if anything) on
    exit. The one place Steps 5/6 are expected to call this: once per
    request, immediately after ``resolve_bearer``/``resolve_session``
    succeeds.

    ALWAYS bind through this contextmanager — never call
    ``current_principal.set()`` directly. That is not a style preference;
    on a pooled worker thread it is a cross-tenant identity leak. Verified
    behaviour (Gate 1 Step 4 security review):

    * ``ThreadPoolExecutor.submit`` does NOT copy contextvars into the
      worker, so a principal bound on the submitting side is simply absent
      there — surprising, but safe.
    * A bare ``current_principal.set()`` inside a job SURVIVES that job and
      is visible to the NEXT job scheduled onto the same reused thread. In
      ``app.analysis.background`` (``MAX_WORKERS = 2``, long-lived pool)
      that means tenant A's analysis job could hand its identity to tenant
      B's. This function's ``try/finally: reset(token)`` is what prevents
      it, in a reused thread as much as anywhere else.

    Nothing binds a principal inside a background job today
    (``app.analysis.background`` never touches this module). Step 7's owner
    predicate is where the temptation first appears — bind here, or not at
    all."""
    token = current_principal.set(principal)
    try:
        yield
    finally:
        current_principal.reset(token)
