"""Single-user context — the open edition's replacement for the cloud adapter.

In the cloud edition this module resolves the caller's identity from a bearer
token and admits heavy operations against a per-tenant daily quota. This build
has exactly one user, so both collapse: every call belongs to the bootstrap
account, and the only concurrency bounds are the ones the analysis scheduler
and your own hardware impose.

The module keeps the SAME public surface (``mcp_session``, ``to_worker_thread``,
``mcp_metered_slots``, ``quota_refusal``, the two refusal exceptions) so that
``app.mcp.server`` is byte-identical between editions — composition, not
forking, is the whole point of the seam.
"""
from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import ParamSpec, TypeVar

import anyio
from sqlalchemy.orm import Session

from app.auth.principal import Principal, bind_principal
from app.db.base import session_scope
from app.db.repositories.users import ensure_bootstrap_user
from app.editor.errors import EditorError


class HeavyOpBusyError(EditorError):
    """Never raised in the single-user edition; ``app.mcp.server`` still
    catches it, and the class exists so that code stays identical."""


class DailyQuotaExceededError(EditorError):
    """Never raised in the single-user edition — there is no daily quota."""

    budget: object = None
    retry_after_seconds: int = 0


def caller_principal(session: Session) -> Principal:
    """Everything belongs to the bootstrap account — one owner, not "no
    owner". The owner-scoped repository layer stays exercised exactly as in
    the cloud edition; only the answer to "who is calling" is constant."""
    return Principal(
        user_id=ensure_bootstrap_user(session), token_id=None, scopes=()
    )


_P = ParamSpec("_P")
_R = TypeVar("_R")


def to_worker_thread(fn: Callable[_P, _R]) -> Callable[_P, Awaitable[_R]]:
    """Register a sync tool body with the SDK as async, running it in a worker
    thread. The SDK dispatches a synchronous tool function directly on the
    event loop, so a plain ``def`` wrapper whose body runs ffmpeg or whisper
    would freeze every session for the duration of the call — up to the
    render timeout. ``functools.wraps`` keeps ``inspect.signature`` (the SDK's
    schema source) and ``__doc__`` (the tool description) intact."""

    @functools.wraps(fn)
    async def _run_in_worker_thread(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))

    return _run_in_worker_thread


@contextmanager
def mcp_session() -> Iterator[Session]:
    """A database session with the (single) owner's ``Principal`` bound for
    its whole scope — every ``@mcp_server.tool()`` wrapper goes through this,
    never a bare ``session_scope()``. The session opens first because
    resolving the principal can need it: on a fresh database the bootstrap
    account is created here."""
    with session_scope() as session:
        with bind_principal(caller_principal(session)):
            yield session


@contextmanager
def mcp_metered_slots(session: Session, op: str) -> Iterator[None]:
    """Admission gate for one heavy tool body. Single user — nothing to meter,
    nobody to protect from you but yourself; renders queue on your own CPU."""
    yield


def quota_refusal(exc: DailyQuotaExceededError) -> dict:
    """Shape kept for ``app.mcp.server``; unreachable in this edition."""
    return {"error": str(exc)}
