"""Shared logging setup used by the editor server and CLI.

R-18 (minimum observability): one JSON object per log record, always carrying
a ``request_id`` field, so a real incident can be filtered to "everything this
one caller triggered" instead of grepped by eye out of interleaved plain-text
lines. This matters specifically BECAUSE the server is single-process but
multi-surface (MCP + ``/api`` + renders share one event loop, per CLAUDE.md) —
without a correlation id, three different callers' log lines are
indistinguishable from one another.

``request_id_var`` is read DIRECTLY by the formatter at format() time rather
than pushed onto each record via a ``logging.Filter``. A filter would only run
for handlers it is explicitly attached to (this module's own
``StreamHandler``), and ``pytest``'s ``caplog`` attaches its OWN handler
straight to the logger, bypassing ours entirely — reading the contextvar in
the formatter instead means ANY handler that happens to use
``JsonLogFormatter`` sees the same id, and ``caplog`` (which never uses this
formatter) is simply unaffected either way.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

# Bound once per inbound request by app.observability.RequestIdMiddleware (the
# one ASGI-level choke point covering /mcp, /api, /ui, /auth, /legal and the
# SPA — see that module and app.web.app.build_app). ``default=None`` rather
# than "-" so the formatter's fallback below is the ONE place that decides how
# an absent id is spelled, not something duplicated at every call site.
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

# The literal reserved LogRecord attributes, computed from a throwaway record
# rather than hand-copied from the stdlib docs — so a Python version that
# changes the set (it has, across 3.x releases: `taskName` is 3.12+) is
# reflected automatically instead of silently under- or over-filtering in
# ``JsonLogFormatter``'s "extra fields" pass below.
_RESERVED_RECORD_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {"message", "asctime"}


def new_request_id() -> str:
    """A fresh correlation id. Hex (no dashes) purely for a shorter log line;
    it carries no meaning beyond uniqueness, unlike a token or user id, so
    there is nothing here for a secret-scrubber to redact."""
    return uuid.uuid4().hex


@contextmanager
def bind_request_id(value: str) -> Iterator[None]:
    """Bind ``value`` to ``request_id_var`` for the wrapped scope only.

    A context manager (not a bare ``.set()``) so a test — or a future non-ASGI
    caller, e.g. a CLI entry point — cannot forget the reset and leak one
    request's id into the next log line on a reused thread. Mirrors
    ``app.auth.principal.bind_principal``'s shape for the same reason that
    module gives: the alternative already caused an identity to leak across
    requests on this codebase's shared worker pool once.
    """
    token = request_id_var.set(value)
    try:
        yield
    finally:
        request_id_var.reset(token)


class JsonLogFormatter(logging.Formatter):
    """One JSON object per record: ``timestamp``, ``level``, ``logger``,
    ``message``, ``request_id``, plus ``exception`` when the record carries
    one. Deliberately does NOT emit the record's raw ``args`` tuple or
    anything from ``record.__dict__`` beyond the fixed fields above and
    caller-supplied ``extra=`` keys. This formatter does no scrubbing of its
    own: a caller logging something worth remembering is expected to pass
    exact, deliberate ``extra=`` keys, never a raw request/environ/headers
    dump that could carry a token or password along for the ride.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get() or "-",
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_ATTRS
        }
        if extra:
            payload["extra"] = extra
        # default=str: a caller's `extra=` value need not be JSON-native (a
        # Path, an Enum, ...) for the line to still ship rather than raise
        # out of the logging call that produced it.
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
