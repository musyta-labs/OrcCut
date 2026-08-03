"""Request identity (Gate 1 Step 4): resolving either an MCP bearer token
(``resolve_bearer``) or a browser cookie session (``resolve_session``) to
the SAME ``Principal`` type for the same human — see ``app.auth.principal``,
``app.auth.resolve``, and ``app.auth.sessions``.

Purely additive package: nothing in ``app/web`` or ``app/mcp`` calls into it
yet. That wiring is Gate 1 Steps 5 (MCP resource server) and 6 (``/api``
auth).
"""
from __future__ import annotations
