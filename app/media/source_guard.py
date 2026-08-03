"""Whether an UNTRUSTED caller may name a server-side FILE as project media.

``add_media`` and ``add_music`` both take "a local path or an http(s) URL", and
for the CLI twin, the test fixtures and a single-operator self-host that is
exactly the right contract: the caller and the server are the same person, so a
path is a convenience, not a capability.

On the multi-tenant service they are NOT the same person, and the same argument
becomes a file-read primitive. A registered asset is streamed back verbatim by
``GET /api/projects/{id}/assets/{media_id}/stream`` — ``app.web.media.
resolve_asset_stream_path`` serves the stored path as-is, by design — so
"register a path, then stream it" reads any file this container can open,
including another tenant's upload under ``media_dir/uploads/<uuid>/``.

The rule is enforced at the two untrusted ENTRY boundaries:

* **HTTP** (``/api/projects/{id}/ops``) — unconditional. Nothing legitimate
  names a server path over HTTP: the browser shares no filesystem with this
  container, so a real file arrives through ``POST .../media/upload`` or
  ``POST .../media/import-url``, and those hand the stored path to the tool
  SERVER-SIDE, past this boundary.
* **MCP** (``/mcp``) — for non-operator callers only (``app.mcp.trust.
  caller_is_operator``). ``editor_add_media`` is operator-tier already;
  ``editor_add_music`` is PUBLIC and carries the identical ``source``
  primitive, which is the hole this closes. An operator token — or
  ``MCP_AUTH_ENABLED=false``, i.e. self-host — keeps paths, because that is the
  documented local workflow and the whole meaning of the tier.

Deliberately NOT enforced inside ``app.mcp.tools``: the CLI twin, ~110 test
call sites and every internal caller (upload ingest, ``generate_voiceover``'s
wavs) name real local paths while a plain tenant ``Principal`` is bound, so a
check there would have to tell "trusted code" from "untrusted request" — which
is precisely what the entry boundary knows and the tool does not.
"""
from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "PATH_CAPABLE_ARGS",
    "REMOTE_SCHEMES",
    "is_remote_source",
    "local_path_arg",
    "refusal_message",
]

# The MCP tool names are the op names with this prefix; both boundaries feed
# their own vocabulary to ``local_path_arg`` and it normalises here, so neither
# has to keep a second copy of the table.
_TOOL_PREFIX = "editor_"

# op name -> the arguments whose value the caller could point at a local file.
# An allowlist of ARGUMENTS keyed by op, mirroring ``app.web.api.
# HTTP_FORBIDDEN_ARGS``: a new op with a path-shaped argument has to be listed
# here to be guarded, which is why ``tests/test_untrusted_media_sources.py``
# asserts the table covers every tool signature that takes a ``source``.
PATH_CAPABLE_ARGS: dict[str, frozenset[str]] = {
    "add_media": frozenset({"source"}),
    "add_music": frozenset({"source"}),
}

# EXACTLY the prefixes ``app.media.downloader.resolve_media_source`` and
# ``app.web.media.resolve_asset_stream_path`` test for, case-sensitively and
# deliberately so. Those two decide "URL or local file" for real, and a guard
# that normalised case (or accepted more schemes) would classify ``HTTPS://x``
# as remote while the resolver reads it as a relative PATH — the guard would
# wave through the exact string it exists to stop. Parity with the resolver is
# the requirement here, not politeness about URL syntax.
REMOTE_SCHEMES = ("http://", "https://")


def is_remote_source(value: object) -> bool:
    """Whether ``value`` is a source the server will FETCH rather than open off
    its own disk. Non-strings are not remote: they cannot be a URL, and failing
    closed hands them to the refusal rather than to the filesystem (their type
    error is then raised by the arg validator, which runs after this)."""
    return isinstance(value, str) and value.startswith(REMOTE_SCHEMES)


def local_path_arg(op_or_tool_name: str, args: Mapping[str, object]) -> str | None:
    """The first path-capable argument ``args`` carries that is NOT an http(s)
    URL, or ``None`` when the call names no server-side file.

    Accepts either an op name (``"add_music"``) or its MCP tool name
    (``"editor_add_music"``). Checked on the argument's VALUE, so a path
    smuggled in beside otherwise-valid arguments is caught too — the same
    posture as ``app.web.api._http_forbidden_arg``.
    """
    op = op_or_tool_name.removeprefix(_TOOL_PREFIX)
    for name in sorted(PATH_CAPABLE_ARGS.get(op, frozenset())):
        if name in args and not is_remote_source(args[name]):
            return name
    return None


def refusal_message(arg: str) -> str:
    """What to tell the caller. Names the argument and the way in that DOES
    work, because "not allowed" alone leaves a legitimate user stuck: the
    browser cannot reach this container's filesystem, so uploading is not a
    workaround for the refusal — it is the actual supported path."""
    return (
        f"{arg!r} must be an http(s) URL here, not a path on the server — "
        "upload the file instead (Media panel → Upload)"
    )
