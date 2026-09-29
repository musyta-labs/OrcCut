"""Standalone MCP server exposing the agent video editor to any agent.

No niche/video/compilation concept anywhere — a client passes an opaque
``metadata`` tag set and path/URL media sources. Streamable HTTP transport by
default so any transport-reachable agent can connect (not just a local stdio
client).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from app.auth.verifier import PrincipalTokenVerifier

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP, Image
from mcp.types import Tool as MCPTool
from pydantic import AnyHttpUrl

from app.auth.principal import Principal
from app.common import ratelimit
from app.common.logging import setup_logging
from app.config import get_settings
from app.db.base import init_db
from app.editor import mutations
from app.mcp import tools

# The nested-payload shapes a bare ``list[dict]`` annotation cannot express.
# FastMCP derives each tool's validation model from these signatures, so
# annotating with the SAME TypedDicts the HTTP/CLI validator uses is what keeps
# an agent from writing a crafted string into a render expression (R-04) — and
# it publishes the real shape in the tool's JSON schema as a side effect.
from app.mcp.argspec import (
    CaptionStyleArgs,
    TextUpdateArgs,
    TransformKeyframeArgs,
    VolumeKeyframeArgs,
)
from app.mcp.context import (
    DailyQuotaExceededError,
    HeavyOpBusyError,
    mcp_metered_slots,
    mcp_session,
    quota_refusal,
    to_worker_thread,
)
from app.mcp.trust import caller_is_operator, visible_tool_names
from app.media.source_guard import local_path_arg, refusal_message

# The bare RFC 9728 path. The SDK only ever serves the SUFFIXED form
# (``/.well-known/oauth-protected-resource/mcp`` for a resource at
# ``.../mcp``, per RFC 9728 §3.1); a client that probes only the bare path
# would find nothing at all. Both are served — see ``_register_metadata_alias``.
BARE_PROTECTED_RESOURCE_PATH = "/.well-known/oauth-protected-resource"


def _build_auth() -> tuple[AuthSettings | None, PrincipalTokenVerifier | None]:
    """``(auth_settings, token_verifier)`` for ``FastMCP``, or ``(None, None)``
    when authentication is switched off.

    FastMCP requires these two to be supplied together or not at all, so they
    are built together here.

    ``required_scopes`` is deliberately ``None`` — Step 5 authenticates, it
    does not yet authorize. Every token minted so far defaults to no scopes
    at all (``mint_token``'s ``scopes=()``), so demanding a scope here would
    reject every credential this system is currently able to issue: a
    lock-out dressed as a security control. Scope enforcement arrives with
    Gate 1 Step 11's trust tiers, against tokens minted to carry them.

    ``client_registration_options`` and ``revocation_options`` stay ``None``
    because both are authorization-SERVER concerns, and this process is a
    resource server only (see ``app.auth.verifier``'s module docstring).
    """
    settings = get_settings()
    if not settings.mcp_auth_enabled:
        return None, None
    # Lazy and guarded: the bearer-token verifier chain (verifier → resolve →
    # sessions/tokens repos) is closed core the public extract (E-01) does not
    # ship. There auth simply cannot be enabled, and the honest response to
    # trying is a clear refusal at startup, not an ImportError traceback.
    try:
        from app.auth.verifier import PrincipalTokenVerifier
    except ImportError as exc:
        raise RuntimeError(
            "MCP_AUTH_ENABLED=true requires the cloud edition's auth modules, "
            "which this build does not include. Set MCP_AUTH_ENABLED=false — "
            "a single-user deployment is the operator unconditionally."
        ) from exc
    auth_settings = AuthSettings(
        issuer_url=AnyHttpUrl(settings.mcp_issuer_url),
        resource_server_url=AnyHttpUrl(settings.mcp_resource_url),
        required_scopes=None,
        client_registration_options=None,
        revocation_options=None,
    )
    verifier = PrincipalTokenVerifier(
        issuer_url=settings.mcp_issuer_url,
        resource_url=settings.mcp_resource_url,
    )
    return auth_settings, verifier


_auth_settings, _token_verifier = _build_auth()


def _current_caller_principal() -> Principal | None:
    """The caller of the request being handled, or ``None`` when there is no
    verified token in context.

    Used by the trust-tier filter, NOT for owner resolution — so unlike
    ``app.mcp.context.caller_principal`` it takes no session and never falls
    back to the bootstrap account. ``None`` means "no authenticated token
    here": with auth ON that yields the public tier (the safe default);
    ``app.mcp.trust.caller_is_operator`` handles auth-OFF as single-tenant
    operator on its own, without needing a principal.
    """
    access_token = get_access_token()
    if access_token is None:
        return None
    # Unreachable without the cloud auth modules: a verified token can only
    # exist when _build_auth() imported the verifier successfully.
    from app.auth.verifier import principal_from_access_token

    return principal_from_access_token(access_token)


class McpRateLimitExceeded(RuntimeError):
    """Raised from ``call_tool`` when a token exceeds its MCP call budget.

    Carries ``retry_after_seconds`` — the JSON-RPC-native equivalent of an HTTP
    ``Retry-After`` header. The MCP transport multiplexes many tool calls over
    one authenticated ``POST /mcp`` and the token identity is only available
    INSIDE the SDK-verified call, so a per-call HTTP 429 is not representable at
    the transport edge; enforcement therefore lives at ``call_tool`` and
    surfaces as this raised error, mirroring how the trust-tier check raises to
    refuse a hidden tool.
    """

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"MCP rate limit exceeded; retry after {retry_after_seconds}s"
        )


def _enforce_mcp_rate_limit(principal: Principal | None) -> None:
    """Throttle the current MCP tool call PER TOKEN.

    Keyed on ``token_id`` (never ``owner_id``): a single runaway token must not
    exhaust another token of the SAME owner (Gate-1 finding "Р10"). When there
    is no token to key on — no verified ``AccessToken`` (``principal is None``)
    or the single-tenant auth-off mode (``token_id is None``) — there is nothing
    a per-token limiter can bound, so the call passes through. That mode is the
    self-hosted one-operator deployment where an unbounded local agent is the
    intended behaviour.
    """
    if principal is None or principal.token_id is None:
        return
    decision = ratelimit.MCP_LIMITER.check(f"token:{principal.token_id}")
    if not decision.allowed:
        raise McpRateLimitExceeded(decision.retry_after_seconds)


def _refuse_untrusted_local_path(name, arguments, principal) -> None:
    """Refuse a NON-OPERATOR call that names a file on this container's disk.

    The tier filter above hides ``editor_add_media`` from the public, but
    ``editor_add_music`` is public tier and takes the identical ``source``
    primitive — register a path, then read it back through the asset stream
    route. That is the hole (see ``app.media.source_guard``).

    Operators keep paths: with ``MCP_AUTH_ENABLED=false`` the sole caller IS the
    operator (the self-host workflow adds media by path), and an ``editor:
    operator`` token is a deliberate grant of exactly this primitive. So this is
    a tier rule, not a blanket ban — unlike the HTTP boundary, where no caller
    is ever an operator and the ban is unconditional.

    ``ValueError`` rather than a tool-level error, so it is raised BEFORE the
    tool body runs and no path is opened, probed, or journaled.
    """
    if caller_is_operator(principal):
        return
    offending = local_path_arg(name, arguments or {})
    if offending is not None:
        raise ValueError(refusal_message(offending))


class TrustTieredMCP(FastMCP):
    """``FastMCP`` that filters the tool dictionary by the caller's trust tier
    (Gate 1 Step 11).

    Two overrides, one predicate (``app.mcp.trust.visible_tool_names``):

    * ``list_tools`` hides tools the caller may not reach, so a public caller's
      ``tools/list`` never even names an operator tool.
    * ``call_tool`` refuses a hidden tool AS IF it did not exist — the same
      "absent, not forbidden" answer the tenant-isolation reads give, so a
      probe cannot use the error to confirm the tool is real. Enforced
      independently of ``list_tools`` because a caller can name a tool without
      listing first.
    """

    async def list_tools(self) -> list[MCPTool]:
        all_tools = await super().list_tools()
        visible = visible_tool_names(
            [tool.name for tool in all_tools], _current_caller_principal()
        )
        return [tool for tool in all_tools if tool.name in visible]

    async def call_tool(self, name, arguments):
        principal = _current_caller_principal()
        # Throttle PER TOKEN before anything else — a hammering caller is bounded
        # regardless of which tool (even a hidden one) it names.
        _enforce_mcp_rate_limit(principal)
        visible = visible_tool_names([name], principal)
        if name not in visible:
            # Deliberately indistinguishable from an unregistered tool.
            raise ValueError(f"Unknown tool: {name}")
        _refuse_untrusted_local_path(name, arguments, principal)
        return await super().call_tool(name, arguments)


mcp_server = TrustTieredMCP(
    "orccut",
    host=get_settings().mcp_host,
    port=get_settings().mcp_port,
    auth=_auth_settings,
    token_verifier=_token_verifier,
)


def _register_metadata_alias(server: FastMCP, auth_settings: AuthSettings | None) -> None:
    """Serve the protected-resource metadata at the BARE well-known path too.

    The alias reuses the SDK's own endpoint object rather than rebuilding the
    metadata document, so the two paths cannot drift into answering
    differently — which would be worse than serving only one of them.

    ``custom_route`` is the right tool here for a reason the SDK states
    outright: "Routes using this decorator will not require authorization."
    Discovery metadata MUST be reachable unauthenticated — it is what tells
    an unauthenticated client how to authenticate. That same property is why
    ``custom_route`` must not be used for anything else in this file.
    """
    if auth_settings is None or auth_settings.resource_server_url is None:
        return
    from mcp.server.auth.routes import (
        build_resource_metadata_url,
        create_protected_resource_routes,
    )

    suffixed_url = build_resource_metadata_url(auth_settings.resource_server_url)
    if urlparse(str(suffixed_url)).path == BARE_PROTECTED_RESOURCE_PATH:
        # The resource sits at the host root, so the SDK's own route already
        # IS the bare path — registering it again would shadow it.
        return
    route = create_protected_resource_routes(
        resource_url=auth_settings.resource_server_url,
        authorization_servers=[auth_settings.issuer_url],
        scopes_supported=auth_settings.required_scopes,
    )[0]
    server.custom_route(
        BARE_PROTECTED_RESOURCE_PATH,
        methods=["GET", "OPTIONS"],
        name="bare_protected_resource_metadata",
    )(route.endpoint)


_register_metadata_alias(mcp_server, _auth_settings)


@mcp_server.tool()
@to_worker_thread
def editor_create_project(
    metadata: dict | None = None,
    aspect_w: int = 1080,
    aspect_h: int = 1920,
    fps: int = 30,
    target_sec: float | None = None,
    min_clips: int = 1,
    max_clips: int | None = None,
    min_duration_sec: float = 0.0,
    max_duration_sec: float | None = None,
) -> dict:
    """Create a fresh empty timeline project. ``metadata`` is an opaque
    client-owned tag set (e.g. {"client": "annotator", "niche": "cats"}) —
    the editor never interprets it. Returns the project summary including its
    project_id."""
    with mcp_session() as session:
        return tools.create_project(
            session,
            metadata=metadata,
            aspect=(aspect_w, aspect_h),
            fps=fps,
            target_sec=target_sec,
            min_clips=min_clips,
            max_clips=max_clips,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
        )


@mcp_server.tool()
@to_worker_thread
def editor_get_project(project_id: str) -> dict:
    """Full serialized timeline project JSON — the agent's 'eyes'."""
    with mcp_session() as session:
        return tools.get_project(session, project_id)


@mcp_server.tool()
@to_worker_thread
def editor_list_projects(metadata: dict | None = None) -> list[dict]:
    """Project summaries, newest first, optionally filtered to those whose
    metadata is a superset of the given filter."""
    with mcp_session() as session:
        return tools.list_projects_tool(session, metadata)


@mcp_server.tool()
@to_worker_thread
def editor_add_media(project_id: str, source: str, duration_sec: float | None = None) -> dict:
    """Register a media source (local path or URL) as a project asset.
    Duration is probed automatically when not supplied. Returns the summary
    plus the new media_id."""
    with mcp_session() as session:
        return tools.add_media(session, project_id, source, duration_sec=duration_sec)


@mcp_server.tool()
@to_worker_thread
def editor_add_clip(
    project_id: str,
    media_id: str,
    track_type: str = "video",
    start_time: float | None = None,
    duration: float | None = None,
    trim_start: float = 0.0,
    trim_end: float | None = None,
    track_id: str | None = None,
) -> dict:
    """Place a media asset on a track (duration defaults to the asset's own).
    ``track_id`` targets one exact track (a V2+ upper video track); None keeps
    the first-track-of-type behavior. Returns the summary plus the new
    element_id."""
    with mcp_session() as session:
        return tools.add_clip(
            session, project_id, media_id,
            track_type=track_type, start_time=start_time, duration=duration,
            trim_start=trim_start, trim_end=trim_end, track_id=track_id,
        )


@mcp_server.tool()
@to_worker_thread
def editor_trim_clip(
    project_id: str,
    element_id: str,
    trim_start: float | None = None,
    trim_end: float | None = None,
) -> dict:
    """Adjust a clip's source in/out points."""
    with mcp_session() as session:
        return tools.trim_clip(session, project_id, element_id, trim_start=trim_start, trim_end=trim_end)


@mcp_server.tool()
@to_worker_thread
def editor_resize_clip(
    project_id: str,
    element_id: str,
    start_time: float | None = None,
    duration: float | None = None,
    trim_start: float | None = None,
    trim_end: float | None = None,
) -> dict:
    """Atomic placement + source-window edit (the GUI edge-drag gesture) — the
    combined twin of ``editor_trim_clip`` + ``editor_move_clip`` in one
    journaled op. ``None`` keeps the current value for every field."""
    with mcp_session() as session:
        return tools.resize_clip(
            session, project_id, element_id,
            start_time=start_time, duration=duration,
            trim_start=trim_start, trim_end=trim_end,
        )


@mcp_server.tool()
@to_worker_thread
def editor_split_clip(project_id: str, element_id: str, at_time: float) -> dict:
    """Split one clip into two adjacent clips at absolute timeline time."""
    with mcp_session() as session:
        return tools.split_clip(session, project_id, element_id, at_time)


@mcp_server.tool()
@to_worker_thread
def editor_move_clip(
    project_id: str, element_id: str, start_time: float, track_id: str | None = None
) -> dict:
    """Move an element to a new absolute start time. ``track_id`` (optional)
    also moves it onto that exact track — the vertical lane-drag across video
    lanes."""
    with mcp_session() as session:
        return tools.move_clip(
            session, project_id, element_id, start_time=start_time, track_id=track_id
        )


@mcp_server.tool()
@to_worker_thread
def editor_add_video_track(project_id: str) -> dict:
    """Add a new, empty V2+ upper video track (its clips composite over the
    main V1 track — arbitrary positions, gaps allowed, per-clip opacity).
    Returns the summary plus the new ``track_id``."""
    with mcp_session() as session:
        return tools.add_video_track(session, project_id)


@mcp_server.tool()
@to_worker_thread
def editor_add_audio_track(project_id: str) -> dict:
    """Add a new, empty audio track (inserted after the last audio track). A
    voiceover bed lands here so it never collides with the music bed on the
    first audio track. Returns the summary plus the new ``track_id``."""
    with mcp_session() as session:
        return tools.add_audio_track(session, project_id)


@mcp_server.tool()
@to_worker_thread
def editor_remove_track(project_id: str, track_id: str) -> dict:
    """Remove a track and its elements. The first (V1/main) video track cannot
    be removed."""
    with mcp_session() as session:
        return tools.remove_track(session, project_id, track_id)


@mcp_server.tool()
@to_worker_thread
def editor_delete_element(project_id: str, element_id: str) -> dict:
    """Remove an element from whatever track holds it."""
    with mcp_session() as session:
        return tools.delete_element(session, project_id, element_id)


@mcp_server.tool()
@to_worker_thread
def editor_set_transform(
    project_id: str,
    element_id: str,
    scale: float | None = None,
    pos_x: float | None = None,
    pos_y: float | None = None,
    speed: float | None = None,
    crop_zoom: float | None = None,
    opacity: float | None = None,
) -> dict:
    """Update a clip's transform (scale/pos/zoom/speed) and/or its composite
    ``opacity`` (0..1, for a V2+ upper-track clip). Only provided fields change
    — the MLT engine applies scale/pos_x/pos_y via an affine transform."""
    with mcp_session() as session:
        return tools.set_transform(
            session, project_id, element_id,
            scale=scale, pos_x=pos_x, pos_y=pos_y, speed=speed,
            crop_zoom=crop_zoom, opacity=opacity,
        )


@mcp_server.tool()
@to_worker_thread
def editor_set_clip_audio(
    project_id: str,
    element_id: str,
    muted: bool | None = None,
    volume: float | None = None,
    gain_db: float | None = None,
) -> dict:
    """Update a clip's (mute / volume) or a music bed's (volume / gain_db)
    audio settings — ``muted`` is clip-only, ``gain_db`` is music-bed-only."""
    with mcp_session() as session:
        return tools.set_clip_audio(
            session, project_id, element_id, muted=muted, volume=volume, gain_db=gain_db,
        )


@mcp_server.tool()
@to_worker_thread
def editor_set_audio_envelope(
    project_id: str,
    element_id: str,
    fade_in_sec: float | None = None,
    fade_out_sec: float | None = None,
    volume_keyframes: list[VolumeKeyframeArgs] | None = None,
) -> dict:
    """Set a fade-in/fade-out and/or a volume-keyframe envelope on a clip's
    own audio or a music bed (mlt engine only). Only provided fields change;
    ``volume_keyframes`` is ``[{"time": t, "volume": v}, ...]``, replacing the
    whole envelope."""
    with mcp_session() as session:
        return tools.set_audio_envelope(
            session, project_id, element_id,
            fade_in_sec=fade_in_sec, fade_out_sec=fade_out_sec,
            volume_keyframes=volume_keyframes,
        )


@mcp_server.tool()
@to_worker_thread
def editor_set_fit(project_id: str, element_id: str, fit: str) -> dict:
    """Set a clip's canvas fill mode: ``"cover"`` (default, fill-scale +
    center-crop) or ``"contain_blur"`` (blurred cover-scaled fill behind a
    sharp, contain-fit copy — mlt engine only)."""
    with mcp_session() as session:
        return tools.set_fit(session, project_id, element_id, fit=fit)


@mcp_server.tool()
@to_worker_thread
def editor_set_cover(project_id: str, at_time: float) -> dict:
    """Set the absolute timeline time exported as the project's cover PNG
    alongside the mp4 on the next ``editor_export`` call."""
    with mcp_session() as session:
        return tools.set_cover(session, project_id, at_time=at_time)


@mcp_server.tool()
@to_worker_thread
def editor_auto_captions(
    project_id: str,
    style_overrides: CaptionStyleArgs | None = None,
    max_chars_per_line: int = 42,
) -> dict:
    """Transcribe every video clip's own trimmed audio window (speech
    recognition, lazy — degrades to a clear error when the 'asr' extra/model is
    unavailable) and add the result as caption TextElements (bottom position
    by default; ``style_overrides`` merges over the defaults). Returns the
    summary plus ``captions_added``."""
    with mcp_session() as session:
        try:
            with mcp_metered_slots(session, "auto_captions"):
                return tools.auto_captions(
                    session, project_id,
                    style_overrides=style_overrides, max_chars_per_line=max_chars_per_line,
                )
        except DailyQuotaExceededError as exc:
            return quota_refusal(exc)
        except HeavyOpBusyError as exc:
            return {"error": str(exc)}


@mcp_server.tool()
@to_worker_thread
def editor_generate_voiceover(
    project_id: str,
    voice: str | None = None,
    duck_music: bool = True,
) -> dict:
    """Synthesize a local TTS voiceover for every auto-caption (run
    ``editor_auto_captions`` first) and lay it on a dedicated new audio track,
    ducking any existing music bed underneath. The voice defaults by the
    project's detected ``captions_lang`` (``ru``/``en``); any other language
    needs an explicit ``voice``. Each caption is fitted to its slot; a wav that
    overruns the gap to the next caption is clamped. ONE journaled op. Returns
    the summary plus ``voiceover_added`` / ``fitted`` / ``track_id``."""
    with mcp_session() as session:
        try:
            with mcp_metered_slots(session, "generate_voiceover"):
                return tools.generate_voiceover(
                    session, project_id, voice=voice, duck_music=duck_music,
                )
        except DailyQuotaExceededError as exc:
            return quota_refusal(exc)
        except HeavyOpBusyError as exc:
            return {"error": str(exc)}


@mcp_server.tool()
@to_worker_thread
def editor_add_text(
    project_id: str,
    content: str,
    start: float,
    duration: float | None = None,
    pos: str = "center",
    size: int = 64,
    color: str = "#FFFFFF",
    pos_x: float | None = None,
    pos_y: float | None = None,
    role: str | None = None,
    font_id: str | None = None,
) -> dict:
    """Add a text overlay (hook title / caption). ``duration`` None = persistent:
    hold from ``start`` to the end of the timeline (a header that stays all
    video). ``role`` optionally marks the element's origin (e.g. ``"caption"``).
    ``font_id`` picks a typeface from YOUR media library (a ``.ttf``/``.otf``
    you uploaded); omit it for the service's own font. Returns the new
    element_id."""
    with mcp_session() as session:
        return tools.add_text(
            session, project_id, content, start, duration,
            pos=pos, size=size, color=color, pos_x=pos_x, pos_y=pos_y, role=role,
            font_id=font_id,
        )


@mcp_server.tool()
@to_worker_thread
def editor_update_text(
    project_id: str,
    element_id: str,
    content: str | None = None,
    start_time: float | None = None,
    duration: float | None | object = mutations.UNSET,
    size: int | None = None,
    color: str | None = None,
    pos: str | None = None,
    pos_x: float | None | object = mutations.UNSET,
    pos_y: float | None | object = mutations.UNSET,
    font_id: str | None | object = mutations.UNSET,
) -> dict:
    """Partial in-place text edit (id stable) — the single-element twin of
    ``editor_update_texts``, and the only one that can touch placement.
    Fields left out of the call keep their current value; for ``duration``,
    ``pos_x``, ``pos_y`` and ``font_id`` an explicit ``null`` is different from
    leaving the field out — it CLEARS the value (``duration=null`` → persistent
    header, ``pos_x``/``pos_y`` null → fall back to the named ``pos`` preset,
    ``font_id=null`` → back to the service's own font). ``font_id`` is the id of
    a ``.ttf``/``.otf`` in YOUR media library."""
    with mcp_session() as session:
        return tools.update_text(
            session, project_id, element_id,
            content=content, start_time=start_time, duration=duration,
            size=size, color=color, pos=pos, pos_x=pos_x, pos_y=pos_y,
            font_id=font_id,
        )


@mcp_server.tool()
@to_worker_thread
def editor_update_texts(
    project_id: str,
    updates: list[TextUpdateArgs],
    reason: str | None = None,
) -> dict:
    """Atomically edit the text of MANY elements in ONE journaled op — the
    batch twin of an in-place text edit (content-only). ``updates`` is a list
    of ``{"element_id": str, "content": str}``; an unknown id or a non-text
    element fails the WHOLE batch (nothing applied), so undo is one step. Used
    e.g. to write a whole translated caption track back at once. ``reason`` is
    an optional note recorded in the op journal."""
    with mcp_session() as session:
        return tools.update_texts(session, project_id, updates=updates, reason=reason)


@mcp_server.tool()
@to_worker_thread
def editor_add_music(
    project_id: str,
    source: str,
    volume: float = 0.15,
    start: float = 0.0,
    duration: float | None = None,
) -> dict:
    """Register a media source as the music bed on the audio track (duration
    defaults to the video-track total)."""
    with mcp_session() as session:
        return tools.add_music(session, project_id, source, volume=volume, start=start, duration=duration)


@mcp_server.tool()
@to_worker_thread
def editor_add_transition(
    project_id: str,
    to_element: str,
    kind: str = "dissolve",
    duration: float = 0.5,
) -> dict:
    """Cross-dissolve from ``to_element``'s immediate predecessor on the same
    video track into it (the predecessor is derived automatically, not
    named). ``duration`` consumes the last ``duration`` seconds of the
    predecessor and the first ``duration`` seconds of ``to_element`` — see
    ``model.TransitionSpec``. MLT-engine only."""
    with mcp_session() as session:
        return tools.add_transition(session, project_id, to_element, kind=kind, duration=duration)


@mcp_server.tool()
@to_worker_thread
def editor_remove_transition(project_id: str, element_id: str) -> dict:
    """Clear ``element_id``'s incoming transition — the off switch for
    ``editor_add_transition``."""
    with mcp_session() as session:
        return tools.remove_transition(session, project_id, element_id)


@mcp_server.tool()
@to_worker_thread
def editor_set_keyframes(
    project_id: str,
    element_id: str,
    keyframes: list[TransformKeyframeArgs],
) -> dict:
    """Replace a clip's animated-transform keyframes. Each item is a dict
    with a required ``time`` (seconds relative to the clip's own start) and
    optional ``scale``/``pos_x``/``pos_y``/``opacity``/``rotation``. An empty
    list clears them, falling back to the clip's static transform.
    MLT-engine only."""
    with mcp_session() as session:
        return tools.set_keyframes(session, project_id, element_id, keyframes=keyframes)


@mcp_server.tool()
@to_worker_thread
def editor_set_color(
    project_id: str,
    element_id: str,
    brightness: float | None = None,
    contrast: float | None = None,
    saturation: float | None = None,
    gamma: float | None = None,
) -> dict:
    """Update a clip's color grade (brightness/contrast/saturation/gamma).
    Only provided fields change. MLT-engine only."""
    with mcp_session() as session:
        return tools.set_color(
            session, project_id, element_id,
            brightness=brightness, contrast=contrast, saturation=saturation, gamma=gamma,
        )


@mcp_server.tool()
@to_worker_thread
def editor_add_overlay(
    project_id: str,
    media_id: str,
    duration: float,
    x: float,
    y: float,
    w: float,
    h: float,
    start_time: float | None = None,
    opacity: float = 1.0,
) -> dict:
    """Place a PiP/sticker overlay (image or video, registered via
    editor_add_media) on the overlay track. x/y/w/h are normalized [0,1]
    frame fractions (top-left anchored). MLT-engine only."""
    with mcp_session() as session:
        return tools.add_overlay(
            session, project_id, media_id,
            start_time=start_time, duration=duration, x=x, y=y, w=w, h=h, opacity=opacity,
        )


@mcp_server.tool()
@to_worker_thread
def editor_update_overlay(
    project_id: str,
    element_id: str,
    start_time: float | None = None,
    duration: float | None = None,
    x: float | None = None,
    y: float | None = None,
    w: float | None = None,
    h: float | None = None,
    opacity: float | None = None,
) -> dict:
    """Partial in-place overlay edit (id stable) — the canvas-drag/resize path
    for a PiP/sticker placed by ``editor_add_overlay``. Only provided fields
    change; ``x``/``y``/``w``/``h``/``opacity`` are normalized [0,1] frame
    fractions, ``w``/``h`` must stay positive."""
    with mcp_session() as session:
        return tools.update_overlay(
            session, project_id, element_id,
            start_time=start_time, duration=duration,
            x=x, y=y, w=w, h=h, opacity=opacity,
        )


@mcp_server.tool()
@to_worker_thread
def editor_render_preview(project_id: str, at_time: float = 0.0):
    """Render a single frame of the timeline at ``at_time`` and return it as an
    image the agent can actually see — its 'eyes' on the edit before export.
    Runs in-process (this server's container already has ffmpeg + the media
    cache) — no separate render service to hop to."""
    with mcp_session() as session:
        try:
            with mcp_metered_slots(session, "render_preview"):
                result = tools.render_preview(session, project_id, at_time)
        except DailyQuotaExceededError as exc:
            return quota_refusal(exc)
        except HeavyOpBusyError as exc:
            return {"error": str(exc)}
    if "error" in result:
        return result
    return Image(path=result["preview_path"])


@mcp_server.tool()
@to_worker_thread
def editor_analyze_media(
    source: str | None = None,
    project_id: str | None = None,
    media_id: str | None = None,
    max_events: int = 4,
) -> dict:
    """Find the "moments of meaning" in one clip — punchlines, hard cuts — and
    render a keyframe strip for each. Deterministic CPU math (motion-diff over
    downscaled frames + audio RMS, z-scored against a rolling baseline), no
    LLM, ~2s per clip.

    The PROJECT is never changed — no element, no version bump. The result is
    persisted, though, keyed by a hash of the media's own bytes: the first call
    for a given file computes and stores it, and every later call for the same
    bytes (same file, a copy of it, or the same clip in another project) reads
    that back without decoding anything. So this is cheap to call repeatedly,
    and analysis stays visible in the UI without anyone re-running it. A stored
    result computed under an older contract version is recomputed, never
    returned.

    Address the media in exactly ONE of two ways:

    - ``source`` — a local file path or an http(s) URL. Use this when you hold
      a URL and no editor project (agents, external pipelines). The
      clip is fetched into a per-source cache and reused on re-analysis.
    - ``project_id`` + ``media_id`` — an asset already registered on a project
      by ``editor_add_media``. Use this inside an editing session.

    Passing both (or neither) returns ``{"error": ...}``.

    Returns the ``"version": 2`` contract ``{version, duration_sec, fps,
    events, cuts, cut_count, clip_type, payoff_at, keyframes, agent}``:

    - ``events``: up to ``max_events`` moments, each ``{t, motion_z, audio_z,
      combined, score, kind}``. ``t`` is seconds into the SOURCE file, NOT the
      timeline — for ``editor_split_clip`` convert with
      ``at_time = clip.start_time + (t - clip.trim_start)``. ``kind`` is
      ``"action"`` (something happened) or ``"cut"`` (an editing cut already
      present in the source).

      RANK ON ``score``, NEVER ON ``combined``. ``score`` is the normalised
      strength that actually selected the event and is the only field
      comparable across events. ``combined`` is the raw geometric mean of the
      two channels and is legitimately ~0 on a solo-detected event — one where
      the clip was clearly SEEN or HEARD but not both at the same instant
      (v2 detects these; v1 missed them entirely). Sorting on ``combined``
      makes exactly those events detectable but never selectable, which is the
      bug the version bump exists to flag.
    - ``cuts``: the ``t`` of every ``kind=="cut"`` event, for splitting a
      supercut back into its parts.
    - ``clip_type``: ``"event"`` or ``"mood"``. A mood clip has nothing that
      stands out — a texture/breather, not a punchline.
    - ``payoff_at``: the strongest action moment, or null on a mood clip. A
      trim must keep this inside the kept window.
    - ``keyframes``: ``/ui/media/analysis/...`` PNG URLs, one strip per event
      (before / at / after) — look at these instead of watching the clip.
    - ``agent``: all-null placeholders a vision session may judge and fill in
      downstream; deterministic code never writes them.
    """
    with mcp_session() as session:
        try:
            with mcp_metered_slots(session, "analyze_media"):
                return tools.analyze_media(
                    session,
                    source=source,
                    project_id=project_id,
                    media_id=media_id,
                    max_events=max_events,
                )
        except DailyQuotaExceededError as exc:
            return quota_refusal(exc)
        except HeavyOpBusyError as exc:
            return {"error": str(exc)}


@mcp_server.tool()
@to_worker_thread
def editor_validate(project_id: str, profile: str = "structural") -> dict:
    """Validate the project. ``profile="structural"`` (default) checks only
    for a broken/desynchronized render; ``profile="shorts"`` additionally
    enforces this project's own duration/clip-count bounds and audio-coverage
    rules. Returns {errors, ok}."""
    with mcp_session() as session:
        return tools.validate(session, project_id, profile=profile)


@mcp_server.tool()
@to_worker_thread
def editor_export(project_id: str, profile: str = "structural", preset: str = "shorts_1080") -> dict:
    """Validate and, when valid, render the project to an mp4 file (synchronous
    — export IS the render in this product). ``preset`` selects encoder
    knobs + an optional resolution override (``"shorts_1080"`` default,
    ``"preview_720"``, ``"master"`` — see ``app.config.EXPORT_PRESETS``).
    When the project has a ``cover_time`` set, also renders a full-res PNG
    cover frame beside the mp4. Returns {ok, output_path, cover_path,
    version, validation_errors}."""
    with mcp_session() as session:
        try:
            with mcp_metered_slots(session, "export"):
                return tools.export(session, project_id, profile=profile, preset=preset)
        except DailyQuotaExceededError as exc:
            return quota_refusal(exc)
        except HeavyOpBusyError as exc:
            return {"error": str(exc)}


@mcp_server.tool()
@to_worker_thread
def editor_reopen_project(project_id: str) -> dict:
    """Unlock an EXPORTED project for further edits, so it can be re-exported."""
    with mcp_session() as session:
        return tools.reopen_project(session, project_id)


@mcp_server.tool()
@to_worker_thread
def editor_delete_project(project_id: str) -> dict:
    """PERMANENTLY delete a project: its timeline, its operation journal, its
    annotations, its whole version history and its rendered artifacts. THIS
    CANNOT BE UNDONE — the version history goes with it, so there is no
    checkpoint left to restore from.

    The owner's UPLOADED FILES are NOT deleted: they belong to the account, not
    to the project, and stay on their media library shelf (``/library``).

    Refuses while an export is still rendering; retry once it finishes.
    OPERATOR tier — see ``app.mcp.trust``."""
    with mcp_session() as session:
        return tools.delete_project(session, project_id)


@mcp_server.tool()
@to_worker_thread
def editor_restore_snapshot(project_id: str, data: dict, reason: str = "undo") -> dict:
    """Undo/redo: persist a previously serialized snapshot (a full project
    JSON, as returned by ``editor_get_project``) as this project's NEXT
    version. The snapshot itself is not re-journaled — only ``reason`` is
    recorded in the op log."""
    with mcp_session() as session:
        return tools.restore_snapshot(session, project_id, data=data, reason=reason)


@mcp_server.tool()
@to_worker_thread
def editor_save_version(project_id: str) -> dict:
    """Freeze the project's CURRENT state as a version in the server-side
    history, and return it. A version is a CHECKPOINT, not a mutation: the
    editor persists every op immediately, so this is how you mark a state
    worth coming back to. Idempotent — calling it twice without editing in
    between returns the same version with ``created: false``."""
    with mcp_session() as session:
        return tools.create_checkpoint_tool(session, project_id)


@mcp_server.tool()
@to_worker_thread
def editor_list_versions(project_id: str) -> list[dict]:
    """The project's saved versions, newest first, each with the mutation
    number it froze, when it was taken, why (``manual``/``idle``) and the
    operation that produced that state. Version numbers are sparse — they are
    mutation numbers, not a separate 1,2,3 counter."""
    with mcp_session() as session:
        return tools.list_versions_tool(session, project_id)


@mcp_server.tool()
@to_worker_thread
def editor_restore_version(
    project_id: str, version: int, reason: str = "restore"
) -> dict:
    """Roll the project back to a version the SERVER holds, by number — the
    counterpart to ``editor_restore_snapshot``, which needs you to still be
    holding the old document yourself. Like it, this rolls forward: the
    restored state is persisted as the project's NEXT version, never a rewind
    of the counter. An unknown version returns ``{"error": ...}``."""
    with mcp_session() as session:
        return tools.restore_version(
            session, project_id, version=version, reason=reason
        )


@mcp_server.tool()
@to_worker_thread
def editor_get_history(project_id: str) -> list[dict]:
    """The full append-only operation journal for a project, oldest first —
    every mutation ever performed, not just the latest snapshot."""
    with mcp_session() as session:
        return tools.get_history_tool(session, project_id)


@mcp_server.tool()
@to_worker_thread
def editor_add_annotation(
    project_id: str,
    time_sec: float,
    note: str,
    severity: str = "major",
    element_id: str | None = None,
) -> dict:
    """Pin an operator error-marker to a project's timeline at ``time_sec``
    (seconds into the exported render). ``severity`` is ``blocker``/``major``/
    ``minor``; ``element_id`` optionally ties the marker to a specific timeline
    element. Annotations are placed AFTER export (review pass) and deliberately
    bypass the EXPORTED edit-lock — they never touch the timeline or its
    version. Returns the created annotation, or ``{"error": ...}``."""
    with mcp_session() as session:
        return tools.add_annotation(
            session, project_id, time_sec=time_sec, note=note, severity=severity, element_id=element_id
        )


@mcp_server.tool()
@to_worker_thread
def editor_list_annotations(project_id: str, include_resolved: bool = False) -> list[dict]:
    """A project's error-markers, oldest first. Resolved markers are hidden
    unless ``include_resolved`` is True."""
    with mcp_session() as session:
        return tools.list_annotations(session, project_id, include_resolved=include_resolved)


@mcp_server.tool()
@to_worker_thread
def editor_resolve_annotation(project_id: str, annotation_id: int) -> dict:
    """Mark one error-marker resolved so it drops out of the default list.
    ``{"resolved": True}`` on success, ``{"error": ...}`` when the id is not a
    marker on ``project_id``.

    ``project_id`` is required here even though the agent already knows which
    project it is reviewing: this wrapper is written by hand with a fixed
    argument list, so nothing forwards a scope it does not name. Without the
    parameter an agent had no way to express which project it meant, and the
    tool resolved markers by a bare global id.
    """
    with mcp_session() as session:
        return tools.resolve_annotation(session, project_id, annotation_id)


def _uvicorn_proxy_kwargs(settings) -> dict:
    """How PROXY_TRUSTED_IPS maps onto uvicorn (see ``Settings`` for the
    deploy decision itself). Explicit ``proxy_headers=False`` when no proxy
    is declared matters: uvicorn's DEFAULT is True with 127.0.0.1 trusted,
    which would let any localhost peer spoof X-Forwarded-For into the
    rate-limit key — the opposite of what app/web/ratelimit.py documents."""
    if settings.proxy_trusted_ips:
        return {
            "proxy_headers": True,
            "forwarded_allow_ips": settings.proxy_trusted_ips,
        }
    return {"proxy_headers": False}


def _uvicorn_capacity_kwargs(settings) -> dict:
    """Connection-capacity bounds for uvicorn (see ``Settings``): a hard cap
    past which new connections get an immediate 503, and a keep-alive idle
    timeout. Split from ``_uvicorn_proxy_kwargs`` because the two answer
    different deploy questions (trust vs. capacity) and are tested apart."""
    return {
        "limit_concurrency": settings.http_limit_concurrency,
        "timeout_keep_alive": settings.http_timeout_keep_alive,
    }


def _compose_http_app(server: FastMCP):
    """The ASGI app for the streamable-http transport — with or without the
    cloud shell.

    In the private repository ``app.web.app.build_app`` wraps the SDK's MCP app
    with the landing page, ``/api``, ``/auth`` and the SPA — one process, one
    port. The public extract (E-01) ships no ``app.web`` at all, and the SDK
    app is self-contained, so the honest composition there is the SDK app
    alone. The ImportError branch is that deployment mode, not a fallback for
    a broken install: everything else in this module is copied into the
    extract verbatim, and this is the one place composition differs.
    """
    try:
        from app.web.app import build_app
    except ImportError:
        return server.streamable_http_app()
    return build_app(server)


def main() -> None:
    setup_logging(get_settings().log_level)
    init_db()
    # Gate 3 Step 9: start the periodic background retention sweep (previews/
    # analysis/voiceover/analysis_cache/uploads via app.mcp.maintenance_cli,
    # plus exports/clips/editor_text via app.mcp.retention). A no-op when
    # Settings.retention_enabled is False. It resolves the ARTIFACT_BACKEND
    # store itself (same app.storage.get_artifact_store the web layer uses),
    # so it sweeps the S3 bucket on an S3 deploy and raises here — before the
    # server starts serving — if that backend is misconfigured. Imported
    # lazily, here rather than
    # at module scope, so importing app.mcp.server (every test in this suite
    # does) never pulls in app.mcp.retention or touches its worker state as
    # a side effect of the import alone.
    from app.mcp.retention import start_retention_worker

    start_retention_worker()
    # The version-history counterpart: freeze a project as a version once the
    # editing stops (Settings.project_checkpoint_idle_minutes). Same lazy import
    # for the same reason, and a no-op when checkpoint_sweep_enabled is False.
    from app.editor.checkpoints import start_checkpoint_worker

    start_checkpoint_worker()
    transport = os.environ.get("MCP_TRANSPORT", get_settings().mcp_transport)
    if transport == "streamable-http":
        import uvicorn

        settings = get_settings()
        app = _compose_http_app(mcp_server)
        uvicorn.run(
            app,
            host=settings.mcp_host,
            port=settings.mcp_port,
            log_level=settings.log_level.lower(),
            **_uvicorn_proxy_kwargs(settings),
            **_uvicorn_capacity_kwargs(settings),
        )
    else:
        mcp_server.run(transport=transport)


if __name__ == "__main__":
    main()
