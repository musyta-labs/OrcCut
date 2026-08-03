"""Runtime type validation for tool arguments that arrive as untrusted JSON.

Python does not enforce annotations at runtime: ``set_transform(..., pos_x=…)``
happily accepts the string ``"1;movie=/etc/passwd"``, stores it in a frozen
dataclass and lets ``app.editor.ffmpeg_graph`` concatenate it into an
``overlay=x=W*{pos_x}`` expression. That is an injection into ffmpeg's own
filter DSL — not a shell injection (every subprocess is a strict argv list),
but ``movie=``/``amovie=`` sources read local files and sidestep the
``app.net.egress`` SSRF guard completely. The op-name and arg-name allowlists
in ``app.web.api`` check WHICH op and WHICH arguments; nothing checked the
values. This module does (R-04).

The validator is derived from the tool's own signature, so it cannot drift from
the implementation: one pydantic model per op, built once and cached, mirroring
what FastMCP already does for the MCP surface (see
``mcp.server.fastmcp.utilities.func_metadata``). Applied by
``app.web.api._invoke_tool`` and ``app.mcp.cli._run_op`` — the two surfaces that
call the tools with raw JSON kwargs.

Two deliberate design points:

* **Lax, not strict.** ``JSON.stringify(2.0)`` is ``2``, so the SPA sends ints
  for float fields on nearly every gesture; strict mode would reject those and
  break the editor. Laxness costs nothing here because the security property is
  that the value REACHING the model is a real number — a string only survives
  if it parses as a whole number, and ``"1;movie=…"`` does not. ``allow_inf_nan``
  is off on top: ``json.loads`` accepts the non-standard ``Infinity``/``NaN``
  literals, and ``W*inf`` is a filter expression that only wastes a render slot.
* **Signatures are not always precise enough.** ``keyframes: list[dict]`` says
  nothing about the floats inside, and those floats are concatenated into MLT
  property strings the same way. ``ARG_OVERRIDES`` supplies the missing shape as
  a ``TypedDict``; the MCP wrappers in ``app.mcp.server`` annotate with the SAME
  TypedDicts, so all three surfaces validate one description.
"""
from __future__ import annotations

import inspect
import types
from functools import lru_cache
from typing import Any, NotRequired, TypedDict, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from app.mcp import tools

# Shared by every generated model AND by the TypedDicts below: unknown keys are
# refused outright (an unexpected kwarg would only TypeError one frame later,
# but an ALLOWLIST posture is what keeps a same-named server-side kwarg from
# being smuggled in), and non-finite floats never reach a filter expression.
_ARG_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    allow_inf_nan=False,
    # ``analyze_media`` takes a ``store_scope`` callable and ``delete_project`` a
    # ``Path``/``ArtifactStore``; both are injected server-side and never
    # request-reachable, but the model still has to be constructible.
    arbitrary_types_allowed=True,
)


class TransformKeyframeArgs(TypedDict):
    """One item of ``set_keyframes(keyframes=…)`` — mirrors
    ``app.editor.model.TransformKeyframe``, whose values become MLT
    ``frame=value;…`` property strings."""

    __pydantic_config__ = _ARG_MODEL_CONFIG  # type: ignore[misc]

    time: float
    scale: NotRequired[float]
    pos_x: NotRequired[float]
    pos_y: NotRequired[float]
    opacity: NotRequired[float]
    rotation: NotRequired[float]


class VolumeKeyframeArgs(TypedDict):
    """One item of ``set_audio_envelope(volume_keyframes=…)`` — mirrors
    ``app.editor.model.VolumeKeyframe``."""

    __pydantic_config__ = _ARG_MODEL_CONFIG  # type: ignore[misc]

    time: float
    volume: float


class TextUpdateArgs(TypedDict):
    """One item of ``update_texts(updates=…)`` (content-only batch edit)."""

    __pydantic_config__ = _ARG_MODEL_CONFIG  # type: ignore[misc]

    element_id: str
    content: str


class CaptionStyleArgs(TypedDict):
    """``auto_captions(style_overrides=…)``, splatted into ``TextStyle(**…)``.

    ``font_path`` is deliberately ABSENT: it is a server-resolved path that
    becomes an ffmpeg input, so a caller that could set it would gain an
    arbitrary local-file read through the caption rasterizer — and, before
    R-33, a way to ask whether any given file exists on the server.

    ``font_id`` is the supported way to choose a typeface: an id of an asset on
    the CALLER's own media-library shelf, which the server resolves to a path
    (``app.mcp.tools._font_resolver``). An id is safe where a path is not
    precisely because resolving it is owner-scoped and refuses anything else."""

    __pydantic_config__ = _ARG_MODEL_CONFIG  # type: ignore[misc]

    size: NotRequired[int]
    color: NotRequired[str]
    pos: NotRequired[str]
    pos_x: NotRequired[float | None]
    pos_y: NotRequired[float | None]
    font_id: NotRequired[str | None]


# Parameters whose declared annotation is structurally too loose to validate
# (a bare ``dict``/``list[dict]`` passes any content). Keyed by tool function
# name, then parameter name.
ARG_OVERRIDES: dict[str, dict[str, Any]] = {
    "set_keyframes": {"keyframes": list[TransformKeyframeArgs]},
    "set_audio_envelope": {"volume_keyframes": list[VolumeKeyframeArgs] | None},
    "update_texts": {"updates": list[TextUpdateArgs]},
    "auto_captions": {"style_overrides": CaptionStyleArgs | None},
}

# The session is the caller's, never the payload's.
#
# NOTE ``restore_snapshot(data=…)`` is only shape-checked here (it must be a
# dict): it is a WHOLE project snapshot, and describing it in this module would
# duplicate ``app.editor.serialization`` and drift from it. Its scalars are
# checked structurally against the model dataclasses instead — see
# ``app.editor.typecheck``, which covers the MCP surface in the same stroke.
_INJECTED_PARAMS = frozenset({"session"})


class ArgValidationError(Exception):
    """Raised with a caller-safe, single-line summary. Never carries pydantic's
    own rendering (URLs, class names, the offending value echoed back)."""


def _strip_sentinel(annotation: Any) -> Any:
    """Drop ``object`` from a union.

    ``tools.update_text`` annotates its tri-state fields ``float | None | object``
    because an ``_UNSET`` sentinel distinguishes "omitted" from "explicit null".
    ``object`` accepts literally anything, so leaving it in would make those
    fields — ``pos_x``/``pos_y`` among them — unvalidated. No caller can send a
    sentinel over JSON, so removing it is lossless.
    """
    if get_origin(annotation) not in (Union, types.UnionType):
        return annotation
    members = tuple(arg for arg in get_args(annotation) if arg is not object)
    if not members or len(members) == len(get_args(annotation)):
        return annotation
    return Union[members]  # noqa: UP007 - built dynamically from a tuple


@lru_cache(maxsize=None)
def _arg_model(op: str) -> type[BaseModel]:
    """The cached pydantic model for one tool's keyword arguments.

    Every field defaults to ``None`` regardless of the tool's own default: this
    model only type-checks the keys the caller actually SENT (see
    ``validate_args``). Missing required arguments stay the callee's business —
    the ``TypeError`` the call raises already maps to the same 422.
    """
    func = getattr(tools, op)
    hints = get_type_hints(func)
    overrides = ARG_OVERRIDES.get(op, {})
    fields: dict[str, Any] = {
        name: (overrides.get(name) or _strip_sentinel(hints.get(name, Any)), None)
        for name in inspect.signature(func).parameters
        if name not in _INJECTED_PARAMS
    }
    return create_model(f"{op}_args", __config__=_ARG_MODEL_CONFIG, **fields)


def _summarize(exc: ValidationError) -> str:
    """One flat ``field: reason`` line. ``include_url=False`` keeps
    errors.pydantic.dev links out of an API response, and the offending input is
    never echoed — it is attacker-chosen text."""
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or '<args>'}: {error['msg']}"
        for error in exc.errors(include_url=False)
    )


def validate_args(op: str, args: dict) -> dict:
    """Type-check ``args`` against ``tools.<op>``'s signature.

    Returns a NEW dict of the same keys with values coerced to their declared
    types (``2`` in a ``float`` field becomes ``2.0``); raises
    ``ArgValidationError`` when a value cannot be that type. Keys the caller did
    not send are never added, so each tool's own defaults still apply.
    """
    try:
        validated = _arg_model(op).model_validate(args)
    except ValidationError as exc:
        raise ArgValidationError(_summarize(exc)) from exc
    return {name: getattr(validated, name) for name in args}
