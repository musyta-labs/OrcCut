"""Immutable timeline-project data model for the agent video editor.

Every structure here is a frozen dataclass — edits never mutate in place;
mutation helpers (``app.editor.mutations``) return a new ``EditorProject`` with
an incremented ``version``. Collections are tuples, not lists, so a project is
fully hashable/immutable. Defaults target a 9:16 (1080x1920) Short at 30 fps.

This model is client-agnostic: a project only knows an opaque ``metadata`` tag
set and path/URL media sources — it has no notion of a niche, a video row, or
a compilation. Shorts-specific bounds (clip count, duration) are per-project
data (``min_clips``/``max_clips``/``min_duration_sec``/``max_duration_sec``),
not hardcoded config, so a client that doesn't care about them just leaves the
defaults in place.
"""
from __future__ import annotations

from dataclasses import dataclass

# 9:16 vertical Short defaults.
DEFAULT_ASPECT_W = 1080
DEFAULT_ASPECT_H = 1920
DEFAULT_FPS = 30

# Layer/audio defaults.
DEFAULT_CLIP_VOLUME = 1.0
DEFAULT_MUSIC_VOLUME = 0.15

# Text defaults.
DEFAULT_TEXT_SIZE = 64
DEFAULT_TEXT_COLOR = "#FFFFFF"
DEFAULT_TEXT_POS = "center"

TEXT_POSITIONS = ("top", "center", "bottom")
TRACK_TYPES = ("video", "overlay", "text", "audio")

# TransitionSpec.kind values. MLT 7.30 has no "dissolve" mlt_service (smoke-
# tested 2026-07-17: `melt` logs "failed to load transition 'dissolve'" and
# exits 0 anyway — see mlt_graph._transition_mlt_service) — a plain
# cross-dissolve is `mlt_service="luma"` with no resource property. "dissolve"
# is kept as the public kind name (the intuitive term for an agent) and maps
# internally to that same luma-with-no-resource transition.
TRANSITION_KINDS = ("dissolve",)
DEFAULT_TRANSITION_DURATION = 0.5
# ClipElement.fit values (canvas fill mode). "cover" is today's fill-scale +
# center-crop (unchanged default); "contain_blur" is the mlt-only blur-fill
# canvas (app.editor.mlt_graph._blur_fill_xml).
FIT_VALUES = ("cover", "contain_blur")

# Export presets live in app.config (EXPORT_PRESETS / DEFAULT_EXPORT_PRESET)
# — a render/output concern, not part of the timeline data model itself.

# Loudness defaults for the finished mix. Measured on 6 trending "Ranking"
# Shorts (2026-07-15): integrated -10..-14 LUFS with LRA <= 6 — 4/6 are pushed
# through a limiter (LRA 2.4-6.1), and the three tightest are the top three by
# views. -12 LUFS sits mid-window; -1 dBTP is the standard delivery ceiling.
# (n=6: the LRA/views link is a correlation, so this is a target, not a law.)
DEFAULT_LOUDNESS_I = -12.0
DEFAULT_LOUDNESS_LRA = 6.0
DEFAULT_LOUDNESS_TP = -1.0


@dataclass(frozen=True)
class VolumeKeyframe:
    """One control point of a clip/audio volume envelope: at ``time`` seconds
    (relative to the element's OWN ``start_time``, i.e. its local timeline),
    the effective linear volume multiplier is ``volume`` (0.0 = silent, 1.0 =
    unchanged). Consecutive keyframes are linearly interpolated by the
    renderer; see ``app.editor.mlt_graph`` for how this combines with
    ``fade_in_sec``/``fade_out_sec`` into one animated MLT ``level`` (dB)
    property."""

    time: float
    volume: float


@dataclass(frozen=True)
class Transform:
    """Per-clip geometry/speed transform. Positions are normalized offsets
    (0.0 = centered); ``speed`` is a playback multiplier; ``crop_zoom`` a
    normalized zoom-in factor (0.0 = no crop)."""

    scale: float = 1.0
    pos_x: float = 0.0
    pos_y: float = 0.0
    speed: float = 1.0
    crop_zoom: float = 0.0


@dataclass(frozen=True)
class LoudnessTarget:
    """EBU R128 normalization target for the finished mix (``ffmpeg loudnorm``).

    ``i`` is integrated loudness in LUFS, ``lra`` the loudness range, ``tp`` the
    true-peak ceiling in dBTP. This is a property of the decoded mix, so it
    cannot be expressed with the per-element linear ``volume`` multipliers —
    it is applied once, as the last stage of the audio chain.
    """

    i: float = DEFAULT_LOUDNESS_I
    lra: float = DEFAULT_LOUDNESS_LRA
    tp: float = DEFAULT_LOUDNESS_TP


@dataclass(frozen=True)
class TextStyle:
    """Rendering style for a text element.

    Placement has two tiers. ``pos`` is a named preset in ``TEXT_POSITIONS``
    (horizontally centred, at a fixed inset from the top/middle/bottom) and
    stays the default. ``pos_x``/``pos_y`` override it per axis with a
    normalized [0,1] fraction of the frame, anchoring the text box's TOP-LEFT
    corner — which is what lets labels of differing widths line up their left
    edges into a board. Either may be set alone: ``pos_x`` with no ``pos_y``
    keeps the preset's vertical placement.

    THE TYPEFACE HAS TWO FIELDS AND ONLY ONE OF THEM IS A CHOICE. ``font_id``
    is the choice: ``None`` means the deployment's configured font, anything
    else names an asset on the OWNER's media-library shelf. ``font_path`` is the
    RESULT — a local path the server derived from that id (see
    ``app.editor.fonts.with_resolved_font_paths``), rewritten on every ingress
    and again before every render, never taken from whatever a client sent. It
    used to be the only field, which made it a client-writable path into
    ``ImageFont.truetype`` and so an oracle for which files exist on the server
    (R-33).
    """

    font_path: str
    size: int = DEFAULT_TEXT_SIZE
    color: str = DEFAULT_TEXT_COLOR
    pos: str = DEFAULT_TEXT_POS
    pos_x: float | None = None
    pos_y: float | None = None
    font_id: str | None = None


@dataclass(frozen=True)
class TransitionSpec:
    """A cross-dissolve between a clip and its immediate predecessor on the
    same video track (see ``ClipElement.transition_in``).

    ``duration`` is the full overlap length in seconds: it consumes the last
    ``duration`` seconds of the PREVIOUS clip's own span and the first
    ``duration`` seconds of THIS clip's own span, playing them simultaneously
    while dissolving between them (the standard NLE cross-dissolve). This
    shrinks the total rendered length by ``duration`` versus a plain back-to-
    back cut — see ``app.editor.mlt_graph``'s transition builder and
    ``app.editor.validation._check_transitions`` for the exact bound.
    """

    kind: str = "dissolve"
    duration: float = DEFAULT_TRANSITION_DURATION


@dataclass(frozen=True)
class TransformKeyframe:
    """One keyframe of an animated per-clip affine transform.

    ``time`` is seconds relative to the CLIP's own start (0 = the clip's first
    frame), not absolute timeline time. ``scale``/``pos_x``/``pos_y`` mirror
    ``Transform``'s static fields (normalized offsets, 1.0 = no zoom);
    ``opacity`` is 0..1; ``rotation`` is degrees. A ``ClipElement`` with a
    single keyframe (or none, falling back to its static ``transform`` fields)
    renders a non-animated affine rect — see ``ClipElement.keyframes``.
    """

    time: float
    scale: float = 1.0
    pos_x: float = 0.0
    pos_y: float = 0.0
    opacity: float = 1.0
    rotation: float = 0.0


@dataclass(frozen=True)
class ColorAdjust:
    """Per-clip color grade, applied via MLT's ``avfilter.eq`` bridge filter.

    ``brightness`` is -1.0..1.0 (0 = unchanged), ``contrast``/``saturation``/
    ``gamma`` are multipliers (1.0 = unchanged) — the same ranges ffmpeg's
    ``eq`` filter itself accepts, since avfilter.eq wraps it directly.
    """

    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    gamma: float = 1.0


@dataclass(frozen=True)
class ClipElement:
    """A video clip placed on a track. ``trim_end`` None = play to media end."""

    id: str
    media_id: str
    start_time: float
    duration: float
    kind: str = "clip"
    trim_start: float = 0.0
    trim_end: float | None = None
    transform: Transform = Transform()
    volume: float = DEFAULT_CLIP_VOLUME
    muted: bool = False
    #: Composite opacity 0..1 for an UPPER (V2+) video-track clip — the mix
    #: field of its affine composite transition over the main track (see
    #: ``app.editor.mlt_graph``). Ignored on the first (V1) video track, which
    #: is the opaque base. Serialization back-compat: absent = 1.0.
    opacity: float = 1.0
    #: Cross-dissolve from the previous clip on the same video track into this
    #: one. ``None`` = a plain cut (today's behavior). MLT-engine only.
    transition_in: TransitionSpec | None = None
    #: Animated affine transform. Empty = static (falls back to ``transform``'s
    #: own scale/pos_x/pos_y as a single, non-animated point). MLT-engine only.
    keyframes: tuple[TransformKeyframe, ...] = ()
    #: Color grade. ``None`` = untouched. MLT-engine only.
    color: ColorAdjust | None = None
    #: Canvas fill mode for a source whose aspect differs from the project's.
    #: "cover" (default) is today's fill-scale + center-crop, byte-for-byte
    #: unchanged. "contain_blur" letterboxes the source at full extent over a
    #: blurred, cover-scaled copy of itself (mlt engine only — see
    #: app.editor.mlt_graph._blur_fill_xml).
    fit: str = "cover"
    #: Audio envelope on this clip's OWN audio (independent of ``volume``,
    #: which stays the sustained/base level the envelope ramps to/from). See
    #: ``VolumeKeyframe``; MLT engine only (app.editor.mlt_graph).
    fade_in_sec: float = 0.0
    fade_out_sec: float = 0.0
    volume_keyframes: tuple[VolumeKeyframe, ...] = ()


@dataclass(frozen=True)
class TextElement:
    """A text overlay (hook title / caption). ``duration`` None = persistent:
    hold from ``start_time`` to the end of the timeline (a header/"шапка")."""

    id: str
    content: str
    start_time: float
    duration: float | None
    style: TextStyle
    #: Optional role marker (e.g. ``"caption"`` for an auto-caption line) that
    #: lets a consumer tell machine-generated text apart from a hand-authored
    #: hook/board slot. ``None`` = unmarked. Serialization back-compat: absent
    #: = None (mirrors ``ClipElement.opacity``).
    role: str | None = None
    kind: str = "text"


@dataclass(frozen=True)
class AudioElement:
    """A music/audio bed. ``gain_db`` optional level adjustment."""

    id: str
    media_id: str
    start_time: float
    duration: float
    kind: str = "audio"
    volume: float = DEFAULT_MUSIC_VOLUME
    gain_db: float = 0.0
    #: Same envelope contract as ``ClipElement`` (see ``VolumeKeyframe``);
    #: mlt engine only.
    fade_in_sec: float = 0.0
    fade_out_sec: float = 0.0
    volume_keyframes: tuple[VolumeKeyframe, ...] = ()


@dataclass(frozen=True)
class OverlayElement:
    """A PiP/sticker image or video clip on the ``overlay`` track (its own
    track type, exempt from the video-track overlap/contiguity rules exactly
    like text — see ``app.editor.validation``).

    ``x``/``y``/``w``/``h`` are normalized [0,1] fractions of the frame (top-
    left anchored, same convention as ``TextStyle.pos_x``/``pos_y``): the
    rect this overlay occupies. ``opacity`` is 0..1. It always plays its full
    source from time 0 for ``duration`` seconds (no trim fields) — MLT-engine
    only, composited via a bare-property ``affine`` transition mirroring
    ``mlt_graph._text_affine_transition_xml`` (the 7bcee79 lesson: bare
    ``rect``/``fill``/``distort``, no ``transition.`` prefix, on the
    transition itself).
    """

    id: str
    media_id: str
    start_time: float
    duration: float
    x: float
    y: float
    w: float
    h: float
    kind: str = "overlay"
    opacity: float = 1.0


Element = ClipElement | TextElement | AudioElement | OverlayElement


@dataclass(frozen=True)
class Track:
    """A typed track holding an ordered tuple of elements."""

    id: str
    type: str
    elements: tuple[Element, ...] = ()


@dataclass(frozen=True)
class MediaAsset:
    """A registered media source (local path or URL), resolved on demand.
    ``local_path``/``duration_sec`` fill in once known."""

    id: str
    source: str
    duration_sec: float | None = None
    local_path: str | None = None


@dataclass(frozen=True)
class EditorProject:
    """The full immutable timeline project. Persisted as JSON; each edit
    yields a new version (history/undo for free).

    ``metadata`` is an opaque client-owned tag set (e.g.
    ``(("client", "annotator"), ("niche", "cats"))``) — the editor never
    interprets it. It MUST stay a tuple of pairs, not a dict: a dict field on a
    frozen dataclass breaks the "fully hashable" invariant this model commits
    to (a dict is unhashable, and the auto-derived ``__hash__`` would raise the
    first time anyone hashes a project holding one). Convert to/from a plain
    dict only at the MCP tool boundary via ``metadata_to_dict``/
    ``metadata_from_dict``.
    """

    id: str
    metadata: tuple[tuple[str, str], ...] = ()
    version: int = 1
    aspect_w: int = DEFAULT_ASPECT_W
    aspect_h: int = DEFAULT_ASPECT_H
    fps: int = DEFAULT_FPS
    target_sec: float | None = None
    #: Shorts-specific(ish) editorial bounds live as per-project DATA, not
    #: hardcoded config — a client that doesn't care leaves the defaults.
    min_clips: int = 1
    max_clips: int | None = None
    min_duration_sec: float = 0.0
    max_duration_sec: float | None = None
    tracks: tuple[Track, ...] = ()
    assets: tuple[MediaAsset, ...] = ()
    #: Normalization applied to the final mix. On by default — an un-normalized
    #: Short is a defect, so opting out (``None``) must be deliberate.
    loudness: LoudnessTarget | None = LoudnessTarget()
    #: Absolute timeline time (seconds) of the frame exported as the cover PNG
    #: alongside the mp4 (see ``app.editor.render.render_project_file`` /
    #: ``editor_set_cover``). ``None`` = no cover frame requested.
    cover_time: float | None = None


def metadata_to_dict(metadata: tuple[tuple[str, str], ...]) -> dict[str, str]:
    """Convert the internal tuple-of-pairs representation to a plain dict, for
    the MCP tool boundary (JSON in/out)."""
    return dict(metadata)


def metadata_from_dict(data: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    """Convert a plain dict (or None) from the MCP tool boundary to the
    internal, hashable tuple-of-pairs representation. Sorted so equal metadata
    always compares equal regardless of the caller's key order."""
    return tuple(sorted((data or {}).items()))
