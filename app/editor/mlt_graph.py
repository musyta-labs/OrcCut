"""Pure MLT XML builder for the MLT render engine.

Mirrors ``app.editor.ffmpeg_graph`` in spirit (a pure, snapshot-testable
compiler from an ``EditorProject`` into a render description) but targets
melt's XML producer/playlist/tractor/consumer format instead of an ffmpeg
``-filter_complex`` graph. No I/O — this module never touches ``melt``,
``ffmpeg`` or the filesystem beyond the paths it is handed; the orchestration
(resolve media, rasterize text, run ``melt``, verify + atomically replace the
output) lives in ``app.editor.render._render_via_mlt``.

Only stdlib is imported at module level (``xml.sax.saxutils``) so importing
this module stays cheap, mirroring the LAZY_IMPORTS convention elsewhere in
``app.editor``.

Deliberately does NOT import ``app.editor.render`` (render.py imports this
module for the mlt render path; the reverse would be a cycle) — the small
ordered-element collectors and ``_video_timeline_end`` below are therefore
reimplemented here rather than shared, mirroring their render.py originals
byte-for-byte in behavior.
"""
from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from app.editor.errors import EditorError
from app.editor.ffmpeg_graph import (
    TEXT_MARGIN,
    _clip_out_point,
)
from app.editor.model import (
    AudioElement,
    ClipElement,
    ColorAdjust,
    EditorProject,
    OverlayElement,
    TextElement,
    VolumeKeyframe,
)

# avformat consumer output sample rate. loudnorm's avfilter bridge internally
# upsamples to 192kHz, so the consumer must force a sane final rate.
CONSUMER_SAMPLE_RATE = 44100
MLT_VERSION = "7.12"

# Audio envelope (fade/keyframe) floor in dB — silence would be -inf, which
# an animated MLT property cannot express; -60dB is inaudible against any
# realistic mix (the loudness target ceiling is -1 dBTP).
ENVELOPE_FLOOR_DB = -60.0


# --------------------------------------------------------------------------- #
# Ordered element collectors (pure) — mirror app.editor.render's originals.
# --------------------------------------------------------------------------- #
def _video_tracks(project: EditorProject) -> list:
    return [track for track in project.tracks if track.type == "video"]


def _main_video_clips(project: EditorProject) -> list[ClipElement]:
    """Clips of the FIRST (V1/main) video track, timeline-ordered — the base
    ``playlist0``/track 0. This is the contiguous main track that defines the
    film's length; V2+ upper tracks composite over it (see ``build_mlt_xml``)."""
    tracks = _video_tracks(project)
    if not tracks:
        return []
    clips = [el for el in tracks[0].elements if isinstance(el, ClipElement)]
    return sorted(clips, key=lambda clip: clip.start_time)


def _upper_video_tracks(project: EditorProject) -> list:
    """Every video track AFTER the first — the V2+ tracks composited over V1."""
    return _video_tracks(project)[1:]


def _upper_video_clips(track) -> list[ClipElement]:
    """One upper (V2+) video track's clips, timeline-ordered."""
    clips = [el for el in track.elements if isinstance(el, ClipElement)]
    return sorted(clips, key=lambda clip: clip.start_time)


def _ordered_music(project: EditorProject) -> list[AudioElement]:
    music = [
        element
        for track in project.tracks
        for element in track.elements
        if isinstance(element, AudioElement)
    ]
    return sorted(music, key=lambda element: element.start_time)


def _ordered_texts(project: EditorProject) -> list[TextElement]:
    texts = [
        element
        for track in project.tracks
        for element in track.elements
        if isinstance(element, TextElement)
    ]
    return sorted(texts, key=lambda element: element.start_time)


def _ordered_overlays(project: EditorProject) -> list[OverlayElement]:
    overlays = [
        element
        for track in project.tracks
        for element in track.elements
        if isinstance(element, OverlayElement)
    ]
    return sorted(overlays, key=lambda element: element.start_time)


def _video_timeline_end(clips: list[ClipElement]) -> float:
    """The playback duration of the video timeline: the latest clip out-point.
    Mirrors ``app.editor.render._video_timeline_end``."""
    return max(clip.start_time + clip.duration for clip in clips)


def _transition_shrink_total(clips: list[ClipElement]) -> float:
    """Sum of every ``TransitionSpec.duration`` on ``clips``. Each one shrinks
    the rendered timeline by that much versus a plain back-to-back cut (see
    ``TransitionSpec``'s docstring for the exact overlap semantics)."""
    return sum(
        clip.transition_in.duration for clip in clips if clip.transition_in is not None
    )


def effective_timeline_end(clips: list[ClipElement]) -> float:
    """The ACTUAL rendered length of the video timeline, accounting for
    transitions. Public (unlike its siblings above) because
    ``app.editor.render._render_via_mlt`` needs the identical number to
    verify the melt output's probed duration — a transition's shrink is not
    something render.py can re-derive from clip start_time/duration alone
    without duplicating this exact formula."""
    return _video_timeline_end(clips) - _transition_shrink_total(clips)


# --------------------------------------------------------------------------- #
# Time / frame formatting
# --------------------------------------------------------------------------- #
def seconds_to_timecode(seconds: float) -> str:
    """Format ``seconds`` as an MLT fractional timecode ``HH:MM:SS.mmm``."""
    total_ms = max(round(seconds * 1000), 0)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _blank_length_frames(start_time: float, fps: int) -> int:
    """Frame count for a leading ``<blank>`` before an overlay/music entry."""
    return round(start_time * fps)


# --------------------------------------------------------------------------- #
# Lookup guards (mirror app.editor.render's _require_media/_require_text_png)
# --------------------------------------------------------------------------- #
def _require_media(resolved_media: dict[str, Path], media_id: str, owner_id: str) -> Path:
    path = resolved_media.get(media_id)
    if path is None:
        raise EditorError(
            f"No resolved media for element {owner_id!r} (media_id {media_id!r})"
        )
    return path


def _require_text_png(text_pngs: dict[str, Path], text_id: str) -> Path:
    path = text_pngs.get(text_id)
    if path is None:
        raise EditorError(f"No rasterized PNG for text element {text_id!r}")
    return path


def _require_png_size(
    text_png_sizes: dict[str, tuple[int, int]], text_id: str
) -> tuple[int, int]:
    size = text_png_sizes.get(text_id)
    if size is None:
        raise EditorError(f"No rasterized PNG size for text element {text_id!r}")
    return size


def _require_clip_dimensions(
    clip_dimensions: dict[str, tuple[int, int]], clip_id: str
) -> tuple[int, int]:
    size = clip_dimensions.get(clip_id)
    if size is None:
        raise EditorError(f"No source dimensions for blur-fill clip {clip_id!r}")
    return size


# --------------------------------------------------------------------------- #
# XML fragment builders
# --------------------------------------------------------------------------- #
def _prop(name: str, value: object) -> str:
    return f'<property name="{name}">{escape(str(value))}</property>'


def _volume_filter(gain: object) -> str:
    return f'<filter mlt_service="volume">{_prop("gain", gain)}</filter>'


def _affine_zoom_filter(zoom: float) -> str:
    """Anti-reuse crop-zoom as an ``affine`` filter attached to the clip's own
    producer (mltframework.org/plugins/FilterAffine): a centred zoom-in crop,
    the MLT equivalent of the ffmpeg engine's ``scale``+``crop`` pair."""
    pct = f"{(1.0 + zoom) * 100:.4f}%"
    rect = f"0%/0%:{pct}x{pct}"
    return (
        '<filter mlt_service="affine">'
        f'{_prop("transition.rect", rect)}'
        f'{_prop("transition.halign", "center")}'
        f'{_prop("transition.valign", "center")}'
        f'{_prop("transition.fill", "1")}'
        "</filter>"
    )


def _rect_pct(scale: float, pos_x: float, pos_y: float, opacity: float) -> str:
    """One affine ``rect`` value: a WxH-scaled, pos_x/pos_y-shifted rect that
    is CENTERED when ``pos_x == pos_y == 0`` — computed directly (rather than
    via the crop-zoom filter's ``halign``/``valign=center`` trick, which would
    discard any pos_x/pos_y shift by always re-centering the literal rect).
    ``pos_x``/``pos_y`` are normalized offsets from center (0.0 = centered,
    matching ``Transform``'s docstring); ``opacity`` is 0..1, encoded as the
    rect's trailing 0..100 mix field."""
    size_pct = scale * 100
    x_pct = (1.0 - scale) * 50 + pos_x * 100
    y_pct = (1.0 - scale) * 50 + pos_y * 100
    return f"{x_pct:.4f}%/{y_pct:.4f}%:{size_pct:.4f}%x{size_pct:.4f}%:{opacity * 100:.4f}"


def _has_transform_override(clip: ClipElement) -> bool:
    """True when the clip needs an animated/static affine transform filter at
    all — either explicit keyframes, or a static scale/pos_x/pos_y away from
    the neutral 1.0/0.0/0.0 (the degenerate single-keyframe case)."""
    if clip.keyframes:
        return True
    transform = clip.transform
    return transform.scale != 1.0 or transform.pos_x != 0.0 or transform.pos_y != 0.0


def _transform_affine_filter(clip: ClipElement, fps: int) -> str:
    """Build the affine FILTER for a clip's scale/pos_x/pos_y/rotation.

    Property names carry the ``transition.`` prefix HERE — the OPPOSITE
    convention from a bare affine TRANSITION (see
    ``_text_affine_transition_xml``'s docstring): on a FILTER, ``rect`` with
    no prefix is silently ignored and the clip renders unscaled (smoke-tested
    2026-07-17, the mirror image of the comp-9 transition bug).

    Animated (``clip.keyframes`` non-empty): each keyframe's ``time`` (seconds
    relative to the clip's own start) becomes a frame number
    (``round(time * fps)``); MLT property-animation syntax is
    ``frame=value;frame=value;...``. Static (no keyframes): a single value
    from ``clip.transform`` — the degenerate one-keyframe case, no rotation
    (``Transform`` has no rotation field; that is keyframe-only).

    ``transition.fill=1`` (NOT 0): a source SMALLER than the canvas (real
    TikTok is 576×1024 on a 1080×1920 canvas) renders at its native size
    pinned top-left with ``fill=0`` — never scaled up to the frame (comp-11
    "Cat Crimes" P1, smoke-tested 2026-07-17). With ``fill=1`` MLT first fits
    the producer to the frame, so the percentage rect then measures against
    the full canvas and a centered zoom/pos stays centered (the same reason
    the crop-zoom filter above already uses ``fill=1``). This also fixes the
    dissolve "rainbow" (P2): that garble was a luma blend over a half-black
    native-top-left frame produced by this very filter, not a transition bug.
    """
    if clip.keyframes:
        rect_parts = []
        rotate_parts = []
        for kf in clip.keyframes:
            frame = round(kf.time * fps)
            rect_parts.append(f"{frame}={_rect_pct(kf.scale, kf.pos_x, kf.pos_y, kf.opacity)}")
            rotate_parts.append(f"{frame}={kf.rotation}")
        rect_value = ";".join(rect_parts)
        rotate_value = ";".join(rotate_parts)
    else:
        transform = clip.transform
        rect_value = _rect_pct(transform.scale, transform.pos_x, transform.pos_y, 1.0)
        rotate_value = "0"
    return (
        '<filter mlt_service="affine">'
        f'{_prop("transition.rect", rect_value)}'
        f'{_prop("transition.fill", "1")}'
        f'{_prop("transition.fix_rotate_x", rotate_value)}'
        "</filter>"
    )


def _color_eq_filter(color: ColorAdjust) -> str:
    """MLT's ``avfilter.eq`` bridge (ffmpeg's ``eq`` filter, no resampling —
    bridge-safe for a video filter). Property names carry the ``av.`` prefix
    (smoke-tested 2026-07-17: a visibly darkened frame confirmed
    ``av.brightness`` takes effect)."""
    return (
        '<filter mlt_service="avfilter.eq">'
        f'{_prop("av.brightness", color.brightness)}'
        f'{_prop("av.contrast", color.contrast)}'
        f'{_prop("av.saturation", color.saturation)}'
        f'{_prop("av.gamma", color.gamma)}'
        "</filter>"
    )


def _has_envelope(element: ClipElement | AudioElement) -> bool:
    """True when an element carries a fade/keyframe audio envelope (as
    opposed to only the plain static ``volume``/``gain_db`` this codebase
    already supported)."""
    return (
        element.fade_in_sec > 0
        or element.fade_out_sec > 0
        or bool(element.volume_keyframes)
    )


def _linear_to_db(volume: float) -> float:
    """Convert a linear multiplier to dB, floored at ``ENVELOPE_FLOOR_DB``
    (0 or a near-zero value would be -inf, which the animated ``level``
    property cannot express)."""
    if volume <= 0:
        return ENVELOPE_FLOOR_DB
    return max(20.0 * math.log10(volume), ENVELOPE_FLOOR_DB)


def _envelope_points_db(
    *,
    base_volume: float,
    base_gain_db: float,
    duration: float,
    fade_in_sec: float,
    fade_out_sec: float,
    volume_keyframes: tuple[VolumeKeyframe, ...],
) -> list[tuple[float, float]]:
    """(local_time_sec, dB) control points for the animated ``level``
    property, ``local_time`` measured from the element's own ``start_time``
    (0 = the element begins, ``duration`` = it ends).

    ``base_volume``/``base_gain_db`` are the element's existing static level
    (``ClipElement.volume`` or ``AudioElement.volume``/``gain_db``) — the
    'sustain' loudness a fade ramps to/from and a keyframe's implicit
    baseline. A fade contributes two synthetic boundary points (silence at
    the very edge, the sustain level at the fade's far end); an explicit
    ``VolumeKeyframe`` at the SAME instant overrides a fade's synthetic point
    there (last-write-wins via the dict), since the caller asked for that
    exact level, not the fade's assumption.
    """
    base_db = _linear_to_db(base_volume) + base_gain_db
    points: dict[float, float] = {
        0.0: ENVELOPE_FLOOR_DB if fade_in_sec > 0 else base_db,
        duration: ENVELOPE_FLOOR_DB if fade_out_sec > 0 else base_db,
    }
    if fade_in_sec > 0:
        points[fade_in_sec] = base_db
    if fade_out_sec > 0:
        points[duration - fade_out_sec] = base_db
    for kf in volume_keyframes:
        points[kf.time] = _linear_to_db(kf.volume) + base_gain_db
    return sorted(points.items())


def _volume_envelope_filter(points: list[tuple[float, float]], *, fps: int, to_source_time) -> str:
    """Build the animated ``volume`` filter from (local_time, dB) control
    points. ``to_source_time`` maps a LOCAL time (0..duration) to the
    producer's own absolute source-time axis — the same axis its ``in``/
    ``out`` timecodes use.

    Empirically confirmed by smoke render: an animated filter property's keyframe
    FRAME NUMBERS are counted against the producer's absolute source frame
    axis, not restarted at 0 for a trimmed playback window — a clip trimmed
    to start at 2s and keyframed "0=0" does NOT mean "0dB at the start of
    playback"; it means "0dB at the source's frame 0" (2s before playback
    even begins). Every keyframe time must therefore go through
    ``to_source_time`` before being turned into a frame number, mirroring how
    ``_clip_producer_xml`` computes the producer's own ``in``/``out``.
    """
    keyframes = ";".join(
        f"{round(to_source_time(t) * fps)}={db:.4f}" for t, db in points
    )
    return f'<filter mlt_service="volume">{_prop("level", keyframes)}</filter>'


def _clip_source_time(local_time: float, clip: ClipElement) -> float:
    """Map a clip-local envelope time (0..duration, in TIMELINE seconds) to
    the clip's own producer source-time axis — ``trim_start + local_time`` at
    speed 1.0; for a ``timewarp:`` producer (speed != 1.0) that axis is
    RETIMED (divided by speed), the same transform ``_clip_producer_xml``
    applies to the producer's own ``in``/``out`` timecodes."""
    speed = clip.transform.speed
    if speed == 1.0:
        return clip.trim_start + local_time
    return clip.trim_start / speed + local_time


def _clip_resource_and_timecodes(path: Path, clip: ClipElement) -> tuple[str, str, str]:
    """Resolve a clip's producer ``resource`` + ``(in, out)`` timecodes.

    Speed == 1.0 uses the plain source path with trim_start/out_point as
    in/out. Speed != 1.0 uses a ``timewarp:`` resource — its in/out are in the
    RETIMED timebase (divided by speed), the one place the two engines'
    trim semantics structurally diverge (mltframework.org/plugins/ProducerTimewarp).

    A muted clip (or one with volume != 1.0) gets a ``volume`` filter; unlike
    the ffmpeg engine, a source with no audio stream needs nothing here — MLT
    plays silence natively for a producer lacking an audio stream.

    Also attaches, in order: the scale/pos_x/pos_y(/keyframe) affine transform
    (``_transform_affine_filter``, only when non-default), the anti-reuse
    crop-zoom affine (unchanged), and the color grade (``_color_eq_filter``,
    only when ``clip.color`` is set) — three independent filters stacked in
    sequence, each operating on the previous one's output.
    Shared by the plain clip producer and the blur-fill background producer
    (same clip, same window — only the visual filters differ).
    """
    speed = clip.transform.speed
    out_point = _clip_out_point(clip)
    if speed == 1.0:
        return str(path), seconds_to_timecode(clip.trim_start), seconds_to_timecode(out_point)
    return (
        f"timewarp:{speed}:{path}",
        seconds_to_timecode(clip.trim_start / speed),
        seconds_to_timecode(out_point / speed),
    )


def _clip_audio_filters(clip: ClipElement, *, fps: int) -> str:
    """A muted clip (or one with volume != 1.0) gets a ``volume`` filter;
    unlike the ffmpeg engine, a source with no audio stream needs nothing
    here — MLT plays silence natively for a producer lacking an audio
    stream. A clip carrying a fade/keyframe envelope (see ``_has_envelope``)
    gets the animated ``level`` variant instead (mute still wins over an
    envelope — silence is silence); ``fps`` is only needed for that path.

    Shared by the plain clip producer AND the blur-fill background producer:
    MLT tractor audio comes from track 0 regardless of which VISUAL fit mode
    occupies it, so the background producer must carry the clip's own audio
    (the foreground producer, positioned via a video-only transition, never
    contributes audio — see ``_blur_fill_bg_producer_xml``).
    """
    if clip.muted:
        return _volume_filter(0)
    if _has_envelope(clip):
        points = _envelope_points_db(
            base_volume=clip.volume,
            base_gain_db=0.0,
            duration=clip.duration,
            fade_in_sec=clip.fade_in_sec,
            fade_out_sec=clip.fade_out_sec,
            volume_keyframes=clip.volume_keyframes,
        )
        return _volume_envelope_filter(
            points, fps=fps, to_source_time=lambda t: _clip_source_time(t, clip)
        )
    if clip.volume != 1.0:
        return _volume_filter(clip.volume)
    return ""


def _clip_producer_xml(
    producer_id: str, path: Path, clip: ClipElement, *, fps: int
) -> tuple[str, str, str]:
    """Build the ``<producer>`` for one video clip. Returns
    ``(producer_xml, in_timecode, out_timecode)`` — the in/out are returned so
    the caller can reuse the identical values on the playlist ``<entry>``.
    """
    resource, in_tc, out_tc = _clip_resource_and_timecodes(path, clip)

    filters = _clip_audio_filters(clip, fps=fps)
    # Visual filters, in phase-4's documented stacking order: scale/pos/
    # keyframe affine transform, then anti-reuse crop-zoom, then color grade.
    if _has_transform_override(clip):
        filters += _transform_affine_filter(clip, fps)
    if clip.transform.crop_zoom > 0:
        filters += _affine_zoom_filter(clip.transform.crop_zoom)
    if clip.color is not None:
        filters += _color_eq_filter(clip.color)

    producer_xml = (
        f'<producer id="{producer_id}" in={quoteattr(in_tc)} out={quoteattr(out_tc)}>'
        f'{_prop("resource", resource)}'
        f"{filters}"
        "</producer>"
    )
    return producer_xml, in_tc, out_tc


def _producer_in_out_seconds(clip: ClipElement) -> tuple[float, float]:
    """The producer's own (in, out) in SECONDS, in whatever timebase its
    resource uses — real time for ``speed == 1.0``, the RETIMED timebase for
    a ``timewarp:`` producer (mirrors ``_clip_producer_xml``'s in/out
    formula exactly). That timebase is, by construction, 1 second = 1
    TIMELINE second (``clip.duration`` == ``(out - in)`` in this domain
    regardless of speed) — which is what lets the transition math below
    treat ``TransitionSpec.duration`` (a timeline quantity) as a raw offset
    from this producer's own in/out, with no separate speed handling."""
    speed = clip.transform.speed
    out_point = _clip_out_point(clip)
    if speed == 1.0:
        return clip.trim_start, out_point
    return clip.trim_start / speed, out_point / speed


def _transition_mlt_service(kind: str) -> str:
    """Map a public ``TransitionSpec.kind`` to its MLT transition service.

    MLT 7.30 has no ``dissolve`` mlt_service (smoke-tested 2026-07-17: `melt`
    logs ``failed to load transition "dissolve"`` and exits 0 anyway — the
    boundary just hard-cuts, no error surfaces). A plain cross-dissolve is
    ``luma`` with no ``resource`` property; ``dissolve`` is kept as the
    public-facing kind name (the intuitive term for an agent) and mapped to
    it here."""
    if kind == "dissolve":
        return "luma"
    raise EditorError(f"Unsupported transition kind: {kind!r}")


def _clip_entry_xml(producer_id: str, in_sec: float, out_sec: float) -> str:
    return (
        f'<entry producer="{producer_id}" '
        f'in={quoteattr(seconds_to_timecode(in_sec))} '
        f'out={quoteattr(seconds_to_timecode(out_sec))}/>'
    )


# --------------------------------------------------------------------------- #
# Canvas blur-fill (ClipElement.fit == "contain_blur")
# --------------------------------------------------------------------------- #
# Gaussian blur sigma for the cover-scaled background layer (avfilter bridge,
# mltframework.org avfilter.gblur -> libavfilter's gblur). ~20 matches the
# plan's target: soft enough to read as an ambient fill, not a legible replay
# of the cropped edges.
BLUR_FILL_SIGMA = 20.0


def _cover_rect_px(src_w: float, src_h: float, canvas_w: int, canvas_h: int) -> str:
    """Pixel rect (``"x/y:wxh"``) that scales a ``src_w x src_h`` source to
    COVER a ``canvas_w x canvas_h`` frame, centered, preserving aspect (the
    overflow is cropped by the frame boundary itself — no ``fill``/``halign``/
    ``valign`` needed once the box is computed exactly).

    Empirically confirmed (2026-07-17 smoke): percentage rects on an
    ``affine`` FILTER measure against whatever MLT already fit the producer
    to, not the raw source — computing PIXEL values directly here, the same
    way the text overlay's rect is computed, sidesteps that ambiguity
    entirely.
    """
    scale = max(canvas_w / src_w, canvas_h / src_h)
    w, h = src_w * scale, src_h * scale
    x, y = (canvas_w - w) / 2, (canvas_h - h) / 2
    return f"{x:.4f}/{y:.4f}:{w:.4f}x{h:.4f}"


def _contain_rect_px(src_w: float, src_h: float, canvas_w: int, canvas_h: int) -> str:
    """Pixel rect (``"x/y:wxh:100"``) that fits a ``src_w x src_h`` source
    WITHIN a ``canvas_w x canvas_h`` frame, centered, preserving aspect (the
    foreground layer of the blur-fill canvas) — same rect grammar as
    ``_text_affine_transition_xml``'s (``:100`` = fully opaque)."""
    scale = min(canvas_w / src_w, canvas_h / src_h)
    w, h = src_w * scale, src_h * scale
    x, y = (canvas_w - w) / 2, (canvas_h - h) / 2
    return f"{round(x)}/{round(y)}:{round(w)}x{round(h)}:100"


def _blur_fill_bg_producer_xml(
    producer_id: str,
    path: Path,
    clip: ClipElement,
    *,
    fps: int,
    src_w: int,
    src_h: int,
    canvas_w: int,
    canvas_h: int,
) -> tuple[str, str, str]:
    """Build the blur-fill canvas's BACKGROUND producer: the clip's own
    resource/in/out/audio (identical to ``_clip_producer_xml``'s, since this
    producer occupies the clip's own slot on the main video track), PLUS a
    cover-scale ``affine`` filter and an ``avfilter.gblur`` blur — the ambient
    fill visible in the letterbox bars behind the sharp foreground (see
    ``_contain_rect_px`` / the transition built in ``build_mlt_xml``).

    Crop-zoom (anti-reuse jitter) is deliberately NOT applied here when
    ``contain_blur`` is active — see the phase-5 report for the reasoning
    (crop_zoom's percentage-of-frame geometry targets the "cover" path; it
    stays inert on a blur-fill clip rather than fight the two-track layout).
    """
    resource, in_tc, out_tc = _clip_resource_and_timecodes(path, clip)
    filters = _clip_audio_filters(clip, fps=fps)
    filters += (
        '<filter mlt_service="affine">'
        f'{_prop("transition.rect", _cover_rect_px(src_w, src_h, canvas_w, canvas_h))}'
        f'{_prop("transition.fill", "1")}'
        "</filter>"
        '<filter mlt_service="avfilter.gblur">'
        f'{_prop("av.sigma", BLUR_FILL_SIGMA)}'
        "</filter>"
    )
    producer_xml = (
        f'<producer id="{producer_id}" in={quoteattr(in_tc)} out={quoteattr(out_tc)}>'
        f'{_prop("resource", resource)}'
        f"{filters}"
        "</producer>"
    )
    return producer_xml, in_tc, out_tc


def _blur_fill_fg_producer_and_playlist_xml(
    clip: ClipElement, path: Path, *, fps: int
) -> tuple[str, str]:
    """Build the blur-fill canvas's FOREGROUND producer + its own playlist
    (leading blank = the clip's ``start_time``, exactly like a text/music
    track) — plain and unfiltered; sharp positioning happens entirely in the
    ``affine`` TRANSITION built by the caller (mirrors
    ``_text_affine_transition_xml``). No audio filters: MLT tractor audio
    comes from track 0 (the background producer already carries it) — a
    video-only ``affine`` transition never mixes in a b_track's audio, so
    this producer contributing none is exactly the point, not a gap."""
    producer_id = f"blur_fg_producer_{clip.id}"
    playlist_id = f"blur_fg_playlist_{clip.id}"
    resource, in_tc, out_tc = _clip_resource_and_timecodes(path, clip)
    producer = (
        f'<producer id="{producer_id}" in={quoteattr(in_tc)} out={quoteattr(out_tc)}>'
        f'{_prop("resource", resource)}'
        "</producer>"
    )
    blank_len = _blank_length_frames(clip.start_time, fps)
    blank = f'<blank length="{blank_len}"/>' if blank_len > 0 else ""
    playlist = (
        f'<playlist id="{playlist_id}">'
        f"{blank}"
        f'<entry producer="{producer_id}" in={quoteattr(in_tc)} out={quoteattr(out_tc)}/>'
        "</playlist>"
    )
    return producer, playlist


def _blur_fill_transition_xml(*, a_track: int, b_track: int, rect: str) -> str:
    """Composite the blur-fill foreground (contain-fit, sharp) over the
    background (cover-scaled, blurred) at ``rect`` — same bare (unprefixed)
    property convention as ``_text_affine_transition_xml`` (the ``transition.``
    prefix belongs to the affine FILTER, not the transition)."""
    return (
        '<transition mlt_service="affine">'
        f'{_prop("a_track", a_track)}'
        f'{_prop("b_track", b_track)}'
        f'{_prop("rect", rect)}'
        f'{_prop("fill", "0")}'
        f'{_prop("distort", "0")}'
        "</transition>"
    )


def _upper_clip_producer_xml(
    producer_id: str, path: Path, clip: ClipElement, *, fps: int
) -> tuple[str, str, str]:
    """Producer for one UPPER (V2+) video-track clip. Same resource/in-out
    (trim + speed) and audio (volume/mute/envelope) as ``_clip_producer_xml``,
    plus an optional color grade — but deliberately WITHOUT the transform
    affine filter or the crop-zoom filter: on an upper track the clip's
    ``Transform`` scale/pos is expressed by the composite affine TRANSITION
    that places the whole track over V1 (see ``_upper_track_transition_xml``),
    and a producer-level affine filter would fight it for the rect (keyframes/
    crop_zoom/contain_blur are banned on upper tracks by validation). Returns
    ``(producer_xml, in_tc, out_tc)``."""
    resource, in_tc, out_tc = _clip_resource_and_timecodes(path, clip)
    filters = _clip_audio_filters(clip, fps=fps)
    if clip.color is not None:
        filters += _color_eq_filter(clip.color)
    producer_xml = (
        f'<producer id="{producer_id}" in={quoteattr(in_tc)} out={quoteattr(out_tc)}>'
        f'{_prop("resource", resource)}'
        f"{filters}"
        "</producer>"
    )
    return producer_xml, in_tc, out_tc


def _upper_track_playlist_xml(
    playlist_id: str,
    clips: list[ClipElement],
    producer_ids: list[str],
    in_tcs: list[str],
    out_tcs: list[str],
    *,
    fps: int,
) -> str:
    """Playlist for one upper video track: each clip entry preceded by a
    ``<blank>`` covering the gap since the previous clip's end (or 0), so the
    clip lands at its declared ``start_time`` on the timeline (gaps ARE allowed
    on upper tracks, unlike V1). Same blank-then-entry shape as a text/music
    playlist, generalized to several entries with gaps between them."""
    parts: list[str] = []
    cursor = 0.0
    for clip, pid, in_tc, out_tc in zip(clips, producer_ids, in_tcs, out_tcs, strict=True):
        blank_frames = round((clip.start_time - cursor) * fps)
        if blank_frames > 0:
            parts.append(f'<blank length="{blank_frames}"/>')
        parts.append(f'<entry producer="{pid}" in={quoteattr(in_tc)} out={quoteattr(out_tc)}/>')
        cursor = clip.start_time + clip.duration
    return f'<playlist id="{playlist_id}">{"".join(parts)}</playlist>'


def _upper_track_transition_xml(
    *, a_track: int, b_track: int, clip: ClipElement, fps: int
) -> str:
    """One bare-property affine composite transition placing an upper-track clip
    over the base track (a_track=0) at its ``Transform`` rect, for exactly this
    clip's timeline window.

    One transition PER CLIP (not per track), bounded with ``in``/``out`` frame
    attributes: the rect differs per clip, and a single per-track transition
    carries only one static rect — so per-clip rects demand per-clip
    transitions, each active only during its own [start, start+duration) frames
    (no intra-track overlap makes these windows disjoint). The rect reuses the
    ``_rect_pct`` math (percentages against the fitted frame, ``fill="1"`` — the
    same convention ``_transform_affine_filter`` relies on) with the clip's
    ``opacity`` as the trailing 0..100 mix field. Bare (unprefixed) ``rect``/
    ``fill``/``distort``, per the 7bcee79/comp-9 lesson for a TRANSITION."""
    rect = _rect_pct(clip.transform.scale, clip.transform.pos_x, clip.transform.pos_y, clip.opacity)
    in_frame = round(clip.start_time * fps)
    out_frame = round((clip.start_time + clip.duration) * fps) - 1
    return (
        f'<transition mlt_service="affine" in="{in_frame}" out="{out_frame}">'
        f'{_prop("a_track", a_track)}'
        f'{_prop("b_track", b_track)}'
        f'{_prop("rect", rect)}'
        f'{_prop("fill", "1")}'
        f'{_prop("distort", "0")}'
        "</transition>"
    )


def _build_video_playlist_and_transitions(
    clips: list[ClipElement], producer_ids: list[str]
) -> tuple[str, str]:
    """Build the outer video playlist AND any nested-tractor transition
    producers it references, splitting a boundary into
    [prev solo][crossfade zone][this solo] wherever a clip declares
    ``transition_in`` (see ``TransitionSpec`` for the exact overlap
    semantics: ``duration`` consumes the previous clip's own tail and this
    clip's own head, played simultaneously, dissolving over that shared
    span). The Shotcut same-track approach: the crossfade zone is an embedded
    tractor (two 1-entry playlists + a transition) referenced as an ordinary
    producer by the outer playlist — smoke-tested 2026-07-17 (pixel colors
    at the zone's midpoint confirmed an actual blend, not a hard cut).

    Returns ``(video_playlist_xml, extra_producers_xml)`` — the extra
    producers (the nested tractors + their two sub-playlists) must be
    declared at the top level like any other producer, so the caller appends
    them alongside the per-clip ``<producer>`` blocks.
    """
    in_out = [_producer_in_out_seconds(clip) for clip in clips]

    # How much of clip i's own tail is claimed by clip (i+1)'s transition_in.
    tail_claimed = [0.0] * len(clips)
    for i in range(1, len(clips)):
        transition = clips[i].transition_in
        if transition is not None:
            tail_claimed[i - 1] = transition.duration

    entries: list[str] = []
    extra_producers: list[str] = []
    for i, clip in enumerate(clips):
        transition = clip.transition_in
        if transition is not None:
            prev_in_sec, prev_out_sec = in_out[i - 1]
            in_sec, _out_sec = in_out[i]
            d = transition.duration
            tractor_id = f"transition_tractor_{i}"
            a_playlist_id = f"{tractor_id}_a"
            b_playlist_id = f"{tractor_id}_b"
            extra_producers.append(
                f'<playlist id="{a_playlist_id}">'
                f"{_clip_entry_xml(producer_ids[i - 1], prev_out_sec - d, prev_out_sec)}"
                "</playlist>"
            )
            extra_producers.append(
                f'<playlist id="{b_playlist_id}">'
                f"{_clip_entry_xml(producer_ids[i], in_sec, in_sec + d)}"
                "</playlist>"
            )
            service = _transition_mlt_service(transition.kind)
            zone_out_tc = seconds_to_timecode(d)
            extra_producers.append(
                f'<tractor id="{tractor_id}" in="00:00:00.000" out={quoteattr(zone_out_tc)}>'
                f'<track producer="{a_playlist_id}"/>'
                f'<track producer="{b_playlist_id}"/>'
                f'<transition mlt_service="{service}">'
                f'{_prop("a_track", 0)}'
                f'{_prop("b_track", 1)}'
                f'{_prop("always_active", "1")}'
                "</transition>"
                "</tractor>"
            )
            entries.append(
                f'<entry producer="{tractor_id}" in="00:00:00.000" out={quoteattr(zone_out_tc)}/>'
            )

        in_sec, out_sec = in_out[i]
        head_claim = transition.duration if transition is not None else 0.0
        solo_in = in_sec + head_claim
        solo_out = out_sec - tail_claimed[i]
        if solo_out > solo_in:
            entries.append(_clip_entry_xml(producer_ids[i], solo_in, solo_out))

    playlist = f'<playlist id="playlist0">{"".join(entries)}</playlist>'
    return playlist, "".join(extra_producers)


def _text_pixel_position(
    text: TextElement, *, width: int, height: int, png_w: int, png_h: int
) -> tuple[int, int]:
    """Pixel (x, y) of the text box's top-left corner. Same position math as
    ``ffmpeg_graph._overlay_position`` (named presets center horizontally at a
    fixed vertical inset; explicit ``pos_x``/``pos_y`` anchor the corner
    directly), computed in pixels instead of an ffmpeg expression string."""
    style = text.style
    if style.pos_x is not None:
        x = round(width * style.pos_x)
    else:
        x = round((width - png_w) / 2)

    if style.pos_y is not None:
        y = round(height * style.pos_y)
    elif style.pos == "top":
        y = TEXT_MARGIN
    elif style.pos == "bottom":
        y = height - png_h - TEXT_MARGIN
    else:
        y = round((height - png_h) / 2)
    return x, y


def _text_producer_and_playlist_xml(
    text: TextElement, png_path: Path, *, fps: int, timeline_end: float
) -> tuple[str, str]:
    """Build the text overlay's own single-image producer + playlist. A
    windowed text (finite ``duration``) plays for that long; a persistent one
    (``duration`` None) runs to the end of the timeline."""
    producer_id = f"text_producer_{text.id}"
    playlist_id = f"text_playlist_{text.id}"
    duration = (
        text.duration if text.duration is not None else timeline_end - text.start_time
    )
    out_tc = seconds_to_timecode(duration)

    producer = (
        f'<producer id="{producer_id}" in="00:00:00.000" out={quoteattr(out_tc)}>'
        f'{_prop("resource", png_path)}'
        f'{_prop("eof", "pause")}'
        "</producer>"
    )
    blank_len = _blank_length_frames(text.start_time, fps)
    blank = f'<blank length="{blank_len}"/>' if blank_len > 0 else ""
    playlist = (
        f'<playlist id="{playlist_id}">'
        f"{blank}"
        f'<entry producer="{producer_id}" in="00:00:00.000" out={quoteattr(out_tc)}/>'
        "</playlist>"
    )
    return producer, playlist


def _text_affine_transition_xml(
    *, a_track: int, b_track: int, x: int, y: int, w: int, h: int
) -> str:
    """One-track-per-overlay composite (mltframework.org/docs/mltxml/): the
    ``affine`` transition places the text track's PNG at its pixel rect; the
    text track's own leading blank is what makes this "windowed".

    Property names carry NO ``transition.`` prefix here: that prefix belongs
    to the affine FILTER (which wraps this transition — see the crop-zoom
    filter above). On the bare transition the properties are ``rect``/
    ``fill``/``distort``; with the prefix they are silently ignored and every
    overlay composites centered (caught by the comp-9 live gate 2026-07-17)."""
    rect = f"{x}/{y}:{w}x{h}:100"
    return (
        '<transition mlt_service="affine">'
        f'{_prop("a_track", a_track)}'
        f'{_prop("b_track", b_track)}'
        f'{_prop("rect", rect)}'
        f'{_prop("fill", "0")}'
        f'{_prop("distort", "0")}'
        "</transition>"
    )


def _overlay_producer_and_playlist_xml(
    overlay: OverlayElement, path: Path, *, fps: int
) -> tuple[str, str]:
    """Build one PiP/sticker overlay's producer + playlist (leading blank =
    its start delay). Always plays its full source from time 0 for
    ``duration`` seconds — mirrors ``_text_producer_and_playlist_xml`` but for
    a media clip/image instead of a rasterized text PNG (no persistent/None
    duration case: an ``OverlayElement`` always has a finite duration)."""
    producer_id = f"overlay_producer_{overlay.id}"
    playlist_id = f"overlay_playlist_{overlay.id}"
    out_tc = seconds_to_timecode(overlay.duration)

    producer = (
        f'<producer id="{producer_id}" in="00:00:00.000" out={quoteattr(out_tc)}>'
        f'{_prop("resource", path)}'
        f'{_prop("eof", "pause")}'
        "</producer>"
    )
    blank_len = _blank_length_frames(overlay.start_time, fps)
    blank = f'<blank length="{blank_len}"/>' if blank_len > 0 else ""
    playlist = (
        f'<playlist id="{playlist_id}">'
        f"{blank}"
        f'<entry producer="{producer_id}" in="00:00:00.000" out={quoteattr(out_tc)}/>'
        "</playlist>"
    )
    return producer, playlist


def _overlay_affine_transition_xml(
    *, a_track: int, b_track: int, x: int, y: int, w: int, h: int, opacity: float
) -> str:
    """Same bare-property affine transition as ``_text_affine_transition_xml``
    (the 7bcee79/comp-9 lesson: no ``transition.`` prefix on a TRANSITION),
    generalized with ``opacity`` in the rect's trailing mix field instead of
    a hardcoded 100 — a PiP/sticker overlay can be semi-transparent."""
    rect = f"{x}/{y}:{w}x{h}:{opacity * 100:.4f}"
    return (
        '<transition mlt_service="affine">'
        f'{_prop("a_track", a_track)}'
        f'{_prop("b_track", b_track)}'
        f'{_prop("rect", rect)}'
        f'{_prop("fill", "0")}'
        f'{_prop("distort", "0")}'
        "</transition>"
    )


def _music_producer_and_playlist_xml(
    music: AudioElement, path: Path, *, fps: int
) -> tuple[str, str]:
    """Build one music bed's producer (volume + optional dB trim) + playlist
    (leading blank = its start delay)."""
    producer_id = f"music_producer_{music.id}"
    playlist_id = f"music_playlist_{music.id}"
    out_tc = seconds_to_timecode(music.duration)

    if _has_envelope(music):
        # The music producer has no trim (its own resource starts at 0), so
        # its source-time axis IS the local envelope axis directly.
        points = _envelope_points_db(
            base_volume=music.volume,
            base_gain_db=music.gain_db,
            duration=music.duration,
            fade_in_sec=music.fade_in_sec,
            fade_out_sec=music.fade_out_sec,
            volume_keyframes=music.volume_keyframes,
        )
        filters = _volume_envelope_filter(points, fps=fps, to_source_time=lambda t: t)
    else:
        filters = _volume_filter(music.volume)
        if music.gain_db != 0.0:
            filters += _volume_filter(f"{music.gain_db}dB")

    producer = (
        f'<producer id="{producer_id}" in="00:00:00.000" out={quoteattr(out_tc)}>'
        f'{_prop("resource", path)}'
        f"{filters}"
        "</producer>"
    )
    blank_len = _blank_length_frames(music.start_time, fps)
    blank = f'<blank length="{blank_len}"/>' if blank_len > 0 else ""
    playlist = (
        f'<playlist id="{playlist_id}">'
        f"{blank}"
        f'<entry producer="{producer_id}" in="00:00:00.000" out={quoteattr(out_tc)}/>'
        "</playlist>"
    )
    return producer, playlist


def _mix_transition_xml(*, a_track: int, b_track: int) -> str:
    """Mix a music track into the base video track's audio
    (mltframework.org/plugins/TransitionMix)."""
    return (
        '<transition mlt_service="mix">'
        f'{_prop("a_track", a_track)}'
        f'{_prop("b_track", b_track)}'
        f'{_prop("always_active", "1")}'
        f'{_prop("sum", "1")}'
        "</transition>"
    )


def _profile_xml(*, width: int, height: int, fps: int) -> str:
    return (
        f'<profile description="mcpcut" width="{width}" height="{height}" '
        'progressive="1" sample_aspect_num="1" sample_aspect_den="1" '
        f'frame_rate_num="{fps}" frame_rate_den="1" colorspace="709"/>'
    )


def _consumer_xml(output_path: Path, *, crf: int, preset: str, pcm_audio: bool) -> str:
    if pcm_audio:
        # Intermediate for the hybrid audio post-pass: PCM in a mov container,
        # so the final AAC encode happens exactly once (in the post-pass) —
        # same single lossy generation as the ffmpeg engine.
        audio = 'f="mov" acodec="pcm_s16le" '
    else:
        audio = 'f="mp4" acodec="aac" movflags="+faststart" '
    return (
        '<consumer mlt_service="avformat" '
        f'target={quoteattr(str(output_path))} '
        'vcodec="libx264" '
        f'crf="{crf}" preset={quoteattr(preset)} '
        f'pix_fmt="yuv420p" {audio}'
        f'ar="{CONSUMER_SAMPLE_RATE}" '
        'rescale="bicubic" real_time="-1" terminate_on_pause="1"/>'
    )


# --------------------------------------------------------------------------- #
# Top-level pure builder
# --------------------------------------------------------------------------- #
def build_mlt_xml(
    project: EditorProject,
    *,
    resolved_media: dict[str, Path],
    text_pngs: dict[str, Path],
    text_png_sizes: dict[str, tuple[int, int]],
    output_path: Path,
    crf: int,
    preset: str,
    pcm_audio: bool = False,
    clip_dimensions: dict[str, tuple[int, int]] | None = None,
) -> str:
    """Build the full MLT XML document for ``project``. PURE — no I/O.

    ``pcm_audio=True`` targets the hybrid intermediate (PCM audio in mov, see
    ``_consumer_xml``) — used when loudness normalization runs as an ffmpeg
    audio post-pass, because MLT 7.30 can express the recipe neither through
    its avfilter bridge (loudnorm/dynaudnorm break its buffer sink) nor with
    native filters (dynamic_loudness measured -9.5 LUFS / TP +1.9 / LRA 10 on
    real 79s material — comp-9 live gate, 2026-07-17). No loudness filters
    are ever emitted into the XML; audio leaves melt un-normalized.

    ``resolved_media`` maps ``media_id`` -> local file (clips, music, AND
    overlays share this one map); ``text_pngs`` maps a text element id -> its
    rasterized PNG; ``text_png_sizes`` maps a text element id -> its PNG's
    ``(width, height)`` in pixels (needed for the ``affine`` transition's
    pixel rect). Raises ``EditorError`` when a referenced clip/music/overlay
    media or text PNG/size is missing, or the project has no video clips
    (mirrors ``build_ffmpeg_command``'s contract exactly).

    Track order in the outer tractor: video(0), overlay(s), text(s), music —
    overlays composite BELOW text so captions stay readable over a PiP/badge;
    all three layer types transition against ``a_track=0`` directly (each
    subsequent transition composites onto whatever is already in that slot,
    not onto the previous layer's own track — this is what lets several
    stacked overlays/texts layer correctly in declaration order).
    ``resolved_media`` maps ``media_id`` -> local file; ``text_pngs`` maps a
    text element id -> its rasterized PNG; ``text_png_sizes`` maps a text
    element id -> its PNG's ``(width, height)`` in pixels (needed for the
    ``affine`` transition's pixel rect). ``clip_dimensions`` maps a CLIP
    element id -> its source's native ``(width, height)`` — only required for
    a clip using ``fit="contain_blur"`` (see ``_blur_fill_bg_producer_xml``);
    every other clip ignores it. Raises ``EditorError`` when a referenced
    clip/music media, text PNG/size, or blur-fill clip's dimensions are
    missing, or the project has no video clips (mirrors
    ``build_ffmpeg_command``'s contract exactly).
    """
    clips = _main_video_clips(project)
    if not clips:
        raise EditorError("Project has no video clips to render")
    upper_tracks = _upper_video_tracks(project)
    music = _ordered_music(project)
    texts = _ordered_texts(project)
    overlays = _ordered_overlays(project)
    clip_dimensions = clip_dimensions or {}

    width, height, fps = project.aspect_w, project.aspect_h, project.fps
    # The RENDERED length, not the naive sum of clip durations: a transition
    # shrinks the timeline by its duration (see effective_timeline_end).
    timeline_end = effective_timeline_end(clips)

    producers: list[str] = []

    clip_producer_ids: list[str] = []
    blur_fill_clips: list[ClipElement] = []
    for position, clip in enumerate(clips):
        path = _require_media(resolved_media, clip.media_id, clip.id)
        producer_id = f"clip_producer_{position}"
        if clip.fit == "contain_blur":
            src_w, src_h = _require_clip_dimensions(clip_dimensions, clip.id)
            producer_xml, in_tc, out_tc = _blur_fill_bg_producer_xml(
                producer_id, path, clip, fps=fps,
                src_w=src_w, src_h=src_h, canvas_w=width, canvas_h=height,
            )
            blur_fill_clips.append(clip)
        else:
            producer_xml, in_tc, out_tc = _clip_producer_xml(producer_id, path, clip, fps=fps)
        producers.append(producer_xml)
        clip_producer_ids.append(producer_id)
    video_playlist, transition_producers = _build_video_playlist_and_transitions(
        clips, clip_producer_ids
    )
    producers.append(transition_producers)

    # --- Upper (V2+) video tracks: producers + playlist + per-clip composite +
    # one audio mix, on track indices 1..len(upper_tracks) (between the base
    # track 0 and the overlay/text/music groups, which shift down accordingly).
    upper_video_playlists: list[str] = []
    upper_video_transitions: list[str] = []
    for track_offset, vtrack in enumerate(upper_tracks):
        track_index = 1 + track_offset
        vclips = _upper_video_clips(vtrack)
        prod_ids: list[str] = []
        in_tcs: list[str] = []
        out_tcs: list[str] = []
        for position, clip in enumerate(vclips):
            path = _require_media(resolved_media, clip.media_id, clip.id)
            producer_id = f"v{track_index}_clip_{position}"
            producer_xml, in_tc, out_tc = _upper_clip_producer_xml(
                producer_id, path, clip, fps=fps
            )
            producers.append(producer_xml)
            prod_ids.append(producer_id)
            in_tcs.append(in_tc)
            out_tcs.append(out_tc)
        playlist_id = f"video_playlist_{vtrack.id}"
        upper_video_playlists.append(
            _upper_track_playlist_xml(playlist_id, vclips, prod_ids, in_tcs, out_tcs, fps=fps)
        )
        for clip in vclips:
            upper_video_transitions.append(
                _upper_track_transition_xml(
                    a_track=0, b_track=track_index, clip=clip, fps=fps
                )
            )
        # The whole upper track's audio mixes into the master, like a music bed.
        upper_video_transitions.append(_mix_transition_xml(a_track=0, b_track=track_index))

    overlay_track_start = 1 + len(upper_tracks)
    overlay_playlists: list[str] = []
    overlay_transitions: list[str] = []
    for offset, overlay in enumerate(overlays):
        track_index = overlay_track_start + offset
        path = _require_media(resolved_media, overlay.media_id, overlay.id)
        producer_xml, playlist_xml = _overlay_producer_and_playlist_xml(
            overlay, path, fps=fps
        )
        producers.append(producer_xml)
        overlay_playlists.append(playlist_xml)
        x, y = round(width * overlay.x), round(height * overlay.y)
        w, h = round(width * overlay.w), round(height * overlay.h)
        overlay_transitions.append(
            _overlay_affine_transition_xml(
                a_track=0, b_track=track_index, x=x, y=y, w=w, h=h, opacity=overlay.opacity
            )
        )

    text_track_start = overlay_track_start + len(overlays)
    text_playlists: list[str] = []
    text_transitions: list[str] = []
    for offset, text in enumerate(texts):
        track_index = text_track_start + offset
        png_path = _require_text_png(text_pngs, text.id)
        png_w, png_h = _require_png_size(text_png_sizes, text.id)
        producer_xml, playlist_xml = _text_producer_and_playlist_xml(
            text, png_path, fps=fps, timeline_end=timeline_end
        )
        producers.append(producer_xml)
        text_playlists.append(playlist_xml)
        x, y = _text_pixel_position(text, width=width, height=height, png_w=png_w, png_h=png_h)
        text_transitions.append(
            _text_affine_transition_xml(
                a_track=0, b_track=track_index, x=x, y=y, w=png_w, h=png_h
            )
        )

    music_track_start = text_track_start + len(texts)
    music_playlists: list[str] = []
    music_transitions: list[str] = []
    for offset, item in enumerate(music):
        path = _require_media(resolved_media, item.media_id, item.id)
        producer_xml, playlist_xml = _music_producer_and_playlist_xml(item, path, fps=fps)
        producers.append(producer_xml)
        music_playlists.append(playlist_xml)
        track_index = music_track_start + offset
        music_transitions.append(_mix_transition_xml(a_track=0, b_track=track_index))

    # Blur-fill foreground tracks: one per contain_blur clip, added after
    # text/music so their track indices never disturb those groups' math.
    blur_fill_track_start = music_track_start + len(music)
    blur_fill_playlists: list[str] = []
    blur_fill_transitions: list[str] = []
    for offset, clip in enumerate(blur_fill_clips):
        path = _require_media(resolved_media, clip.media_id, clip.id)
        producer_xml, playlist_xml = _blur_fill_fg_producer_and_playlist_xml(clip, path, fps=fps)
        producers.append(producer_xml)
        blur_fill_playlists.append(playlist_xml)
        src_w, src_h = _require_clip_dimensions(clip_dimensions, clip.id)
        rect = _contain_rect_px(src_w, src_h, width, height)
        track_index = blur_fill_track_start + offset
        blur_fill_transitions.append(
            _blur_fill_transition_xml(a_track=0, b_track=track_index, rect=rect)
        )

    tracks_xml = (
        '<track producer="playlist0"/>'
        + "".join(f'<track producer="video_playlist_{t.id}"/>' for t in upper_tracks)
        + "".join(f'<track producer="overlay_playlist_{o.id}"/>' for o in overlays)
        + "".join(f'<track producer="text_playlist_{text.id}"/>' for text in texts)
        + "".join(f'<track producer="music_playlist_{item.id}"/>' for item in music)
        + "".join(f'<track producer="blur_fg_playlist_{clip.id}"/>' for clip in blur_fill_clips)
    )

    tractor = (
        '<tractor id="tractor0">'
        f"{tracks_xml}"
        f"{''.join(upper_video_transitions)}"
        f"{''.join(overlay_transitions)}"
        f"{''.join(text_transitions)}"
        f"{''.join(music_transitions)}"
        f"{''.join(blur_fill_transitions)}"
        "</tractor>"
    )

    profile = _profile_xml(width=width, height=height, fps=fps)
    consumer = _consumer_xml(output_path, crf=crf, preset=preset, pcm_audio=pcm_audio)

    body = "".join(
        [profile, *producers, video_playlist, *upper_video_playlists,
         *overlay_playlists, *text_playlists,
         *music_playlists, *blur_fill_playlists, tractor, consumer]
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<mlt LC_NUMERIC="C" version="{MLT_VERSION}">{body}</mlt>'
    )
