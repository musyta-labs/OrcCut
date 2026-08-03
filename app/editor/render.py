"""Server-side render engines: compile an ``EditorProject`` -> 9:16 mp4.

Two render engines, selected by ``render_project_file(..., engine=...)``:

* ``"ffmpeg"`` (default): ``build_ffmpeg_command`` is PURE — it turns a project
  plus resolved media/text paths into the full ``ffmpeg`` argv (a single
  ``-filter_complex`` graph). No I/O, no execution, so it is fully
  snapshot-testable without ffmpeg installed. ``_render_via_ffmpeg`` is the
  orchestration: resolve media (``resolve_media_source``), rasterize text
  overlays to PNGs (PIL), build the command, run it streaming to a temp file,
  then atomically ``os.replace`` into ``output_path``.
* ``"mlt"`` (phase 3): ``app.editor.mlt_graph.build_mlt_xml`` is the pure
  equivalent, compiling the project to MLT XML instead. ``_render_via_mlt``
  shares the same resolve/rasterize/atomic-replace shape, but runs ``melt``
  against the XML and verifies the output's probed duration before trusting
  it (melt is known to exit 0 on some kill paths — mlt#547).

FFmpeg/PIL/yt-dlp are imported lazily (inside functions) so importing this
module stays cheap for images that never render.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from app.common.logging import get_logger
from app.editor.errors import EditorError
from app.editor.ffmpeg_graph import (
    amix_filter,
    clip_audio_filter,
    concat_filter,
    loudness_filter,
    music_filter,
    overlay_filter,
    preview_clip_filter,
    preview_overlay_filter,
    video_clip_filter,
)
from app.editor.mlt_graph import build_mlt_xml, effective_timeline_end
from app.editor.model import (
    AudioElement,
    ClipElement,
    EditorProject,
    LoudnessTarget,
    MediaAsset,
    OverlayElement,
    TextElement,
)
from app.editor.text_raster import rasterize_text_png
from app.media.downloader import resolve_media_source
from app.net.egress import EgressBlockedError, assert_url_egress_allowed
from app.storage.capacity import guard_disk_space, no_space_becomes_a_sentence

logger = get_logger(__name__)

VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"
PIX_FMT = "yuv420p"
# Encoder knobs made explicit on the argv (previously ffmpeg's own defaults,
# which happened to be crf=23/preset=medium — 20 is a deliberate quality bump
# to keep this pure builder's config-independent defaults close to the
# service default in app.config.Settings without importing config here).
DEFAULT_CRF = 20
DEFAULT_PRESET = "medium"

# Wall-clock ceiling for a single ffmpeg invocation. A hung ffmpeg (a corrupt
# input, a stalled filter) would otherwise wedge a render queue forever; on
# timeout subprocess.run kills the child and raises TimeoutExpired.
SUBPROCESS_TIMEOUT_SEC = 15 * 60

# Wall-clock ceiling for a single ffprobe invocation (metadata-only, never
# decodes) — the probes below (``probe_has_audio``, ``probe_video_dimensions``,
# ``probe_duration_sec``) and ``app.analysis.ffmpeg_io.probe_fps`` all share
# this one constant. ffprobe is normally instant, but a hung/unreadable source
# (a FIFO nothing writes to, a stalled network read on a URL) would otherwise
# block whichever request or worker called it forever; on timeout
# subprocess.run kills the child and raises TimeoutExpired, which every probe
# here turns into an ``EditorError``.
PROBE_TIMEOUT_SEC = 30

CLIPS_SUBDIR = "clips"   # keyed by the asset's own uuid, not any client-side row id
TEXT_SUBDIR = "editor_text"
# NOTE (follow-up, not fixed here): nothing on this repo's schedule ever prunes
# media_dir/CLIPS_SUBDIR/<asset_id>/ or media_dir/TEXT_SUBDIR/ — both grow
# without bound as projects render. A retention/cleanup task is recommended
# follow-up work, deliberately out of scope for this extraction.

# Text rasterization constants (mirrors the source repo's overlay look).
TEXT_STROKE = 6
TEXT_MAX_WIDTH_RATIO = 0.9
TEXT_LINE_SPACING = 16
_BLACK = (0, 0, 0)


# --------------------------------------------------------------------------- #
# Ordered element collectors (pure)
# --------------------------------------------------------------------------- #
def _ordered_video_clips(project: EditorProject) -> list[ClipElement]:
    clips = [
        element
        for track in project.tracks
        if track.type == "video"
        for element in track.elements
        if isinstance(element, ClipElement)
    ]
    return sorted(clips, key=lambda clip: clip.start_time)


def _video_tracks(project: EditorProject) -> list:
    return [track for track in project.tracks if track.type == "video"]


def _main_video_clips(project: EditorProject) -> list[ClipElement]:
    """Clips of the FIRST (V1/main) video track, timeline-ordered. V1 defines
    the film's length, so the mlt render's expected/verified duration is based
    on THIS set alone — an upper (V2+) clip overhanging past V1's end does not
    stretch the render (mirrors ``mlt_graph._main_video_clips``)."""
    tracks = _video_tracks(project)
    if not tracks:
        return []
    clips = [el for el in tracks[0].elements if isinstance(el, ClipElement)]
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


# --------------------------------------------------------------------------- #
# Pure command builder
# --------------------------------------------------------------------------- #
def build_ffmpeg_command(
    project: EditorProject,
    *,
    resolved_media: dict[str, Path],
    text_pngs: dict[str, Path],
    output_path: Path,
    font_path: str,
    media_has_audio: dict[str, bool] | None = None,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
) -> list[str]:
    """Build the full ``ffmpeg`` argv for ``project``. PURE — no I/O.

    ``resolved_media`` maps ``media_id`` -> local file; ``text_pngs`` maps a text
    element id -> its rasterized PNG. ``media_has_audio`` maps a clip's
    ``media_id`` -> whether its resolved file has an audio stream (a media_id
    absent from the map is assumed to have audio, preserving prior behaviour
    for every caller that does not probe). Probing a file is I/O, so this pure
    builder never does it itself — the orchestration layer computes the map
    (see ``probe_has_audio``) and passes it in; a clip whose source has none
    gets synthesized silence instead of a ``[N:a]`` reference that would not
    exist. Raises ``EditorError`` when a referenced clip/music media or text
    PNG is missing, or the project has no video clips.
    """
    clips = _ordered_video_clips(project)
    if not clips:
        raise EditorError("Project has no video clips to render")
    music = _ordered_music(project)
    texts = _ordered_texts(project)

    inputs: list[tuple[list[str], str]] = []  # (pre-flags, path)
    filter_parts: list[str] = []

    clip_indices = [
        _register_input(inputs, _require_media(resolved_media, clip.media_id, clip.id))
        for clip in clips
    ]
    music_indices = [
        _register_input(inputs, _require_media(resolved_media, item.media_id, item.id))
        for item in music
    ]
    text_indices = [
        _register_input(
            inputs, _require_text_png(text_pngs, text.id), pre_flags=["-loop", "1"]
        )
        for text in texts
    ]

    width, height, fps = project.aspect_w, project.aspect_h, project.fps

    final_v = _build_video_chain(
        filter_parts, clips, clip_indices, texts, text_indices, width, height, fps
    )
    final_a = _build_audio_chain(
        filter_parts,
        clips,
        clip_indices,
        music,
        music_indices,
        project.loudness,
        media_has_audio or {},
    )

    filter_complex = ";".join(filter_parts)
    duration = _video_timeline_end(clips)
    return _assemble_argv(
        inputs, filter_complex, final_v, final_a, fps, duration, output_path,
        crf=crf, preset=preset,
    )


def _video_timeline_end(clips: list[ClipElement]) -> float:
    """The playback duration of the video timeline: the latest clip out-point.

    Used to cap the output with ``-t`` — without it, the looped (``-loop 1``)
    text-overlay image is an infinite input and ``overlay`` (default
    ``shortest=0``) would render forever, freezing on the last video frame."""
    return max(clip.start_time + clip.duration for clip in clips)


def _register_input(
    inputs: list[tuple[list[str], str]], path: Path, *, pre_flags: list[str] | None = None
) -> int:
    """Append an input and return its ffmpeg input index."""
    index = len(inputs)
    inputs.append((pre_flags or [], str(path)))
    return index


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


def _build_video_chain(
    filter_parts: list[str],
    clips: list[ClipElement],
    clip_indices: list[int],
    texts: list[TextElement],
    text_indices: list[int],
    width: int,
    height: int,
    fps: int,
) -> str:
    """Per-clip normalize -> concat -> stacked text overlays. Returns final label."""
    labels: list[str] = []
    for position, (clip, index) in enumerate(zip(clips, clip_indices, strict=True)):
        label = f"v{position}"
        filter_parts.append(
            video_clip_filter(index, clip, label, width=width, height=height, fps=fps)
        )
        labels.append(label)

    filter_parts.append(concat_filter(labels, "vcat", is_video=True))
    current = "vcat"
    for position, (text, index) in enumerate(zip(texts, text_indices, strict=True)):
        out_label = f"vo{position}"
        filter_parts.append(overlay_filter(current, index, text, out_label))
        current = out_label
    return current


def _build_audio_chain(
    filter_parts: list[str],
    clips: list[ClipElement],
    clip_indices: list[int],
    music: list[AudioElement],
    music_indices: list[int],
    loudness: LoudnessTarget | None,
    media_has_audio: dict[str, bool],
) -> str:
    """Per-clip audio (silence when muted or when the source has no audio
    stream) -> concat -> amix music -> loudnorm.

    Normalization is last: integrated loudness is a property of the whole mix,
    so it can only be measured once every source and bed is in it. Returns the
    final label.
    """
    labels: list[str] = []
    for position, (clip, index) in enumerate(zip(clips, clip_indices, strict=True)):
        label = f"a{position}"
        has_audio = media_has_audio.get(clip.media_id, True)
        filter_parts.append(clip_audio_filter(index, clip, label, has_audio=has_audio))
        labels.append(label)

    filter_parts.append(concat_filter(labels, "acat", is_video=False))
    mixed = "acat"

    if music:
        mix_labels = ["acat"]
        for position, (item, index) in enumerate(zip(music, music_indices, strict=True)):
            label = f"m{position}"
            filter_parts.append(music_filter(index, item, label))
            mix_labels.append(label)
        filter_parts.append(amix_filter(mix_labels, "amixout"))
        mixed = "amixout"

    if loudness is None:
        return mixed
    filter_parts.append(loudness_filter(mixed, loudness, "anorm"))
    return "anorm"


def _assemble_argv(
    inputs: list[tuple[list[str], str]],
    filter_complex: str,
    final_v: str,
    final_a: str,
    fps: int,
    duration: float,
    output_path: Path,
    *,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
) -> list[str]:
    argv = ["ffmpeg", "-y"]
    for pre_flags, path in inputs:
        argv.extend(pre_flags)
        argv.extend(["-i", path])
    argv.extend(["-filter_complex", filter_complex])
    argv.extend(["-map", f"[{final_v}]", "-map", f"[{final_a}]"])
    argv.extend(["-c:v", VIDEO_CODEC, "-c:a", AUDIO_CODEC])
    argv.extend(["-crf", str(crf), "-preset", preset])
    argv.extend(["-r", str(fps), "-pix_fmt", PIX_FMT, "-movflags", "+faststart"])
    # Cap the output to the timeline length; the looped text-overlay image is an
    # infinite input, so without -t ffmpeg would never reach EOF.
    argv.extend(["-t", f"{duration}"])
    argv.append(str(output_path))
    return argv


# --------------------------------------------------------------------------- #
# Single-frame preview (pure)
# --------------------------------------------------------------------------- #
def _clip_at(project: EditorProject, at_time: float) -> ClipElement | None:
    """The video clip on-screen at timeline time ``at_time`` (half-open range
    ``[start, start+duration)``), or ``None`` when nothing is playing."""
    for clip in _ordered_video_clips(project):
        if clip.start_time <= at_time < clip.start_time + clip.duration:
            return clip
    return None


def _active_texts_at(project: EditorProject, at_time: float) -> list[TextElement]:
    """Text overlays visible at ``at_time`` (same half-open window). A persistent
    text (``duration`` None) is active from ``start_time`` onward."""
    return [
        text
        for text in _ordered_texts(project)
        if text.start_time <= at_time
        and (text.duration is None or at_time < text.start_time + text.duration)
    ]


def _source_seek(clip: ClipElement, at_time: float) -> float:
    """Map a timeline time to the clip's source-media seek point (accounts for the
    clip's trim in-point and playback speed)."""
    return clip.trim_start + (at_time - clip.start_time) * clip.transform.speed


def _active_overlays_at(project: EditorProject, at_time: float) -> list[OverlayElement]:
    """Overlay elements visible at ``at_time`` (same half-open window as clips)."""
    return [
        overlay
        for overlay in _ordered_overlays(project)
        if overlay.start_time <= at_time < overlay.start_time + overlay.duration
    ]


def _clip_static_offenders(clip: ClipElement) -> list[str]:
    """Per-clip MLT-only features that offend regardless of the moment in
    time: keyframed transforms, a static scale/pos away from neutral, and a
    color grade.

    THE single place to add a new static per-clip MLT-only feature — shared by
    ``_mlt_only_offenders`` (whole-project ffmpeg-export guard) and
    ``_preview_unsupported_at`` (per-moment preview guard) so neither can be
    forgotten. Two per-clip features are deliberately NOT here:
    ``transition_in`` (the export guard flags it unconditionally, the preview
    guard only inside the dissolve-overlap window) and ``fit="contain_blur"``
    (the export path rejects it in ``_check_mlt_only_features`` with its own
    message format, the preview guard flags it per moment).
    """
    offenders: list[str] = []
    if clip.keyframes:
        offenders.append(f"clip {clip.id!r} (keyframes)")
    transform = clip.transform
    if transform.scale != 1.0 or transform.pos_x != 0.0 or transform.pos_y != 0.0:
        offenders.append(f"clip {clip.id!r} (transform.scale/pos_x/pos_y)")
    if clip.color is not None:
        offenders.append(f"clip {clip.id!r} (color)")
    return offenders


def _preview_unsupported_at(project: EditorProject, at_time: float) -> list[str]:
    """Name every feature VISIBLE at ``at_time`` that ``build_preview_command``
    would silently drop from the frame — the per-moment narrowing of
    ``_mlt_only_offenders``: the active clip's static transform / keyframes /
    color grade / contain_blur fit, the cross-dissolve overlap window of its
    ``transition_in``, any overlay (sticker/PiP) on screen then, and a second
    video-track clip compositing at the same moment.

    ``render_preview_frame`` refuses to render when this is non-empty: an
    "exact" frame that quietly un-moves a dragged clip or hides a sticker is
    worse than no frame (the UI shows 'no frame here' on the resulting 422).
    Features elsewhere on the timeline do NOT block the frame.
    """
    offenders: list[str] = []
    clip = _clip_at(project, at_time)
    if clip is not None:
        # Inside the dissolve window the real render blends this clip with the
        # previous clip's tail; past it the clip plays alone and its
        # transition_in must not block the rest of the clip's frames.
        if (
            clip.transition_in is not None
            and at_time < clip.start_time + clip.transition_in.duration
        ):
            offenders.append(f"clip {clip.id!r} (transition_in)")
        offenders.extend(_clip_static_offenders(clip))
        # contain_blur letterboxes over a blurred fill; the preview graph
        # always fill-scales + center-crops, so its framing would silently
        # differ from the export.
        if clip.fit == "contain_blur":
            offenders.append(f"clip {clip.id!r} (fit=contain_blur)")
        active_clips = [
            element
            for track in _video_tracks(project)
            for element in track.elements
            if isinstance(element, ClipElement)
            and element.start_time <= at_time < element.start_time + element.duration
        ]
        if len(active_clips) > 1:
            offenders.append("multiple video-track clips active (multitrack composite)")
    offenders.extend(
        f"overlay {overlay.id!r}" for overlay in _active_overlays_at(project, at_time)
    )
    return offenders


def build_preview_command(
    project: EditorProject,
    *,
    resolved_media: dict[str, Path],
    text_pngs: dict[str, Path],
    at_time: float,
    output_path: Path,
    font_path: str,
) -> list[str]:
    """Build the ``ffmpeg`` argv that renders ONE PNG frame of the composed
    timeline at ``at_time``. PURE — no I/O.

    Only the single clip on-screen at ``at_time`` and the text overlays active
    then are composited (a still frame needs no concat/audio). The source frame
    is picked with an input ``-ss`` seek, then framed with the same
    scale/crop-to-``aspect`` as the full render. Raises ``EditorError`` when no
    clip is active or its media/text PNG is missing.
    """
    clip = _clip_at(project, at_time)
    if clip is None:
        raise EditorError(f"No clip active at t={at_time}")

    width, height = project.aspect_w, project.aspect_h
    inputs: list[tuple[list[str], str]] = []
    filter_parts: list[str] = []

    clip_path = _require_media(resolved_media, clip.media_id, clip.id)
    clip_idx = _register_input(
        inputs, clip_path, pre_flags=["-ss", str(_source_seek(clip, at_time))]
    )
    filter_parts.append(
        preview_clip_filter(clip_idx, clip, "pbase", width=width, height=height)
    )

    current = "pbase"
    for position, text in enumerate(_active_texts_at(project, at_time)):
        text_idx = _register_input(
            inputs, _require_text_png(text_pngs, text.id), pre_flags=["-loop", "1"]
        )
        out_label = f"pvo{position}"
        filter_parts.append(preview_overlay_filter(current, text_idx, text, out_label))
        current = out_label

    return _assemble_preview_argv(inputs, ";".join(filter_parts), current, output_path)


def _assemble_preview_argv(
    inputs: list[tuple[list[str], str]],
    filter_complex: str,
    final_v: str,
    output_path: Path,
) -> list[str]:
    argv = ["ffmpeg", "-y"]
    for pre_flags, path in inputs:
        argv.extend(pre_flags)
        argv.extend(["-i", path])
    argv.extend(["-filter_complex", filter_complex])
    argv.extend(["-map", f"[{final_v}]"])
    argv.extend(["-frames:v", "1", "-update", "1"])
    argv.append(str(output_path))
    return argv


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _assets_by_id(project: EditorProject) -> dict[str, MediaAsset]:
    return {asset.id: asset for asset in project.assets}


def _used_media_ids(project: EditorProject) -> list[str]:
    """media_ids referenced by clips + music + overlays, de-duplicated,
    order-preserving."""
    seen: dict[str, None] = {}
    for clip in _ordered_video_clips(project):
        seen.setdefault(clip.media_id, None)
    for music in _ordered_music(project):
        seen.setdefault(music.media_id, None)
    for overlay in _ordered_overlays(project):
        seen.setdefault(overlay.media_id, None)
    return list(seen)


def _resolve_asset_path(asset: MediaAsset, media_dir: Path) -> Path:
    """Resolve one asset to a local file (reuse ``local_path`` or
    ``resolve_media_source``, keyed by the asset's own uuid — this model has
    no client-side row id to key by)."""
    if asset.local_path:
        return Path(asset.local_path)
    dest_dir = media_dir / CLIPS_SUBDIR / asset.id
    path = resolve_media_source(asset.source, dest_dir)
    if path is None:
        raise EditorError(
            f"Could not resolve media for asset {asset.id!r} (source {asset.source!r})"
        )
    return path


def _resolve_media(project: EditorProject, media_dir: Path) -> dict[str, Path]:
    """Resolve every used media_id to a local file. Raises ``EditorError`` when a
    referenced media_id is unknown or a required clip cannot be resolved."""
    assets = _assets_by_id(project)
    resolved: dict[str, Path] = {}
    for media_id in _used_media_ids(project):
        asset = assets.get(media_id)
        if asset is None:
            raise EditorError(f"Unknown media_id referenced: {media_id!r}")
        resolved[media_id] = _resolve_asset_path(asset, media_dir)
    return resolved


def resolve_project_media(project: EditorProject, media_dir: Path) -> dict[str, Path]:
    """Public entry point for a caller OUTSIDE this module that needs every
    used media_id resolved to a local file (e.g. ``app.captions.
    build_captions``, which lives outside ``app.editor`` and so must not
    reach into ``_resolve_media`` directly). Thin wrapper, same contract."""
    return _resolve_media(project, media_dir)


def resolve_asset_media(asset: MediaAsset, media_dir: Path) -> Path:
    """Public entry point for resolving ONE asset to a local file, for a caller
    outside this module (e.g. ``app.mcp.tools.analyze_media``, which analyses a
    library asset that may not be on the timeline yet and so never shows up in
    ``resolve_project_media``'s used-media set). Thin wrapper, same contract:
    raises ``EditorError`` when the media cannot be resolved."""
    return _resolve_asset_path(asset, media_dir)


def probe_has_audio(path: Path) -> bool:
    """True when the media file at ``path`` has at least one audio stream.

    Uses ``ffprobe`` (bundled with ffmpeg, already a hard dependency of this
    render engine) rather than guessing from the container/extension: a
    video-only fallback download can land without an audio stream, and only
    the file itself can say so for certain.

    Raises ``EditorError`` when ffprobe exceeds ``PROBE_TIMEOUT_SEC`` (a hung
    or unreadable source must not wedge the caller forever) rather than
    returning a guessed ``False``.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=PROBE_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        raise EditorError(
            f"ffprobe timed out after {PROBE_TIMEOUT_SEC}s probing audio streams in {path}"
        ) from exc
    return bool(result.stdout.strip())


def _probe_clip_audio(
    project: EditorProject, resolved_media: dict[str, Path]
) -> dict[str, bool]:
    """Probe each distinct clip media file once for an audio stream.

    Only clips are probed (music beds are audio content by construction). A
    media_id missing from ``resolved_media`` is left for ``_require_media`` to
    raise on inside ``build_ffmpeg_command`` rather than reported here.
    """
    availability: dict[str, bool] = {}
    for clip in _ordered_video_clips(project):
        if clip.media_id in availability:
            continue
        path = resolved_media.get(clip.media_id)
        availability[clip.media_id] = True if path is None else probe_has_audio(path)
    return availability


def probe_video_dimensions(path: Path) -> tuple[int, int] | None:
    """Probe a media file's native (width, height) via ``ffprobe``. Returns
    ``None`` when it cannot be determined instead of raising — the caller
    (``_clip_dimensions_for_blur_fill``) turns a miss into a clear
    ``EditorError`` naming the offending clip, the same shape as
    ``_require_media``.

    Only needed for ``ClipElement.fit == "contain_blur"`` (the blur-fill
    canvas needs the source's real aspect to size the cover-scaled
    background and the contain-fit foreground) — every other clip renders
    without ever calling this, so it costs nothing for the common case.

    Raises ``EditorError`` when ffprobe exceeds ``PROBE_TIMEOUT_SEC`` (a hung
    or unreadable source must not wedge the caller forever) rather than
    returning a guessed ``None``.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=PROBE_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        raise EditorError(
            f"ffprobe timed out after {PROBE_TIMEOUT_SEC}s probing dimensions of {path}"
        ) from exc
    value = result.stdout.strip()
    if "x" not in value:
        return None
    width_str, _, height_str = value.partition("x")
    try:
        return int(width_str), int(height_str)
    except ValueError:
        return None


def probe_duration_sec(source: str) -> float | None:
    """Probe a media source's duration in seconds via ``ffprobe``. Returns
    ``None`` when the source is not yet resolvable/probeable (e.g. an
    unreachable URL) instead of raising — duration is best-effort metadata,
    not required to register an asset.

    A URL source is egress-checked first: if its host resolves to a private /
    loopback / link-local / metadata address, we return ``None`` WITHOUT running
    ffprobe rather than let ffprobe's network protocols reach an internal host
    (SSRF). Returning ``None`` keeps the best-effort contract — a blocked probe
    is indistinguishable from an unreachable one, and duration is optional.

    Raises ``EditorError`` when ffprobe exceeds ``PROBE_TIMEOUT_SEC`` (a hung
    or unreadable source — including a stalled read against a URL that passed
    the egress check — must not wedge the caller forever) rather than returning
    a guessed ``None``.
    """
    if source.startswith(("http://", "https://")):
        try:
            assert_url_egress_allowed(source)
        except EgressBlockedError:
            logger.warning("refusing to probe non-public URL: %s", source)
            return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                source,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=PROBE_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        raise EditorError(
            f"ffprobe timed out after {PROBE_TIMEOUT_SEC}s probing duration of {source!r}"
        ) from exc
    value = result.stdout.strip()
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _mlt_only_offenders(project: EditorProject) -> list[str]:
    """Name every element/feature the ffmpeg engine cannot express: a
    transition, a keyframed transform (or a static scale/pos_x/pos_y away
    from neutral — the ffmpeg filter graph has no notion of a sub-frame
    position/size), a color grade, or any overlay-track element. Used by
    ``_render_via_ffmpeg`` to fail with a clear, named error instead of
    silently dropping them. The per-clip static checks live in
    ``_clip_static_offenders`` (shared with ``_preview_unsupported_at``)."""
    offenders: list[str] = []
    for clip in _ordered_video_clips(project):
        if clip.transition_in is not None:
            offenders.append(f"clip {clip.id!r} (transition_in)")
        offenders.extend(_clip_static_offenders(clip))
    offenders.extend(f"overlay {overlay.id!r}" for overlay in _ordered_overlays(project))
    if len(_video_tracks(project)) > 1:
        offenders.append("project has >1 video track (multitrack is mlt-only)")
    return offenders


def _require_ffmpeg_compatible(project: EditorProject) -> None:
    offenders = _mlt_only_offenders(project)
    if offenders:
        raise EditorError(
            "This project uses MLT-only features and cannot render on the "
            f"ffmpeg engine: {'; '.join(offenders)}. Use engine=\"mlt\" "
            "(the default since phase 4) or remove them."
        )


def _rasterize_texts(project: EditorProject, media_dir: Path) -> dict[str, Path]:
    text_dir = media_dir / TEXT_SUBDIR
    pngs: dict[str, Path] = {}
    for text in _ordered_texts(project):
        out_path = text_dir / f"{project.id}_{text.id}.png"
        pngs[text.id] = rasterize_text_png(
            text.content, text.style, width=project.aspect_w, out_path=out_path
        )
    return pngs


def render_project_file(
    project: EditorProject,
    *,
    media_dir: Path,
    output_path: Path,
    font_path: str,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
    engine: str = "ffmpeg",
    melt_binary: str = "melt",
) -> Path:
    """Render ``project`` to ``output_path`` (atomic) and return it.

    ``engine`` selects the render path: ``"ffmpeg"`` (default, today's
    hand-built filter_complex graph) or ``"mlt"`` (XML -> ``melt`` subprocess,
    phase 3). Both paths resolve media/rasterize text the same way and share
    the same atomic-replace + timeout contract, so ``app.mcp.tools`` needs no
    per-engine branching of its own.

    R-20's floor is checked HERE, at the one entry both engines share, so every
    surface that renders — the HTTP routes, the MCP tools, the CLI — is behind
    it without each having to remember. Before the engines, so a volume with no
    room costs no subprocess at all: a render that dies at minute fourteen of a
    fifteen-minute encode has spent the compute AND still has nothing to show.
    """
    output_path = Path(output_path)
    guard_disk_space(output_path.parent, purpose="render")
    with no_space_becomes_a_sentence("render"):
        if engine == "mlt":
            return _render_via_mlt(
                project,
                media_dir=media_dir,
                output_path=output_path,
                crf=crf,
                preset=preset,
                melt_binary=melt_binary,
            )
        return _render_via_ffmpeg(
            project,
            media_dir=media_dir,
            output_path=output_path,
            font_path=font_path,
            crf=crf,
            preset=preset,
        )


def _render_via_ffmpeg(
    project: EditorProject,
    *,
    media_dir: Path,
    output_path: Path,
    font_path: str,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
) -> Path:
    """Resolves media, rasterizes text overlays, builds the ffmpeg command and
    runs it streaming to a temp file, then ``os.replace`` into place.

    Raises ``EditorError`` naming the offending elements when the project
    uses any MLT-only feature — wave 1's transitions/keyframes/color/
    overlays/non-default scale-pos (``_require_ffmpeg_compatible``) and
    wave 2's audio envelopes/contain_blur (``_check_mlt_only_features``) —
    rather than silently rendering a Short that is missing what the agent
    asked for.
    """
    _require_ffmpeg_compatible(project)
    _check_mlt_only_features(project)
    media_dir = Path(media_dir)
    output_path = Path(output_path)

    resolved_media = _resolve_media(project, media_dir)
    text_pngs = _rasterize_texts(project, media_dir)
    media_has_audio = _probe_clip_audio(project, resolved_media)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.mp4")
    argv = build_ffmpeg_command(
        project,
        resolved_media=resolved_media,
        text_pngs=text_pngs,
        output_path=tmp_path,
        font_path=font_path,
        media_has_audio=media_has_audio,
        crf=crf,
        preset=preset,
    )
    logger.info("rendering editor project %s -> %s", project.id, output_path)
    try:
        subprocess.run(argv, check=True, timeout=SUBPROCESS_TIMEOUT_SEC)
        os.replace(tmp_path, output_path)
    finally:
        # A failed/timed-out ffmpeg (or a failed replace) must not leave
        # <id>.tmp.mp4 behind forever; os.replace already moved it away on
        # success, so this is a no-op then.
        tmp_path.unlink(missing_ok=True)
    return output_path


def _clip_dimensions_for_blur_fill(
    project: EditorProject, resolved_media: dict[str, Path]
) -> dict[str, tuple[int, int]]:
    """Probe native (width, height) for every clip using ``fit ==
    "contain_blur"`` — the only fit mode that needs it (see
    ``app.editor.mlt_graph._blur_fill_bg_filters``). Keyed by clip element id
    (not media_id): two clips can share one asset with different fits."""
    dimensions: dict[str, tuple[int, int]] = {}
    for clip in _ordered_video_clips(project):
        if clip.fit != "contain_blur":
            continue
        path = resolved_media.get(clip.media_id)
        if path is None:
            continue  # _require_media inside build_mlt_xml raises on this
        probed = probe_video_dimensions(path)
        if probed is None:
            raise EditorError(
                f"Could not determine video dimensions for clip {clip.id!r} "
                f"(media {clip.media_id!r}) — required for its contain_blur fit"
            )
        dimensions[clip.id] = probed
    return dimensions


def _check_mlt_only_features(project: EditorProject) -> None:
    """Raise a clear ``EditorError`` naming the offending elements when the
    project uses a feature the ffmpeg engine cannot render (audio envelopes,
    the contain_blur canvas fit) — never silently degrade/ignore them."""
    envelope_ids = [
        el.id
        for track in project.tracks
        for el in track.elements
        if isinstance(el, (ClipElement, AudioElement))
        and (el.fade_in_sec > 0 or el.fade_out_sec > 0 or el.volume_keyframes)
    ]
    if envelope_ids:
        raise EditorError(
            "audio envelopes (fade_in_sec/fade_out_sec/volume_keyframes) "
            f"require the mlt engine: elements {envelope_ids}"
        )
    blur_fill_ids = [
        el.id
        for track in project.tracks
        for el in track.elements
        if isinstance(el, ClipElement) and el.fit == "contain_blur"
    ]
    if blur_fill_ids:
        raise EditorError(
            f"fit='contain_blur' requires the mlt engine: elements {blur_fill_ids}"
        )


def _text_png_sizes(text_pngs: dict[str, Path]) -> dict[str, tuple[int, int]]:
    """Read each rasterized text PNG's pixel size (PIL, lazy import — mirrors
    the rest of this module's I/O-on-demand convention)."""
    from PIL import Image

    return {text_id: Image.open(path).size for text_id, path in text_pngs.items()}


def _render_via_mlt(
    project: EditorProject,
    *,
    media_dir: Path,
    output_path: Path,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
    melt_binary: str = "melt",
) -> Path:
    """Resolve media/rasterize text (same helpers as the ffmpeg path), compile
    the timeline to MLT XML, run ``melt`` against it, then verify the output
    before the atomic replace.

    The XML's consumer target is baked to point at the TMP path (melt writes
    its own target, unlike ffmpeg which we invoke with an explicit output
    path argument) — so ``build_mlt_xml`` must receive ``tmp_path``, never
    ``output_path``. The ``.mlt`` XML file itself is kept next to the export
    on success, as the artifact of record for debugging.

    melt is known to exit 0 on some kill/SIGTERM-adjacent paths (mlt#547), so
    exit code alone is not trusted: the tmp output must exist and probe to at
    least the timeline's expected duration (0.5s slack) before it is treated
    as a real render and replaces ``output_path``.

    HYBRID loudness (comp-9 live gate, 2026-07-17): MLT 7.30 cannot express
    the loudness recipe itself (avfilter bridge breaks on loudnorm/dynaudnorm;
    native dynamic_loudness measured -9.5 LUFS / TP +1.9 dBTP / LRA 10 on real
    material). So when ``project.loudness`` is set, melt renders to a PCM-audio
    intermediate (single AAC generation overall) and an ffmpeg audio-only
    post-pass applies the EXACT ffmpeg-engine chain (``loudness_chain``) with
    ``-c:v copy`` — the video stream is never re-encoded. Loudness=None skips
    the post-pass entirely (single melt pass straight to mp4).
    """
    media_dir = Path(media_dir)
    output_path = Path(output_path)

    resolved_media = _resolve_media(project, media_dir)
    text_pngs = _rasterize_texts(project, media_dir)
    text_png_sizes = _text_png_sizes(text_pngs)
    clip_dimensions = _clip_dimensions_for_blur_fill(project, resolved_media)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalize = project.loudness is not None
    tmp_path = output_path.with_suffix(".tmp.mp4")
    melt_target = output_path.with_suffix(".pcm.tmp.mov") if normalize else tmp_path
    xml_path = output_path.with_suffix(".mlt")
    xml = build_mlt_xml(
        project,
        resolved_media=resolved_media,
        text_pngs=text_pngs,
        text_png_sizes=text_png_sizes,
        clip_dimensions=clip_dimensions,
        output_path=melt_target,
        crf=crf,
        preset=preset,
        pcm_audio=normalize,
    )
    xml_path.write_text(xml, encoding="utf-8")

    logger.info("rendering editor project %s via mlt -> %s", project.id, output_path)
    # effective_timeline_end, not _video_timeline_end: a transition shrinks
    # the rendered length by its duration (see mlt_graph.TransitionSpec). Over
    # the FIRST (V1) video track only — it defines the film's length, so an
    # upper (V2+) clip overhanging past V1's end must not inflate this baseline.
    expected_duration = effective_timeline_end(_main_video_clips(project))
    try:
        subprocess.run(
            [melt_binary, str(xml_path), "-loglevel", "error", "-silent"],
            check=True,
            timeout=SUBPROCESS_TIMEOUT_SEC,
        )
        _verify_mlt_output(melt_target, expected_duration)
        if normalize:
            subprocess.run(
                build_audio_postpass_command(
                    melt_target, tmp_path, loudness=project.loudness
                ),
                check=True,
                timeout=SUBPROCESS_TIMEOUT_SEC,
            )
            actual = probe_duration_sec(str(tmp_path)) if tmp_path.exists() else None
            if actual is None or actual < expected_duration - 0.5:
                raise EditorError(
                    "audio post-pass produced a missing/short output: expected >= "
                    f"{expected_duration - 0.5:.2f}s, probed {actual!r} at {tmp_path}"
                )
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)
        if normalize:
            melt_target.unlink(missing_ok=True)
    return output_path


def _verify_mlt_output(path: Path, expected_duration: float) -> None:
    """melt can exit 0 on some kill paths (mlt#547) — never trust the exit
    code alone; the output must exist and probe to the timeline's length."""
    actual_duration = probe_duration_sec(str(path)) if path.exists() else None
    if actual_duration is None or actual_duration < expected_duration - 0.5:
        raise EditorError(
            "mlt render produced a missing/short output: expected >= "
            f"{expected_duration - 0.5:.2f}s (timeline {expected_duration:.2f}s), "
            f"probed {actual_duration!r} at {path} "
            "(melt can exit 0 on some kill paths — verified explicitly, see mlt#547)"
        )


def build_audio_postpass_command(
    intermediate: Path, output: Path, *, loudness
) -> list[str]:
    """PURE — the hybrid's audio-only normalization pass: video stream copied
    byte-for-byte, audio run through the exact ffmpeg-engine chain
    (``ffmpeg_graph.loudness_chain``) and encoded to AAC once."""
    from app.editor.ffmpeg_graph import loudness_chain

    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(intermediate),
        "-c:v", "copy",
        "-af", loudness_chain(loudness),
        "-c:a", AUDIO_CODEC,
        "-ar", "44100",
        "-movflags", "+faststart",
        str(output),
    ]


def _rasterize_active_texts(
    project: EditorProject, at_time: float, media_dir: Path
) -> dict[str, Path]:
    """Rasterize only the text overlays active at ``at_time`` (preview needs no more)."""
    text_dir = media_dir / TEXT_SUBDIR
    pngs: dict[str, Path] = {}
    for text in _active_texts_at(project, at_time):
        out_path = text_dir / f"{project.id}_{text.id}.png"
        pngs[text.id] = rasterize_text_png(
            text.content, text.style, width=project.aspect_w, out_path=out_path
        )
    return pngs


def render_preview_frame(
    project: EditorProject,
    *,
    media_dir: Path,
    output_path: Path,
    at_time: float,
    font_path: str,
) -> Path:
    """Render a single PNG frame of ``project`` at ``at_time`` (atomic) and return
    it. Mirrors ``render_project_file`` but resolves only the media for the clip
    active at ``at_time`` and rasterizes only the text overlays visible then.
    Raises ``EditorError`` when no clip is active or its media can't be resolved.

    Carries R-20's floor for the same reason ``render_project_file`` does. A
    preview is small, but it resolves (and can DOWNLOAD) its clip's media and
    rasterizes text before it encodes anything — on a volume with no room those
    are exactly the writes that die halfway. The guard-then-wrap shell is kept
    separate from the work for the same reason it is there, one entry point
    behind which nothing has to remember.
    """
    output_path = Path(output_path)
    guard_disk_space(output_path.parent, purpose="preview")
    with no_space_becomes_a_sentence("preview"):
        return _render_preview_frame(
            project,
            media_dir=Path(media_dir),
            output_path=output_path,
            at_time=at_time,
            font_path=font_path,
        )


def _render_preview_frame(
    project: EditorProject,
    *,
    media_dir: Path,
    output_path: Path,
    at_time: float,
    font_path: str,
) -> Path:
    """The preview render itself — see ``render_preview_frame`` for the
    contract; this is the body its disk guard wraps."""
    clip = _clip_at(project, at_time)
    if clip is None:
        raise EditorError(f"No clip active at t={at_time}")
    # The single-clip ffmpeg preview graph cannot express everything the real
    # (MLT) render shows. Refuse — before resolving media or spawning ffmpeg —
    # rather than serve a frame that silently drops what is on screen at this
    # moment (the "dragged element snapped back" bug).
    unsupported = _preview_unsupported_at(project, at_time)
    if unsupported:
        raise EditorError(
            f"Exact frame unavailable at t={at_time}: the preview renderer "
            f"cannot composite {'; '.join(unsupported)}"
        )
    asset = _assets_by_id(project).get(clip.media_id)
    if asset is None:
        raise EditorError(f"Unknown media_id referenced: {clip.media_id!r}")

    resolved_media = {clip.media_id: _resolve_asset_path(asset, media_dir)}
    text_pngs = _rasterize_active_texts(project, at_time, media_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.png")
    argv = build_preview_command(
        project,
        resolved_media=resolved_media,
        text_pngs=text_pngs,
        at_time=at_time,
        output_path=tmp_path,
        font_path=font_path,
    )
    logger.info("rendering preview of %s @ %ss -> %s", project.id, at_time, output_path)
    try:
        subprocess.run(argv, check=True, timeout=SUBPROCESS_TIMEOUT_SEC)
        os.replace(tmp_path, output_path)
    finally:
        # Same rationale as render_project_file: don't leak <id>.tmp.png on a
        # failed/timed-out ffmpeg run.
        tmp_path.unlink(missing_ok=True)
    return output_path
