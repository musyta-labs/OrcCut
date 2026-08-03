"""Pure ``filter_complex`` string builders for the FFmpeg render engine.

Every function here is pure: it takes model elements plus input indices/labels
and returns a filter-chain string. No I/O, no FFmpeg execution — so the whole
graph is snapshot-testable without ffmpeg installed. ``app.editor.render``
assembles these fragments into the full ``ffmpeg`` argv.
"""
from __future__ import annotations

from app.editor.model import AudioElement, ClipElement, LoudnessTarget, TextElement

# Audio silence source parameters for muted clips.
SAMPLE_RATE = 44100
SILENCE_LAYOUT = "stereo"

# Pre-levelling window for the loudness chain (see ``loudness_filter``). Tuned
# by measurement, not taste: at f=500:g=11 the same mixes land at LRA 7.7 — over
# the <=6 target — while f=250:g=7 reaches 4.9/5.2.
LEVEL_FRAME_MS = 250
LEVEL_GAUSS_SIZE = 7

# Vertical inset (px) for top/bottom text overlays.
TEXT_MARGIN = 120


def _clip_out_point(clip: ClipElement) -> float:
    """Source out-point in seconds. ``trim_end`` None = derive from the clip's
    timeline duration and playback speed so the segment lasts ``duration``s."""
    if clip.trim_end is not None:
        return clip.trim_end
    return clip.trim_start + clip.duration * clip.transform.speed


def video_clip_filter(
    input_idx: int,
    clip: ClipElement,
    out_label: str,
    *,
    width: int,
    height: int,
    fps: int,
) -> str:
    """Trim, speed-adjust, fill-scale + center-crop to ``width x height`` and
    normalize (fps/sar/pix_fmt) a single video segment for later concat."""
    speed = clip.transform.speed
    parts = [
        f"[{input_idx}:v]trim=start={clip.trim_start}:end={_clip_out_point(clip)}",
        "setpts=PTS-STARTPTS",
    ]
    if speed != 1.0:
        parts.append(f"setpts=PTS/{speed}")

    zoom = 1.0 + clip.transform.crop_zoom
    scale_w = int(round(width * zoom))
    scale_h = int(round(height * zoom))
    parts.append(f"scale={scale_w}:{scale_h}:force_original_aspect_ratio=increase")
    parts.append(f"crop={width}:{height}")
    parts.append(f"fps={fps}")
    parts.append("setsar=1")
    parts.append("format=yuv420p")
    return ",".join(parts) + f"[{out_label}]"


def clip_audio_filter(
    input_idx: int, clip: ClipElement, out_label: str, *, has_audio: bool = True
) -> str:
    """Per-clip audio segment. A muted clip, or one whose source has no audio
    stream at all, contributes generated silence (no reference to its input's
    ``[N:a]`` stream, which may not exist); otherwise trim/speed/volume the
    source audio.

    ``has_audio`` defaults to True so every existing caller keeps the prior
    behaviour: probing a file for an audio stream is I/O, so it is the
    orchestration layer's job (see ``app.editor.render.probe_has_audio``), not
    this pure builder's — a video-only source (e.g. the downloader's fallback
    format ``b``) would otherwise emit ``[N:a]atrim=...`` against a stream
    that does not exist, and ffmpeg exits non-zero, failing the whole render.
    """
    if clip.muted or not has_audio:
        return (
            f"anullsrc=r={SAMPLE_RATE}:cl={SILENCE_LAYOUT},"
            f"atrim=duration={clip.duration},asetpts=PTS-STARTPTS[{out_label}]"
        )
    speed = clip.transform.speed
    parts = [
        f"[{input_idx}:a]atrim=start={clip.trim_start}:end={_clip_out_point(clip)}",
        "asetpts=PTS-STARTPTS",
    ]
    if speed != 1.0:
        parts.append(f"atempo={speed}")
    parts.append(f"volume={clip.volume}")
    return ",".join(parts) + f"[{out_label}]"


def concat_filter(labels: list[str], out_label: str, *, is_video: bool) -> str:
    """Concat ordered segment labels into one stream."""
    joined = "".join(f"[{label}]" for label in labels)
    va = "v=1:a=0" if is_video else "v=0:a=1"
    return f"{joined}concat=n={len(labels)}:{va}[{out_label}]"


def _overlay_position(text: TextElement) -> tuple[str, str]:
    """Return the (x, y) ffmpeg overlay expressions for a text element.

    Named presets (``pos``) centre horizontally at a fixed vertical inset;
    ``pos_x``/``pos_y`` override an axis with a normalized fraction of the frame
    anchored at the box's top-left corner. In an ``overlay`` expression ``W``/``H``
    are the frame and ``w``/``h`` the text box, so the preset forms subtract the
    box size to stay inside the frame while the explicit forms do not — an
    explicit offset means "put the corner exactly here".
    """
    style = text.style
    if style.pos_x is not None:
        x = f"W*{style.pos_x}"
    else:
        x = "(W-w)/2"

    if style.pos_y is not None:
        y = f"H*{style.pos_y}"
    elif style.pos == "top":
        y = str(TEXT_MARGIN)
    elif style.pos == "bottom":
        y = f"H-h-{TEXT_MARGIN}"
    else:
        y = "(H-h)/2"
    return x, y


def overlay_filter(
    in_label: str, text_idx: int, text: TextElement, out_label: str
) -> str:
    """Composite a pre-rasterized text PNG (input ``text_idx``) over ``in_label``
    at the style position. A finite ``duration`` shows the text within its
    start..start+duration window; ``duration`` None keeps it on from ``start``
    to the end of the render (a persistent header — the output is capped by
    ``-t`` so ``gte`` never runs forever)."""
    start = text.start_time
    x, y = _overlay_position(text)
    if text.duration is None:
        enable = f"gte(t,{start})"
    else:
        enable = f"between(t,{start},{start + text.duration})"
    return (
        f"[{in_label}][{text_idx}:v]overlay=x={x}:y={y}:"
        f"enable='{enable}'[{out_label}]"
    )


def preview_clip_filter(
    input_idx: int,
    clip: ClipElement,
    out_label: str,
    *,
    width: int,
    height: int,
) -> str:
    """Fill-scale + center-crop a single already-seeked source frame to
    ``width x height`` (same framing as ``video_clip_filter``, without the
    trim/fps needed only for a moving concat). Used by the still-frame preview."""
    zoom = 1.0 + clip.transform.crop_zoom
    scale_w = int(round(width * zoom))
    scale_h = int(round(height * zoom))
    parts = [
        f"[{input_idx}:v]scale={scale_w}:{scale_h}:force_original_aspect_ratio=increase",
        f"crop={width}:{height}",
        "setsar=1",
        "format=yuv420p",
    ]
    return ",".join(parts) + f"[{out_label}]"


def preview_overlay_filter(
    in_label: str, text_idx: int, text: TextElement, out_label: str
) -> str:
    """Composite a text PNG over ``in_label`` for the still-frame preview. No
    ``enable`` window: the caller only passes texts already active at the frame."""
    x, y = _overlay_position(text)
    return f"[{in_label}][{text_idx}:v]overlay=x={x}:y={y}[{out_label}]"


def music_filter(input_idx: int, music: AudioElement, out_label: str) -> str:
    """Trim a music bed to its duration, apply volume + gain, delay to its start.

    ``volume`` is a linear multiplier and ``gain_db`` a decibel trim; both are
    honoured (a no-op 0 dB gain emits no filter, keeping the graph readable).
    """
    parts = [
        f"[{input_idx}:a]atrim=start=0:end={music.duration}",
        "asetpts=PTS-STARTPTS",
        f"volume={music.volume}",
    ]
    if music.gain_db != 0.0:
        parts.append(f"volume={music.gain_db}dB")
    if music.start_time > 0:
        delay_ms = int(round(music.start_time * 1000))
        parts.append(f"adelay={delay_ms}|{delay_ms}")
    return ",".join(parts) + f"[{out_label}]"


def loudness_filter(in_label: str, target: LoudnessTarget, out_label: str) -> str:
    """Normalize the finished mix to an integrated-loudness + loudness-range target.

    Two stages, because ``loudnorm`` alone provably does not reach the target.
    Measured on two 6-clip mixes of real cached TikTok sources (2026-07-15):

    ==========================================  =======  ======  =====
    chain                                             I     LRA     TP
    ==========================================  =======  ======  =====
    raw mix A / B                                -16.90 / -16.43   18.90 / 8.40
    ``loudnorm`` alone                           -13.65 / -11.37   12.00 / 6.40
    ``dynaudnorm`` -> ``loudnorm`` (this)        -12.13 / -11.43    4.90 / 5.20
    ==========================================  =======  ======  =====

    A compilation's loudness range is dominated by the JUMPS BETWEEN clips, not
    by dynamics inside one clip. ``loudnorm``'s single-pass gating rides those
    jumps far too slowly to close them, so it lands ~2x over the LRA target.
    ``dynaudnorm`` levels the mix in a moving window first (that is what pulls
    LRA under 6), and ``loudnorm`` then sets the integrated level and true-peak
    ceiling. The order matters: levelling after ``loudnorm`` would undo its
    integrated target.
    """
    return (
        f"[{in_label}]"
        f"{loudness_chain(target)}"
        f"[{out_label}]"
    )


def loudness_chain(target) -> str:
    """The raw (label-free) normalization chain — shared by the in-graph
    ``loudness_filter`` above and the MLT engine's audio post-pass, so the
    recipe literally cannot drift between engines."""
    return (
        f"dynaudnorm=f={LEVEL_FRAME_MS}:g={LEVEL_GAUSS_SIZE},"
        f"loudnorm=I={target.i}:LRA={target.lra}:TP={target.tp}"
    )


def amix_filter(labels: list[str], out_label: str) -> str:
    """Mix clip audio with one or more music beds."""
    joined = "".join(f"[{label}]" for label in labels)
    return (
        f"{joined}amix=inputs={len(labels)}:duration=first:"
        f"dropout_transition=0[{out_label}]"
    )
