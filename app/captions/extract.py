"""Extract a clip's own trimmed source window to a standalone audio file.

Transcribing the WHOLE source file would be wasteful (most clips are a small
slice of a much longer download) and wrong when ``trim_start`` > 0 (segment
times would be measured against the wrong origin). Cutting the exact window
first with ffmpeg keeps the transcriber's own segment times trim-relative —
see ``app.captions.mapping.window_time_to_timeline_time`` for how those map
back onto the timeline.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from app.editor.errors import EditorError
from app.editor.model import ClipElement

logger = logging.getLogger(__name__)

# 16kHz mono is what faster-whisper/whisper resamples to internally anyway;
# extracting straight to that format keeps the intermediate file small and
# skips redundant resampling work.
ASR_SAMPLE_RATE = 16000
EXTRACT_TIMEOUT_SEC = 5 * 60


def _clip_out_point(clip: ClipElement) -> float:
    """Source out-point in seconds. Mirrors
    ``app.editor.ffmpeg_graph._clip_out_point`` (duplicated rather than
    imported: that name is module-private, and this module lives outside
    ``app.editor`` — the same reasoning ``app.editor.validation`` already
    documents for its own copy). Keep the two in sync by hand if trim/speed
    semantics ever change."""
    if clip.trim_end is not None:
        return clip.trim_end
    return clip.trim_start + clip.duration * clip.transform.speed


def extract_clip_window(source_path: Path, clip: ClipElement, out_path: Path) -> Path:
    """Cut ``[clip.trim_start, clip's own source out-point)`` from
    ``source_path`` into a 16kHz mono WAV at ``out_path``. Raises
    ``EditorError`` when ffmpeg fails (corrupt source, no audio stream)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_point = _clip_out_point(clip)
    argv = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source_path),
        "-vn",
        "-ss", str(clip.trim_start),
        "-to", str(out_point),
        "-ar", str(ASR_SAMPLE_RATE),
        "-ac", "1",
        str(out_path),
    ]
    try:
        subprocess.run(argv, check=True, timeout=EXTRACT_TIMEOUT_SEC, capture_output=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        # The exception's own str() quotes the whole argv, and the source path
        # is this container's filesystem layout — neither may travel to the
        # client in the ``EditorError`` message the API returns verbatim
        # (R-05). The caller keeps the clip id, which is theirs already and is
        # what makes the failure actionable.
        logger.error(
            "audio window extraction failed for clip %s from %s: %s",
            clip.id, source_path, exc,
        )
        raise EditorError(
            f"could not extract the audio window for clip {clip.id!r}"
        ) from exc
    return out_path
