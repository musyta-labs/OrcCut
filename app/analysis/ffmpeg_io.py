"""Shared ffmpeg/ffprobe plumbing for the analysis engine.

``frames.py`` and ``audio.py`` both pipe a whole decoded stream out of ffmpeg
on stdout, and both must turn a non-zero exit into an ``EditorError`` naming
the stage that failed rather than a bare ``CalledProcessError``. That is one
behaviour, so it lives in one place (DRY) — the argv itself stays with the
caller that knows what it is decoding.

ffmpeg's stderr is logged, never returned (R-05): an ``EditorError`` message
travels verbatim to the client in the tool/API error envelope, and ffmpeg's
diagnostics quote absolute media paths and the argv, i.e. this container's
filesystem layout. The operator gets the detail in the log; the caller gets a
stable sentence naming what was being done.

Mirrors the subprocess conventions of ``app.captions.extract`` (explicit argv,
``check=True``, an explicit timeout, ``capture_output``) and the ffprobe
conventions of ``app.editor.render.probe_duration_sec``.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from app.editor.errors import EditorError
from app.editor.render import PROBE_TIMEOUT_SEC

logger = logging.getLogger(__name__)

# Decoding a whole clip is seconds of CPU for the short-form media this engine
# handles; the ceiling only exists so a pathological input cannot hang a
# worker. Same order of magnitude as ``captions.extract.EXTRACT_TIMEOUT_SEC``.
DECODE_TIMEOUT_SEC = 5 * 60
# How much of ffmpeg's stderr to keep in the SERVER LOG — enough to name the
# real failure, short enough to stay readable in a log line.
STDERR_TAIL_CHARS = 500
# Fallback when ffprobe cannot report a frame rate (odd container, image
# sequence). Only ever used to keep time math finite; never silently correct.
FALLBACK_FPS = 30.0


class FfmpegFailure(EditorError):
    """A non-zero ffmpeg exit, with the process's stderr tail kept OUT of the
    message and reachable only through ``.stderr``.

    The split is the whole point of R-05: ``str(exc)`` is what boundary code
    hands the client, while a caller that must react to a specific ffmpeg
    diagnostic (``app.analysis.audio`` treats "no audio stream" as an empty
    buffer, not a failure) reads the attribute instead of scraping the message.
    """

    def __init__(self, message: str, *, stderr: str) -> None:
        super().__init__(message)
        self.stderr = stderr


def _stderr_tail(raw: bytes | None) -> str:
    if not raw:
        return "<no stderr>"
    return raw.decode("utf-8", errors="replace").strip()[-STDERR_TAIL_CHARS:]


def run_ffmpeg_capture(argv: list[str], *, what: str) -> bytes:
    """Run ``argv`` and return its raw stdout bytes.

    Raises ``EditorError`` naming ``what`` when ffmpeg exits non-zero or
    exceeds ``DECODE_TIMEOUT_SEC`` — never swallowed, never a silent empty
    buffer. ffmpeg's own diagnostics go to the log, not into the exception
    (R-05): see this module's docstring.
    """
    try:
        result = subprocess.run(
            argv, check=True, timeout=DECODE_TIMEOUT_SEC, capture_output=True
        )
    except subprocess.CalledProcessError as exc:
        stderr = _stderr_tail(exc.stderr)
        logger.error(
            "ffmpeg exited %s while %s: argv=%r stderr=%s", exc.returncode, what, argv, stderr
        )
        raise FfmpegFailure(f"ffmpeg failed while {what}", stderr=stderr) from exc
    except subprocess.TimeoutExpired as exc:
        logger.error("ffmpeg timed out while %s: argv=%r", what, argv)
        raise EditorError(
            f"ffmpeg timed out after {DECODE_TIMEOUT_SEC}s while {what}"
        ) from exc
    return result.stdout


def probe_fps(path: Path) -> float:
    """Probe a media file's average frame rate via ``ffprobe``.

    Deliberately probed rather than assumed: every timestamp this package
    emits is ``frame_index / fps``, so a hardcoded 30 would silently mis-place
    every event of a 24fps or 60fps source. Mirrors the ffprobe argv style of
    ``app.editor.render.probe_video_dimensions``; falls back to
    ``FALLBACK_FPS`` only when ffprobe reports nothing usable.

    Raises ``EditorError`` when ffprobe exceeds ``PROBE_TIMEOUT_SEC`` (mirrors
    ``app.editor.render``'s own probes) rather than silently falling back to
    ``FALLBACK_FPS`` — a hang here must surface, not masquerade as a guess.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate",
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
            f"ffprobe timed out after {PROBE_TIMEOUT_SEC}s probing frame rate of {path}"
        ) from exc
    return _parse_frame_rate(result.stdout.strip())


def _parse_frame_rate(value: str) -> float:
    """Parse ffprobe's ``num/den`` (or plain float) frame-rate notation."""
    if not value:
        return FALLBACK_FPS
    numerator, _, denominator = value.partition("/")
    try:
        fps = float(numerator) / float(denominator) if denominator else float(numerator)
    except (ValueError, ZeroDivisionError):
        return FALLBACK_FPS
    return fps if fps > 0 else FALLBACK_FPS
