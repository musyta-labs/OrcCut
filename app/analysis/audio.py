"""Decode a clip's audio to mono PCM and measure loudness per video frame.

The audio channel exists to separate a real beat (a smack, a yelp, a punchline)
from mere camera movement: a motion spike that also lands on a loudness spike
is what the combined score in ``peaks.py`` rewards. Sampling is deliberately
per VIDEO frame so the two signals share one index space.

A clip with no audio stream is normal (video-only downloads happen), so it
yields zeros here and the caller falls back to motion-only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.analysis.ffmpeg_io import FfmpegFailure, run_ffmpeg_capture

# 16kHz mono matches app.captions.extract.ASR_SAMPLE_RATE: plenty for an
# envelope measurement and a small buffer to pipe.
ANALYSIS_SAMPLE_RATE = 16000
# int16 PCM: the sample width `-f s16le` emits.
_PCM_DTYPE = np.int16


def read_rms(
    path: Path,
    *,
    fps: float,
    n_frames: int,
    sample_rate: int = ANALYSIS_SAMPLE_RATE,
) -> np.ndarray:
    """Root-mean-square loudness of ``path``'s audio, one value per video
    frame, as a float array of exactly ``n_frames`` values.

    Returns zeros when the file has no audio stream (ffmpeg emits an empty
    buffer) or when ``n_frames`` is not positive. Raises ``EditorError`` only
    when ffmpeg itself fails for a reason other than a missing stream.
    """
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float64)
    raw = _decode_pcm(path, sample_rate=sample_rate)
    if not raw:
        return np.zeros(n_frames, dtype=np.float64)
    samples = np.frombuffer(raw, dtype=_PCM_DTYPE).astype(np.float64)
    window = max(int(round(sample_rate / fps)), 1) if fps > 0 else sample_rate
    return _fit_to_frames(_windowed_rms(samples, window), n_frames)


def _decode_pcm(path: Path, *, sample_rate: int) -> bytes:
    """Pipe the file's audio out as raw mono int16 PCM. A file with no audio
    stream decodes to an empty buffer rather than an ffmpeg failure thanks to
    ``-vn`` plus a tolerated empty output."""
    argv = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vn",
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-",
    ]
    try:
        return run_ffmpeg_capture(argv, what=f"decoding audio of {path.name!r}")
    except FfmpegFailure as exc:
        # Read from the attribute, not from ``str(exc)``: the message is the
        # client-facing one and deliberately carries no process output (R-05).
        if _is_missing_audio_stream(exc.stderr):
            return b""
        raise


# ffmpeg's wording when the input simply has nothing to map to the audio
# output. Treated as "no audio", not as a failure.
_NO_AUDIO_MARKERS = ("does not contain any stream", "Output file does not contain")


def _is_missing_audio_stream(message: str) -> bool:
    return any(marker in message for marker in _NO_AUDIO_MARKERS)


def _windowed_rms(samples: np.ndarray, window: int) -> np.ndarray:
    """RMS over consecutive non-overlapping windows of ``window`` samples.
    The trailing partial window is dropped (it would be measured over fewer
    samples and read as a spurious level change)."""
    usable = (samples.size // window) * window
    if usable == 0:
        return np.zeros(0, dtype=np.float64)
    blocks = samples[:usable].reshape(-1, window)
    return np.sqrt(np.mean(np.square(blocks), axis=1))


def _fit_to_frames(values: np.ndarray, n_frames: int) -> np.ndarray:
    """Trim or edge-pad ``values`` to exactly ``n_frames`` entries. Returns a
    new array; never mutates the input."""
    if values.size == 0:
        return np.zeros(n_frames, dtype=np.float64)
    if values.size >= n_frames:
        return np.array(values[:n_frames], dtype=np.float64)
    return np.pad(values, (0, n_frames - values.size), mode="edge")
