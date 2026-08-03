"""Decode a clip to a stack of tiny grayscale frames and measure per-frame
motion.

Downscaling to 160x90 gray before any math is the whole trick: a whole short
clip decodes in one ffmpeg process into a few megabytes, and frame-to-frame
mean absolute difference at that size still tracks "something happened" while
being immune to compression noise at full resolution. Validated live on four
cats clips (2026-07-20 spike).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.analysis.ffmpeg_io import probe_fps, run_ffmpeg_capture
from app.editor.errors import EditorError

# Analysis resolution. Small enough that a whole clip fits in memory
# comfortably, large enough that a subject moving in one third of the frame
# still moves the mean.
ANALYSIS_W = 160
ANALYSIS_H = 90


def read_gray_frames(
    path: Path, *, width: int = ANALYSIS_W, height: int = ANALYSIS_H
) -> np.ndarray:
    """Decode the whole video at ``path`` into an ``(n, height, width)``
    ``int16`` array of grayscale frames.

    ``int16`` (not ``uint8``) because the very next thing every caller does is
    a signed difference — promoting here once avoids an easy wraparound bug
    downstream.

    Forces constant-frame-rate decode at the source's own PROBED (average)
    fps via ``-vsync cfr -r {fps}``. Every timestamp downstream is
    ``frame_index / fps`` (``app.analysis.peaks.pick_events``), so that math
    is only correct when the decoded frame index actually IS spaced at
    ``fps`` seconds apart. On a variable-frame-rate source (TikTok downloads
    commonly are) that is not true of the raw decode: measured live, an
    un-resampled decode of a VFR fixture padded a 10fps segment up to the
    container's higher 50fps ``r_frame_rate`` (300 raw frames instead of the
    container's own 80 ``nb_frames``), which put a planted event at wall-clock
    3.0s at computed index/fps time 11.25s — nowhere close, and past the
    clip's own duration. ``-vsync cfr -r {fps}`` resamples decode onto the
    probed rate's own grid (confirmed live: frame count then matches
    ``nb_frames`` and the same event lands within one frame period of the
    real time) so the index space this package's math assumes is the index
    space ffmpeg actually produces. See ``tests/test_ffmpeg_io_vfr.py``.
    """
    fps = probe_fps(path)
    argv = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vsync", "cfr",
        "-r", str(fps),
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-s", f"{width}x{height}",
        "-",
    ]
    raw = run_ffmpeg_capture(argv, what=f"decoding frames of {path.name!r}")
    return _frames_from_buffer(raw, width=width, height=height, source=path)


def _frames_from_buffer(
    raw: bytes, *, width: int, height: int, source: Path
) -> np.ndarray:
    """Reshape a rawvideo byte buffer into frames, discarding a truncated
    trailing frame (a killed/short-decoded stream can end mid-frame, and
    ``reshape`` would raise on the ragged tail)."""
    frame_size = width * height
    usable = (len(raw) // frame_size) * frame_size
    if usable == 0:
        raise EditorError(
            f"no complete video frames decoded from {str(source)!r} "
            f"(got {len(raw)} bytes, need {frame_size} per frame)"
        )
    flat = np.frombuffer(raw[:usable], dtype=np.uint8)
    return flat.reshape(-1, height, width).astype(np.int16)


def motion_diffs(frames: np.ndarray) -> np.ndarray:
    """Mean absolute pixel change between consecutive frames.

    Returns a float array of length ``len(frames) - 1`` (empty when fewer than
    two frames). Element ``i`` describes the transition INTO frame ``i + 1``,
    which is the convention every timestamp in this package follows.
    Never mutates ``frames``.
    """
    if frames.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    return np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2)).astype(np.float64)
