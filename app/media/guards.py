"""Pathological-media guard: reject a file before decode, not just by size.

A byte-size cap (``app.media.uploads``' ``max_upload_mb``) stops an oversized
upload but says nothing about what a SMALL file decodes into — a handful of
KB of crafted video can carry an absurd resolution or an absurd stream count,
and either one turns the very next ffmpeg decode (analysis, export, preview)
into an OOM or a multi-hour hang that no per-process subprocess timeout alone
prevents.

``reject_pathological_media`` probes the container's OWN metadata (one
``ffprobe`` call, timeout-guarded — see
``app.editor.render.PROBE_TIMEOUT_SEC``) and raises ``EditorError`` before any
caller (upload, URL-import, or analysis) ever hands the file to a decoder.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.editor.errors import EditorError
from app.editor.render import PROBE_TIMEOUT_SEC


@dataclass(frozen=True)
class MediaSummary:
    """ffprobe's own account of a file's shape, gathered in ONE call so the
    pathological-media guard costs no more subprocess launches than a plain
    duration probe already did.

    Any field is ``None``/``0`` when ffprobe could not determine it (matches
    ``app.editor.render.probe_duration_sec``'s own best-effort contract) —
    only a TIMEOUT raises, never an unparseable/absent value.
    """

    duration_sec: float | None
    width: int | None
    height: int | None
    stream_count: int


def probe_media_summary(path: Path | str) -> MediaSummary:
    """Probe ``path``'s duration, first video stream's (width, height), and
    total stream count in one ffprobe call.

    Raises ``EditorError`` when ffprobe exceeds ``PROBE_TIMEOUT_SEC`` (mirrors
    every other probe in ``app.editor.render``/``app.analysis.ffmpeg_io``).
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration:stream=codec_type,width,height",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=PROBE_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        raise EditorError(
            f"ffprobe timed out after {PROBE_TIMEOUT_SEC}s probing {path}"
        ) from exc

    try:
        payload = json.loads(result.stdout) if result.stdout.strip() else {}
    except json.JSONDecodeError:
        payload = {}

    streams = payload.get("streams", []) if isinstance(payload, dict) else []
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    width = video_streams[0].get("width") if video_streams else None
    height = video_streams[0].get("height") if video_streams else None

    duration_raw = payload.get("format", {}).get("duration") if isinstance(payload, dict) else None
    try:
        duration_sec = float(duration_raw) if duration_raw else None
    except (TypeError, ValueError):
        duration_sec = None

    return MediaSummary(
        duration_sec=duration_sec,
        width=width,
        height=height,
        stream_count=len(streams),
    )


def reject_pathological_media(path: Path | str) -> MediaSummary:
    """Raise ``EditorError`` when ``path`` exceeds the configured resolution,
    duration, or stream-count bounds (``app.config.Settings.max_media_*``).

    Returns the probed ``MediaSummary`` on success so a caller that already
    needs the duration (e.g. ``add_media``) does not pay for a second ffprobe
    call just to get it.
    """
    summary = probe_media_summary(path)
    settings = get_settings()

    if summary.width is not None and summary.height is not None:
        pixels = summary.width * summary.height
        if pixels > settings.max_media_pixels:
            raise EditorError(
                f"media resolution {summary.width}x{summary.height} "
                f"({pixels} px) exceeds the {settings.max_media_pixels} px limit"
            )

    if summary.duration_sec is not None and summary.duration_sec > settings.max_media_duration_sec:
        raise EditorError(
            f"media duration {summary.duration_sec:.1f}s exceeds the "
            f"{settings.max_media_duration_sec:.0f}s limit"
        )

    if summary.stream_count > settings.max_media_streams:
        raise EditorError(
            f"media has {summary.stream_count} streams, exceeding the "
            f"{settings.max_media_streams} stream limit"
        )

    return summary


def reject_undecodable_media(path: Path | str) -> MediaSummary:
    """``reject_pathological_media`` PLUS a positive "is this media at all?"
    check: raise ``EditorError`` when ffprobe understood nothing about the file.

    An extension allowlist only proves what a file is CALLED. A text file (or a
    zip, or anything else) renamed to ``.mp4`` sails through it, and through the
    pathological-media bounds too — those are all upper bounds, and a file with
    no streams, no duration and no dimensions violates none of them. It then
    sits in the library forever as an undecodable "video".

    Deliberately a SEPARATE entry point rather than a stricter
    ``reject_pathological_media``. That function's existing callers — project
    ingest and the analysis engine — tolerate an unprobeable source on purpose
    (a URL-imported asset may not be probeable at ingest time, and
    ``tests/test_media_guards.py`` pins that contract). Tightening it in place
    would change their behavior; the media library is a different bargain,
    where a human hands over a file and expects to be told immediately if it is
    not usable media.

    "Understood nothing" is the conjunction of all three signals being absent,
    not the absence of any one: an audio-only file legitimately has no
    dimensions, and a stream ffprobe cannot time legitimately has no duration.
    """
    summary = reject_pathological_media(path)
    if (
        summary.stream_count == 0
        and summary.duration_sec is None
        and summary.width is None
        and summary.height is None
    ):
        raise EditorError(
            "not a media file: no audio or video stream could be read from it "
            "(the extension says media, the contents do not)"
        )
    return summary
