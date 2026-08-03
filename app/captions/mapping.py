"""Pure timing/text helpers that turn ASR segments into caption TextElements.

No I/O here — ``app.mcp.tools.auto_captions`` is the orchestration (extract
each clip's trimmed audio window, transcribe it, call these helpers, then
``app.editor.mutations.add_text`` for each resulting caption).
"""
from __future__ import annotations

from app.editor.model import ClipElement

DEFAULT_MAX_CHARS_PER_LINE = 42
# Bottom-third caption position, matching the CapCut-style convention this
# codebase's own exemplar research already settled on for labels (see
# app.editor.model.TEXT_POSITIONS / the compile-short skill).
DEFAULT_CAPTION_POS = "bottom"


def window_time_to_timeline_time(window_time: float, clip: ClipElement) -> float:
    """Map a time WITHIN a clip's own extracted trim window (0 = the
    window's start, i.e. ``clip.trim_start`` in the source) to absolute
    TIMELINE time.

    This is the inverse of the mapping the MLT graph builder uses to go the
    other way (timeline time -> source time, see
    ``app.editor.mlt_graph._clip_source_time``): since the audio handed to
    the transcriber is a window ALREADY extracted starting at
    ``trim_start`` (see ``app.captions.extract.extract_clip_window``), a
    segment's own ``start``/``end`` (seconds from 0) are already
    trim-relative — only the playback speed and the clip's own
    ``start_time`` remain to convert:

        timeline_time = window_time / speed + clip.start_time

    A segment beginning at the window's very start (``window_time=0``)
    therefore lands exactly at ``clip.start_time``, and one ending at the
    window's last second lands at ``clip.start_time + clip.duration``
    (window duration == ``clip.duration * speed``, so dividing by speed
    brings it back to the clip's own timeline duration).
    """
    return window_time / clip.transform.speed + clip.start_time


def split_caption_text(text: str, max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE) -> str:
    """Word-wrap ``text`` into ``\\n``-joined lines no longer than
    ``max_chars_per_line`` (the existing ``add_text`` convention: ``\\n`` in
    ``content`` forces a line break). A single word longer than the limit is
    kept whole on its own line rather than broken mid-word."""
    words = text.split()
    if not words:
        return text

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars_per_line:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)
