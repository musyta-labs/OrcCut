"""Orchestrate auto-captions: extract each clip's own trimmed audio window,
transcribe it, map segments onto the timeline, and add them as caption
TextElements. The one place this package touches ``app.editor.mutations`` —
``app.mcp.tools.auto_captions`` is the DB/session wiring around this.
"""
from __future__ import annotations

from pathlib import Path

from app.captions.extract import extract_clip_window
from app.captions.mapping import (
    DEFAULT_MAX_CHARS_PER_LINE,
    split_caption_text,
    window_time_to_timeline_time,
)
from app.captions.transcribe import ASR_MODELS_SUBDIR, transcribe_audio
from app.editor import mutations
from app.editor.errors import EditorError
from app.editor.model import ClipElement, EditorProject, TextStyle

CAPTION_WINDOWS_SUBDIR = "caption_windows"
# A near-zero segment (ASR sometimes emits a 0-length boundary artifact)
# would otherwise become a zero/negative-duration TextElement.
MIN_CAPTION_DURATION_SEC = 0.05


def _ordered_video_clips(project: EditorProject) -> list[ClipElement]:
    """Mirrors ``app.editor.render._ordered_video_clips`` (duplicated, not
    imported — module-private, and this package lives outside
    ``app.editor``; same reasoning ``app.editor.validation`` already
    documents for its own copy)."""
    clips = [
        element
        for track in project.tracks
        if track.type == "video"
        for element in track.elements
        if isinstance(element, ClipElement)
    ]
    return sorted(clips, key=lambda clip: clip.start_time)


def add_auto_captions(
    project: EditorProject,
    *,
    resolved_media: dict[str, Path],
    media_dir: Path,
    models_dir: Path,
    asr_model_size: str,
    asr_device: str,
    asr_compute_type: str,
    style: TextStyle,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
) -> tuple[EditorProject, int, str | None]:
    """Transcribe every video clip's own trimmed window and add the result as
    caption TextElements (each stamped ``role="caption"``). Returns
    ``(updated_project, captions_added, detected_language)`` — the language is
    the first ISO code ASR reported across the clips (``None`` when none did),
    so a caller can record it for a later translation pass.

    ``media_dir`` and ``models_dir`` are deliberately separate (Gate 3 Step
    10): the extracted per-clip audio windows are a transient USER artifact
    under ``media_dir``, while the ASR model weights are a shared, possibly
    read-only/pre-baked resource under ``models_dir`` — see
    ``app.captions.transcribe``.

    Raises ``EditorError`` when the project has no video clips, a clip's
    media isn't in ``resolved_media``, or ASR itself is unavailable
    (``transcribe_audio`` — missing 'asr' extra, model download failure) —
    the caller turns this into ``{"error": ...}`` rather than a raw crash.
    """
    clips = _ordered_video_clips(project)
    if not clips:
        raise EditorError("Project has no video clips to caption")

    media_dir = Path(media_dir)
    download_root = Path(models_dir) / ASR_MODELS_SUBDIR
    windows_dir = media_dir / CAPTION_WINDOWS_SUBDIR

    added = 0
    detected_language: str | None = None
    for clip in clips:
        source_path = resolved_media.get(clip.media_id)
        if source_path is None:
            raise EditorError(
                f"No resolved media for clip {clip.id!r} (media_id {clip.media_id!r})"
            )
        window_path = extract_clip_window(source_path, clip, windows_dir / f"{clip.id}.wav")
        segments, language = transcribe_audio(
            window_path,
            model_size=asr_model_size,
            device=asr_device,
            compute_type=asr_compute_type,
            download_root=download_root,
        )
        if detected_language is None:
            detected_language = language
        for segment in segments:
            if not segment.text:
                continue
            start = window_time_to_timeline_time(segment.start, clip)
            end = window_time_to_timeline_time(segment.end, clip)
            duration = max(end - start, MIN_CAPTION_DURATION_SEC)
            content = split_caption_text(segment.text, max_chars_per_line)
            project, _element_id = mutations.add_text(
                project, content=content, start_time=start, duration=duration,
                style=style, role="caption",
            )
            added += 1
    return project, added, detected_language
