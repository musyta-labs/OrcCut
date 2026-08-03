"""faster-whisper wrapper for auto-captions.

Lazy import (mirrors ``app/media/downloader.py``'s yt-dlp convention):
importing this module never touches faster-whisper/ctranslate2, so it stays
cheap for every process that never calls ``transcribe_audio``. The model
weights live under a caller-supplied ``download_root`` — see
``app.config.Settings.models_dir``/``asr_model_size``/``asr_device``/
``asr_compute_type`` and ``app.mcp.tools.auto_captions``.

Two ways ``download_root`` gets populated (Gate 3 Step 10):

1. Writable, download-on-first-use (the default/self-host path, unchanged):
   ``download_root`` does not exist yet or is empty, and is writable — we
   ``mkdir`` it and ``WhisperModel`` downloads the weights into it the first
   time this runs.
2. Read-only, pre-baked (``docker/Dockerfile``'s ``model-prebake`` stage +
   a read-only volume mount): ``download_root`` already contains the
   weights. We detect this — a populated directory — and skip straight to
   loading, WITHOUT ever calling ``mkdir`` or attempting a write, so a
   read-only mount never fails there. If the directory is both unwritable
   AND empty (no weights were baked in), ``transcribe_audio`` raises a
   readable ``EditorError`` naming the missing model and how to prebake it,
   rather than a bare permission error surfacing from deep inside
   faster-whisper/huggingface_hub.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path

from app.common.logging import get_logger
from app.editor.errors import EditorError

logger = get_logger(__name__)

# Where the model weights are cached, relative to models_dir (see
# app.config.Settings.models_dir and the Dockerfile's model-prebake stage).
ASR_MODELS_SUBDIR = "asr_models"

# ``WhisperModel(...)`` loads ~460MB of weights (and ~1GiB RSS) — reloading it
# on every ``transcribe_audio`` call is what Gate 2 Step 2 exists to stop.
# Keyed by the full config tuple (not a bare singleton) so a caller that
# genuinely asks for a different model_size/device/compute_type gets ITS OWN
# instance rather than silently reusing an unrelated one; in production the
# config is fixed by ``Settings``, so this collapses to one entry in practice.
_MODEL_CACHE: dict[tuple[str, str, str, str], object] = {}
_MODEL_CACHE_LOCK = threading.Lock()

# How long a single ``transcribe`` call may run before we give up waiting and
# surface a timeout rather than hang the caller forever on a stuck decode.
# Generous: whisper on CPU can take minutes for a long clip. Python cannot
# forcibly kill the worker thread on timeout, so a truly stuck backend keeps
# running in the background — this bounds the CALLER's wait, not the work.
ASR_TRANSCRIBE_TIMEOUT_SEC = 10 * 60


def _weights_dir_populated(path: Path) -> bool:
    """True iff ``path`` already exists and holds at least one entry — the
    signal that model weights were pre-baked (or previously downloaded)
    there. When this is true, ``transcribe_audio`` must never mkdir or write
    into ``path`` again, read-only mount or not."""
    return path.is_dir() and any(path.iterdir())


def _writable_or_creatable(path: Path) -> bool:
    """True iff ``path`` is itself writable, or does not exist yet but its
    nearest existing ancestor is — i.e. ``path.mkdir(parents=True)`` would
    actually succeed rather than raise ``PermissionError``."""
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:  # walked off the filesystem root
            return False
        probe = parent
    return os.access(probe, os.W_OK)


def _ensure_download_root(download_root: Path, model_size: str) -> None:
    """Prepare ``download_root`` for ``WhisperModel`` without ever touching a
    directory that already holds pre-baked weights.

    - Already populated (pre-baked or a prior download): do nothing — no
      mkdir, no write, whether or not the mount is writable. This is the
      read-only hot path.
    - Empty but writable: ``mkdir`` it, exactly as before — download-on-
      first-use keeps working unchanged.
    - Empty AND not writable: there is no way to get the weights, so raise a
      readable ``EditorError`` naming the missing model and how to fix it,
      instead of letting a bare ``PermissionError``/``OSError`` surface from
      deep inside faster-whisper/huggingface_hub.
    """
    if _weights_dir_populated(download_root):
        return
    if _writable_or_creatable(download_root):
        download_root.mkdir(parents=True, exist_ok=True)
        return
    raise EditorError(
        f"ASR model {model_size!r} is missing from {str(download_root)!r} "
        "and that path is not writable, so it cannot be downloaded there. "
        "Either point MODELS_DIR (app.config.Settings.models_dir) at a "
        "writable directory, or prebake the weights into the image first "
        "(see docker/Dockerfile's `model-prebake` build target)."
    )


def _get_model(
    model_size: str,
    device: str,
    compute_type: str,
    download_root: Path,
    whisper_model_cls,
):
    """Return the cached ``WhisperModel`` for this exact config, building it
    under a double-checked lock so N concurrent first-callers construct
    (and download the weights for) exactly ONE instance, not N."""
    key = (model_size, device, compute_type, str(download_root))

    model = _MODEL_CACHE.get(key)
    if model is not None:
        return model

    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)  # re-check: someone may have built it
        if model is None:
            model = whisper_model_cls(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=str(download_root),
            )
            _MODEL_CACHE[key] = model
        return model


@dataclass(frozen=True)
class TranscribedSegment:
    """One ASR segment. ``start``/``end`` are seconds relative to the START
    of whatever audio file was handed to ``transcribe_audio`` — the caller
    (``app.captions.mapping``) maps that into absolute timeline time."""

    start: float
    end: float
    text: str


def _transcribe_and_collect(
    model, path: str
) -> tuple[list[TranscribedSegment], str | None]:
    """Run the actual (blocking) decode and eagerly consume the segment
    generator. Split out of ``transcribe_audio`` so it can be run inside a
    worker thread with a timeout — ``model.transcribe`` itself returns a lazy
    generator, so the real work only happens once we iterate it, and the
    timeout must cover that iteration too."""
    # word_timestamps forces cross-attention alignment per word. Without it
    # whisper pins an utterance to the start of its 30s decode window on
    # music-heavy audio (measured 2026-07-19: speech at 13.2-14.9s emitted
    # as 0.0-2.0), so the caption lands where nobody speaks.
    segments, info = model.transcribe(path, word_timestamps=True)
    parsed = [
        TranscribedSegment(start=seg.start, end=seg.end, text=seg.text.strip())
        for seg in segments
    ]
    return parsed, getattr(info, "language", None)


def transcribe_audio(
    path: Path,
    *,
    model_size: str,
    device: str,
    compute_type: str,
    download_root: Path,
) -> tuple[list[TranscribedSegment], str | None]:
    """Transcribe ``path`` (any ffmpeg-readable media file) into ordered
    segments, returning ``(segments, detected_language)`` — the ISO code
    faster-whisper auto-detected (e.g. ``"en"``), or ``None`` when the backend
    did not report one. The language is what an agent needs to translate the
    captions afterwards, so it is surfaced rather than discarded.

    Raises ``EditorError`` — never a raw ``ImportError``/``OSError``/timeout —
    when faster-whisper is not installed, the model cannot be loaded or
    downloaded, or the decode itself does not finish within
    ``ASR_TRANSCRIBE_TIMEOUT_SEC`` — so a caller (``editor_auto_captions``)
    degrades gracefully with a clear message instead of an unhandled crash or
    hang.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise EditorError(
            "auto-captions requires the 'asr' extra — "
            "install with: pip install -e '.[asr]'"
        ) from exc

    download_root = Path(download_root)
    _ensure_download_root(download_root, model_size)

    try:
        # Building the model (and, on first use, downloading its weights) is
        # itself the thing the first-load lock in ``_get_model`` serialises —
        # a second concurrent caller for the SAME config blocks there instead
        # of racing a second download / holding a second copy of the weights.
        model = _get_model(model_size, device, compute_type, download_root, WhisperModel)
        # NOT a ``with ThreadPoolExecutor(...)`` block: that calls
        # ``shutdown(wait=True)`` on exit, which blocks until the worker
        # thread finishes — exactly the unbounded wait we are trying to
        # avoid. On timeout we shut down WITHOUT waiting and let the
        # already-running decode finish on its own in the background.
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_transcribe_and_collect, model, str(path))
        try:
            parsed, language = future.result(timeout=ASR_TRANSCRIBE_TIMEOUT_SEC)
        except FutureTimeoutError as exc:
            executor.shutdown(wait=False)
            raise EditorError(
                f"auto-captions transcription timed out after "
                f"{ASR_TRANSCRIBE_TIMEOUT_SEC}s for {path!r}"
            ) from exc
        executor.shutdown(wait=True)
        return parsed, language
    except EditorError:
        raise
    except Exception as exc:  # faster-whisper/ctranslate2/huggingface_hub errors
        raise EditorError(
            f"auto-captions transcription failed for {path!r}: {exc}"
        ) from exc
