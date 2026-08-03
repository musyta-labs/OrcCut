"""Local text-to-speech synthesis (Piper).

The heavy runtime dependency (``piper-tts``) and the voice model weights are
both optional: the package is installed via the ``tts`` extra and imported
lazily, while voices download on first use to a caller-supplied cache dir —
mirroring ``app.captions.transcribe``'s treatment of faster-whisper.
"""
from __future__ import annotations

from app.tts.synthesize import (
    TTS_VOICES_SUBDIR,
    ensure_voice,
    synthesize_caption,
)

__all__ = ["TTS_VOICES_SUBDIR", "ensure_voice", "synthesize_caption"]
