"""Piper TTS: synthesize a caption's voiceover locally.

Lazy import (mirrors ``app.captions.transcribe``'s faster-whisper convention):
importing this module never touches ``piper``/``onnxruntime``, so it stays
cheap for every process that never synthesizes speech. Voice weights live
under a caller-supplied cache dir (``{models_dir}/tts_voices`` — see
``app.config.Settings.models_dir``), populated one of two ways (Gate 3
Step 10, mirrors ``app.captions.transcribe``'s):

1. Writable, download-on-first-use (the default/self-host path, unchanged):
   ``ensure_voice`` downloads the missing ``.onnx``/``.onnx.json`` pair from
   HuggingFace the first time a voice is used.
2. Read-only, pre-baked (``docker/Dockerfile``'s ``model-prebake`` stage + a
   read-only volume mount): both files already exist, so ``ensure_voice``
   returns immediately WITHOUT ever calling ``mkdir`` or downloading —
   whether or not the mount happens to be writable. A read-only mount
   missing the requested voice raises a readable ``EditorError`` naming it
   and how to prebake it, rather than a bare permission error.

``piper-tts`` (the OHF-Voice/piper1-gpl fork) is GPL-3.0; it is a runtime
dependency of this internal service, imported lazily and never redistributed.
"""
from __future__ import annotations

import os
import re
import shutil
import threading
import urllib.request
import wave
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

from app.common.logging import get_logger
from app.editor.errors import EditorError
from app.editor.render import probe_duration_sec

logger = get_logger(__name__)

# Where voice weights are cached, relative to models_dir (see
# app.config.Settings.models_dir, the Dockerfile's model-prebake stage, and
# app.captions.transcribe's twin).
TTS_VOICES_SUBDIR = "tts_voices"

# HuggingFace layout for rhasspy/piper-voices. A voice name like
# ``ru_RU-dmitri-medium`` decomposes into lang_family ``ru`` / lang_code
# ``ru_RU`` / voice ``dmitri`` / quality ``medium``, and each voice ships two
# files: ``<name>.onnx`` (weights) and ``<name>.onnx.json`` (config).
_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
_VOICE_NAME_RE = re.compile(
    r"^(?P<lang_code>[a-z]{2,3}_[A-Z]{2})-(?P<voice>[a-z0-9]+)-"
    r"(?P<quality>x_low|low|medium|high)$"
)

# Documented, verified voices (README). Format validation — not this list —
# is the gate, so a well-formed unlisted voice still resolves if HF has it.
SUPPORTED_VOICES = (
    "ru_RU-dmitri-medium",
    "ru_RU-irina-medium",
    "ru_RU-ruslan-medium",
    "ru_RU-denis-medium",
    "en_US-lessac-medium",
)

# length_scale is the speech-rate knob: <1.0 speeds speech UP. We only ever
# speed up to fit a slot (never slow down to fill it), and never past this
# floor — beyond it speech turns unnatural.
LENGTH_SCALE_NEUTRAL = 1.0
LENGTH_SCALE_FLOOR = 0.85

# Loaded PiperVoice objects, keyed by onnx path, so N captions sharing a voice
# do not reload the model N times. Bounded LRU: an agent can be pointed at
# many distinct voices over a long-lived process, and each loaded PiperVoice
# holds a full onnxruntime session — unbounded growth here is a real memory
# leak, not a theoretical one.
VOICE_CACHE_MAX_ENTRIES = 8
_VOICE_CACHE: OrderedDict[str, object] = OrderedDict()
_VOICE_CACHE_LOCK = threading.Lock()

# Guards the check-and-download of voice weights end to end (existence check
# THROUGH the fetch, not just the final atomic move) so two concurrent callers
# for the same — or a different — voice cannot both see "missing" and both
# start a download. A single global lock (not one per voice) is deliberate:
# downloads are rare (first use only) and this keeps the guard trivially
# correct instead of needing a second bounded per-voice lock registry.
_DOWNLOAD_LOCK = threading.Lock()

# A single voice file over HF is a few MB to ~100MB; refuse to hang forever on
# a stalled connection.
DOWNLOAD_TIMEOUT_SEC = 60.0

# A single caption's synthesis should finish in well under a second of audio
# per second of wall clock on CPU; this is a generous ceiling against a wedged
# onnxruntime call, not a tuned expectation. Python cannot forcibly kill the
# worker thread on timeout — see ``_synthesize_wav``'s docstring.
SYNTHESIZE_TIMEOUT_SEC = 60.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _parse_voice_name(voice_name: str) -> re.Match[str]:
    match = _VOICE_NAME_RE.match(voice_name)
    if match is None:
        raise EditorError(
            f"invalid TTS voice name {voice_name!r} — expected "
            "'<lang>_<REGION>-<voice>-<quality>' (e.g. 'ru_RU-dmitri-medium')"
        )
    return match


def _voice_file_url(voice_name: str, suffix: str) -> str:
    """Build the HF download URL for one voice file (``.onnx`` or
    ``.onnx.json``)."""
    parts = _parse_voice_name(voice_name)
    lang_code = parts["lang_code"]
    lang_family = lang_code.split("_", 1)[0]
    voice = parts["voice"]
    quality = parts["quality"]
    return (
        f"{_HF_BASE}/{lang_family}/{lang_code}/{voice}/{quality}/"
        f"{voice_name}{suffix}"
    )


def _download_file(url: str, dest: Path) -> None:
    """Atomically download ``url`` to ``dest`` (temp file then ``os.replace``),
    so a partial download can never masquerade as a complete voice file.
    Bounded by ``DOWNLOAD_TIMEOUT_SEC`` — a stalled HF connection must fail,
    not hang the caller (and, since this runs under ``_DOWNLOAD_LOCK``, hang
    every other voice download behind it too)."""
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SEC) as response:  # noqa: S310 — fixed HF https host
            with open(tmp, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
    except Exception as exc:  # network/HTTP/timeout errors
        tmp.unlink(missing_ok=True)
        raise EditorError(
            f"failed to download TTS voice file from {url}: {exc}"
        ) from exc
    os.replace(tmp, dest)


def _writable_or_creatable(path: Path) -> bool:
    """True iff ``path`` is itself writable, or does not exist yet but its
    nearest existing ancestor is — mirrors
    ``app.captions.transcribe``'s helper of the same name."""
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:  # walked off the filesystem root
            return False
        probe = parent
    return os.access(probe, os.W_OK)


def ensure_voice(voice_name: str, voices_dir: Path) -> Path:
    """Return the local path to ``{voices_dir}/{voice_name}.onnx``, downloading
    the weights + config from HuggingFace on first use.

    The existence check AND the download it may trigger run under
    ``_DOWNLOAD_LOCK`` as one atomic unit — not just the final
    ``os.replace`` — so two concurrent callers for the same voice cannot both
    observe "missing" and both fetch it; the second simply finds the files the
    first one just wrote.

    When both files already exist (pre-baked into a read-only mount, or a
    prior download) this returns immediately — no ``mkdir``, no write —
    whether or not ``voices_dir`` happens to be writable. Only when the voice
    is genuinely missing AND ``voices_dir`` is not writable does this raise a
    readable ``EditorError`` explaining that a read-only, un-prebaked mount
    cannot be downloaded into.

    Validates the voice name format up front and raises ``EditorError`` — never
    a raw network/OS error — on a bad name or a failed download.
    """
    _parse_voice_name(voice_name)  # fail fast on a malformed name

    voices_dir = Path(voices_dir)
    onnx_path = voices_dir / f"{voice_name}.onnx"
    config_path = voices_dir / f"{voice_name}.onnx.json"

    with _DOWNLOAD_LOCK:
        if onnx_path.exists() and config_path.exists():
            return onnx_path  # pre-baked or previously downloaded — read-only hot path

        if not _writable_or_creatable(voices_dir):
            raise EditorError(
                f"TTS voice {voice_name!r} is missing from {str(voices_dir)!r} "
                "and that path is not writable, so it cannot be downloaded "
                "there. Either point MODELS_DIR (app.config.Settings."
                "models_dir) at a writable directory, or prebake this voice "
                "into the image first (see docker/Dockerfile's "
                "`model-prebake` build target)."
            )

        voices_dir.mkdir(parents=True, exist_ok=True)
        if not config_path.exists():
            _download_file(_voice_file_url(voice_name, ".onnx.json"), config_path)
        if not onnx_path.exists():
            _download_file(_voice_file_url(voice_name, ".onnx"), onnx_path)

        return onnx_path


def _load_voice(voice_path: Path):
    """Load (and LRU-cache) a PiperVoice from an onnx path. Lazy-imports piper
    so the ImportError→EditorError guard covers the caller.

    Bounded at ``VOICE_CACHE_MAX_ENTRIES``: on a miss that would overflow the
    cap, the least-recently-used voice is evicted first — each entry is a
    live onnxruntime session, not a few bytes, so this cannot be left to grow
    for the life of the process."""
    key = str(voice_path)

    with _VOICE_CACHE_LOCK:
        cached = _VOICE_CACHE.get(key)
        if cached is not None:
            _VOICE_CACHE.move_to_end(key)
            return cached

    try:
        from piper import PiperVoice
    except ImportError as exc:
        raise EditorError(
            "TTS voiceover requires the 'tts' extra — "
            "install with: pip install -e '.[tts]'"
        ) from exc

    try:
        voice = PiperVoice.load(str(voice_path))
    except Exception as exc:  # onnxruntime / missing-file errors
        raise EditorError(
            f"failed to load TTS voice {voice_path!r}: {exc}"
        ) from exc

    with _VOICE_CACHE_LOCK:
        # Another thread may have loaded (and cached) the same voice while we
        # were loading ours outside the lock — prefer its entry so a shared
        # voice never has two live copies in the cache.
        cached = _VOICE_CACHE.get(key)
        if cached is not None:
            _VOICE_CACHE.move_to_end(key)
            return cached
        if len(_VOICE_CACHE) >= VOICE_CACHE_MAX_ENTRIES:
            _VOICE_CACHE.popitem(last=False)  # evict least-recently-used
        _VOICE_CACHE[key] = voice
        return voice


def _synthesize_wav(voice, text: str, out_wav: Path, length_scale: float) -> None:
    """Render ``text`` to a 22.05kHz mono WAV at the given speech rate.

    Bounded by ``SYNTHESIZE_TIMEOUT_SEC``: the actual synthesis runs in a
    worker thread so a wedged onnxruntime call surfaces as an ``EditorError``
    instead of hanging the caller forever. Python cannot forcibly kill that
    worker thread on timeout, so on a real timeout it keeps running (writing
    into an already-closed file, which it will simply fail to do) — this
    bounds the CALLER's wait, not the work itself.
    """
    try:
        from piper import SynthesisConfig
    except ImportError as exc:
        raise EditorError(
            "TTS voiceover requires the 'tts' extra — "
            "install with: pip install -e '.[tts]'"
        ) from exc

    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # Opened WITHOUT a ``with`` block on purpose: on a timeout, the wav header
    # was never finalised (setnchannels/etc never ran), so a plain
    # ``wave.Wave_write.close()`` raises its OWN error ("# channels not
    # specified") — inside a ``with`` block's ``__exit__`` that error would
    # silently replace the EditorError we are trying to raise. The ``finally``
    # below closes explicitly and swallows exactly that expected close-time
    # error, but only once we already have a real one to report.
    wav_file = wave.open(str(out_wav), "wb")
    failed = False
    try:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            voice.synthesize_wav,
            text,
            wav_file,
            syn_config=SynthesisConfig(length_scale=length_scale),
        )
        try:
            future.result(timeout=SYNTHESIZE_TIMEOUT_SEC)
        except FutureTimeoutError as exc:
            executor.shutdown(wait=False)
            failed = True
            raise EditorError(
                f"TTS synthesis timed out after {SYNTHESIZE_TIMEOUT_SEC}s "
                f"for {out_wav!r}"
            ) from exc
        executor.shutdown(wait=True)
    except EditorError:
        raise
    except Exception as exc:  # piper/onnxruntime synthesis errors
        failed = True
        raise EditorError(
            f"TTS synthesis failed for {out_wav!r}: {exc}"
        ) from exc
    finally:
        if failed:
            try:
                wav_file.close()
            except Exception:
                pass  # the header is expected to be incomplete here
        else:
            wav_file.close()


def synthesize_caption(
    text: str,
    voice_path: Path,
    out_wav: Path,
    *,
    target_sec: float | None = None,
) -> tuple[Path, float, bool]:
    """Synthesize ``text`` to ``out_wav`` and, if it overruns ``target_sec``,
    resynthesize ONCE faster to fit.

    Returns ``(out_wav, final_duration, fitted)`` where ``fitted`` is True iff
    the second, faster pass ran. Only ever speeds speech up (clamped at
    ``LENGTH_SCALE_FLOOR``); never slows it down to pad a slot.
    """
    if not text or not text.strip():
        raise EditorError("cannot synthesize empty caption text")

    out_wav = Path(out_wav)
    voice = _load_voice(Path(voice_path))

    _synthesize_wav(voice, text, out_wav, LENGTH_SCALE_NEUTRAL)
    actual = probe_duration_sec(str(out_wav))
    if actual is None:
        raise EditorError(
            f"could not probe synthesized voiceover duration for {out_wav!r}"
        )

    if target_sec is None or actual <= target_sec:
        return out_wav, actual, False

    length_scale = _clamp(target_sec / actual, LENGTH_SCALE_FLOOR, LENGTH_SCALE_NEUTRAL)
    _synthesize_wav(voice, text, out_wav, length_scale)
    fitted = probe_duration_sec(str(out_wav))
    if fitted is None:
        raise EditorError(
            f"could not probe refitted voiceover duration for {out_wav!r}"
        )
    return out_wav, fitted, True
