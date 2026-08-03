"""Resolve a media source (local path or URL) to a local file.

yt-dlp is imported lazily so importing this module (or anything that imports
it transitively) stays cheap when the caller never actually needs a download.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from app.common.logging import get_logger
from app.config import get_settings
from app.editor.errors import EditorError
from app.net.egress import assert_url_egress_allowed

logger = get_logger(__name__)

# The format selector used when the caller does not request a specific quality —
# unchanged from the original inline literal.
_DEFAULT_FORMAT = "bv*[height<=1920]+ba/b"

# Merged downloads can land in any of these containers.
_MEDIA_GLOBS = ("*.mp4", "*.mkv", "*.webm", "*.mov")

# yt-dlp's own in-progress-download artifacts: fragment/part files and its
# resume-state sidecar. Left behind whenever a previous attempt raised
# DownloadError before a merge completed (see the except clause below) since
# that path returns before yt-dlp does its own cleanup.
_PARTIAL_GLOBS = ("*.part", "*.ytdl")

# NOTE (follow-up, not fixed here): dest_dir (media_dir/clips/<asset_id>/) is
# never pruned once a clip lands here — nothing on this repo's schedule does
# retention on the media volume. A cleanup/retention task is recommended
# follow-up work, deliberately out of scope for this extraction.


def _existing_clip(dest_dir: Path) -> Path | None:
    for pattern in _MEDIA_GLOBS:
        found = sorted(dest_dir.glob(pattern))
        if found:
            return found[0]
    return None


def _clean_stale_partials(dest_dir: Path) -> None:
    """Remove fragment/state files left by a previous failed download attempt.

    ``dest_dir`` is exclusive to one asset id (see ``resolve_media_source``),
    so anything matching here belongs to this clip alone and is safe to remove
    before a fresh attempt — otherwise a clip whose every retry hits
    ``DownloadError`` accumulates ``.part``/``.ytdl`` files forever.
    """
    for pattern in _PARTIAL_GLOBS:
        for stale in dest_dir.glob(pattern):
            stale.unlink(missing_ok=True)


def _percent_from_hook(d: dict) -> float:
    """Translate a yt-dlp progress dict into a 0..100 percentage.

    ``total_bytes`` is exact; ``total_bytes_estimate`` is yt-dlp's guess before
    the real size is known — either is good enough for a progress bar. When
    neither is present (or is 0) the percentage is unknowable, so it stays 0.0
    rather than dividing by nothing."""
    total = d.get("total_bytes") or d.get("total_bytes_estimate")
    if not total:
        return 0.0
    downloaded = d.get("downloaded_bytes") or 0
    pct = 100.0 * downloaded / total
    return max(0.0, min(100.0, pct))


def _build_progress_hook(callback: Callable[[float, str], None]):
    """Wrap ``callback`` in a yt-dlp progress hook that can never fail the
    download. A hook that raises would abort an otherwise-healthy fetch, so the
    whole body is defensive: any exception (a bad status dict, a callback that
    itself throws) is swallowed and logged, never propagated."""

    def _hook(d: dict) -> None:
        try:
            callback(_percent_from_hook(d), d.get("status"))
        except Exception:  # a progress glitch must never sink a good download
            logger.debug("progress hook error (ignored)", exc_info=True)

    return _hook


def download_clip(
    url: str,
    dest_dir: Path,
    *,
    timeout_sec: int = 300,
    format_selector: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> Path | None:
    """Return the downloaded file path, or None if the clip is unavailable.

    Re-uses an already-downloaded file in ``dest_dir`` so a retried render does
    not fetch the same clip twice.

    Enforces a WALL-CLOCK ceiling on the whole fetch (metadata + download),
    not just yt-dlp's own ``socket_timeout`` — that only bounds a single
    socket operation, so a source that keeps trickling bytes (or a DNS/
    extractor hang that never touches a socket at all) just fast enough to
    keep resetting it could otherwise run unbounded. Python cannot force-kill
    a blocked native thread, so the ceiling is enforced from the OUTSIDE:
    yt-dlp runs on its own daemon watchdog thread, and this function raises
    ``EditorError`` (rather than block) the instant ``timeout_sec`` elapses.
    The thread itself is abandoned at that point — bounded on its own by
    ``socket_timeout`` in the common case, and, being a daemon thread, never
    the reason this process fails to exit even in the uncommon case where it
    is not.

    Also caps the download's byte size via yt-dlp's own ``max_filesize``,
    tied to ``Settings.max_upload_bytes`` — the same ceiling a browser upload
    or URL-import gets, so a single hostile source cannot fill the disk
    before anything downstream even looks at what landed.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    cached = _existing_clip(dest_dir)
    if cached is not None:
        logger.info("reusing cached clip: %s", cached)
        return cached

    _clean_stale_partials(dest_dir)

    import yt_dlp  # lazy: heavy optional dependency

    options = {
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "format": format_selector if format_selector is not None else _DEFAULT_FORMAT,
        "quiet": True,
        "noplaylist": True,
        "socket_timeout": timeout_sec,
        "max_filesize": get_settings().max_upload_bytes,
    }
    if progress_callback is not None:
        options["progress_hooks"] = [_build_progress_hook(progress_callback)]

    outcome: dict = {}

    def _fetch() -> None:
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)
                outcome["path"] = Path(ydl.prepare_filename(info))
        except Exception as exc:  # relayed to, and handled on, the caller's thread
            outcome["error"] = exc

    worker = threading.Thread(target=_fetch, daemon=True)
    worker.start()
    worker.join(timeout=timeout_sec)
    if worker.is_alive():
        raise EditorError(f"Media download timed out after {timeout_sec}s fetching {url!r}")

    error = outcome.get("error")
    if error is not None:
        if isinstance(error, yt_dlp.utils.DownloadError):
            logger.warning("clip unavailable, skipping: %s (%s)", url, error)
            return None
        raise error

    path = outcome["path"]
    if path.exists():
        return path
    # After a merge the real extension may differ from prepare_filename's guess.
    return _existing_clip(dest_dir)


def resolve_media_source(
    source: str,
    dest_dir: Path,
    *,
    timeout_sec: int = 300,
    format_selector: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> Path | None:
    """``source`` is a local file path or an http(s) URL. A local path that
    exists is returned as-is (no copy, no download). A URL is fetched with
    yt-dlp into ``dest_dir``, reusing an already-downloaded file there.

    A URL source is egress-checked BEFORE yt-dlp runs: its host must resolve
    only to public addresses, or ``EgressBlockedError`` is raised. This is a
    pre-flight guard — yt-dlp owns its own sockets and redirect handling, so it
    cannot pin the connection the way the import path does — but it closes the
    obvious SSRF cases (private literals, the metadata endpoint, a host that
    resolves private) before the extractor ever connects."""
    if not source.startswith(("http://", "https://")):
        local = Path(source)
        return local if local.exists() else None
    assert_url_egress_allowed(source)
    return download_clip(
        source,
        dest_dir,
        timeout_sec=timeout_sec,
        format_selector=format_selector,
        progress_callback=progress_callback,
    )
