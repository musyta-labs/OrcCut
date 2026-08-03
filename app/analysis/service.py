"""Cache-aware clip analysis: the one place that decides compute vs. reuse.

``app.analysis.analyze.analyze_media_file`` is the pure engine — it always
computes. This module wraps it with the media-keyed store so a result is
computed ONCE per distinct media per contract version, and every later caller
reads it back instead of spending ~2s of CPU (plus, for a URL, a download)
producing the same numbers again.

Two properties this module exists to guarantee:

- **Exactly one computation per media.** A per-content-hash lock serialises
  check -> compute -> write, so a background analysis triggered by ``add_clip``
  and a foreground ``editor_analyze_media`` for the same file cannot both
  decode it, nor race each other into a double write. The second arrival
  blocks, then re-checks the store and takes the cache hit.
- **No DB connection held across the decode.** Sessions are opened for the
  lookup, closed, and re-opened for the write. A minutes-long yt-dlp +ffmpeg
  run holding a pooled connection is what lets a handful of concurrent
  analyses exhaust the pool and 500 every unrelated request.
"""
from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.analysis.analyze import ANALYSIS_VERSION, analyze_media_file
from app.analysis.digest import content_digest
from app.analysis.peaks import MAX_EVENTS
from app.common.logging import get_logger
from app.db.repositories import media_analysis as store

logger = get_logger(__name__)

# One lock per content hash, created on demand. A long-lived process that
# sees enough DISTINCT media would otherwise grow this dict forever — each
# entry is only a few dozen bytes, but "forever" is still unbounded. Bounded
# LRU: on a miss that would overflow the cap, the least-recently-used FREE
# lock is evicted (never one currently held — see ``_evict_locked``). This is
# safe because all reads/writes of ``_locks`` happen under ``_locks_guard``,
# so "not currently held" and "removed from the dict" are one atomic step; no
# caller can ever observe two different Lock objects for the same content
# hash while either is actually locked.
_LOCKS_MAX_ENTRIES = 4096
_locks: OrderedDict[str, threading.Lock] = OrderedDict()
_locks_guard = threading.Lock()


@dataclass(frozen=True)
class AnalysisOutcome:
    """One resolved analysis plus HOW it was resolved.

    ``from_cache`` is not part of the wire contract — ``analyze_media`` returns
    ``data`` alone, unchanged, because the annotation client's
    the annotation client reads that documented payload. It exists so
    callers and logs can tell a reuse from a recompute.
    """

    data: dict
    content_hash: str
    from_cache: bool


def _evict_free_locks_locked() -> None:
    """Remove least-recently-used, currently-UNLOCKED entries until
    ``_locks`` is back under the cap. Must be called while holding
    ``_locks_guard``. A lock actually held by another in-flight
    ``analyze_with_cache`` call is never evicted — if every entry happens to
    be held (many distinct media analysing at once), the dict may briefly
    exceed the cap by a few entries rather than break mutual exclusion."""
    for content_hash in list(_locks.keys()):
        if len(_locks) < _LOCKS_MAX_ENTRIES:
            return
        if not _locks[content_hash].locked():
            del _locks[content_hash]


def _lock_for(content_hash: str) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(content_hash)
        if lock is not None:
            _locks.move_to_end(content_hash)
            return lock
        if len(_locks) >= _LOCKS_MAX_ENTRIES:
            _evict_free_locks_locked()
        lock = threading.Lock()
        _locks[content_hash] = lock
        return lock


def analyze_with_cache(
    path: Path | str,
    *,
    media_dir: Path,
    session_scope: Callable,
    owner_id: int,
    max_events: int = MAX_EVENTS,
) -> AnalysisOutcome:
    """Return the analysis for the media at ``path``, computing it only if no
    current-version result is stored.

    ``path`` must already be a LOCAL file: a URL is resolved to disk by the
    caller first (``app.mcp.tools._resolve_analysis_source``), because the key
    is a hash of the bytes and there are no bytes to hash until the download
    has landed. Resolving first also means a URL and its downloaded copy
    collapse onto one stored record instead of two.

    ``session_scope`` is injected rather than imported so the web app, the MCP
    path and the tests each supply their own — no module-level DB coupling.

    ``owner_id`` is the tenant asking, and it is a REQUIRED argument passed
    explicitly rather than read from ``get_principal()`` inside: the background
    worker calls this on a pool thread, where the principal contextvar does not
    propagate. It is recorded against the media on every call, hit or miss —
    see below.
    """
    path = Path(path)
    digest = content_digest(path)

    with _lock_for(digest):
        cached = _load_cached(session_scope, digest)
        if cached is not None:
            logger.info(
                "analysis cache HIT for %s (sha256=%s…, version=%s)",
                path.name, digest[:12], ANALYSIS_VERSION,
            )
            # A HIT is precisely when a tenant arrives at bytes ANOTHER tenant
            # already analysed — the shared-cache case Step 10 exists to make
            # safe. Skipping the grant here would hand this caller a payload
            # naming strips it is then refused, which is the cross-tenant leak
            # inverted into a broken page.
            _grant(session_scope, owner_id=owner_id, content_hash=digest)
            return AnalysisOutcome(data=cached, content_hash=digest, from_cache=True)

        logger.info(
            "analysis cache MISS for %s (sha256=%s…) — computing",
            path.name, digest[:12],
        )
        analysis_id = uuid.uuid4().hex
        result = analyze_media_file(
            path,
            media_dir=Path(media_dir),
            analysis_id=analysis_id,
            max_events=max_events,
        )
        data = result.to_dict()
        with session_scope() as session:
            store.save(
                session, digest, version=ANALYSIS_VERSION, data=data,
                analysis_id=analysis_id,
            )
            store.grant_access(session, user_id=owner_id, content_hash=digest)
        logger.info(
            "analysis stored for %s (sha256=%s…, events=%d)",
            path.name, digest[:12], len(data.get("events", [])),
        )
        return AnalysisOutcome(data=data, content_hash=digest, from_cache=False)


def _grant(session_scope: Callable, *, owner_id: int, content_hash: str) -> None:
    """Record this tenant's access in its own short session — the cache-hit
    path holds none, and must not open one around the decode that a miss would
    follow with."""
    with session_scope() as session:
        store.grant_access(session, user_id=owner_id, content_hash=content_hash)


def _load_cached(session_scope: Callable, digest: str) -> dict | None:
    """The stored payload for ``digest`` at the CURRENT contract version, or
    ``None``. Opened and closed on its own so nothing is held during the
    decode that may follow."""
    with session_scope() as session:
        row = store.get_current(session, digest, version=ANALYSIS_VERSION)
        return dict(row.data) if row is not None else None


def stored_analysis(session_scope: Callable, path: Path | str) -> tuple[str, dict | None]:
    """``(content_hash, payload-or-None)`` for a local file, WITHOUT ever
    computing. The read path behind the HTTP status endpoint: it must be able
    to say "nothing stored yet" cheaply rather than triggering a 2s decode
    inside a request."""
    digest = content_digest(path)
    return digest, _load_cached(session_scope, digest)
