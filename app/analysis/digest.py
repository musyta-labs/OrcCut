"""Content digest for a media file — the key every stored analysis hangs off.

Deliberately hashes the BYTES, not the source string. A source-string digest
(what ``app.mcp.tools._analysis_cache_dir`` uses for its *download* cache, and
correctly so for that job) has two holes that matter once results are
persisted rather than recomputed every time:

- a local path overwritten in place keeps its key, so a stale analysis would be
  served for entirely new content. ``ANALYSIS_VERSION`` cannot catch this — it
  only catches changes to the contract, never changes to the media.
- a URL and its downloaded local copy hash to two different keys, so the same
  clip would be analysed twice — the exact duplication the media-keyed design
  exists to prevent.

The digest of the BYTES is still the identity — but repeat calls answer from a
``(size, mtime_ns)``-validated cache. The staleness hole above stays closed:
any real in-place overwrite moves ``mtime_ns`` (or the size), which kills the
entry and forces a re-hash. Without the cache, the browser's 3-second
``GET /analysis`` poll re-reads every asset's full bytes on every request —
a 10-clip project of 200 MB files is 2 GB of disk reads per poll per viewer.
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

from app.editor.errors import EditorError

# Read size for the streaming hash. 1 MiB keeps peak memory flat regardless of
# clip size while staying large enough that the loop overhead is irrelevant.
DIGEST_CHUNK_BYTES = 1024 * 1024

# Entries are ~200 bytes; the cap only guards against a pathological churn of
# distinct paths. Eviction is FIFO (dict insertion order) — good enough for a
# working set that is "every asset currently on some open timeline".
_CACHE_MAX_ENTRIES = 4096

# absolute path -> (size, mtime_ns, hex digest). Guarded by _CACHE_LOCK; the
# hash itself runs OUTSIDE the lock so one large file cannot serialize every
# other caller's cache hit.
_cache: dict[str, tuple[int, int, str]] = {}
_CACHE_LOCK = threading.Lock()


def content_digest(path: Path | str) -> str:
    """Full lowercase hex sha256 of the file's bytes.

    Streams in ``DIGEST_CHUNK_BYTES`` chunks so a large file never lands in
    memory. Raises ``EditorError`` when the file cannot be read — a caller that
    cannot hash the media must not fall back to some other key, because a
    second key for the same bytes is precisely the duplication this module
    exists to prevent.

    The stat is taken BEFORE the read: if the file mutates mid-hash, the
    stored stat no longer matches on the next call and the entry self-heals
    with a recompute.
    """
    file_path = Path(path)
    try:
        stat = file_path.stat()
    except OSError as exc:
        raise EditorError(f"cannot hash media file {str(path)!r}: {exc}") from exc
    key = os.path.abspath(str(file_path))
    with _CACHE_LOCK:
        entry = _cache.get(key)
    if entry is not None and entry[0] == stat.st_size and entry[1] == stat.st_mtime_ns:
        return entry[2]

    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as handle:
            while chunk := handle.read(DIGEST_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise EditorError(f"cannot hash media file {str(path)!r}: {exc}") from exc
    hex_digest = digest.hexdigest()
    with _CACHE_LOCK:
        while len(_cache) >= _CACHE_MAX_ENTRIES:
            _cache.pop(next(iter(_cache)))
        _cache[key] = (stat.st_size, stat.st_mtime_ns, hex_digest)
    return hex_digest
