"""Key helpers shared by the storage backends and their callers.

A storage key is a POSIX-relative path (``subdir/name`` or ``subdir/a/b``).
These helpers build and validate keys so callers never hand-format a string
that could smuggle a leading slash, a ``..`` hop, or a backslash into a
backend. Pure functions, no I/O.
"""
from __future__ import annotations

from pathlib import PurePosixPath


def artifact_key(*parts: str) -> str:
    """Join ``parts`` into a single POSIX key, rejecting any part that is
    empty or contains a path separator/``..`` — every part is one plain path
    component. ``artifact_key("voiceover", pid)`` -> ``"voiceover/<pid>"``.

    Raises ``ValueError`` on a malformed part rather than silently producing a
    key that escapes its subtree: the callers pass ids/uuids/hashes that are
    validated elsewhere, so a separator here is a bug, not untrusted input."""
    cleaned: list[str] = []
    for part in parts:
        if not part or part in (".", ".."):
            raise ValueError(f"invalid key component: {part!r}")
        if "/" in part or "\\" in part or part != PurePosixPath(part).name:
            raise ValueError(f"key component is not a single path segment: {part!r}")
        cleaned.append(part)
    return "/".join(cleaned)


def normalize_key(key: str) -> str:
    """Normalize an already-built key: strip a leading slash and collapse the
    POSIX path so ``a//b`` -> ``a/b``. Raises ``ValueError`` if the result
    would escape its root (a ``..`` that climbs above the first component) —
    the last line of defense for a key assembled from a filesystem-relative
    path (see the maintenance sweep, which relativizes scanned paths)."""
    posix = PurePosixPath(key.lstrip("/"))
    if ".." in posix.parts or posix.is_absolute():
        raise ValueError(f"key escapes its root: {key!r}")
    normalized = posix.as_posix()
    if normalized in ("", "."):
        raise ValueError(f"empty key: {key!r}")
    return normalized
