"""The local-filesystem artifact store — today's behavior, byte for byte.

Keys map straight onto ``media_dir/<key>``. Every operation is the same
filesystem call the code used before this seam existed: ``put`` writes a file
(creating parent dirs), ``delete`` unlinks, ``delete_prefix`` rmtrees a
subtree. This is the DEFAULT backend and the only one a self-hosted single
node needs; ``S3ArtifactStore`` is the opt-in alternative for a shared bucket.

Containment: a key is normalized (``app.storage.keys.normalize_key`` rejects
``..``/absolute) and then resolved under the media root, and every resolved
path is re-checked to be inside the root before any write or delete. The keys
these callers pass are built from validated ids, but the guard is kept so a
future caller cannot turn a key into a path-traversal write/delete.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import BinaryIO

from app.storage.base import ArtifactStore
from app.storage.capacity import no_space_becomes_a_sentence
from app.storage.keys import normalize_key


class LocalArtifactStore(ArtifactStore):
    """Store artifacts as files under a single ``media_dir`` root."""

    def __init__(self, media_dir: Path | str) -> None:
        self._root = Path(media_dir)

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, key: str) -> Path:
        """Resolve ``key`` to an absolute path inside the media root, or raise
        ``ValueError`` if it would land outside it."""
        normalized = normalize_key(key)
        base = self._root.resolve()
        candidate = (self._root / normalized).resolve()
        if candidate != base and base not in candidate.parents:
            raise ValueError(f"key escapes media root: {key!r}")
        return candidate

    def put(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # R-20: a write that dies on a full volume says so, and the fragment it
        # left goes away — a half-written object left in place is one a later
        # ``exists``/``get`` would read as a finished artifact.
        with no_space_becomes_a_sentence("store", cleanup=path):
            path.write_bytes(data)

    def put_path(self, key: str, path, *, content_type: str | None = None) -> None:
        """Copy the file at ``path`` to ``media_dir/<key>`` in bounded chunks.

        ``shutil.copyfile`` streams (and uses the platform's own fast-copy path
        where one exists), so a multi-hundred-MB upload never becomes a
        multi-hundred-MB allocation the way ``put(path.read_bytes())`` would.
        """
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        with no_space_becomes_a_sentence("store", cleanup=target):
            shutil.copyfile(path, target)

    def get(self, key: str) -> bytes:
        path = self._resolve(key)
        try:
            return path.read_bytes()
        except (FileNotFoundError, IsADirectoryError) as exc:
            raise KeyError(key) from exc

    def open_stream(self, key: str) -> BinaryIO:
        path = self._resolve(key)
        try:
            return path.open("rb")
        except (FileNotFoundError, IsADirectoryError) as exc:
            raise KeyError(key) from exc

    def exists(self, key: str) -> bool:
        try:
            return self._resolve(key).is_file()
        except ValueError:
            return False

    def delete(self, key: str) -> None:
        # missing_ok makes a repeat pass a no-op — deletion is idempotent.
        self._resolve(key).unlink(missing_ok=True)

    def delete_prefix(self, prefix: str) -> int:
        """rmtree ``media_dir/<prefix>`` (a directory) or unlink it (a single
        file), returning the number of files removed — 0 when the prefix does
        not exist, so a second pass reports nothing removed."""
        target = self._resolve(prefix)
        if target.is_dir():
            removed = sum(1 for p in target.rglob("*") if p.is_file())
            shutil.rmtree(target, ignore_errors=True)
            # An empty directory still counts as one removed subtree, so a
            # caller counting "dirs cleaned" sees a nonzero result.
            return removed or 1
        if target.is_file():
            target.unlink(missing_ok=True)
            return 1
        return 0

    @property
    def supports_presigned_url(self) -> bool:
        return False

    def presigned_url(
        self,
        key: str,
        *,
        expires_sec: int | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> str:
        raise NotImplementedError(
            "the local artifact store has no external URL; the app serves the "
            "file itself via FileResponse"
        )
