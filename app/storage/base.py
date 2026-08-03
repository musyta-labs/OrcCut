"""The artifact-storage interface (Gate 3 Step 8).

Every media byte this editor produces — exported renders, preview frames,
keyframe analysis strips, per-project voiceover, browser uploads, the download
cache — has until now lived as a file under one local ``media_dir``. That is
correct and complete for the self-hosted single-node path, and it stays the
DEFAULT. This module adds the seam that lets the SAME artifacts live in an
S3-compatible object store instead, so a multi-replica deployment can share
one durable bucket rather than a node-local disk.

``ArtifactStore`` is that seam. Two implementations back it:

- ``LocalArtifactStore`` (``app.storage.local``): the current behavior, byte
  for byte. Keys map to ``media_dir/<key>`` and every operation is a plain
  filesystem call. This is the default and the only backend a self-host needs.
- ``S3ArtifactStore`` (``app.storage.s3``): an S3-compatible object store
  (MinIO by default; the same client reaches AWS S3 or Cloudflare R2 by
  changing only the endpoint) reached over the S3 API via ``minio-py``.

KEYS. A key is a POSIX-relative path with no leading slash and no tenant
prefix: ``exports/<pid>_1.mp4``, ``previews/<pid>_0.png``, ``uploads/<uuid>``,
``voiceover/<pid>``, ``analysis/<analysis_id>_overview.png``,
``clips/<asset_id>/…``, ``analysis_cache/<digest>/…``. This mirrors the
existing on-disk subpath layout exactly — the local backend maps a key
straight onto ``media_dir/<key>``, and the object backend uses it verbatim as
the object name — so the two backends address the identical artifact by the
identical key.

Analysis strips are DELIBERATELY content-addressed and shared across tenants
(Gate 1's design): their key is derived from the media's content, never from
the tenant. The store never adds a per-tenant prefix, so a strip that two
tenants legitimately share resolves to ONE object. The cross-tenant reference
count that decides when such a shared object may be deleted lives in the DB
(``app.db.repositories.accounts``), not here — the store only performs the
physical put/get/delete it is told to.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import BinaryIO


class ArtifactStore(ABC):
    """A content store for media artifacts, addressed by POSIX-relative key.

    Implementations MUST be safe to share across threads (the web app and the
    background analysis threads both reach the same instance): every method is
    a discrete operation with no per-instance mutable state beyond an immutable
    configuration (a root path or a client handle).
    """

    @abstractmethod
    def put(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        """Write ``data`` at ``key``, creating any parent structure and
        overwriting any existing object. ``content_type`` is advisory metadata
        (used by the object backend for a served ``Content-Type``; ignored by
        the local backend, which infers type at serve time)."""

    @abstractmethod
    def put_path(self, key: str, path, *, content_type: str | None = None) -> None:
        """Store the FILE at ``path`` under ``key``, streaming it — the bytes
        must never be materialized in the caller's process.

        The ingest twin of ``put``. Every upload path in this app already
        streams its body to a temp file under a byte cap before anything else
        touches it, so by the time the store is involved the artifact is a path,
        not a buffer. ``put(path.read_bytes())`` would undo that: a 500 MB
        library file would become a 500 MB allocation per concurrent upload,
        which is how a handful of simultaneous uploads exhausts the process.

        Overwrites any existing object and creates whatever parent structure the
        backend needs, exactly like ``put``. Raises ``OSError`` when ``path``
        cannot be read. ``content_type`` is advisory metadata, same as ``put``.
        """

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Return the full bytes stored at ``key``. Raises ``KeyError`` when no
        object exists at ``key`` — callers that expect a possibly-absent object
        check ``exists`` first or catch ``KeyError``."""

    @abstractmethod
    def open_stream(self, key: str) -> BinaryIO:
        """Open ``key`` for streaming reads and return a binary file-like the
        caller must close (use it as a context manager). Raises ``KeyError``
        when no object exists at ``key``. This is the backend-agnostic read
        path for serving/consuming an artifact without buffering it whole."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Whether an object is stored at ``key``."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove the single object at ``key``. A no-op when it is already
        absent — deletion is idempotent, so a repeated account/maintenance
        sweep never raises on a second pass."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int:
        """Remove every object whose key is ``prefix`` itself or lies under
        ``prefix/``. Returns the number of objects removed (0 when nothing
        matched). This is the tree-delete the local backend implements as an
        ``rmtree`` of ``media_dir/<prefix>`` and the object backend as a
        list-then-delete — used to drop a whole ``voiceover/<pid>`` or
        ``uploads/<uuid>`` directory at once."""

    @property
    @abstractmethod
    def supports_presigned_url(self) -> bool:
        """Whether ``presigned_url`` yields a client-followable URL. The object
        backend returns ``True`` (serving offloads the bytes to the store via a
        short-lived signed URL); the local backend returns ``False`` (there is
        no external URL — the app serves the file itself)."""

    @abstractmethod
    def presigned_url(
        self,
        key: str,
        *,
        expires_sec: int | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> str:
        """A short-lived URL a client may GET directly to fetch ``key``,
        bypassing the app. Only meaningful when ``supports_presigned_url`` is
        True; the local backend raises ``NotImplementedError``. The caller is
        responsible for its OWN authorization check BEFORE minting a URL — a
        presigned URL is a bearer capability, so it must never be produced for
        a caller that has not already passed the route's ownership check.

        ``response_headers`` are S3 response-header OVERRIDES
        (``response-content-type``, ``response-content-disposition``, …) folded
        into the signature, so the object store applies them to the response it
        serves. Without them a redirect would silently drop whatever headers the
        app intended: a download would arrive with the bucket's stored type and
        no ``attachment`` disposition. They are part of the signed request, so a
        client cannot alter them after the fact.
        """
