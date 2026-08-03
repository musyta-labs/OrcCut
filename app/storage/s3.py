"""The S3-compatible artifact store, over the S3 API via ``minio-py``.

Backed by MinIO in the bundled compose stack, but the client is written to the
plain S3 API: point ``endpoint`` at ``s3.amazonaws.com`` (AWS) or an R2
endpoint and the same code reaches those instead — no MinIO-specific calls.
Chosen over boto3 for a far smaller dependency footprint (minio + pycryptodome
+ urllib3 vs. boto3's botocore/jmespath/s3transfer/dateutil/six), which keeps
``constraints.txt`` lean.

``minio`` is imported lazily inside ``__init__`` — mirroring the yt-dlp / ASR /
TTS convention in this codebase — so the default local-backend deployment (and
the whole test suite when it uses the local store) never needs the package
installed, and importing ``app.storage`` costs nothing on the self-host path.

Bucket keys are the SAME POSIX keys the local backend uses, verbatim as the
object name (S3 has a flat namespace; ``/`` in a key is just a character, and
``list_objects(recursive=True)`` treats a key prefix as a folder). Content
addressing and the absence of any tenant prefix are preserved — a shared
analysis strip is one object under ``analysis/<id>_*.png`` for every tenant.
"""
from __future__ import annotations

import io
from typing import BinaryIO

from app.storage.base import ArtifactStore
from app.storage.keys import normalize_key

# Default lifetime of a presigned GET (seconds). Short — a serving redirect is
# followed immediately by the browser, so the capability need not outlive the
# click by much.
DEFAULT_PRESIGN_EXPIRY_SEC = 900


class S3ArtifactStore(ArtifactStore):
    """Store artifacts as objects in one S3-compatible bucket."""

    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = True,
        region: str | None = None,
        presign_expiry_sec: int = DEFAULT_PRESIGN_EXPIRY_SEC,
        ensure_bucket: bool = True,
    ) -> None:
        # Lazy import: keep the local/self-host path free of the dependency.
        from minio import Minio

        self._bucket = bucket
        self._presign_expiry_sec = presign_expiry_sec
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region or None,
        )
        if ensure_bucket and not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket, location=region or None)

    @property
    def bucket(self) -> str:
        return self._bucket

    def put(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        name = normalize_key(key)
        self._client.put_object(
            self._bucket,
            name,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type or "application/octet-stream",
        )

    def put_path(self, key: str, path, *, content_type: str | None = None) -> None:
        """Upload the file at ``path`` via ``fput_object``.

        minio streams the file itself (multipart for large objects), so the
        bytes never pass through this process' heap — the whole reason this
        method exists alongside ``put``.
        """
        self._client.fput_object(
            self._bucket,
            normalize_key(key),
            str(path),
            content_type=content_type or "application/octet-stream",
        )

    def get(self, key: str) -> bytes:
        response = self._open_response(key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def open_stream(self, key: str) -> BinaryIO:
        # The minio urllib3 response IS a readable binary stream; the caller
        # closes it. We wrap close() to also release the pooled connection.
        response = self._open_response(key)
        original_close = response.close

        def _close() -> None:
            original_close()
            response.release_conn()

        response.close = _close  # type: ignore[method-assign]
        return response

    def _open_response(self, key: str):
        from minio.error import S3Error

        name = normalize_key(key)
        try:
            return self._client.get_object(self._bucket, name)
        except S3Error as exc:
            if exc.code in ("NoSuchKey", "NoSuchObject"):
                raise KeyError(key) from exc
            raise

    def exists(self, key: str) -> bool:
        from minio.error import S3Error

        name = normalize_key(key)
        try:
            self._client.stat_object(self._bucket, name)
            return True
        except S3Error as exc:
            if exc.code in ("NoSuchKey", "NoSuchObject", "NoSuchBucket"):
                return False
            raise

    def delete(self, key: str) -> None:
        # remove_object is idempotent on S3: deleting an absent key succeeds.
        self._client.remove_object(self._bucket, normalize_key(key))

    def delete_prefix(self, prefix: str) -> int:
        from minio.deleteobjects import DeleteObject

        name = normalize_key(prefix)
        # Match the prefix itself and everything beneath ``prefix/`` — but not
        # a sibling like ``prefix-2`` that merely shares the string prefix.
        objects = [
            obj
            for obj in self._client.list_objects(
                self._bucket, prefix=name, recursive=True
            )
            if obj.object_name == name or obj.object_name.startswith(name + "/")
        ]
        if not objects:
            return 0
        errors = list(
            self._client.remove_objects(
                self._bucket,
                (DeleteObject(obj.object_name) for obj in objects),
            )
        )
        # remove_objects yields only FAILED deletions; surface none-silently.
        if errors:
            raise RuntimeError(f"failed to delete {len(errors)} object(s) under {prefix!r}")
        return len(objects)

    @property
    def supports_presigned_url(self) -> bool:
        return True

    def presigned_url(
        self,
        key: str,
        *,
        expires_sec: int | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> str:
        """Sign a GET for ``key``, folding any response-header overrides into
        the signature so the bucket serves them (see ``ArtifactStore``)."""
        from datetime import timedelta

        seconds = expires_sec if expires_sec is not None else self._presign_expiry_sec
        return self._client.presigned_get_object(
            self._bucket,
            normalize_key(key),
            expires=timedelta(seconds=seconds),
            response_headers=response_headers or None,
        )
