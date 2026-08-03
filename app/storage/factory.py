"""Backend selection — mirrors ``app.config`` style: the default is local.

``get_artifact_store`` reads the effective ``Settings`` and returns the
``LocalArtifactStore`` unless ``artifact_backend == "s3"``, in which case it
builds the ``S3ArtifactStore`` from the ``s3_*`` settings. The local default is
load-bearing: a deployment that sets no storage env var behaves exactly as it
did before this seam existed.
"""
from __future__ import annotations

from pathlib import Path

from app.storage.base import ArtifactStore
from app.storage.local import LocalArtifactStore

LOCAL_BACKEND = "local"
S3_BACKEND = "s3"


def get_artifact_store(settings) -> ArtifactStore:
    """The configured artifact store. Defaults to local; only builds the S3
    client (and imports minio) when explicitly asked for the ``s3`` backend."""
    backend = (settings.artifact_backend or LOCAL_BACKEND).strip().lower()
    if backend == LOCAL_BACKEND:
        return LocalArtifactStore(Path(settings.media_dir))
    if backend == S3_BACKEND:
        return _build_s3_store(settings)
    raise ValueError(
        f"unknown ARTIFACT_BACKEND {backend!r}: expected {LOCAL_BACKEND!r} or {S3_BACKEND!r}"
    )


def _build_s3_store(settings) -> ArtifactStore:
    """Construct the S3 store, failing loudly if a required credential is
    missing — the S3 backend was explicitly requested, so an empty endpoint or
    key is an operator error, never a silent fall-back to local."""
    from app.storage.s3 import S3ArtifactStore

    missing = [
        name
        for name in ("s3_endpoint", "s3_access_key", "s3_secret_key", "s3_bucket")
        if not getattr(settings, name)
    ]
    if missing:
        raise ValueError(
            "ARTIFACT_BACKEND=s3 requires "
            + ", ".join(n.upper() for n in missing)
            + " to be set"
        )
    return S3ArtifactStore(
        endpoint=settings.s3_endpoint,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        bucket=settings.s3_bucket,
        secure=settings.s3_secure,
        region=settings.s3_region or None,
        presign_expiry_sec=settings.s3_presign_expiry_sec,
    )
