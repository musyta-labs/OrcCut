"""Artifact storage: a backend-agnostic interface for every media byte the
editor produces, with a local-filesystem default and an S3-compatible option
(Gate 3 Step 8). See ``app.storage.base`` for the interface contract."""
from __future__ import annotations

from app.storage.base import ArtifactStore
from app.storage.factory import (
    LOCAL_BACKEND,
    S3_BACKEND,
    get_artifact_store,
)
from app.storage.keys import artifact_key, normalize_key
from app.storage.local import LocalArtifactStore

__all__ = [
    "ArtifactStore",
    "LocalArtifactStore",
    "get_artifact_store",
    "artifact_key",
    "normalize_key",
    "LOCAL_BACKEND",
    "S3_BACKEND",
]
