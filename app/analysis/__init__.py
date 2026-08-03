"""Deterministic clip event detection ("moments of meaning").

Motion-diff over downscaled grayscale frames + audio RMS, z-scored against a
rolling baseline, non-maximum suppressed into a handful of timestamped events
plus keyframe strips. ~2s of CPU per clip, zero tokens, no LLM anywhere on
this path.

``analyze_media_file`` is the only entry point callers need; ``AnalysisResult.
to_dict()`` is the wire contract, ``"version": 2``.

Version 2 added the solo-channel rule (an event that was clearly SEEN or HEARD,
but not both on the same frame — v1's geometric mean crushed these below the
bar and missed them) and, with it, the ``score`` field on every event.
**Consumers must rank on ``score``, never on ``combined``.** ``score`` is the
normalised strength that actually selected the event and is the only field
comparable across events; ``combined`` is the honest raw geometric mean of the
two channels and is legitimately ~0 on a solo-detected event. Sorting on
``combined`` therefore makes exactly the events v2 was built to find
detectable but never selectable.
"""
from __future__ import annotations

from app.analysis.analyze import (
    ANALYSIS_SUBDIR,
    ANALYSIS_URL_PREFIX,
    ANALYSIS_VERSION,
    AnalysisResult,
    analyze_media_file,
)

__all__ = [
    "ANALYSIS_SUBDIR",
    "ANALYSIS_URL_PREFIX",
    "ANALYSIS_VERSION",
    "AnalysisResult",
    "analyze_media_file",
]
