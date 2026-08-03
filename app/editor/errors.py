"""Editor-specific exceptions."""
from __future__ import annotations


class EditorError(Exception):
    """Raised for invalid editor operations (unknown element/track ids, bad
    arguments to a mutation, etc.). Boundary code turns this into a
    user-friendly message."""
