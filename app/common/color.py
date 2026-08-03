"""Hex-colour parsing shared by the text rasterizer.

Split out because a "#RRGGBB" -> (r, g, b) conversion assuming exactly 6 hex
digits raises ``ValueError`` on a valid CSS short form like ``color="#FFF"``
deep inside PIL rasterization and fails the whole render. One helper means
that fix only has to happen once.
"""
from __future__ import annotations

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Parse a CSS-style hex colour to an ``(r, g, b)`` triple.

    Accepts both the 3-digit short form (``#FFF``, expanded digit-by-digit to
    ``#FFFFFF``) and the full 6-digit form, with or without a leading ``#``.
    Anything else — wrong length, non-hex characters — raises ``ValueError``
    naming the offending value, so a malformed colour fails with a clear
    message at the point it is parsed instead of crashing inside PIL.
    """
    stripped = value.lstrip("#")
    if len(stripped) == 3:
        stripped = "".join(digit * 2 for digit in stripped)
    if len(stripped) != 6 or any(char not in _HEX_DIGITS for char in stripped):
        raise ValueError(
            f"invalid hex colour {value!r}: expected '#RGB' or '#RRGGBB'"
        )
    return (
        int(stripped[0:2], 16),
        int(stripped[2:4], 16),
        int(stripped[4:6], 16),
    )
