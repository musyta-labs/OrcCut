"""Rasterize an overlay text element to a transparent PNG (PIL).

Split out of ``app.editor.render`` because it grew a second font: the text face
(DejaVu) has no emoji glyphs and the colour-emoji face (Noto Color Emoji) has no
latin, so a line like ``"Gun 🔫"`` must be composited from BOTH — PIL has no font
fallback of its own. A line is segmented into text runs and emoji clusters; text
runs are drawn with a stroke, emoji clusters are rasterized from the bitmap
strike and scaled down to sit with the text.

Both faces are loaded from ``assets/`` (bundled in this repo/image so a preview
rendered anywhere renders identically to the full export).

PIL/`Image` is imported lazily inside functions, mirroring the rest of the
editor: importing this module stays cheap for images without the render extras.
"""
from __future__ import annotations

from pathlib import Path

from app.common.color import hex_to_rgb as _hex_to_rgb
from app.common.logging import get_logger
from app.editor.errors import EditorError
from app.editor.model import TextStyle

logger = get_logger(__name__)

_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "fonts"
TEXT_FONT = _ASSETS / "DejaVuSans-Bold.ttf"
EMOJI_FONT = _ASSETS / "NotoColorEmoji.ttf"

# Noto Color Emoji is a CBDT/CBLC *bitmap* font: PIL accepts only its native
# strike size and rejects anything else with "invalid pixel size". Glyphs
# therefore always rasterize into this fixed box and are scaled afterwards.
EMOJI_STRIKE_PX = 109
EMOJI_BITMAP_W = 136
EMOJI_BITMAP_H = 128

# Look constants.
TEXT_STROKE = 6
TEXT_MAX_WIDTH_RATIO = 0.9
TEXT_LINE_SPACING = 16
_BLACK = (0, 0, 0)

# Emoji-ish codepoint ranges. Deliberately coarse: a false positive only routes
# a glyph to the emoji face (where it renders or is dropped), never corrupts text.
_EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),  # pictographs, emoticons, transport, flags, extended-A
    (0x2600, 0x27BF),    # misc symbols + dingbats
    (0x2B00, 0x2BFF),    # stars/arrows
    (0x1F900, 0x1F9FF),  # supplemental symbols
)
_ZWJ = 0x200D
_VARIATION_SELECTOR_16 = 0xFE0F
_SKIN_TONES = (0x1F3FB, 0x1F3FF)


def _is_emoji_base(char: str) -> bool:
    cp = ord(char)
    return any(low <= cp <= high for low, high in _EMOJI_RANGES)


def _is_modifier(char: str) -> bool:
    cp = ord(char)
    return cp == _VARIATION_SELECTOR_16 or _SKIN_TONES[0] <= cp <= _SKIN_TONES[1]


def segment_line(line: str) -> list[tuple[str, bool]]:
    """Split ``line`` into ``(run, is_emoji)`` pieces, in order.

    An emoji cluster is a base codepoint plus its modifiers, plus any
    ZWJ-joined continuation (so "👨‍👩‍👧" stays one glyph while "🔫💀" splits into
    two). Text runs are accumulated verbatim.
    """
    segments: list[tuple[str, bool]] = []
    text: list[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if not _is_emoji_base(char):
            text.append(char)
            index += 1
            continue

        if text:
            segments.append(("".join(text), False))
            text = []

        start = index
        index += 1
        while index < len(line):
            if _is_modifier(line[index]):
                index += 1
            elif (
                ord(line[index]) == _ZWJ
                and index + 1 < len(line)
                and _is_emoji_base(line[index + 1])
            ):
                index += 2
            else:
                break
        segments.append((line[start:index], True))

    if text:
        segments.append(("".join(text), False))
    return segments


def _load_font(font_path: str, size: int, *, chosen: bool):
    """Load the text face.

    TWO REGIMES, and the difference is who picked the path (R-33).

    ``chosen`` is False for the deployment's configured font — nobody selected
    it, every text element gets it by default, and the configured path
    legitimately may not exist on this host (an interactive preview run outside
    the container names the container's path). So it falls back: first to the
    vendored DejaVu, so a preview still looks like production, and only then to
    PIL's bitmap default.

    ``chosen`` is True for a font the OWNER picked out of their media library.
    That one is never substituted: a fallback would render the user's project in
    a typeface they did not choose, and — the reason this parameter exists at
    all — a silent fallback is what turned this function into an oracle for
    which files exist on the server. The path itself is server-resolved
    (``app.editor.fonts``), so reaching here with a face that will not open
    means the asset is genuinely gone, which the caller is told.
    """
    from PIL import ImageFont

    if chosen:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError as exc:
            raise EditorError(
                "the font chosen for this text is no longer available; "
                "pick another font in the text panel"
            ) from exc

    for candidate in (font_path, str(TEXT_FONT)):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    logger.warning(
        "no usable text font (tried %s, %s), using PIL default", font_path, TEXT_FONT
    )
    return ImageFont.load_default()


def _load_emoji_font():
    """Load the colour-emoji face at its only supported size, or None."""
    from PIL import ImageFont

    try:
        return ImageFont.truetype(str(EMOJI_FONT), EMOJI_STRIKE_PX)
    except OSError:
        logger.warning("no colour-emoji font at %s; emoji will be dropped", EMOJI_FONT)
        return None


def _emoji_height(size: int) -> int:
    return max(int(size), 1)


def _emoji_width(size: int) -> int:
    return max(int(round(EMOJI_BITMAP_W / EMOJI_BITMAP_H * _emoji_height(size))), 1)


def _emoji_image(cluster: str, size: int):
    """Rasterize one emoji cluster and scale it to ``size``-tall. None when the
    emoji face is unavailable."""
    from PIL import Image, ImageDraw

    font = _load_emoji_font()
    if font is None:
        return None
    tile = Image.new("RGBA", (EMOJI_BITMAP_W, EMOJI_BITMAP_H), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((0, 0), cluster, font=font, embedded_color=True)
    return tile.resize(
        (_emoji_width(size), _emoji_height(size)), Image.LANCZOS
    )


def _run_width(draw, run: str, is_emoji: bool, font, size: int) -> int:
    if is_emoji:
        return _emoji_width(size) * len(segment_line(run))
    box = draw.textbbox((0, 0), run, font=font, stroke_width=TEXT_STROKE)
    return box[2] - box[0]


def _line_width(draw, line: str, font, size: int) -> int:
    return sum(
        _run_width(draw, run, is_emoji, font, size)
        for run, is_emoji in segment_line(line)
    )


def _wrap(draw, line: str, font, size: int, max_width: int) -> list[str]:
    """Greedy word-wrap of ONE already-explicit line to ``max_width``."""
    words = line.split()
    if not words:
        return [""]
    wrapped: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and _line_width(draw, candidate, font, size) > max_width:
            wrapped.append(current)
            current = word
        else:
            current = candidate
    if current:
        wrapped.append(current)
    return wrapped


def _layout_lines(draw, content: str, font, size: int, max_width: int) -> list[str]:
    """Explicit ``\\n`` breaks first, then auto-wrap each resulting line.

    Splitting on whitespace alone (the previous behaviour) silently destroyed
    every explicit break, so a deliberate 2-line header was impossible.
    """
    lines: list[str] = []
    for explicit in content.split("\n"):
        lines.extend(_wrap(draw, explicit, font, size, max_width))
    return lines or [""]


def _draw_line(canvas, draw, line: str, font, style: TextStyle, top: int, left: int):
    """Composite one line's text runs and emoji clusters left-to-right."""
    fill = _hex_to_rgb(style.color)
    cursor = left
    for run, is_emoji in segment_line(line):
        if not is_emoji:
            box = draw.textbbox((0, 0), run, font=font, stroke_width=TEXT_STROKE)
            draw.text(
                (cursor - box[0], top - box[1]),
                run,
                font=font,
                fill=(*fill, 255),
                stroke_width=TEXT_STROKE,
                stroke_fill=(*_BLACK, 255),
            )
            cursor += box[2] - box[0]
            continue
        for cluster, _ in segment_line(run):
            image = _emoji_image(cluster, style.size)
            if image is not None:
                canvas.alpha_composite(image, (int(cursor), int(top)))
            cursor += _emoji_width(style.size)


def rasterize_text_png(
    content: str, style: TextStyle, *, width: int, out_path: Path
) -> Path:
    """Render ``content`` to a tight transparent RGBA PNG and return ``out_path``.

    Text is stroked in black and filled with ``style.color``; emoji keep their
    own colours. Lines break on explicit ``\\n`` and then auto-wrap to 90% of
    ``width``.
    """
    from PIL import Image, ImageDraw

    font = _load_font(style.font_path, style.size, chosen=style.font_id is not None)
    max_width = int(width * TEXT_MAX_WIDTH_RATIO)

    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines = _layout_lines(measure, content, font, style.size, max_width)

    widths = [_line_width(measure, line, font, style.size) for line in lines]
    heights = [_line_height(measure, line, font, style.size) for line in lines]
    canvas_w = max(max(widths, default=1), 1)
    canvas_h = max(
        sum(heights) + TEXT_LINE_SPACING * (len(lines) - 1), 1
    )

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    top = 0
    for line, line_w, line_h in zip(lines, widths, heights, strict=True):
        _draw_line(canvas, draw, line, font, style, top, (canvas_w - line_w) // 2)
        top += line_h + TEXT_LINE_SPACING

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def _line_height(draw, line: str, font, size: int) -> int:
    """Tallest run on the line: text (with stroke) or a scaled emoji."""
    box = draw.textbbox((0, 0), line or "X", font=font, stroke_width=TEXT_STROKE)
    text_h = box[3] - box[1]
    has_emoji = any(is_emoji for _, is_emoji in segment_line(line))
    return max(text_h, _emoji_height(size) if has_emoji else 0)
