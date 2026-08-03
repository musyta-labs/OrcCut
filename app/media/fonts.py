"""Fonts as media-library assets (R-33).

A text overlay's ``font_path`` used to be a plain string that rode a client
snapshot straight into ``app.editor.text_raster``'s ``ImageFont.truetype`` call.
A path that was not a font fell back to the bundled face and a real font did
not, which made the pair "does this file exist, and is it a font?" answerable by
any tenant. The answer is not to freeze the field to the configured font but to
make it RESOLVABLE from exactly two places — the deployment's own font and an
asset on the CALLER's shelf — with everything else refused. This module owns the
asset half of that: what counts as a font file, whether a given upload really is
one, and where the renderer finds its bytes.

THREE THINGS, and no more:

* ``FONT_EXTENSIONS`` — the library's font allowlist, kept SEPARATE from
  ``app.media.uploads.ALLOWED_EXTENSIONS`` so only the library grows a font
  shelf. A project upload is media that gets decoded, and a font landing there
  would only fail the media guard one step later.
* ``reject_unloadable_font`` — the content check. An extension proves what a
  file is CALLED; this proves it PARSES, exactly as
  ``app.media.guards.reject_undecodable_media`` does for media. Without it a
  ``.ttf``-named blob would be accepted at upload and blow up at render time, on
  a code path a tenant chose.
* ``ensure_local_font`` — a font is stored in the ``ArtifactStore`` like every
  other library object, but ``melt``/PIL need a FILE. The bytes are materialized
  once under ``media_dir/fonts/<owner>/<file_id>/<name>`` and reused: keyed by
  the immutable file id (a rename never moves the object, so the key stays
  valid), owner-scoped so one tenant's cache directory can never be addressed by
  another's id, and re-created on demand if the directory is swept.

The local copy is a CACHE, not a second source of truth: the row and the object
remain the only durable state, and deleting the library file drops the cache too
(``discard_local_font``).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from app.editor.errors import EditorError
from app.storage import ArtifactStore

# Sibling of ``app.editor.render.CLIPS_SUBDIR`` and ``uploads``: where the
# renderer's local copies of library fonts live under the media root.
FONTS_SUBDIR = "fonts"

# What may be uploaded as a font. TrueType and OpenType only — the two formats
# FreeType (and therefore PIL, and therefore every render path here) reads
# without a shaping engine. Web-only wrappers (.woff/.woff2) are deliberately
# absent: they would have to be unwrapped before use, which is a decode step
# this service has no reason to grow.
FONT_EXTENSIONS: frozenset[str] = frozenset({".ttf", ".otf"})

# A hard ceiling on a font, far below the upload cap. A real face is well under
# a megabyte; CJK and emoji faces reach a few. Anything larger is either not a
# font or is a font nobody is typesetting a caption with — and it would be
# copied to local disk on every render node that touches the project.
MAX_FONT_BYTES = 20 * 1024 ** 2

__all__ = [
    "FONTS_SUBDIR",
    "FONT_EXTENSIONS",
    "MAX_FONT_BYTES",
    "discard_local_font",
    "ensure_local_font",
    "font_cache_path",
    "is_font_filename",
    "reject_unloadable_font",
]


def is_font_filename(name: str) -> bool:
    """Whether ``name``'s extension is one this service accepts as a font.

    Extension only, and deliberately so — it is the CHEAP half of the check,
    used to route an upload to the right guard and to answer "is this library
    row usable as a font?" without reading bytes. The expensive half is
    ``reject_unloadable_font``, which every ingested font also passes.
    """
    return Path(name).suffix.lower() in FONT_EXTENSIONS


def reject_unloadable_font(path: Path) -> None:
    """Raise ``EditorError`` unless the file at ``path`` really parses as a font.

    The font twin of ``app.media.guards.reject_undecodable_media``, and it exists
    for the same reason that one does: the extension allowlist only proves what a
    file is CALLED. A ``.ttf``-named blob that reached the shelf would be
    offered in the font picker and then fail — or silently change the typeface —
    inside a render the user is paying for.

    PIL is the checker rather than a hand-rolled sfnt parser because PIL (via
    FreeType) is the SAME code that will open the file at render time: a file
    this accepts is by construction a file the rasterizer can use, which no
    header sniff can promise. It is imported lazily, matching the rest of the
    editor's PIL usage.

    Size is checked BEFORE parsing: handing FreeType an arbitrarily large
    attacker-supplied buffer is exactly the sort of thing a cheap bound should
    stand in front of.
    """
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise EditorError("could not read the uploaded font") from exc
    if size_bytes > MAX_FONT_BYTES:
        raise EditorError(
            f"font exceeds the {MAX_FONT_BYTES // 1024 ** 2} MB font limit"
        )

    from PIL import ImageFont

    try:
        # The size argument is irrelevant to whether the face parses; any legal
        # value does, and this one costs nothing to rasterize because nothing
        # here draws with it.
        ImageFont.truetype(str(path), 16)
    except Exception as exc:  # noqa: BLE001 — PIL raises OSError, but a corrupt
        # face can surface as ValueError/struct.error from deeper in FreeType,
        # and every one of them means the same thing to the caller.
        raise EditorError("this file is not a usable font (.ttf/.otf)") from exc


def font_cache_path(media_dir: Path, owner_id: int, file_id: str, filename: str) -> Path:
    """Where THIS owner's font with THIS id is kept for the renderer.

    Owner-scoped on purpose, mirroring ``media_library.file_prefix``: the
    directory an id maps to is inside its owner's own subtree, so a path built
    from one tenant's id can never name another tenant's cached bytes even if
    the two ids somehow collided.
    """
    return Path(media_dir) / FONTS_SUBDIR / str(owner_id) / file_id / filename


def ensure_local_font(
    store: ArtifactStore,
    *,
    media_dir: Path,
    owner_id: int,
    file_id: str,
    filename: str,
    storage_key: str,
) -> Path:
    """Return a local path holding this library font's bytes, materializing it
    from the artifact store on first use.

    Idempotent and cheap on the hot path: an already-materialized font costs one
    ``stat``. The copy is needed because the store may be an S3 bucket while PIL
    and ``melt`` can only open a file — and it is safe to keep because the bytes
    behind a library file id never change (a rename moves no object).

    Written to a temp name and then ``replace``d, exactly as the render output
    is: two concurrent renders of the same project must never have one of them
    reading a half-copied face.

    Raises ``EditorError`` when the object is gone — a deleted font must fail
    LOUDLY here rather than leave a path that the rasterizer would quietly
    substitute the default face for.
    """
    target = font_cache_path(media_dir, owner_id, file_id, filename)
    if target.is_file():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.partial")
    try:
        # ``try/finally`` rather than ``with``: the object backend hands back a
        # urllib3 response, which ``app.web.media_library._iter_object`` closes
        # the same way for the same reason.
        source = store.open_stream(storage_key)
        try:
            with staging.open("wb") as handle:
                shutil.copyfileobj(source, handle)
        finally:
            source.close()
    except KeyError as exc:  # the object is not in the store
        staging.unlink(missing_ok=True)
        raise EditorError(f"Unknown font id: {file_id!r}") from exc
    except OSError as exc:
        staging.unlink(missing_ok=True)
        raise EditorError(f"could not read font {file_id!r}") from exc
    staging.replace(target)
    return target


def discard_local_font(media_dir: Path, owner_id: int, file_id: str) -> None:
    """Drop one font's materialized copy, best-effort.

    Called when the library file itself is deleted. Best-effort because the row
    and the object are already gone by then: a cache directory that survives is
    unreferenced bytes, while raising here would tell the caller a delete failed
    that in fact succeeded.
    """
    shutil.rmtree(Path(media_dir) / FONTS_SUBDIR / str(owner_id) / file_id,
                  ignore_errors=True)
