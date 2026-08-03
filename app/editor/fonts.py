"""Server-resolution of every text element's typeface (R-33).

``TextStyle`` carries a ``font_id`` (the user's CHOICE — ``None`` for the
deployment's configured font, otherwise a media-library asset) and a
``font_path`` (the RESULT — a local file the renderer opens). This module is the
one place the second is computed from the first.

It is pure, like everything else in ``app.editor``: the caller supplies a
``resolve`` callable that knows about sessions, owners and object storage (see
``app.mcp.tools._font_resolver``), and gets back a NEW project. Nothing here
reads a database and nothing here mutates.

APPLIED AT BOTH ENDS, deliberately:

* on INGRESS (``tools.restore_snapshot``), so a client's ``font_path`` never
  survives into storage and an unusable ``font_id`` is refused while the user is
  still looking at the editor;
* before every RENDER, so a snapshot written by an older build — or by any
  future path that forgets — still hands ffmpeg a path this server derived
  rather than one it merely found in a row.

The second one is what makes the guarantee independent of what is already in the
database, which is why there is no data migration for R-33.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from app.editor.model import EditorProject, TextElement

#: ``font_id -> local font path``. Raises ``EditorError`` for an id that does not
#: belong to the caller, rather than falling back to a default face: the quiet
#: fallback is precisely what made a wrong font path observable (R-33).
FontPathResolver = Callable[[str | None], str]

__all__ = ["FontPathResolver", "with_resolved_font_paths"]


def with_resolved_font_paths(
    project: EditorProject, resolve: FontPathResolver
) -> EditorProject:
    """Return a copy of ``project`` whose every text style carries the path
    ``resolve`` gives for its ``font_id``.

    The version is NOT bumped: this is a normalization of a value the server
    owns, not an edit the user made. Callers that are inside a mutation have
    already bumped; callers that are about to render want a throwaway copy, the
    same way ``tools.export`` builds one to override the aspect.
    """
    resolved_by_id: dict[str | None, str] = {}

    def _path_for(font_id: str | None) -> str:
        # One lookup per distinct font, not per element: a caption track is
        # hundreds of text elements sharing one face, and each resolution can
        # touch the database and the object store.
        if font_id not in resolved_by_id:
            resolved_by_id[font_id] = resolve(font_id)
        return resolved_by_id[font_id]

    tracks = tuple(
        replace(track, elements=tuple(_resolved_element(e, _path_for) for e in track.elements))
        for track in project.tracks
    )
    return replace(project, tracks=tracks)


def _resolved_element(element, path_for: FontPathResolver):
    if not isinstance(element, TextElement):
        return element
    return replace(element, style=replace(element.style, font_path=path_for(element.style.font_id)))
