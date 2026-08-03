"""One finished export is not one file, and every path that removes an export
has to know that.

``app.mcp.tools.export`` writes the render itself as
``exports/{project_id}_{version}[_{preset}].mp4``. Beside it,
``app.editor.render._render_via_mlt`` keeps the ``.mlt`` XML it handed to melt
(``output_path.with_suffix(".mlt")``), and a project carrying a ``cover_time``
also gets ``{stem}_cover.png`` from ``_render_cover_if_requested``. Three files,
one deliverable.

Deleting only the ``.mp4`` therefore leaves sidecars that describe a video that
no longer exists — dead weight that no sweep would ever collect afterwards,
because every other rule in this codebase is keyed on the render's own name.
So the SET is named exactly once, here, and the three callers that remove an
export (the request-path release below, the media-library hand-off in
``app.web.media_library``, and the periodic TTL sweep in ``app.mcp.retention``)
all read it from this module rather than each re-deriving it.

TWO WAYS TO ASK, because the two kinds of caller know different things:

* ``bundle_for(export_path)`` — "I have this render, what else belongs to it?"
  Builds the two sidecar names from the render's own name and keeps the ones
  that exist. No directory scan: the caller already resolved one exact file.
* ``bundles_in(exports_dir)`` — "group EVERYTHING in here into deliverables".
  Used by the age sweep, which must also collect a sidecar whose render is
  already gone (removed by hand, or by an older build that deleted only the
  mp4). Grouping by ``bundle_key`` means such a stray forms a bundle of its own
  and ages out normally instead of becoming immortal.

Both agree by construction because both derive membership from the same
``bundle_key``.

PHYSICAL REMOVAL goes through the ``ArtifactStore`` seam (Gate 3 Step 8) via
``app.mcp.maintenance_cli.delete_files`` — the one place the store-key
derivation and the dry-run branch live — rather than calling ``Path.unlink``
here. That is what lets an ``ARTIFACT_BACKEND=s3`` deployment delete the same
bundle out of its bucket with no second code path.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from app.mcp import maintenance_cli as cli
from app.storage import ArtifactStore

__all__ = [
    "COVER_SUFFIX",
    "MLT_SUFFIX",
    "bundle_for",
    "bundle_key",
    "bundles_in",
    "delete_bundle",
]

# The two sidecars, spelled the way their writers spell them:
# ``app.editor.render`` uses ``output_path.with_suffix(".mlt")`` and
# ``app.mcp.tools._render_cover_if_requested`` uses
# ``f"{output_path.stem}_cover.png"``.
MLT_SUFFIX = ".mlt"
COVER_SUFFIX = "_cover.png"


def bundle_key(filename: str) -> str:
    """The deliverable ``filename`` belongs to — ``"pid_3.mp4"``,
    ``"pid_3.mlt"`` and ``"pid_3_cover.png"`` all answer ``"pid_3"``.

    The cover is checked FIRST and by its whole compound suffix: stripping only
    the extension would leave ``"pid_3_cover"``, a fourth key that no other file
    shares, and the cover would then survive its own render's deletion.

    Anything else this directory happens to hold (a ``.tmp.mp4`` a killed render
    left behind, say) simply keys on its own stem and forms a one-file bundle —
    it still ages out, it just never drags a live render with it.
    """
    if filename.endswith(COVER_SUFFIX):
        return filename[: -len(COVER_SUFFIX)]
    return Path(filename).stem


def bundle_for(export_path: Path) -> tuple[Path, ...]:
    """Every file of the deliverable ``export_path`` is the render of — itself
    plus whichever sidecars exist — with the render FIRST.

    Order is load-bearing on the delete path: if removal dies partway, having
    dropped the render before its sidecars leaves debris that
    ``bundles_in``'s stray-collection can still age out, whereas the reverse
    would leave a playable-looking export missing the XML it was built from.

    A non-existent sidecar is simply absent from the result rather than an
    error — most exports have no cover at all (only a project with a
    ``cover_time`` gets one), and the ffmpeg render engine writes no ``.mlt``.
    """
    candidates = (
        export_path,
        export_path.with_suffix(MLT_SUFFIX),
        export_path.parent / f"{export_path.stem}{COVER_SUFFIX}",
    )
    return tuple(path for path in candidates if path.is_file())


def bundles_in(exports_dir: Path) -> list[tuple[Path, ...]]:
    """Every file directly under ``exports_dir``, grouped into deliverables by
    ``bundle_key`` and sorted by key so a sweep's order is deterministic.

    Returns ``[]`` when the directory does not exist — a deployment that has
    never exported anything is not an error. Subdirectories are ignored;
    ``exports/`` is flat by construction (see ``app.mcp.tools.export``).
    """
    if not exports_dir.is_dir():
        return []
    grouped: dict[str, list[Path]] = {}
    for entry in exports_dir.iterdir():
        if not entry.is_file():
            continue
        grouped.setdefault(bundle_key(entry.name), []).append(entry)
    return [tuple(sorted(grouped[key])) for key in sorted(grouped)]


def delete_bundle(
    paths: Iterable[Path],
    *,
    media_dir: Path,
    store: ArtifactStore,
    dry_run: bool = False,
) -> int:
    """Remove one bundle through ``store`` and return how many files were (or,
    under ``dry_run``, would have been) removed.

    A thin, deliberate wrapper over ``maintenance_cli.delete_files``: that
    function already owns the path→store-key derivation and the dry-run branch,
    and having a second copy of either is exactly how a local deployment and an
    S3 one end up deleting different things. ``ArtifactStore.delete`` is
    documented a no-op on an absent key, so this is idempotent — a bundle two
    callers race to release is deleted once and reported twice, never an error.
    """
    return cli.delete_files(
        list(paths), dry_run=dry_run, store=store, media_dir=media_dir
    )
