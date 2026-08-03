"""Media-dir cleanup CLI: prune stale render byproducts on a schedule.

    .venv/bin/python -m app.mcp.maintenance_cli [--dry-run] [--days N] [--delete-uploads]

Six categories, and ONLY these six — enforced by construction, since
``run_cleanup``/``run_upload_cleanup`` never construct a path outside them:

- ``{media_dir}/previews/`` — frame PNGs from ``editor_render_preview``
  (``PREVIEWS_SUBDIR`` in ``app.mcp.tools``). Age-based: anything older than
  the threshold is disposable, a fresh preview is one call away.
- ``{media_dir}/caption_windows/`` — per-clip audio windows extracted for ASR
  (``CAPTION_WINDOWS_SUBDIR`` in ``app.captions.build_captions``). Same
  age-based reasoning.
- ``{media_dir}/analysis/`` — keyframe-strip PNGs from clip event detection
  (``ANALYSIS_SUBDIR`` in ``app.analysis.analyze``). Age-based, BUT strips
  that a stored ``media_analyses`` record still references are excluded
  however old they are (``protected_analysis_names``, sourced from
  ``app.db.repositories.media_analysis.referenced_keyframe_names``).

  This exclusion is required for correctness: an analysis payload holds
  ``/ui/media/analysis/*.png`` URLs, so pruning a referenced strip leaves a
  live record pointing at a missing file — a broken image in the UI.

  It does NOT reintroduce the growth it was added to stop. The measured
  398 MB across 106 PNGs (2026-07-20) came from RE-analysing the same clips,
  each run minting a fresh ``uuid4()`` and orphaning the previous strips.
  Analysis is now persisted and content-keyed, so each distinct media has
  exactly ONE strip set and growth is bounded by the number of distinct clips
  rather than by how often anyone re-analyses. Superseded strips — orphaned
  when a contract-version bump replaces a record — drop out of the referenced
  set and age out normally, as do all the pre-persistence leftovers.
- ``{media_dir}/voiceover/{project_id}/`` — per-project TTS voiceover output.
  Orphan-based, not age-based: a directory survives as long as its
  ``project_id`` is still a known project in the editor's SQLite DB, however
  old it is; it is removed only once the project itself is gone.
- ``{media_dir}/analysis_cache/<source-digest>/`` — per-source clip downloads
  fetched by ``analyze_media`` for a bare ``source`` URL/path
  (``ANALYSIS_CACHE_SUBDIR`` in ``app.mcp.tools``; the download itself is
  ``app.media.downloader.resolve_media_source``, whose own module docstring
  flagged this directory as never pruned). Age-based like ``previews/``: the
  digest is a stable hash of the source, not a project id, so unlike
  ``voiceover/`` there is nothing to check for "still owned" — a directory
  simply stops earning its keep once it has not been re-downloaded in
  ``--days``. A re-analysis of the same source just re-fetches it.
- ``{media_dir}/uploads/<uuid4-hex>/`` — browser-uploaded (and URL-imported)
  media (``UPLOADS_SUBDIR`` in ``app.media.uploads``). Orphan-based too, but
  the directory name is a random uuid, not a project id, so it cannot be
  matched against known project ids the way ``voiceover/`` is. Instead a
  directory is orphaned only when NO project's ``MediaAsset.source`` or
  ``.local_path`` (across every project in the DB) resolves inside it — see
  ``referenced_upload_dir_names``. Deliberately NOT age-based: a file
  uploaded five minutes ago and not yet added to any timeline must survive.
  Deliberately report-only by default (``dry_run=True`` in
  ``run_upload_cleanup``, and ``main`` requires the explicit
  ``--delete-uploads`` flag to delete) — getting this wrong deletes media a
  live project still renders from. A database with zero projects refuses to
  delete anything here at all (see ``run_upload_cleanup``): that shape means
  something is wrong (DB pointed at the wrong place), not that every upload
  is safe to sweep.

``{media_dir}/exports/`` (finished renders), ``{media_dir}/clips/`` (the
yt-dlp download cache, see ``app.editor.render.CLIPS_SUBDIR``) and
``{media_dir}/editor_text/`` (rasterized text overlays, ``TEXT_SUBDIR``) are
never touched HERE — this module has no code path that reads any of the
three names, and that stays true; ``main()``/``run_cleanup``/
``run_upload_cleanup`` below are unchanged from before Gate 3 Step 9.

As of Step 9 those three directories get their OWN sweep instead —
``app.mcp.retention.run_artifact_cleanup``: orphan+TTL for ``editor_text/``,
and for ``exports/`` and ``clips/`` a plain age expiry on their own
hours-scale TTLs that ignores liveness entirely (a clip is refetchable from
the asset's URL, and an export is a delivery the project + its operation
journal outlive — see that module's docstring) — composed with everything in
THIS module via ``app.mcp.retention.run_full_cleanup``, which is what the
new periodic background worker (``app.mcp.retention.start_retention_worker``,
started from ``app.mcp.server.main``) actually calls on an interval. This
module's own ``main()`` remains a valid, narrower manual entry point for the
original five categories only — see ``app.mcp.retention`` for the wider one
and for why exports/clips/editor_text were kept in a separate module rather
than folded in here.

"NEVER TOUCHED HERE" IS A STATEMENT ABOUT THIS MODULE, NOT A PROMISE THAT
THE FILES ARE SAFE. It was both when it was written; it is only the former
now that exports expire on a 24h TTL and clips on a 48h one. Anything
reading this to decide whether an export survives must read
``app.mcp.retention`` — the guarantee ``run_cleanup``/``main()`` still make,
and that ``test_real_run_never_touches_exports_or_clips`` still enforces, is
narrower than it used to imply: it is that THIS entry point deletes nothing
in those three directories, so an operator's cron cannot be the thing that
removed a render.

The subdir names are duplicated here as plain strings rather than imported
from their home modules (``app.mcp.tools`` pulls in the whole editor/render
stack; ``voiceover`` has no home module yet — the feature isn't built) so
this maintenance CLI stays a small, decoupled leaf: it only ever touches
filesystem paths and the ``projects`` table (via raw ``ProjectRow.data``,
never ``app.editor.serialization``'s full model reconstruction), never
editor internals.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import ProjectRow
from app.db.repositories.media_analysis import referenced_keyframe_names
from app.storage import ArtifactStore, LocalArtifactStore, normalize_key

PREVIEWS_SUBDIR = "previews"
CAPTION_WINDOWS_SUBDIR = "caption_windows"
ANALYSIS_SUBDIR = "analysis"
VOICEOVER_SUBDIR = "voiceover"
ANALYSIS_CACHE_SUBDIR = "analysis_cache"
UPLOADS_SUBDIR = "uploads"

# Overridable via --days.
DEFAULT_STALE_DAYS = 7


@dataclass(frozen=True)
class CleanupReport:
    """Per-category counts from one cleanup pass."""

    previews_removed: int
    caption_windows_removed: int
    voiceover_dirs_removed: int
    analysis_removed: int
    analysis_cache_dirs_removed: int

    def describe(self, *, dry_run: bool) -> str:
        verb = "would be removed" if dry_run else "removed"
        return (
            f"previews: {self.previews_removed} {verb}, "
            f"caption_windows: {self.caption_windows_removed} {verb}, "
            f"voiceover: {self.voiceover_dirs_removed} dir {verb}, "
            f"analysis: {self.analysis_removed} {verb}, "
            f"analysis_cache: {self.analysis_cache_dirs_removed} dir {verb}"
        )


@dataclass(frozen=True)
class UploadCleanupReport:
    """One cleanup pass over ``uploads/``. Distinct from ``CleanupReport``
    because its default (``dry_run=True``) and its refusal mode (see
    ``refused_reason``) have no equivalent among the age/orphan-by-project-id
    categories — see the module docstring."""

    orphan_dirs: tuple[Path, ...]
    total_bytes: int
    deleted: bool
    refused_reason: str | None = None

    @property
    def orphan_count(self) -> int:
        return len(self.orphan_dirs)

    def describe(self) -> str:
        if self.refused_reason is not None:
            return f"uploads: refused to scan — {self.refused_reason}"
        verb = "removed" if self.deleted else "would be removed"
        return f"uploads: {self.orphan_count} dir(s) {verb} ({self.total_bytes} bytes)"


# --- pure functions: no filesystem mutation, easy to unit test -------------


def find_stale(directory: Path, cutoff: datetime) -> list[Path]:
    """Files directly under ``directory`` whose mtime is older than
    ``cutoff``. Subdirectories are ignored (neither ``previews/`` nor
    ``caption_windows/`` nests further). Returns ``[]`` when ``directory``
    does not exist — a fresh media dir is not an error."""
    if not directory.is_dir():
        return []
    stale = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc)
        if mtime < cutoff:
            stale.append(entry)
    return stale


def find_stale_dirs(directory: Path, cutoff: datetime) -> list[Path]:
    """Subdirectories directly under ``directory`` whose mtime is older than
    ``cutoff``. Mirrors ``find_stale`` but for directories rather than loose
    files — used for ``analysis_cache/``, whose entries are per-source
    download directories (named by a digest of the source, not a project id,
    so there is no "still owned" check the way ``voiceover/`` has one).
    Returns ``[]`` when ``directory`` does not exist."""
    if not directory.is_dir():
        return []
    stale = []
    for entry in directory.iterdir():
        if not entry.is_dir():
            continue
        mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc)
        if mtime < cutoff:
            stale.append(entry)
    return stale


def find_orphan_voiceover_dirs(voiceover_dir: Path, known_ids: set[str]) -> list[Path]:
    """Subdirectories of ``voiceover_dir`` whose name is not in
    ``known_ids`` (i.e. not a project the editor's DB still knows about).
    Returns ``[]`` when ``voiceover_dir`` does not exist."""
    if not voiceover_dir.is_dir():
        return []
    return [
        entry for entry in voiceover_dir.iterdir()
        if entry.is_dir() and entry.name not in known_ids
    ]


def known_project_ids(session: Session) -> set[str]:
    """Every ``project_id`` currently persisted in the editor's SQLite DB —
    the "is this voiceover dir still owned by something" check."""
    return set(session.execute(select(ProjectRow.project_id)).scalars())


def all_project_rows(session: Session) -> list[ProjectRow]:
    """Every persisted project row, ``data`` included — the source of truth
    for which ``uploads/`` directories are still referenced by a
    ``MediaAsset.source``/``.local_path``. Reads the raw JSON blob rather than
    going through ``app.editor.serialization.project_from_dict``: only two
    string fields per asset are needed here, and this module deliberately
    never imports editor internals (see module docstring)."""
    return list(session.execute(select(ProjectRow)).scalars())


def _upload_dir_name(raw_path: str | None, uploads_dir: Path) -> str | None:
    """The name of the ``uploads/<uuid>`` directory ``raw_path`` resolves
    inside, or ``None`` when it doesn't resolve there at all — a URL, a path
    elsewhere on disk, or plain garbage. Never raises: resolving a URL-shaped
    string as a ``Path`` does not crash, it just fails the containment check
    below, and that is deliberate — an asset that isn't a local upload must
    silently protect nothing rather than aborting the whole scan."""
    if not raw_path:
        return None
    try:
        resolved = Path(raw_path).resolve()
    except (OSError, ValueError):
        return None
    try:
        relative = resolved.relative_to(uploads_dir.resolve())
    except (OSError, ValueError):
        return None
    return relative.parts[0] if relative.parts else None


def referenced_upload_dir_names(rows: list[ProjectRow], uploads_dir: Path) -> set[str]:
    """Names of ``uploads/<uuid>`` directories that at least one
    ``MediaAsset`` — ``source`` OR ``local_path``, across ALL given project
    rows — resolves inside. This is the only thing standing between an
    orphan scan and deleting media a live project still renders from, so
    containment is checked with ``Path.resolve()`` + ``relative_to``, never a
    string/substring match."""
    referenced: set[str] = set()
    for row in rows:
        assets = (row.data or {}).get("assets") or []
        for asset in assets:
            for raw in (asset.get("source"), asset.get("local_path")):
                name = _upload_dir_name(raw, uploads_dir)
                if name is not None:
                    referenced.add(name)
    return referenced


def find_orphan_upload_dirs(uploads_dir: Path, referenced_names: set[str]) -> list[Path]:
    """Subdirectories of ``uploads_dir`` whose name is not in
    ``referenced_names``. Mirrors ``find_orphan_voiceover_dirs`` in shape, but
    ``referenced_names`` comes from resolved ``MediaAsset`` paths across every
    project (see ``referenced_upload_dir_names``) rather than from project
    ids directly — an uploads dir is named with a random uuid4 hex, so there
    is no id to match against. Returns ``[]`` when ``uploads_dir`` does not
    exist."""
    if not uploads_dir.is_dir():
        return []
    return [
        entry for entry in uploads_dir.iterdir()
        if entry.is_dir() and entry.name not in referenced_names
    ]


def _dir_size_bytes(path: Path) -> int:
    """Total size of every file under ``path``, recursively — best-effort
    (a file racing to disappear mid-scan is skipped, not fatal)."""
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


# --- I/O: physical removal, routed through the artifact store ---------------
#
# The candidate SCAN (find_stale / find_orphan_*) stays on the local
# filesystem — it reads mtimes and directory listings, which is the reference
# model this module owns and Step 8 deliberately does not change. Only the
# physical REMOVAL is moved to the ``ArtifactStore`` interface: a scanned path
# under ``media_dir`` is relativized into a store key, so a ``LocalArtifactStore``
# reproduces today's unlink/rmtree byte-for-byte and an injected S3 store
# receives the same delete calls.


def _key_for(path: Path, media_dir: Path) -> str:
    """The store key for a scanned path under ``media_dir`` (e.g.
    ``media_dir/previews/x.png`` -> ``previews/x.png``)."""
    return normalize_key(path.relative_to(media_dir).as_posix())


def delete_files(
    paths: list[Path], *, dry_run: bool, store: ArtifactStore, media_dir: Path
) -> int:
    """Delete each of ``paths`` (single objects) through ``store`` unless
    ``dry_run``, and return the count either way. Public (not
    underscore-prefixed) because ``app.mcp.retention`` reuses this exact
    store-routed dry-run branching for its own exports/editor_text sweep
    rather than re-implement it — see that module's docstring."""
    if not dry_run:
        for path in paths:
            store.delete(_key_for(path, media_dir))
    return len(paths)


def delete_dirs(
    paths: list[Path], *, dry_run: bool, store: ArtifactStore, media_dir: Path
) -> int:
    """``delete_files``'s twin for whole-prefix (directory) removal. Also
    reused by ``app.mcp.retention`` for its clips/ sweep."""
    if not dry_run:
        for path in paths:
            store.delete_prefix(_key_for(path, media_dir))
    return len(paths)


def run_cleanup(
    media_dir: Path,
    known_ids: set[str],
    *,
    days: int = DEFAULT_STALE_DAYS,
    dry_run: bool = False,
    protected_analysis_names: set[str] | None = None,
    store: ArtifactStore | None = None,
) -> CleanupReport:
    """Run one cleanup pass over ``media_dir`` and report what was (or would
    be) removed. Only ever looks at ``previews/``, ``caption_windows/``,
    ``analysis/``, ``voiceover/`` and ``analysis_cache/`` under ``media_dir``
    — see module docstring.

    ``protected_analysis_names`` are keyframe-strip basenames that a stored
    ``media_analyses`` record still references. They are excluded from the age
    sweep however old they are — deleting one would leave a live record
    pointing at a missing PNG, i.e. a broken image in the UI. See the module
    docstring's ``analysis/`` entry for why this does not reintroduce the
    unbounded growth the age sweep was added to stop.
    """
    protected = protected_analysis_names or set()
    if store is None:
        store = LocalArtifactStore(media_dir)
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    stale_previews = find_stale(media_dir / PREVIEWS_SUBDIR, cutoff)
    stale_caption_windows = find_stale(media_dir / CAPTION_WINDOWS_SUBDIR, cutoff)
    stale_analysis = [
        path for path in find_stale(media_dir / ANALYSIS_SUBDIR, cutoff)
        if path.name not in protected
    ]
    orphan_voiceover_dirs = find_orphan_voiceover_dirs(media_dir / VOICEOVER_SUBDIR, known_ids)
    stale_analysis_cache_dirs = find_stale_dirs(media_dir / ANALYSIS_CACHE_SUBDIR, cutoff)

    return CleanupReport(
        previews_removed=delete_files(
            stale_previews, dry_run=dry_run, store=store, media_dir=media_dir
        ),
        caption_windows_removed=delete_files(
            stale_caption_windows, dry_run=dry_run, store=store, media_dir=media_dir
        ),
        voiceover_dirs_removed=delete_dirs(
            orphan_voiceover_dirs, dry_run=dry_run, store=store, media_dir=media_dir
        ),
        analysis_removed=delete_files(
            stale_analysis, dry_run=dry_run, store=store, media_dir=media_dir
        ),
        analysis_cache_dirs_removed=delete_dirs(
            stale_analysis_cache_dirs, dry_run=dry_run, store=store, media_dir=media_dir
        ),
    )


def run_upload_cleanup(
    media_dir: Path,
    rows: list[ProjectRow],
    *,
    dry_run: bool = True,
    store: ArtifactStore | None = None,
) -> UploadCleanupReport:
    """Run one cleanup pass over ``{media_dir}/uploads/``. Orphan-based, never
    age-based (see module docstring): a directory is removed only when no
    project's ``MediaAsset.source``/``.local_path`` resolves inside it.

    ``dry_run`` defaults to ``True`` here — unlike ``run_cleanup`` — because
    getting this wrong deletes media a live project still renders from, and
    an upload not yet added to any timeline looks identical, on disk, to one
    that is truly abandoned.

    Refuses outright (deletes nothing, ``dry_run`` or not) when ``rows`` is
    empty: zero projects in the DB would make ``referenced_upload_dir_names``
    return an empty set, which would make every uploads dir look orphaned —
    exactly the "DB pointed at the wrong place" failure this guards against,
    not a green light to sweep everything.
    """
    uploads_dir = media_dir / UPLOADS_SUBDIR
    if not rows:
        return UploadCleanupReport(
            orphan_dirs=(),
            total_bytes=0,
            deleted=False,
            refused_reason=(
                "zero projects in the database — refusing to treat every uploads/ "
                "directory as orphaned; this looks like the DB is pointed at the "
                "wrong place, not a genuinely empty install"
            ),
        )
    if store is None:
        store = LocalArtifactStore(media_dir)
    referenced_names = referenced_upload_dir_names(rows, uploads_dir)
    orphan_dirs = find_orphan_upload_dirs(uploads_dir, referenced_names)
    total_bytes = sum(_dir_size_bytes(d) for d in orphan_dirs)
    delete_dirs(orphan_dirs, dry_run=dry_run, store=store, media_dir=media_dir)
    return UploadCleanupReport(
        orphan_dirs=tuple(orphan_dirs), total_bytes=total_bytes, deleted=not dry_run,
    )


# --- CLI entrypoint ----------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.mcp.maintenance_cli",
        description="Prune stale previews/caption_windows/analysis/analysis_cache, "
        "orphaned voiceover directories, and (report-only unless --delete-uploads) "
        "orphaned uploads under media_dir. This entry point never touches "
        "exports/, clips/ or editor_text/ — those have their own TTL sweep in "
        "app.mcp.retention, which the server runs on a timer.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be deleted; delete nothing (including uploads, "
        "overriding --delete-uploads if both are given).",
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_STALE_DAYS,
        help="Age threshold in days for previews/caption_windows/analysis/analysis_cache "
        f"(default {DEFAULT_STALE_DAYS}).",
    )
    parser.add_argument(
        "--delete-uploads", action="store_true",
        help="Actually delete orphaned uploads/ directories. Report-only by default "
        "— unlike previews/caption_windows/analysis/analysis_cache/voiceover, which "
        "delete unless --dry-run is passed, uploads/ requires this explicit opt-in every time because an "
        "orphan scan that gets it wrong deletes media a live project still renders "
        "from. Ignored (stays report-only) when --dry-run is also given.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    media_dir = Path(get_settings().media_dir)
    with session_scope() as session:
        known_ids = known_project_ids(session)
        rows = all_project_rows(session)
        protected = referenced_keyframe_names(session)
    report = run_cleanup(
        media_dir,
        known_ids,
        days=args.days,
        dry_run=args.dry_run,
        protected_analysis_names=protected,
    )
    print(report.describe(dry_run=args.dry_run))
    upload_dry_run = args.dry_run or not args.delete_uploads
    upload_report = run_upload_cleanup(media_dir, rows, dry_run=upload_dry_run)
    print(upload_report.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
