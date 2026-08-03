"""Automatic retention (Gate 3 Step 9): extends ``app.mcp.maintenance_cli``'s
manual sweep to ``exports/``, ``clips/`` and ``editor_text/`` — the three
directories that module's own docstring says it deliberately never
touches — and runs the FULL sweep (everything ``maintenance_cli`` already
knows how to clean, plus these three) on a periodic timer inside the running
server process, so a deployment that never sets up an operator cron still
gets bounded disk growth.

WHY A NEW MODULE, NOT AN EXTENSION OF ``maintenance_cli.py``'s OWN
CATEGORIES. That module's docstring and its own test suite
(``test_real_run_never_touches_exports_or_clips``, and the ``main()``
fixtures that hand it a bare ``media_dir``-only fake ``Settings`` with no
retention fields at all) lock in the promise that ``exports/``/``clips/``
are untouched. Rather than break that contract, this module adds a SEPARATE
sweep for the three previously-untouched directories and composes it with
the existing one via ``run_full_cleanup``. ``maintenance_cli.main()`` is
unchanged, byte for byte, and stays a valid — narrower — manual entry point
on its own; this module is the wider one, and the one the background worker
below actually calls.

TWO DIFFERENT RULES, because the three directories are not the same kind of
thing. ``exports/`` and ``clips/`` are swept ON AGE ALONE — one is a
DELIVERY the user has already been handed, the other a download cache — and
only ``editor_text/`` is swept ORPHAN-AND-OLD, because it is a render INPUT
whose loss a live project would actually feel.

AGE ALONE, 1: ``exports/{project_id}_{version}[_{preset}].{mp4,mlt,png}`` —
the render, the ``.mlt`` XML sidecar ``app.editor.render`` keeps next to it,
and the ``_cover.png`` a requested cover frame adds (all written by
``app.mcp.tools.export``/``_render_cover_if_requested``; the set is named in
``app.media.exports``). Swept once the bundle is older than
``Settings.export_ttl_hours`` (24h), with NO liveness check: a finished
render belonging to a project someone is editing right now is removed just
the same.

  WHY THAT IS SAFE, and why the earlier "an export of a live project is
  untouchable at any age" rule was replaced. That rule was written from the
  premise that an export is a stored artifact of the service. It is not: it
  is a DELIVERY. What the service keeps is the project — the timeline
  snapshot in ``ProjectRow`` plus the append-only ``OperationLogRow``
  journal, neither of which this module has ever touched — and re-running
  ``editor_export`` on the current version reproduces the file. The user
  takes the delivery away (a download, or a copy onto their media library
  shelf via ``POST /api/media/from-export``), and BOTH of those paths delete
  the server copy themselves the moment the hand-off is confirmed. This TTL
  is therefore the backstop for the ONLY case left: someone who never
  chose — closed the tab, lost the network — whose export would otherwise
  sit on disk forever because their project is still alive.

  THE OLD RULE'S JUSTIFICATION IS RETIRED DELIBERATELY, not overlooked. It
  argued from ``AnnotationRow``: a review marker pins to "the project's
  exported timeline" by ``project_id`` alone, with no version column, so no
  DB read can prove an older export is unreferenced — and concluded the file
  must therefore be kept. The conclusion no longer follows, because keeping
  the FILE was never what made an annotation meaningful: the annotation is
  about the timeline at a version, and the timeline is what survives.
  Annotations outlive the file they were written against, on purpose; they
  are journal entries about an edit, not attachments to an mp4.

  Re-rendering assumes the sources are still fetchable, and a clip pulled
  from a URL that has since gone away (or aged out of ``clips/`` below) will
  not come back. That is an ACCEPTED risk, taken with open eyes rather than
  designed around: nothing here tries to pin sources on behalf of an export.

- ``editor_text/<project_id>_<text_id>.png`` (``app.editor.render.
  TEXT_SUBDIR``; one rasterized PNG per text element, re-rasterized on every
  render) is the one category still swept ORPHAN-AND-OLD, on
  ``Settings.retention_artifact_ttl_days``. It is filename-prefixed with its
  owning ``project_id``, so "still owned" is answered from the SAME
  ``ProjectRow`` read ``maintenance_cli`` already does — mirroring
  ``voiceover/``'s "protected by project existence, not by content" rule
  there. The age gate on top means an artifact that JUST lost its owning
  project mid-request cannot vanish before whatever deleted the project has
  finished committing. Only ORPHANED *and* older than the TTL is ever
  removed; a live project's text raster is never removed at any age.

AGE ALONE, 2: ``clips/<asset_id>/``, the yt-dlp download cache for a
URL-sourced ``MediaAsset`` — written by ``app.media.downloader.
download_clip``, keyed by the asset's OWN id. This directory is swept once
it is older than ``Settings.clip_cache_ttl_hours`` (48h), and the sweep does
not consult project liveness AT ALL: a clip belonging to a project someone
is actively editing is removed just the same.

  WHY THAT IS SAFE, and why the earlier "live as long as some project's
  assets list still names that asset id" rule was replaced. A clips
  directory is not the asset — the ``MediaAsset`` row keeps the ORIGINAL
  URL, and ``app.web.api``'s ``resolve_media``/``app.editor.render``
  re-invoke ``download_clip`` with the identical output template whenever
  the file is missing. Deleting a cached clip therefore destroys no data;
  it costs one refetch. The old rule, by contrast, meant a clip was pinned
  to disk for as long as its project existed — i.e. effectively forever, at
  any age, for the largest files this service stores. That is unbounded
  growth with no upper bound an operator can set, which is exactly what a
  retention sweep exists to prevent, and it bought nothing that a refetch
  does not buy back.

  AGE IS READ FROM THE DIRECTORY'S CONTENTS, not from the directory's own
  mtime (``newest_content_mtime`` below). ``mkdir`` stamps the directory
  when it is created and writing a file INTO it does not restamp it on
  Linux/macOS, so a directory's own mtime is the age of the download that
  started, not of the media that is actually there — reading it would expire
  a cache that a longer download had only just finished filling.

The rules deliberately keep separate TTL settings rather than share one
number: hours for a delivery already handed over (``export_ttl_hours``),
hours for a cache that must not outstay its usefulness
(``clip_cache_ttl_hours``), days for the one artifact whose loss a live
project would feel (``retention_artifact_ttl_days``, now governing
``editor_text/`` and nothing else).

PHYSICAL DELETION goes through the SAME ``ArtifactStore`` seam Step 8 wired
into ``maintenance_cli.py`` (``store.delete``/``store.delete_prefix``) — this
module reuses ``maintenance_cli.delete_files``/``delete_dirs`` rather than
re-implement the dry-run/store branching a second time.

MULTI-REPLICA SAFETY (Step 9's requirement — no distributed lock). Every
physical deletion is idempotent by construction: ``ArtifactStore.delete`` is
documented a no-op on an already-absent key, and ``delete_prefix`` returns 0
when nothing matched (see ``app.storage.base``). Two replicas running this
sweep at the same moment therefore never error into each other — each
independently scans the same DB and the same filesystem/object store, so
they compute (near enough) the same candidate set and their deletes simply
overlap into redundant no-ops. No lock is taken, and none is required for
CORRECTNESS — only to avoid duplicate scan work across replicas, which this
deliberately does not optimise for (YAGNI: a low-frequency filesystem scan
is cheap next to the render/analysis work this same process also does).

SCANNING IS NOT ATOMIC WITH DELETING, and for both age-based categories that
gap is load-bearing rather than theoretical. Between "this directory's newest file
is older than the TTL" and the ``delete_prefix`` that acts on it, a
concurrent ``resolve_media``/render can start refilling the very same
``clips/<asset_id>/`` (yt-dlp writes ``.part``/``.ytdl`` fragments there and
then merges them), because the cache path is derived from the asset id alone
and nothing announces that a sweep is in flight. Deleting an IDLE clip is
cheap by design — it costs one refetch — but deleting an ACTIVE download is
a user-visible "resolve failed", recurring exactly once per
``retention_interval_sec``. So ``run_clip_cache_cleanup`` re-reads
``newest_content_mtime`` immediately before removing EACH directory and
skips any that has grown fresher since the scan; the skipped one is simply
re-examined on the next pass. This narrows the window to the single
store call rather than to a whole scan, WITHOUT introducing a lock: the
re-check is per-replica, local, and read-only, so it neither coordinates
between processes nor breaks the idempotence above (a directory another
replica already deleted just re-checks as gone and is skipped).

``run_export_cleanup`` re-checks its bundles the SAME way and for the same
reason, one gap narrower: a render writes ``exports/{project_id}_{version}
.mp4`` directly (plus ``.tmp.mp4`` while ffmpeg/melt is still working), so
re-exporting the SAME version over an expired file, or the cover frame
landing seconds after the mp4, both make a bundle grow fresher between the
scan and its delete. The bundle's age is the NEWEST mtime among its files
for exactly that reason — judging by the mp4 alone would delete a cover that
had only just been written.
"""
from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.common.logging import get_logger
from app.config import get_settings
from app.db.base import session_scope
from app.db.repositories.media_analysis import referenced_keyframe_names
from app.mcp import maintenance_cli as cli
from app.media import exports
from app.storage import ArtifactStore, LocalArtifactStore, get_artifact_store

logger = get_logger(__name__)

# Duplicated as plain strings rather than imported from their home modules
# (``app.mcp.tools.EXPORTS_SUBDIR``, ``app.editor.render.CLIPS_SUBDIR``/
# ``TEXT_SUBDIR``) for the same reason ``maintenance_cli.py`` duplicates its
# own five: ``app.mcp.tools`` pulls in the whole editor/render stack, and
# this module is imported at server STARTUP (to launch the background
# worker below), not only by an operator's occasional CLI invocation.
EXPORTS_SUBDIR = "exports"
CLIPS_SUBDIR = "clips"
EDITOR_TEXT_SUBDIR = "editor_text"
# ``app.media.preview_proxy.PROXY_SUBDIR`` — the light 720p copies the editor
# stage streams instead of a phone original, duplicated here for the same reason
# the three above are.
PROXY_SUBDIR = "proxies"

# Overridable via Settings.retention_artifact_ttl_days. Governs editor_text/
# ONLY — exports moved to their own hours-scale TTL below.
DEFAULT_ARTIFACT_TTL_DAYS = 30
# Overridable via Settings.clip_cache_ttl_hours. In HOURS, not days, and
# deliberately much shorter than DEFAULT_ARTIFACT_TTL_DAYS — see the module
# docstring's "AGE ALONE" sections for why the download cache and the export
# delivery are the two liveness-blind categories.
DEFAULT_CLIP_CACHE_TTL_HOURS = 48
# Overridable via Settings.export_ttl_hours. The shortest TTL here on purpose:
# an export is a delivery, and the two paths that hand it over delete the
# server copy themselves — this only catches the user who never chose.
DEFAULT_EXPORT_TTL_HOURS = 24


@dataclass(frozen=True)
class ArtifactCleanupReport:
    """Per-category counts from one exports/clips/editor_text sweep.

    ``exports_removed`` counts EXPIRED DELIVERABLES — whole bundles (the
    render plus its ``.mlt``/``_cover.png`` sidecars), not files, because one
    bundle is one thing a user would recognise as "my export".
    ``clips_dirs_removed`` counts expired cache directories. Only
    ``editor_text_removed`` counts orphaned-and-old files; the field names
    predate the split and are kept as they are so existing callers and log
    lines keep working."""

    exports_removed: int
    clips_dirs_removed: int
    editor_text_removed: int
    # Defaulted, so every existing caller and test that builds this report
    # positionally or by keyword keeps working. Counts preview proxies
    # (``app.media.preview_proxy``) expired on the clip-cache TTL.
    proxies_removed: int = 0

    def describe(self, *, dry_run: bool) -> str:
        verb = "would be removed" if dry_run else "removed"
        return (
            f"exports: {self.exports_removed} {verb}, "
            f"clips: {self.clips_dirs_removed} dir {verb}, "
            f"editor_text: {self.editor_text_removed} {verb}, "
            f"proxies: {self.proxies_removed} {verb}"
        )


@dataclass(frozen=True)
class FullCleanupReport:
    """Every category from one sweep pass: ``maintenance_cli``'s original
    five (``base``), the three this module adds (``artifacts``), and the
    uploads pass (``uploads``) — the shape both the background worker and
    any manual call to ``run_full_cleanup`` get back."""

    base: cli.CleanupReport
    artifacts: ArtifactCleanupReport
    uploads: cli.UploadCleanupReport

    def describe(self, *, dry_run: bool) -> str:
        return "; ".join((
            self.base.describe(dry_run=dry_run),
            self.artifacts.describe(dry_run=dry_run),
            self.uploads.describe(),
        ))


# --- pure functions: no filesystem mutation, easy to unit test -------------


def _owning_project_id(filename: str) -> str:
    """The ``project_id`` a maintenance-swept filename begins with — the
    segment before the FIRST underscore. Every id this codebase mints
    (``app.editor.mutations._new_id``) is a bare ``uuid.uuid4().hex``, which
    never itself contains an underscore, so splitting on the first one is
    exact regardless of what follows: a version number and optional preset
    suffix (and an optional ``_cover``) for an export, a text element id for
    an ``editor_text`` raster."""
    return filename.split("_", 1)[0]


def find_stale_orphan_files(
    directory: Path, known_project_ids: set[str], cutoff: datetime
) -> list[Path]:
    """Files directly under ``directory`` whose leading ``{project_id}_``
    segment does not name a project in ``known_project_ids``, AND whose
    mtime is older than ``cutoff``. Used for ``editor_text/`` — and, since
    exports moved to a plain TTL, for nothing else here. It is project-scoped
    by filename, and ownership outlives the file's own age (a project
    untouched for months is still a live project), so age alone is never
    sufficient for a render INPUT; see the module docstring. Returns ``[]``
    when ``directory`` does not exist."""
    if not directory.is_dir():
        return []
    stale = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        if _owning_project_id(entry.name) in known_project_ids:
            continue
        mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc)
        if mtime < cutoff:
            stale.append(entry)
    return stale


def newest_content_mtime(directory: Path) -> datetime:
    """The mtime of the most recently written file anywhere under
    ``directory``, falling back to the directory's OWN mtime when it holds no
    files at all.

    The fallback is what lets an empty leftover (a download that died before
    writing anything, or a directory whose files were removed by hand) still
    age out instead of being immortal. The primary rule exists because a
    directory's own mtime is NOT the age of its contents: ``mkdir`` stamps it
    once at creation and writing a file inside does not restamp it, so a
    multi-minute yt-dlp download would otherwise be judged by when it
    STARTED. ``rglob`` rather than ``iterdir`` because a future yt-dlp
    postprocessor writing into a subdirectory must not read as "no content".

    Best-effort per entry: a file racing to disappear mid-scan is skipped
    rather than aborting the sweep (same posture as
    ``maintenance_cli._dir_size_bytes``)."""
    newest: float | None = None
    for entry in directory.rglob("*"):
        try:
            if not entry.is_file():
                continue
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
    if newest is None:
        newest = directory.stat().st_mtime
    return datetime.fromtimestamp(newest, tz=timezone.utc)


def find_expired_clip_dirs(clips_dir: Path, cutoff: datetime) -> list[Path]:
    """Subdirectories of ``clips_dir`` — one per ``MediaAsset`` id — whose
    freshest content predates ``cutoff``.

    Takes NO liveness argument, and that absence is the whole point: unlike
    every other sweep in this module and in ``maintenance_cli``, the download
    cache expires purely by age, so a clip owned by a project someone is
    editing right now is removed exactly like an abandoned one once it is old
    enough. See the module docstring for why that loses nothing (the asset
    keeps its source URL and the clip is refetched on demand).

    Returns ``[]`` when ``clips_dir`` does not exist."""
    if not clips_dir.is_dir():
        return []
    expired = []
    for entry in clips_dir.iterdir():
        if not entry.is_dir():
            continue
        if newest_content_mtime(entry) < cutoff:
            expired.append(entry)
    return expired


def newest_bundle_mtime(paths: Iterable[Path]) -> datetime | None:
    """The mtime of the most recently written file in one export bundle, or
    ``None`` when not one of them can be stat'ed any more.

    NEWEST, not the render's own, for the reason the module docstring gives:
    the cover frame is rendered AFTER the mp4 (a second ffmpeg run), and a
    re-export of the same version rewrites the mp4 in place — judging a bundle
    by any single member would expire files its siblings prove are in use.

    ``None`` rather than a fallback timestamp when everything is gone: unlike
    an empty ``clips/`` directory (which still exists and must still be able to
    age out), a bundle with no readable files is not a thing to delete, it is a
    thing that has already been deleted. Best-effort per entry, matching
    ``newest_content_mtime``.
    """
    newest: float | None = None
    for path in paths:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
    if newest is None:
        return None
    return datetime.fromtimestamp(newest, tz=timezone.utc)


def find_expired_export_bundles(
    exports_dir: Path, cutoff: datetime
) -> list[tuple[Path, ...]]:
    """Export bundles under ``exports_dir`` whose freshest file predates
    ``cutoff`` — the render plus its sidecars, grouped by
    ``app.media.exports.bundles_in``.

    Takes NO liveness argument, deliberately and unlike
    ``find_stale_orphan_files``: an export is a delivery, so a bundle belonging
    to a project someone is actively editing expires exactly like one whose
    project was deleted. See the module docstring for why that loses nothing
    the service promised to keep.

    Returns ``[]`` when ``exports_dir`` does not exist."""
    expired = []
    for bundle in exports.bundles_in(exports_dir):
        newest = newest_bundle_mtime(bundle)
        if newest is not None and newest < cutoff:
            expired.append(bundle)
    return expired


# --- I/O: physical removal, routed through the artifact store --------------


def _is_still_expired(directory: Path, cutoff: datetime) -> bool:
    """Whether ``directory`` STILL looks expired at this instant — the
    re-check the module docstring's "SCANNING IS NOT ATOMIC WITH DELETING"
    section describes, evaluated immediately before the directory is removed
    rather than back when the candidate list was built.

    A directory that vanished in the meantime (another replica's sweep, an
    operator's ``rm``) is reported as NOT expired: there is nothing left to
    delete, so counting it as removed would inflate the report with work this
    process did not do. ``newest_content_mtime`` reads the directory's own
    mtime when it holds no files, which is the call that can raise once the
    directory itself is gone."""
    try:
        return newest_content_mtime(directory) < cutoff
    except OSError:
        return False


def run_clip_cache_cleanup(
    media_dir: Path,
    *,
    ttl_hours: int = DEFAULT_CLIP_CACHE_TTL_HOURS,
    dry_run: bool = False,
    store: ArtifactStore | None = None,
) -> int:
    """Expire ``{media_dir}/clips/<asset_id>/`` directories older than
    ``ttl_hours`` and return how many were (or, under ``dry_run``, would have
    been) removed.

    Its own function rather than a branch inside ``run_artifact_cleanup``
    because it answers a different question with a different unit (hours of
    cache age, not days since the owning project vanished) and takes no
    liveness input at all — keeping them apart is what stops the two rules
    from being read as one. Physical removal still goes through
    ``cli.delete_dirs``/``ArtifactStore``, the same seam every other category
    uses, so ``dry_run`` and an injected S3 store behave identically here.

    Removal is per-directory rather than one batch call so that each
    candidate's age can be RE-VERIFIED against the same cutoff right before
    its own delete (see ``_is_still_expired`` and the module docstring): a
    clip a concurrent download started refilling since the scan is skipped
    and left for the next pass. The re-check is deliberately NOT skipped
    under ``dry_run`` — the dry-run count is supposed to answer "what would a
    real run remove", and a real run would skip that directory too."""
    if store is None:
        store = LocalArtifactStore(media_dir)
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=ttl_hours)
    expired = find_expired_clip_dirs(media_dir / CLIPS_SUBDIR, cutoff)
    removed = 0
    for directory in expired:
        if not _is_still_expired(directory, cutoff):
            logger.info(
                "clip cache %s grew fresher between scan and delete; "
                "leaving it for the next sweep", directory.name,
            )
            continue
        removed += cli.delete_dirs(
            [directory], dry_run=dry_run, store=store, media_dir=media_dir
        )
    return removed


def find_expired_files(directory: Path, cutoff: datetime) -> list[Path]:
    """Regular files directly inside ``directory`` last modified before
    ``cutoff``. Age alone — no project or asset liveness — which is what the
    proxy cache wants: a proxy is derived data whose only cost of being wrong is
    that a preview streams the original until something rebuilds it."""
    if not directory.is_dir():
        return []
    expired = []
    for path in directory.iterdir():
        try:
            if path.is_file() and _mtime(path) < cutoff:
                expired.append(path)
        except OSError:
            continue  # vanished between iterdir and stat
    return expired


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def run_proxy_cache_cleanup(
    media_dir: Path,
    *,
    ttl_hours: int = DEFAULT_CLIP_CACHE_TTL_HOURS,
    dry_run: bool = False,
    store: ArtifactStore | None = None,
) -> int:
    """Expire preview proxies under ``{media_dir}/proxies/`` older than
    ``ttl_hours``, returning how many were (or would have been) removed.

    Shares ``clip_cache_ttl_hours`` rather than introducing a knob of its own:
    both are derived caches of source media with the same bargain — cheap to
    rebuild, worthless once nobody is editing that clip — and a second setting
    would be a second thing to explain for no decision anyone would make
    differently.

    Sweeping a proxy that IS still in use is safe and deliberately unguarded: the
    stream route falls back to the original, so the consequence is a heavier
    preview until the next ``add_clip`` schedules a rebuild, never a broken one.
    """
    if store is None:
        store = LocalArtifactStore(media_dir)
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=ttl_hours)
    expired = find_expired_files(media_dir / PROXY_SUBDIR, cutoff)
    removed = 0
    for path in expired:
        # Re-verified immediately before its own delete, exactly like the clip
        # cache: a proxy rebuilt between the scan and now is left alone.
        try:
            if _mtime(path) >= cutoff:
                continue
        except OSError:
            continue
        removed += cli.delete_files(
            [path], dry_run=dry_run, store=store, media_dir=media_dir
        )
    return removed


def _bundle_is_still_expired(bundle: tuple[Path, ...], cutoff: datetime) -> bool:
    """``_is_still_expired``'s twin for an export bundle, evaluated
    immediately before the bundle is removed rather than back when the
    candidate list was built.

    A bundle whose files have all vanished (another replica's sweep, the
    release route, an operator's ``rm``) reports NOT expired — there is
    nothing left to delete, and counting it would inflate the report with work
    this process did not do. So does one a re-export refreshed since the scan.
    """
    newest = newest_bundle_mtime(bundle)
    return newest is not None and newest < cutoff


def run_export_cleanup(
    media_dir: Path,
    *,
    ttl_hours: int = DEFAULT_EXPORT_TTL_HOURS,
    dry_run: bool = False,
    store: ArtifactStore | None = None,
) -> int:
    """Expire whole export bundles under ``{media_dir}/exports/`` older than
    ``ttl_hours`` and return how many bundles were (or, under ``dry_run``,
    would have been) removed.

    Shaped exactly like ``run_clip_cache_cleanup`` — its own function, no
    liveness input, per-candidate removal so each bundle's age can be
    RE-VERIFIED against the same cutoff right before its own delete (see
    ``_bundle_is_still_expired``). A render that rewrote the same version, or a
    cover frame that landed after the scan, is skipped and left for the next
    pass. The re-check is deliberately NOT skipped under ``dry_run``: the
    dry-run count answers "what would a real run remove", and a real run would
    skip it too.

    Whole bundles, never loose files: ``app.media.exports`` names the set, so a
    ``.mlt`` or ``_cover.png`` is never stranded pointing at a render that is
    gone. The count is BUNDLES for the same reason.
    """
    if store is None:
        store = LocalArtifactStore(media_dir)
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=ttl_hours)
    expired = find_expired_export_bundles(media_dir / EXPORTS_SUBDIR, cutoff)
    removed = 0
    for bundle in expired:
        if not _bundle_is_still_expired(bundle, cutoff):
            logger.info(
                "export %s grew fresher between scan and delete; "
                "leaving it for the next sweep", bundle[0].name,
            )
            continue
        exports.delete_bundle(
            bundle, media_dir=media_dir, store=store, dry_run=dry_run
        )
        removed += 1
    return removed


def run_artifact_cleanup(
    media_dir: Path,
    known_project_ids: set[str],
    *,
    ttl_days: int = DEFAULT_ARTIFACT_TTL_DAYS,
    clip_cache_ttl_hours: int = DEFAULT_CLIP_CACHE_TTL_HOURS,
    export_ttl_hours: int = DEFAULT_EXPORT_TTL_HOURS,
    dry_run: bool = False,
    store: ArtifactStore | None = None,
) -> ArtifactCleanupReport:
    """One sweep pass over ``exports/``, ``clips/`` and ``editor_text/``
    under ``media_dir`` — the three directories ``app.mcp.maintenance_cli``
    deliberately never touches (see that module's docstring).

    ``exports/`` (``export_ttl_hours``) and ``clips/``
    (``clip_cache_ttl_hours``) expire on age alone and take no project/asset
    liveness into account; only ``editor_text/`` is orphan-AND-old
    (``ttl_days``), so ``known_project_ids`` is now consulted for that one
    category. See this module's docstring for why the rules differ."""
    if store is None:
        store = LocalArtifactStore(media_dir)
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=ttl_days)
    stale_text = find_stale_orphan_files(
        media_dir / EDITOR_TEXT_SUBDIR, known_project_ids, cutoff
    )
    return ArtifactCleanupReport(
        exports_removed=run_export_cleanup(
            media_dir, ttl_hours=export_ttl_hours, dry_run=dry_run, store=store
        ),
        clips_dirs_removed=run_clip_cache_cleanup(
            media_dir, ttl_hours=clip_cache_ttl_hours, dry_run=dry_run, store=store
        ),
        editor_text_removed=cli.delete_files(
            stale_text, dry_run=dry_run, store=store, media_dir=media_dir
        ),
        proxies_removed=run_proxy_cache_cleanup(
            media_dir, ttl_hours=clip_cache_ttl_hours, dry_run=dry_run, store=store
        ),
    )


def run_full_cleanup(
    session: Session,
    media_dir: Path,
    *,
    days: int = cli.DEFAULT_STALE_DAYS,
    artifact_ttl_days: int = DEFAULT_ARTIFACT_TTL_DAYS,
    clip_cache_ttl_hours: int = DEFAULT_CLIP_CACHE_TTL_HOURS,
    export_ttl_hours: int = DEFAULT_EXPORT_TTL_HOURS,
    dry_run: bool = False,
    delete_uploads: bool = False,
    store: ArtifactStore | None = None,
) -> FullCleanupReport:
    """The single sweep both the background worker and a manual invocation
    call: everything ``maintenance_cli.main()`` runs (previews/
    caption_windows/analysis/analysis_cache/voiceover, then uploads) PLUS
    this module's exports/clips/editor_text pass — one DB read
    (``known_ids``/``rows``/``protected``), all six-plus-three categories fed
    from it, so the picture of "what is live" cannot drift between categories
    within a single pass. ``exports/`` and ``clips/`` need nothing from that
    read: both expire on age alone (see the module docstring).

    ``delete_uploads`` mirrors ``maintenance_cli.main``'s explicit opt-in:
    ``uploads/`` stays report-only unless this is True (and unless
    ``dry_run`` is True, which always wins), for the exact reason that
    module's docstring gives — getting an uploads orphan scan wrong deletes
    media a live project still renders from.
    """
    known_ids = cli.known_project_ids(session)
    rows = cli.all_project_rows(session)
    protected = referenced_keyframe_names(session)

    base_report = cli.run_cleanup(
        media_dir, known_ids, days=days, dry_run=dry_run,
        protected_analysis_names=protected, store=store,
    )
    artifact_report = run_artifact_cleanup(
        media_dir, known_ids,
        ttl_days=artifact_ttl_days, clip_cache_ttl_hours=clip_cache_ttl_hours,
        export_ttl_hours=export_ttl_hours, dry_run=dry_run, store=store,
    )
    upload_dry_run = dry_run or not delete_uploads
    upload_report = cli.run_upload_cleanup(
        media_dir, rows, dry_run=upload_dry_run, store=store,
    )
    return FullCleanupReport(base=base_report, artifacts=artifact_report, uploads=upload_report)


# --- background worker: periodic sweep inside the running server -----------
#
# Starlette lifespan hooks are not a clean fit here: the app this project
# actually serves (``app.mcp.server.main`` -> ``app.web.app.build_app`` ->
# ``mcp_server.streamable_http_app()``) already sets its OWN ``lifespan=``
# callable at Router construction time (running the MCP SDK's session
# manager) — Starlette honours exactly one ``lifespan`` per app, given
# explicitly at construction, and composing a second one in would mean
# editing that assembly in ``app.web.app.build_app``. A bare interval-driven
# daemon thread avoids that and matches two existing precedents already in
# this codebase: ``app.media.downloader``'s per-fetch
# ``threading.Thread(daemon=True)`` and ``app.analysis.background``'s
# lazily-started, module-singleton executor/scheduler.

_worker_thread: threading.Thread | None = None
_stop_event = threading.Event()
_worker_lock = threading.Lock()


def _run_one_sweep(
    media_dir: Path,
    store: ArtifactStore,
    *,
    days: int,
    artifact_ttl_days: int,
    clip_cache_ttl_hours: int,
    export_ttl_hours: int,
) -> None:
    """One full pass, in its own DB session, with failures CONTAINED — a
    sweep that raises must not kill the worker thread (there would be no one
    left to retry it on the next interval) and must not propagate into
    whatever called ``start_retention_worker``.

    ``store`` is REQUIRED, not defaulted: leaving it out is exactly the bug
    this parameter exists to prevent (``run_full_cleanup`` would fall back to
    a ``LocalArtifactStore`` and cheerfully report deletions from a local
    path while the configured S3/MinIO bucket grew without bound)."""
    try:
        with session_scope() as session:
            report = run_full_cleanup(
                session, media_dir, days=days, artifact_ttl_days=artifact_ttl_days,
                clip_cache_ttl_hours=clip_cache_ttl_hours,
                export_ttl_hours=export_ttl_hours,
                dry_run=False, delete_uploads=False, store=store,
            )
        logger.info("automatic retention sweep: %s", report.describe(dry_run=False))
    except Exception:
        logger.exception("automatic retention sweep failed; retrying next interval")


def _worker_loop(
    media_dir: Path,
    interval_sec: float,
    store: ArtifactStore,
    *,
    days: int,
    artifact_ttl_days: int,
    clip_cache_ttl_hours: int,
    export_ttl_hours: int,
) -> None:
    # ``Event.wait`` doubles as an interruptible sleep: a set ``_stop_event``
    # returns True immediately and ends the loop instead of blocking a full
    # interval on shutdown.
    #
    # ``store`` is built once by the caller and reused for every pass: it is a
    # stateless client handle (an S3ArtifactStore wraps one minio client; the
    # local one wraps a Path), and ``get_artifact_store`` reads the same
    # process-wide ``Settings`` every time, so rebuilding it per pass would
    # reconstruct an identical object — and, on the S3 backend, a fresh
    # connection pool — for no gain.
    while not _stop_event.wait(interval_sec):
        _run_one_sweep(
            media_dir, store, days=days, artifact_ttl_days=artifact_ttl_days,
            clip_cache_ttl_hours=clip_cache_ttl_hours,
            export_ttl_hours=export_ttl_hours,
        )


def start_retention_worker() -> None:
    """Start the periodic background sweep once, idempotently. A no-op when
    ``Settings.retention_enabled`` is False, or when a worker is already
    running (a repeat call — e.g. an app assembled more than once in one
    process — never spawns a second thread).

    THE SWEEP RUNS AGAINST THE CONFIGURED ARTIFACT STORE, obtained from the
    same ``app.storage.get_artifact_store(get_settings())`` helper the HTTP
    layer uses (``app.web.app.build_app``), NOT from ``run_full_cleanup``'s
    local default. Without it an ``ARTIFACT_BACKEND=s3`` deployment swept a
    local directory the real ``exports/``/``clips/``/``editor_text/`` objects
    had nothing to do with — logging encouraging "N removed" counts while the
    bucket, and its 48h clip-cache TTL, went entirely unenforced.

    The store is built ONCE here and captured by the worker thread rather
    than rebuilt per pass: ``get_artifact_store`` is a plain factory over the
    process-wide ``Settings`` (no caching, no refresh semantics), so every
    call in a given process yields an equivalent handle — see ``_worker_loop``.
    A misconfigured backend therefore raises HERE, at startup, exactly as it
    already does when ``build_app`` asks the same helper the same question;
    silently sweeping the wrong place is the failure mode being fixed, so it
    must not be re-introduced as a fall-back.

    MULTI-REPLICA: see the module docstring — no distributed lock, none
    required for correctness, since every physical delete this worker
    triggers is idempotent.
    """
    global _worker_thread
    settings = get_settings()
    if not settings.retention_enabled:
        return
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _stop_event.clear()
        media_dir = Path(settings.media_dir)
        store = get_artifact_store(settings)
        _worker_thread = threading.Thread(
            target=_worker_loop,
            args=(media_dir, settings.retention_interval_sec, store),
            kwargs={
                "days": cli.DEFAULT_STALE_DAYS,
                "artifact_ttl_days": settings.retention_artifact_ttl_days,
                # 48h TTL against a 6h interval: a clip lives at most one
                # extra interval past its TTL, which is the accuracy this
                # category needs — see Settings.clip_cache_ttl_hours.
                "clip_cache_ttl_hours": settings.clip_cache_ttl_hours,
                # Same "TTL plus at most one interval" slack, and harmless for
                # the same reason: this sweep only ever sees exports whose
                # owner never told us where to put them, so nobody is waiting
                # on the deletion — see Settings.export_ttl_hours.
                "export_ttl_hours": settings.export_ttl_hours,
            },
            name="retention-worker",
            daemon=True,
        )
        _worker_thread.start()


def stop_retention_worker(*, timeout: float = 5.0) -> None:
    """Signal the background sweep to stop and wait up to ``timeout``
    seconds for its current sleep/sweep to unwind. Safe to call when no
    worker is running (a no-op). Tests use this in teardown so a started
    worker never bleeds a live thread into the next test; a graceful
    process shutdown may call it too, though the thread is a daemon and
    would not block process exit either way."""
    global _worker_thread
    with _worker_lock:
        thread = _worker_thread
        _worker_thread = None
    if thread is None:
        return
    _stop_event.set()
    thread.join(timeout=timeout)
