"""Deleting ONE project: its row, every row keyed by it, and its own artifacts.

Until this module existed the ONLY way a project could be deleted was to delete
the whole ACCOUNT (``app.db.repositories.accounts.delete_account``), whose
docstring said so in as many words and warned that adding a per-project path
later "means adding it to that list too". This is that path, and rather than
repeat the list it OWNS it: ``delete_project_children`` below is the single
enumeration of the tables keyed by ``project_id``, and the account cascade now
calls it instead of carrying a second copy that would drift on the next table.
(It already had: ``analysis_jobs`` gained a ``project_id`` in Gate 3 Step 6 and
never made it into the account cascade. Unifying the list fixes that too.)

WHAT A PROJECT OWNS, AND WHAT IT ONLY BORROWS
=============================================

DELETED — rows keyed by the project:

* ``projects`` — the snapshot row itself.
* ``operations`` — the append-only op journal.
* ``annotations`` — operator error-markers pinned to this timeline.
* ``project_checkpoints`` — the server-side version history. It has no life of
  its own by design (see ``app.db.models.ProjectCheckpointRow``: "a checkpoint
  dies when the project does"), which is exactly what makes this deletion
  irreversible — there is no state left to roll back to.
* ``analysis_jobs`` — any LIVE background-analysis job for this project's
  assets. A queued job whose project no longer exists can only fail when it is
  claimed, and its unique ``(project_id, media_id)`` key would otherwise sit
  there forever.

DELETED — artifacts under ``media_dir``, all through the ``ArtifactStore`` seam
so an ``ARTIFACT_BACKEND=s3`` deployment behaves identically:

* ``exports/{project_id}_*`` — the WHOLE deliverable, every version of it. The
  glob is deliberately by prefix rather than by ``app.media.exports.bundle_for``
  because all three members of a bundle (the render, its ``.mlt`` sidecar and
  ``{stem}_cover.png``) are built from a stem that starts with
  ``{project_id}_``, so one prefix sweep is a strict SUPERSET of every bundle —
  including the strays that module's ``bundles_in`` exists to collect.
* ``previews/{project_id}_*`` — rendered preview frames.
* ``editor_text/{project_id}_*`` — rasterized text overlays.
* ``voiceover/{project_id}/`` — the whole per-project TTS directory.
* ``clips/<asset_id>/`` — the yt-dlp download cache of this project's assets,
  and ONLY of the asset ids no surviving project still names (the same
  reference-count discipline ``accounts._deletable_upload_dir_names`` applies
  to uploads, for the same reason: deleting a directory another project renders
  from is the one mistake a cache sweep must never make).

KEPT, DELIBERATELY:

* **The owner's media library and their uploaded files.** Owner decision
  2026-07-27: a file the user uploaded belongs to the USER, not to the project
  it happened to be dropped into. It lives under ``uploads/``, it is paid for
  out of ``storage_bytes_used``, and it has a ``MediaLibraryFileRow`` with
  ``charged_to='storage'`` on the ``/library`` screen. Deleting a project must
  therefore not touch ``uploads/``, ``media_library_files``, or the charge that
  paid for them — the user removes those from ``/library`` themselves, through
  ``DELETE /api/media/{id}``, which refunds the right budget.
* ``media_analyses`` / ``analysis_access`` / ``analysis/`` strips — keyed by
  media CONTENT and shared across tenants (see those models). A project is not
  their owner, and ``AnalysisAccessRow``'s docstring already records the
  decision that "the media a tenant already downloaded is not un-seen by
  deleting a project".
* ``caption_windows/<clip_id>.wav`` — keyed by TIMELINE ELEMENT id, not by
  project, so it cannot be enumerated by prefix; it is a transient ASR
  intermediate already collected on age by ``app.mcp.maintenance_cli``. Walking
  every element to issue one delete per clip would cost a store round-trip per
  timeline element for files that are usually already gone.
* ``analysis_cache/<digest>/`` — keyed by a digest of the SOURCE, shared
  between projects, age-swept.

QUOTAS: REFUND EXACTLY WHAT THE DELETED BYTES WERE CHARGED
==========================================================

Traced through the code, not from memory. ``storage_bytes_used`` has exactly
three writers:

* ``app.web.api._register_stored_file`` → ``quotas.charge_storage`` — project
  ingest (``uploads/`` + the shelf row). NOT deleted here, so NOT refunded.
* ``app.web.api.export_project`` → ``_meter_render_output`` →
  ``quotas.meter_storage`` for ``output_path`` and ``cover_path``.
* ``app.web.api.preview_frame`` → the same ``_meter_render_output`` for
  ``preview_path``.

So the ONLY deleted artifacts that ever moved a counter are the ones under
``exports/`` and ``previews/``, and their byte total is handed back with
``quotas.release_storage``. Everything else deleted here was charged to nobody
and is refunded nothing, on the record: ``clips/`` says so outright
("Cache, not owned storage. Nothing here charges the owner's quota" —
``app.web.api._download_into_clip_cache``), and no charge/meter call exists on
the ``voiceover/`` (``app.mcp.tools.generate_voiceover``) or ``editor_text/``
(``app.editor.render``) write paths at all. ``media_bytes_used`` is untouched
because the library is untouched.

THE REFUND CAN BE GENEROUS, and that is the deliberate direction. Metering
lives at the HTTP boundary, so an export driven over MCP (``editor_export``)
writes a file nothing ever charged for — refunding it credits bytes that were
never taken. The alternative is worse and unbounded: never refunding means
every HTTP export a project ever produced stays charged forever, and the owner
loses that space permanently. ``_apply_delta``'s ``CASE`` floor keeps the
counter non-negative either way, so the error is bounded by what was charged,
and this is the same trade ``_register_stored_file`` already documents for its
own over-generous refund. The real fix is metering inside ``tools.export``
rather than beside it, which is a separate change to a separate module.

Sizes are read off the LOCAL filesystem before the store delete, which is the
same thing ``_meter_render_output`` does when it charges them, and the same
thing every artifact-discovery site in this codebase does (``retention``,
``find_latest_export``, ``maintenance_cli``): enumeration is local, only the
physical delete goes through the store. That asymmetry predates this module and
is not introduced by it.

ORDER: DATABASE FIRST, ARTIFACTS SECOND
=======================================

Rows go in the caller's transaction; files go after ``flush`` confirms the rows
are gone. The reverse order is strictly worse, and the two failure modes are
not symmetric:

* **Artifacts fail after the rows are gone** (this order) — the project no
  longer exists and its files are debris, which the retention sweep already
  collects on age: ``exports/`` and ``clips/`` expire on age ALONE, and
  ``editor_text/`` expires orphan-and-old, which orphaned is exactly what it now
  is (``app.mcp.retention``). Bounded, self-healing, invisible to the user. The
  artifact phase therefore never raises — a store that refuses a delete must not
  turn a completed deletion into a 500 the client will retry against a project
  that is already gone.
* **Rows fail after the artifacts are gone** (the reverse) — a LIVE project
  whose renders, text rasters and voiceover have vanished under it. Nothing
  collects that, and nothing tells the user why their project broke.

Neither order survives a crash between the two phases perfectly, and this one
does not pretend to. The residual window is between ``flush`` and the caller's
``commit``: if the commit fails there, the rows come back and the files do not.
That is the identical window ``delete_account`` has carried since Gate 2 Step 9,
its damage is limited to re-renderable/re-fetchable artifacts (and a quota
refund that rolls back with the rows it belongs to), and closing it would take a
two-phase commit against an object store this codebase deliberately does not
have.

IDEMPOTENCE. Every DB delete is by predicate and every store delete is
documented a no-op on an absent key, so a second pass writes nothing. The
second CALL, though, answers "no such project" — because there genuinely is not
one — which is what makes the route's repeat-delete answer 404 rather than 200.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.job_models import AnalysisJobRow
from app.db.models import (
    AnnotationRow, OperationLogRow, ProjectRow, ProjectStatus,
)
from app.db.repositories import project_checkpoints as checkpoints_repo
from app.editor.errors import EditorError
from app.storage import ArtifactStore, LocalArtifactStore, artifact_key

logger = logging.getLogger(__name__)

# Spelled as plain strings rather than imported from their home modules
# (``app.mcp.tools``, ``app.editor.render``) for the reason ``app.mcp.retention``
# and ``app.mcp.maintenance_cli`` already spell out for their own copies: those
# modules pull in the whole editor/render stack, and this one is a leaf a
# repository and the MCP tool layer both import — the latter would make the
# import cycle.
EXPORTS_SUBDIR = "exports"
PREVIEWS_SUBDIR = "previews"
EDITOR_TEXT_SUBDIR = "editor_text"
VOICEOVER_SUBDIR = "voiceover"
CLIPS_SUBDIR = "clips"

# Prefix-scanned directories whose bytes WERE metered against
# ``storage_bytes_used`` (see the module docstring's quota trace) — deleting one
# of these files refunds its size.
_METERED_SUBDIRS = (EXPORTS_SUBDIR, PREVIEWS_SUBDIR)
# Prefix-scanned directories nothing ever charged for.
_UNMETERED_SUBDIRS = (EDITOR_TEXT_SUBDIR,)


class ProjectExportingError(EditorError):
    """Refusal to delete a project while its export is still rendering.

    A DISTINCT TYPE, not a message the caller has to sniff, so the HTTP
    boundary maps it to 423 by type — the same discipline
    ``StorageQuotaExceededError`` documents for its 413. EXPORTED is NOT this
    error: that render has finished and nothing is writing to disk any more, so
    an exported project is perfectly deletable. Only EXPORTING is refused, and
    only because a melt/ffmpeg subprocess is at this moment writing files whose
    project row we would be pulling out from under it.
    """


@dataclass(frozen=True)
class ProjectChildRowsDeleted:
    """Row counts from one pass of ``delete_project_children``."""

    operations: int
    annotations: int
    checkpoints: int
    analysis_jobs: int


@dataclass(frozen=True)
class ProjectDeletionReport:
    """Immutable summary of one ``delete_project`` pass.

    ``bytes_released`` is what was handed back to ``storage_bytes_used``, i.e.
    the size of the deleted ``exports/``+``previews/`` files and nothing else —
    see the module docstring's quota trace.
    """

    project_id: str
    rows: ProjectChildRowsDeleted
    files_deleted: int
    prefixes_deleted: int
    bytes_released: int


def delete_project_children(session: Session, project_ids) -> ProjectChildRowsDeleted:
    """Delete every row keyed by ``project_id`` in the given projects, EXCEPT
    the ``projects`` rows themselves.

    THE SINGLE SOURCE OF TRUTH for that table list. Both deletion paths — this
    module's per-project one and ``accounts.delete_account`` — call it, so a
    table added with a ``project_id`` is wired into both by being added here
    once. Two independent lists is precisely how ``analysis_jobs`` came to be
    cascaded by neither.

    The ``projects`` row is NOT deleted here because the two callers need
    genuinely different predicates for it — "every project of this owner" for
    the account cascade, "this one id, owned by this caller, and not mid-export"
    for ``delete_project`` — and folding both into one function would mean a
    parameterised predicate that is harder to read than the two statements it
    replaces. The parent row is one line at each call site; the CHILD list is
    the part that drifts, and that is what lives here.

    By predicate throughout, so a second run matches nothing rather than
    failing — the idempotence both callers' contracts promise. Deletes nothing
    and returns zeroes for an empty id list.
    """
    ids = list(project_ids)
    if not ids:
        return ProjectChildRowsDeleted(0, 0, 0, 0)
    operations = session.execute(
        delete(OperationLogRow).where(OperationLogRow.project_id.in_(ids))
    ).rowcount
    annotations = session.execute(
        delete(AnnotationRow).where(AnnotationRow.project_id.in_(ids))
    ).rowcount
    checkpoints = checkpoints_repo.delete_for_projects(session, ids)
    analysis_jobs = session.execute(
        delete(AnalysisJobRow).where(AnalysisJobRow.project_id.in_(ids))
    ).rowcount
    return ProjectChildRowsDeleted(
        operations=operations,
        annotations=annotations,
        checkpoints=checkpoints,
        analysis_jobs=analysis_jobs,
    )


def delete_project(
    session: Session,
    project_id: str,
    *,
    owner_id: int,
    media_dir: Path | str,
    store: ArtifactStore | None = None,
) -> ProjectDeletionReport | None:
    """Delete ONE of ``owner_id``'s projects and everything that belongs to it.

    Returns ``None`` when the id names no project OF THIS OWNER — an unknown id
    and another tenant's project are the same answer, deliberately, so this
    cannot be used to probe which project ids exist (the rule every read in
    ``app.db.repositories.projects`` already follows). Callers map ``None`` to
    404, which is also what makes a repeat delete answer 404 instead of 500.

    Raises ``ProjectExportingError`` when the project is EXPORTING. EXPORTED
    deletes normally.

    The caller owns the transaction: this function only ``flush``es, so a
    failure in the DB phase leaves the whole cascade uncommitted. ``store``
    defaults to a ``LocalArtifactStore`` rooted at ``media_dir`` — the
    self-hosted behavior — and an S3-backed store can be injected instead.

    **THE AUTHORIZATION AND THE LOCK CHECK RIDE INSIDE THE DELETE**, not in a
    read before it, for the reason ``projects.save_project`` spells out at
    length: "confirm the caller owns this and it is unlocked" followed by "now
    write" is a TOCTOU window, and what fits in that window here is an export
    starting between the two. ``DELETE ... WHERE project_id = ? AND owner_id = ?
    AND status <> 'exporting'`` carries every precondition into the write, so
    the database evaluates them atomically with it on any backend. The
    zero-row case is then classified by a re-read that is purely diagnostic and
    can no longer affect the write it explains.

    The parent row goes FIRST and the children after — the opposite of
    ``delete_account``'s children-first order, and free to differ because none
    of these tables carries a real foreign key to ``projects`` (they reference
    it by a plain string, deliberately; see ``app.db.models``). Deleting the
    parent first is what lets the EXPORTING refusal raise having written
    NOTHING, instead of depending on the caller to roll back children it had
    already removed.
    """
    media_root = Path(media_dir)
    if store is None:
        store = LocalArtifactStore(media_root)

    # Read the asset ids while the row still exists — the clips cache is keyed
    # by asset id and the row is about to be gone.
    row = session.execute(
        select(ProjectRow).where(
            ProjectRow.project_id == project_id, ProjectRow.owner_id == owner_id
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    asset_ids = _asset_ids(row.data)

    deleted = session.execute(
        delete(ProjectRow).where(
            ProjectRow.project_id == project_id,
            ProjectRow.owner_id == owner_id,
            ProjectRow.status != ProjectStatus.EXPORTING.value,
        )
    ).rowcount
    if deleted == 0:
        _raise_for_failed_delete(session, project_id, owner_id=owner_id)
        return None

    rows = delete_project_children(session, [project_id])

    clip_ids = _deletable_clip_asset_ids(session, asset_ids)
    metered_files = _prefixed_files(media_root, _METERED_SUBDIRS, project_id)
    unmetered_files = _prefixed_files(media_root, _UNMETERED_SUBDIRS, project_id)
    released = sum(size for _, _, size in metered_files)
    if released:
        # Lazy on purpose: the quota tables are closed core the public extract
        # (E-01) does not ship. There nothing ever charged these bytes, so
        # there is nothing to refund — skipping is the correct accounting,
        # not a degraded one.
        try:
            from app.db.repositories import quotas
        except ImportError:
            logger.debug("quota repo absent (single-user build) — nothing charged, "
                         "nothing to refund for %d bytes", released)
        else:
            quotas.release_storage(session, owner_id, released)
    session.flush()

    files_deleted, prefixes_deleted = _delete_artifacts(
        store, project_id, metered_files + unmetered_files, clip_ids
    )
    return ProjectDeletionReport(
        project_id=project_id,
        rows=rows,
        files_deleted=files_deleted,
        prefixes_deleted=prefixes_deleted,
        bytes_released=released,
    )


def _raise_for_failed_delete(
    session: Session, project_id: str, *, owner_id: int
) -> None:
    """Explain a zero-row ``DELETE`` in ``delete_project``, or return so the
    caller can report "no such project".

    Only ONE cause raises: the row is still there and it is EXPORTING. The
    other cause — it vanished between the gathering read and the delete,
    because a concurrent request or account deletion got there first — is not
    an error at all; the caller's intent is satisfied and "no such project" is
    the honest answer.
    """
    row = session.execute(
        select(ProjectRow).where(
            ProjectRow.project_id == project_id, ProjectRow.owner_id == owner_id
        )
    ).scalar_one_or_none()
    if row is None:
        return
    raise ProjectExportingError(
        f"project {project_id!r} is {ProjectStatus(row.status).value}; refusing to "
        "delete while a render is writing to disk — retry once the export finishes"
    )


# --- what the project's artifacts are ---------------------------------------


def _asset_ids(data: dict | None) -> set[str]:
    """Every ``MediaAsset`` id named by a stored project snapshot.

    Reads the raw ``data`` dict rather than deserializing through
    ``project_from_dict``: this is deletion, and it must still work for a row
    an older (or newer) serialization contract wrote. A snapshot with no
    ``assets`` key simply names nothing.
    """
    assets = (data or {}).get("assets") or []
    return {
        str(asset["id"])
        for asset in assets
        if isinstance(asset, dict) and asset.get("id")
    }


def _deletable_clip_asset_ids(session: Session, asset_ids: set[str]) -> set[str]:
    """Of ``asset_ids``, those no SURVIVING project still names.

    ``clips/<asset_id>/`` is a download cache, not owned storage, so the cost of
    deleting one wrongly is a refetch rather than data loss — but the cost of
    deleting one a live project renders from is a broken render right now, and
    that is worth one query to avoid. Asset ids are uuid4 hex minted per
    ``add_media``, so sharing is not the normal case; it becomes possible the
    moment a snapshot is copied between projects (``editor_restore_snapshot``
    takes an arbitrary blob), which is exactly the kind of thing a reference
    count exists to survive.

    Called AFTER the project's own row is deleted, so "surviving" is simply
    "still in the table" — no id predicate to keep in sync.
    """
    if not asset_ids:
        return set()
    referenced: set[str] = set()
    for data in session.execute(select(ProjectRow.data)).scalars():
        referenced |= _asset_ids(data)
    return asset_ids - referenced


def _prefixed_files(
    media_root: Path, subdirs, project_id: str
) -> list[tuple[str, str, int]]:
    """``(subdir, filename, size_bytes)`` for every file directly under each of
    ``subdirs`` whose name starts with ``{project_id}_``.

    The prefix is unambiguous because a project id is a fixed-length uuid4 hex
    (``app.editor.mutations._new_id``), so no project's prefix can be a prefix
    of another's. Sizes are read here, before the deletes, because that is the
    only moment they are still knowable; a file that disappears between the
    scan and the ``stat`` (a concurrent retention sweep) contributes zero
    rather than raising. A directory that does not exist yields nothing — a
    deployment that has never exported anything is not an error.
    """
    found: list[tuple[str, str, int]] = []
    for subdir in subdirs:
        directory = media_root / subdir
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            if not entry.is_file() or not entry.name.startswith(f"{project_id}_"):
                continue
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            found.append((subdir, entry.name, size))
    return found


# --- physical removal (best-effort, idempotent, never fatal) ----------------


def _delete_artifacts(
    store: ArtifactStore,
    project_id: str,
    files: list[tuple[str, str, int]],
    clip_asset_ids: set[str],
) -> tuple[int, int]:
    """Remove the project's artifacts through ``store``, returning
    ``(files_removed, prefixes_removed)``.

    NEVER RAISES — see the module docstring's ordering section. By the time
    this runs the rows are gone and the deletion has succeeded from every
    caller's point of view; a store that refuses a delete has left debris the
    retention sweep collects on age, which is not worth turning into a 500 that
    invites a retry against a project that no longer exists. Logged with the
    key, never swallowed silently.
    """
    files_removed = 0
    for subdir, name, _size in files:
        if _guarded(store.delete, artifact_key(subdir, name)) is not _FAILED:
            files_removed += 1

    prefixes = [artifact_key(VOICEOVER_SUBDIR, project_id)] + [
        artifact_key(CLIPS_SUBDIR, asset_id) for asset_id in sorted(clip_asset_ids)
    ]
    prefixes_removed = 0
    for prefix in prefixes:
        removed = _guarded(store.delete_prefix, prefix)
        # ``delete_prefix`` answers 0 for an absent subtree, so a project that
        # never had a voiceover (the common case) is not counted as one removed
        # — the same accounting ``accounts._delete_named_prefixes`` uses.
        if removed is not _FAILED and removed > 0:
            prefixes_removed += 1
    return files_removed, prefixes_removed


# Sentinel distinguishing "the removal raised" from any value it could return
# (``delete`` returns None, ``delete_prefix`` returns 0 for an absent subtree).
_FAILED = object()


def _guarded(action, key: str):
    """Run one store removal on ``key``, returning its result — or ``_FAILED``
    when it raised. A failure is logged and skipped so the sweep continues with
    the next artifact: one unreachable object must not strand the rest."""
    try:
        return action(key)
    except Exception:  # noqa: BLE001 — see _delete_artifacts' docstring
        logger.exception(
            "could not remove project artifact %s; leaving it for the "
            "retention sweep", key,
        )
        return _FAILED
