"""Logic behind the editor MCP tools.

Each function takes an open ``Session`` first and returns a JSON-serializable
dict. Projects are loaded and saved through the ``app.db.repositories.projects``
repository; edits go through the pure immutable helpers in
``app.editor.mutations``. Every mutation bumps the project version, re-persists
the serialized snapshot, and appends one row to the operation journal.

An ``EditorError`` (unknown project/element id, illegal split, wrong element
kind) is caught at the boundary and surfaced as ``{"error": str(exc)}``.

Client-agnostic by design: no niche/video/compilation lookups anywhere here.
``metadata`` is an opaque client-owned tag set the editor never interprets.
"""
from __future__ import annotations

import dataclasses
import hashlib
import subprocess
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy.orm import Session

from app.analysis.analyze import payload_with_keyframe_urls
from app.analysis.background import schedule_media_analysis
from app.analysis.peaks import MAX_EVENTS
from app.analysis.service import analyze_with_cache
from app.auth.principal import get_principal
from app.db.base import session_scope as default_session_scope
from app.captions.build_captions import add_auto_captions
from app.captions.mapping import DEFAULT_CAPTION_POS, DEFAULT_MAX_CHARS_PER_LINE
from app.config import DEFAULT_EXPORT_PRESET, EXPORT_PRESETS, get_settings
from app.db.models import ProjectStatus
from app.db.repositories import annotations as annotations_repo
from app.db.repositories import project_checkpoints as checkpoints_repo
from app.db.repositories import project_deletion
from app.db.repositories.projects import (
    get_history,
    list_projects,
    load_project,
    locked_status,
    reopen_project as _reopen,
    save_project,
    set_status,
)
from app.editor import mutations
from app.editor.errors import EditorError
from app.editor.fonts import FontPathResolver, with_resolved_font_paths
from app.editor.model import (
    TEXT_POSITIONS,
    AudioElement,
    EditorProject,
    MediaAsset,
    OverlayElement,
    TextElement,
    TextStyle,
    TransformKeyframe,
    VolumeKeyframe,
    metadata_from_dict,
    metadata_to_dict,
)
from app.editor.render import (
    probe_duration_sec,
    render_preview_frame,
    render_project_file,
    resolve_asset_media,
    resolve_project_media,
)
from app.editor.serialization import project_to_dict
from app.editor.validation import validate_project
from app.media.downloader import resolve_media_source
from app.media.fonts import ensure_local_font, is_font_filename
from app.storage import ArtifactStore, get_artifact_store
from app.tts.synthesize import TTS_VOICES_SUBDIR, ensure_voice, synthesize_caption

DEFAULT_ASPECT = (1080, 1920)
DEFAULT_FPS = 30
VIDEO_TRACK_TYPE = "video"
PREVIEWS_SUBDIR = "previews"
EXPORTS_SUBDIR = "exports"
VOICEOVER_SUBDIR = "voiceover"
# Where ``analyze_media`` parks clips it had to fetch itself (bare-``source``
# mode). One directory PER SOURCE, keyed by a hash of the source string,
# because ``resolve_media_source`` reuses whatever file it finds in the dest
# dir — a single shared cache dir would hand back the previous clip's file for
# every later URL.
ANALYSIS_CACHE_SUBDIR = "analysis_cache"
# Hex chars of the source digest used as that directory name. 16 is ample
# against accidental collision and keeps paths readable; hex only, so nothing
# caller-supplied ever reaches the filesystem.
_CACHE_KEY_CHARS = 16
_MS_PER_SEC = 1000

# Default Piper voice by the ``captions_lang`` metadata written by
# ``auto_captions``. Any other/missing language needs an explicit ``voice``.
DEFAULT_VOICE_BY_LANG = {
    "ru": "ru_RU-dmitri-medium",
    "en": "en_US-lessac-medium",
}
# Ducking: while a voiceover plays, a music bed dips to this fraction of its
# own volume, ramping over this many seconds on each side.
DUCK_FACTOR = 0.25
DUCK_RAMP_SEC = 0.2
# Float tolerance for merging touching/overlapping voiceover intervals.
_INTERVAL_EPSILON = 1e-6


# --- shared helpers ---------------------------------------------------------

def _require_project(session: Session, project_id: str, *, owner_id: int) -> EditorProject:
    project = load_project(session, project_id, owner_id=owner_id)
    if project is None:
        raise EditorError(f"Unknown project id: {project_id!r}")
    return project


def _find_asset(project: EditorProject, media_id: str):
    for asset in project.assets:
        if asset.id == media_id:
            return asset
    raise EditorError(f"Unknown media id: {media_id!r}")


def _summary(project: EditorProject) -> dict:
    """Compact project snapshot returned by every mutating tool."""
    return {
        "project_id": project.id,
        "version": project.version,
        "metadata": metadata_to_dict(project.metadata),
        "aspect": [project.aspect_w, project.aspect_h],
        "fps": project.fps,
        "target_sec": project.target_sec,
        "tracks": len(project.tracks),
        "elements": sum(len(track.elements) for track in project.tracks),
        "assets": len(project.assets),
    }


def _edit(
    session: Session,
    project_id: str,
    mutate: Callable[[EditorProject], EditorProject],
    *,
    owner_id: int,
    op_name: str,
    op_args: dict,
) -> dict:
    """Load → ``mutate`` → save (+ journal) → summary, catching ``EditorError``."""
    try:
        project = _require_project(session, project_id, owner_id=owner_id)
        updated = mutate(project)
        save_project(session, updated, owner_id=owner_id, op_name=op_name, op_args=op_args)
        return _summary(updated)
    except EditorError as exc:
        return {"error": str(exc)}


# --- fonts (R-33) -----------------------------------------------------------

def _font_resolver(session: Session, *, owner_id: int) -> FontPathResolver:
    """``font_id -> local font path``, for THIS caller only.

    The one place a typeface id becomes a path, and therefore the one place that
    decides what a text element is allowed to be rendered in. Exactly two things
    resolve:

    * ``None`` — the deployment's configured font. This is what every text
      element has always used and what a project written before R-33 means by
      carrying no id at all.
    * an asset on the CALLER's media-library shelf whose name is a font. The
      lookup is owner-scoped in SQL (``media_library.get_file``), so another
      tenant's id is simply absent.

    Everything else raises ``EditorError``, and every refusal carries the SAME
    message: an unknown id, another tenant's id and a library row that is not a
    font are indistinguishable. That is the point of R-33 — the old behaviour
    quietly substituted the default face, which let a caller tell "this path
    exists on the server" from "it does not" by looking at the render.
    """
    settings = get_settings()
    media_dir = Path(settings.media_dir)
    store: ArtifactStore | None = None

    def _resolve(font_id: str | None) -> str:
        nonlocal store
        if font_id is None:
            return settings.font_path
        # Lazy on purpose: the media library is a closed-core module the public
        # extract (E-01) does not ship. Without it every font id refuses with
        # the SAME message as an unknown id — the R-33 rule that no refusal may
        # reveal whether something exists server-side.
        try:
            from app.db.repositories import media_library as media_library_repo
        except ImportError:
            raise EditorError(f"Unknown font id: {font_id!r}") from None
        row = media_library_repo.get_file(session, font_id, owner_id=owner_id)
        if row is None or not is_font_filename(row.filename):
            raise EditorError(f"Unknown font id: {font_id!r}")
        # Built lazily: a project with no chosen font — the overwhelming
        # majority — must not pay for an object-store client on every edit.
        if store is None:
            store = get_artifact_store(settings)
        return str(
            ensure_local_font(
                store,
                media_dir=media_dir,
                owner_id=owner_id,
                file_id=row.id,
                filename=row.filename,
                storage_key=row.storage_key,
            )
        )

    return _resolve


def _with_resolved_fonts(
    session: Session, project: EditorProject, *, owner_id: int
) -> EditorProject:
    """The project with every text style's ``font_path`` recomputed from its
    ``font_id`` — applied on ingress AND before every render, so a stored path
    is never the thing handed to the rasterizer. See ``app.editor.fonts``."""
    return with_resolved_font_paths(project, _font_resolver(session, owner_id=owner_id))


def _resolved_font_path(session: Session, font_id: str | None, *, owner_id: int) -> str:
    """One font id resolved, for the tools that BUILD a style rather than
    normalize a whole project (``add_text``, ``auto_captions``)."""
    return _font_resolver(session, owner_id=owner_id)(font_id)


def _video_track_total(project: EditorProject) -> float:
    ends = [
        element.start_time + element.duration
        for track in project.tracks
        if track.type == VIDEO_TRACK_TYPE
        for element in track.elements
    ]
    return max(ends) if ends else 0.0


# --- lifecycle --------------------------------------------------------------

def create_project(
    session: Session,
    *,
    metadata: dict[str, str] | None = None,
    aspect: tuple[int, int] = DEFAULT_ASPECT,
    fps: int = DEFAULT_FPS,
    target_sec: float | None = None,
    min_clips: int = 1,
    max_clips: int | None = None,
    min_duration_sec: float = 0.0,
    max_duration_sec: float | None = None,
) -> dict:
    """Create and persist a fresh empty timeline project. Returns its summary."""
    owner_id = get_principal().user_id
    project = mutations.create_project(
        metadata=metadata,
        aspect=tuple(aspect),
        fps=fps,
        target_sec=target_sec,
        min_clips=min_clips,
        max_clips=max_clips,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
    )
    save_project(session, project, owner_id=owner_id, op_name="create_project", op_args={"metadata": metadata})
    return _summary(project)


def get_project(session: Session, project_id: str) -> dict:
    """Full serialized timeline project — the agent's 'eyes'."""
    owner_id = get_principal().user_id
    project = load_project(session, project_id, owner_id=owner_id)
    if project is None:
        return {"error": f"Unknown project id: {project_id!r}"}
    return project_to_dict(project)


def list_projects_tool(session: Session, metadata: dict[str, str] | None = None) -> list[dict]:
    """Project summaries, newest first, optionally filtered by ``metadata``."""
    owner_id = get_principal().user_id
    return [_summary(p) for p in list_projects(session, metadata_filter=metadata, owner_id=owner_id)]


def get_history_tool(session: Session, project_id: str) -> list[dict]:
    """The full append-only operation journal for a project, oldest first."""
    owner_id = get_principal().user_id
    return get_history(session, project_id, owner_id=owner_id)


def delete_project(
    session: Session,
    project_id: str,
    *,
    media_dir: Path | str | None = None,
    store: ArtifactStore | None = None,
) -> dict:
    """PERMANENTLY delete one of the caller's projects and everything it owns.

    A thin boundary over ``project_deletion.delete_project`` — the SAME
    function ``DELETE /api/projects/{id}`` calls, so the tool and the route
    cannot come to remove different things. What is deleted and, just as
    importantly, what is deliberately NOT (the owner's uploads and their media
    library, which belong to the USER and not to the project) is documented
    once, in that module.

    Errors follow this module's convention — ``{"error": ...}`` rather than an
    exception — for both refusals: an unknown project id (which is also the
    answer for another tenant's project, so this cannot probe which ids exist)
    and a project whose export is still rendering. Only the HTTP boundary needs
    to tell those apart, and it does so by TYPE (``ProjectExportingError`` →
    423), never by reading the message back out of this dict.

    ``media_dir``/``store`` exist for callers that already know where the
    artifacts live (the HTTP route, which holds both on ``app.state``, and the
    tests, which point at a temp tree). An explicit ``media_dir`` with no store
    implies the LOCAL backend rooted there — asking the configured factory in
    that case would hand back a store rooted somewhere else entirely — while
    supplying neither resolves both from settings, so an ``ARTIFACT_BACKEND=s3``
    deployment deletes out of its bucket over MCP exactly as it does over HTTP.
    """
    owner_id = get_principal().user_id
    settings = get_settings()
    if media_dir is None:
        media_dir = Path(settings.media_dir)
        if store is None:
            store = get_artifact_store(settings)
    try:
        report = project_deletion.delete_project(
            session, project_id, owner_id=owner_id,
            media_dir=Path(media_dir), store=store,
        )
    except EditorError as exc:  # ProjectExportingError among them
        return {"error": str(exc)}
    if report is None:
        return {"error": f"Unknown project id: {project_id!r}"}
    return {
        "project_id": report.project_id,
        "deleted": True,
        "operations_deleted": report.rows.operations,
        "annotations_deleted": report.rows.annotations,
        "checkpoints_deleted": report.rows.checkpoints,
        "files_deleted": report.files_deleted,
        "released_bytes": report.bytes_released,
    }


def reopen_project(session: Session, project_id: str) -> dict:
    """Unlock an EXPORTED project for further edits (generalizes 'a finished
    export is still recoverable' into an explicit agent action)."""
    owner_id = get_principal().user_id
    project = load_project(session, project_id, owner_id=owner_id)
    if project is None:
        return {"error": f"Unknown project id: {project_id!r}"}
    _reopen(session, project_id, owner_id=owner_id)
    return _summary(project)


def restore_snapshot(
    session: Session, project_id: str, *, data: dict, reason: str = "undo"
) -> dict:
    """Undo/redo: persist a previously serialized snapshot (a full
    ``project_to_dict``) as this project's NEXT version — see
    ``mutations.restore_snapshot``. The snapshot blob is deliberately NOT
    journaled (it is already the row's ``data``); only ``reason`` is recorded
    in ``op_args``.

    THE SNAPSHOT'S FONT PATHS ARE DISCARDED AND RECOMPUTED (R-33). This is the
    one op that carries a whole client-authored project, ``TextStyle`` included,
    and ``font_path`` used to travel from here into ``ImageFont.truetype``
    unchecked — an oracle for which files exist on this server. What survives is
    the ``font_id`` beside it, which resolves to the configured font or to one
    of this owner's own library assets and to nothing else; an id that resolves
    to neither refuses the whole restore rather than silently rendering in the
    default face."""
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda project: _with_resolved_fonts(
            session, mutations.restore_snapshot(project, data=data), owner_id=owner_id
        ),
        owner_id=owner_id,
        op_name="restore_snapshot",
        op_args={"reason": reason},
    )


# --- server-side version history (checkpoints) ------------------------------

def create_checkpoint_tool(session: Session, project_id: str) -> dict:
    """Freeze this project's CURRENT state as a version, or report that it is
    already frozen.

    The explicit half of the version history: there is no "save" button in the
    editor (every op persists immediately), so this is how an operator says
    "this state matters". Idempotent by mutation number — asking twice without
    editing in between returns the same version with ``created=False``, which
    is a no-op and not an error.
    """
    owner_id = get_principal().user_id
    result = checkpoints_repo.create_checkpoint(session, project_id, owner_id=owner_id)
    if result is None:
        return {"error": f"Unknown project id: {project_id!r}"}
    return {
        "project_id": project_id,
        "version": result.version,
        "created_at": result.created_at.isoformat(),
        "trigger": result.trigger,
        "created": result.created,
    }


def list_versions_tool(session: Session, project_id: str) -> list[dict]:
    """This project's checkpoints, newest first, each naming the op that
    produced the state it froze. Empty for a project that is not the caller's —
    absence, not refusal, like every other read here."""
    owner_id = get_principal().user_id
    entries = checkpoints_repo.list_checkpoints(session, project_id, owner_id=owner_id)
    return [
        {
            "version": entry.version,
            "created_at": entry.created_at.isoformat(),
            "trigger": entry.trigger,
            "op_name": entry.op_name,
            "op_args": entry.op_args,
        }
        for entry in entries
    ]


def restore_version(
    session: Session, project_id: str, *, version: int, reason: str = "restore"
) -> dict:
    """SERVER-SIDE rollback: load the checkpoint at mutation number ``version``
    and persist it as this project's NEXT version.

    The same semantics as the client-side ``restore_snapshot`` above — a
    rollback is a new version FORWARD, never a rewind of the counter — and
    deliberately the SAME mutation (``mutations.restore_snapshot``) rather than
    a second implementation of it. The only difference is where the document
    comes from: the caller's memory there, this server's history here.

    The op journal records which version was restored FROM
    (``restored_from_version``), so reading the history afterwards shows not
    just "a rollback happened" but where it went back to.
    """
    owner_id = get_principal().user_id
    found = checkpoints_repo.get_checkpoint(
        session, project_id, version, owner_id=owner_id
    )
    if found is None:
        return {"error": f"Unknown version {version} for project {project_id!r}"}
    _, data = found
    return _edit(
        session,
        project_id,
        lambda project: mutations.restore_snapshot(project, data=data),
        owner_id=owner_id,
        op_name="restore_version",
        op_args={"restored_from_version": version, "reason": reason},
    )


# --- annotations (operator error-markers on an exported timeline) -----------

def _annotation_dict(annotation: annotations_repo.Annotation) -> dict:
    return {
        "id": annotation.id,
        "project_id": annotation.project_id,
        "time_sec": annotation.time_sec,
        "element_id": annotation.element_id,
        "severity": annotation.severity,
        "note": annotation.note,
        "author_user_id": annotation.author_user_id,
        "resolved": annotation.resolved,
        "created_at": annotation.created_at.isoformat() if annotation.created_at else None,
    }


def add_annotation(
    session: Session,
    project_id: str,
    *,
    time_sec: float,
    note: str,
    severity: str = annotations_repo.DEFAULT_SEVERITY,
    element_id: str | None = None,
) -> dict:
    """Pin an error-marker to a project's timeline. Verifies the project exists,
    then delegates to the annotations repo (which bypasses the save_project
    EXPORTED lock — markers are placed after the render). Bad severity / empty
    note surface as ``{"error": ...}``."""
    owner_id = get_principal().user_id
    if load_project(session, project_id, owner_id=owner_id) is None:
        return {"error": f"Unknown project id: {project_id!r}"}
    try:
        annotation = annotations_repo.add_annotation(
            session,
            project_id=project_id,
            time_sec=time_sec,
            note=note,
            owner_id=owner_id,
            author_user_id=owner_id,
            severity=severity,
            element_id=element_id,
        )
    except ValueError as exc:
        return {"error": str(exc)}
    return _annotation_dict(annotation)


def list_annotations(session: Session, project_id: str, *, include_resolved: bool = False) -> list[dict]:
    """A project's error-markers, oldest first; resolved ones hidden unless
    ``include_resolved`` is set."""
    owner_id = get_principal().user_id
    return [
        _annotation_dict(a)
        for a in annotations_repo.list_annotations(
            session, project_id, owner_id=owner_id, include_resolved=include_resolved
        )
    ]


def resolve_annotation(session: Session, project_id: str, annotation_id: int) -> dict:
    """Mark one error-marker resolved. ``{"resolved": True}`` when a row was
    updated, ``{"error": ...}`` when the id is unknown ON THIS PROJECT.

    ``project_id`` is not decoration and must not be dropped as "redundant"
    because the repository also checks it: over HTTP this parameter is what
    causes the path's project to be supplied at all. ``_invoke_tool``
    (``app/web/api.py``) injects ``project_id`` only into tools that declare
    it, so before it was declared here the dispatcher passed the annotation id
    alone and any marker was resolvable from any project's endpoint.
    """
    owner_id = get_principal().user_id
    resolved = annotations_repo.resolve_annotation(
        session, annotation_id, project_id=project_id, owner_id=owner_id
    )
    if resolved:
        return {"resolved": True, "annotation_id": annotation_id}
    return {"error": f"Unknown annotation id: {annotation_id}"}


# --- media ------------------------------------------------------------------

def add_media(
    session: Session, project_id: str, source: str, *, duration_sec: float | None = None
) -> dict:
    """Register a media source (local path or URL) as a project asset.
    Duration is probed via ffprobe when not supplied. Returns the summary plus
    the new media_id."""
    owner_id = get_principal().user_id
    try:
        project = _require_project(session, project_id, owner_id=owner_id)
        resolved_duration = duration_sec if duration_sec is not None else probe_duration_sec(source)
        updated, media_id = mutations.add_media(project, source=source, duration_sec=resolved_duration)
        save_project(session, updated, owner_id=owner_id, op_name="add_media", op_args={"source": source})
        return {**_summary(updated), "media_id": media_id}
    except EditorError as exc:
        return {"error": str(exc)}


def set_media_duration(
    session: Session, project_id: str, media_id: str, *, duration_sec: float
) -> dict:
    """Persist a probed duration onto an already-registered asset (a journaled
    op that bumps the version). Used by the web resolve endpoint after it
    downloads a URL asset whose duration was unknown at ``add_media`` time, so
    the asset can then be placed on the timeline. Returns the summary."""
    owner_id = get_principal().user_id
    try:
        project = _require_project(session, project_id, owner_id=owner_id)
        updated = mutations.set_media_duration(
            project, media_id=media_id, duration_sec=duration_sec
        )
        save_project(
            session,
            updated,
            owner_id=owner_id,
            op_name="set_media_duration",
            op_args={"media_id": media_id, "duration_sec": duration_sec},
        )
        return {**_summary(updated), "media_id": media_id}
    except EditorError as exc:
        return {"error": str(exc)}


# --- edit -------------------------------------------------------------------

def _schedule_preview_proxy_if_available(**kwargs) -> None:
    """Kick off the background preview proxy — when the closed core is present.

    The proxy's contract is that NOTHING else can observe it (no journal entry,
    no version bump, renders always use the original), so its absence must be
    just as unobservable. The public extract (E-01) ships without
    ``app.media.proxy_jobs`` — quota-coupled, closed core — and ``add_clip``
    behaves identically there minus the background transcode.
    """
    try:
        from app.media import proxy_jobs
    except ImportError:
        return
    proxy_jobs.schedule_preview_proxy(**kwargs)


def add_video_track(session: Session, project_id: str) -> dict:
    """Add a new, empty V2+ upper video track (composites over V1). Returns the
    summary plus the new ``track_id`` (mirrors how ``add_clip`` returns
    ``element_id``)."""
    owner_id = get_principal().user_id
    try:
        project = _require_project(session, project_id, owner_id=owner_id)
        updated, track_id = mutations.add_video_track(project)
        save_project(session, updated, owner_id=owner_id, op_name="add_video_track", op_args={})
        return {**_summary(updated), "track_id": track_id}
    except EditorError as exc:
        return {"error": str(exc)}


def add_audio_track(session: Session, project_id: str) -> dict:
    """Add a new, empty audio track (mirrors ``add_video_track`` for audio —
    inserted after the last audio track). Returns the summary plus the new
    ``track_id``. This is the track a voiceover bed lands on so it never
    collides with the music bed on the first audio track."""
    owner_id = get_principal().user_id
    try:
        project = _require_project(session, project_id, owner_id=owner_id)
        updated, track_id = mutations.add_audio_track(project)
        save_project(session, updated, owner_id=owner_id, op_name="add_audio_track", op_args={})
        return {**_summary(updated), "track_id": track_id}
    except EditorError as exc:
        return {"error": str(exc)}


def remove_track(session: Session, project_id: str, track_id: str) -> dict:
    """Remove a track and its elements. Errors on an unknown track or the first
    (V1/main) video track (which cannot be removed)."""
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda p: mutations.remove_track(p, track_id),
        owner_id=owner_id,
        op_name="remove_track",
        op_args={"track_id": track_id},
    )


def add_clip(
    session: Session,
    project_id: str,
    media_id: str,
    *,
    track_type: str = VIDEO_TRACK_TYPE,
    start_time: float | None = None,
    duration: float | None = None,
    trim_start: float = 0.0,
    trim_end: float | None = None,
    track_id: str | None = None,
) -> dict:
    """Place a media asset on a track. ``duration`` defaults to the asset's own
    duration; ``start_time`` None auto-appends after the last element. Returns
    the summary plus the new element_id.

    ``track_id`` targets one exact, already-created track (a V2+ upper video
    track from ``add_video_track``); None keeps today's first-track-of-type
    behavior. An unknown ``track_id`` surfaces as ``{"error": ...}``."""
    owner_id = get_principal().user_id
    try:
        project = _require_project(session, project_id, owner_id=owner_id)
        asset = _find_asset(project, media_id)
        clip_duration = duration if duration is not None else asset.duration_sec
        if clip_duration is None:
            return {"error": f"media {media_id!r} has no known duration; pass duration"}
        updated, element_id = mutations.add_clip(
            project,
            media_id=media_id,
            track_type=track_type,
            start_time=start_time,
            duration=clip_duration,
            trim_start=trim_start,
            trim_end=trim_end,
            track_id=track_id,
        )
        save_project(
            session, updated,
            owner_id=owner_id,
            op_name="add_clip",
            op_args={
                "media_id": media_id, "duration": clip_duration,
                "start_time": start_time, "track_id": track_id,
            },
        )
        # Auto-analysis hangs off HERE, on the shared tool layer, rather than
        # off the MCP server and the HTTP /ops route separately — both funnel
        # through this function, so ONE hook covers both entry paths and the
        # two cannot drift apart. Fire-and-forget and never raises: a
        # scheduling problem must not turn a successful timeline edit into an
        # error. Media that already has a current-version stored analysis
        # costs one content hash and no decode.
        schedule_media_analysis(
            project_id=project_id,
            media_id=media_id,
            # Threaded explicitly, NOT read from the contextvar in the worker:
            # ThreadPoolExecutor does not copy contextvars, so get_principal()
            # would raise there (see app.auth.principal.bind_principal).
            owner_id=owner_id,
            media_dir=Path(get_settings().media_dir),
            # NOT the caller's session: the job runs on a pool thread, and a
            # SQLAlchemy Session must never cross a thread boundary.
            session_scope=default_session_scope,
        )
        # And, on the same hook and the same terms, the PREVIEW PROXY: a phone
        # original is ~15 Mbps, which the editor used to stream in full to draw a
        # 200 px stage (see app.media.preview_proxy). Same fire-and-forget
        # contract — the project never learns a proxy exists, and a failure just
        # leaves the preview on the original file.
        _schedule_preview_proxy_if_available(
            project_id=project_id,
            media_id=media_id,
            owner_id=owner_id,
            media_dir=Path(get_settings().media_dir),
            session_scope=default_session_scope,
        )
        return {**_summary(updated), "element_id": element_id}
    except EditorError as exc:
        return {"error": str(exc)}


def trim_clip(
    session: Session,
    project_id: str,
    element_id: str,
    *,
    trim_start: float | None = None,
    trim_end: float | None = None,
) -> dict:
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda p: mutations.trim_clip(p, element_id, trim_start=trim_start, trim_end=trim_end),
        owner_id=owner_id,
        op_name="trim_clip",
        op_args={"element_id": element_id, "trim_start": trim_start, "trim_end": trim_end},
    )


def resize_clip(
    session: Session,
    project_id: str,
    element_id: str,
    *,
    start_time: float | None = None,
    duration: float | None = None,
    trim_start: float | None = None,
    trim_end: float | None = None,
) -> dict:
    """Atomic placement + source-window edit (the GUI edge-drag gesture);
    ``None`` keeps the current value — see ``mutations.resize_clip``."""
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda p: mutations.resize_clip(
            p,
            element_id,
            start_time=start_time,
            duration=duration,
            trim_start=trim_start,
            trim_end=trim_end,
        ),
        owner_id=owner_id,
        op_name="resize_clip",
        op_args={
            "element_id": element_id,
            "start_time": start_time,
            "duration": duration,
            "trim_start": trim_start,
            "trim_end": trim_end,
        },
    )


def split_clip(session: Session, project_id: str, element_id: str, at_time: float) -> dict:
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda p: mutations.split_clip(p, element_id, at_time),
        owner_id=owner_id,
        op_name="split_clip",
        op_args={"element_id": element_id, "at_time": at_time},
    )


def cut_range(
    session: Session,
    project_id: str,
    element_id: str,
    *,
    start_time: float,
    end_time: float,
) -> dict:
    """Excise the middle range ``[start_time, end_time]`` from a clip in ONE
    atomic op (the "scissors" gesture) — the clip splits into two contiguous
    halves with the range removed. See ``mutations.cut_range``."""
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda p: mutations.cut_range(
            p, element_id, start_time=start_time, end_time=end_time
        ),
        owner_id=owner_id,
        op_name="cut_range",
        op_args={"element_id": element_id, "start_time": start_time, "end_time": end_time},
    )


def move_clip(
    session: Session,
    project_id: str,
    element_id: str,
    *,
    start_time: float,
    track_id: str | None = None,
) -> dict:
    """Move an element to a new absolute start time; ``track_id`` (optional)
    also moves it onto that exact track — the vertical lane-drag gesture. An
    unknown ``track_id`` surfaces as ``{"error": ...}``."""
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda p: mutations.move_clip(p, element_id, start_time=start_time, track_id=track_id),
        owner_id=owner_id,
        op_name="move_clip",
        op_args={"element_id": element_id, "start_time": start_time, "track_id": track_id},
    )


def delete_element(session: Session, project_id: str, element_id: str) -> dict:
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda p: mutations.delete_element(p, element_id),
        owner_id=owner_id,
        op_name="delete_element",
        op_args={"element_id": element_id},
    )


def set_transform(
    session: Session,
    project_id: str,
    element_id: str,
    *,
    scale: float | None = None,
    pos_x: float | None = None,
    pos_y: float | None = None,
    speed: float | None = None,
    crop_zoom: float | None = None,
    opacity: float | None = None,
) -> dict:
    """Update a clip's transform (scale/pos/speed/crop_zoom) and/or its
    composite ``opacity`` (0..1, for a V2+ upper-track clip). Only provided
    fields change."""
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda p: mutations.set_transform(
            p, element_id,
            scale=scale, pos_x=pos_x, pos_y=pos_y, speed=speed,
            crop_zoom=crop_zoom, opacity=opacity,
        ),
        owner_id=owner_id,
        op_name="set_transform",
        op_args={
            "element_id": element_id, "speed": speed,
            "crop_zoom": crop_zoom, "opacity": opacity,
        },
    )


def set_clip_audio(
    session: Session,
    project_id: str,
    element_id: str,
    *,
    muted: bool | None = None,
    volume: float | None = None,
    gain_db: float | None = None,
) -> dict:
    """Update a clip's (mute/volume) or a music bed's (volume/gain_db) audio
    settings — see ``mutations.set_clip_audio`` for the per-kind field
    rejection rules (``gain_db`` is music-bed only, ``muted`` is clip only)."""
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda p: mutations.set_clip_audio(p, element_id, muted=muted, volume=volume, gain_db=gain_db),
        owner_id=owner_id,
        op_name="set_clip_audio",
        op_args={"element_id": element_id, "muted": muted, "volume": volume, "gain_db": gain_db},
    )


# --- audio envelope + canvas fit (phase 5 wave 2) ---------------------------

def _volume_keyframes_from_payload(
    data: list[dict] | None,
) -> tuple[VolumeKeyframe, ...] | None:
    """MCP boundary conversion: a JSON list of {"time":..,"volume":..} dicts
    into the internal ``VolumeKeyframe`` tuple. ``None`` means "leave the
    element's existing keyframes untouched" (mirrors ``set_audio_envelope``'s
    own only-provided-fields-change contract) — an EXPLICIT empty list clears
    them."""
    if data is None:
        return None
    return tuple(VolumeKeyframe(time=kf["time"], volume=kf["volume"]) for kf in data)


def set_audio_envelope(
    session: Session,
    project_id: str,
    element_id: str,
    *,
    fade_in_sec: float | None = None,
    fade_out_sec: float | None = None,
    volume_keyframes: list[dict] | None = None,
) -> dict:
    """Set a fade-in/fade-out and/or a volume-keyframe envelope on a clip's
    own audio or a music bed (mlt engine only — the ffmpeg engine rejects a
    project using this at export/render time with a clear error). Only
    provided fields change; ``volume_keyframes`` is
    ``[{"time": t, "volume": v}, ...]``, replacing the whole envelope."""
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda p: mutations.set_audio_envelope(
            p, element_id,
            fade_in_sec=fade_in_sec, fade_out_sec=fade_out_sec,
            volume_keyframes=_volume_keyframes_from_payload(volume_keyframes),
        ),
        owner_id=owner_id,
        op_name="set_audio_envelope",
        op_args={
            "element_id": element_id, "fade_in_sec": fade_in_sec,
            "fade_out_sec": fade_out_sec, "volume_keyframes": volume_keyframes,
        },
    )


def set_fit(session: Session, project_id: str, element_id: str, *, fit: str) -> dict:
    """Set a clip's canvas fill mode. ``"cover"`` (default) is today's
    fill-scale + center-crop; ``"contain_blur"`` letterboxes the source at
    full extent over a blurred, cover-scaled copy of itself (mlt engine
    only). Must be one of ``app.editor.model.FIT_VALUES``."""
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda p: mutations.set_fit(p, element_id, fit=fit),
        owner_id=owner_id,
        op_name="set_fit",
        op_args={"element_id": element_id, "fit": fit},
    )


def set_cover(session: Session, project_id: str, *, at_time: float) -> dict:
    """Set the absolute timeline time exported as the project's cover PNG
    alongside the mp4 on the next ``export`` call."""
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda p: mutations.set_cover(p, at_time=at_time),
        owner_id=owner_id,
        op_name="set_cover",
        op_args={"at_time": at_time},
    )


def auto_captions(
    session: Session,
    project_id: str,
    *,
    style_overrides: dict | None = None,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
) -> dict:
    """Transcribe every video clip's own trimmed audio window (speech
    recognition, lazy — degrades to a clear error when the 'asr' extra/model is
    unavailable) and add the result as caption TextElements (bottom position
    by default; ``style_overrides`` merges over the defaults, e.g.
    ``{"size": 48, "color": "#FFFF00"}``). Long segments are split at
    ``max_chars_per_line``. Returns the summary plus ``captions_added``."""
    owner_id = get_principal().user_id
    try:
        project = _require_project(session, project_id, owner_id=owner_id)
        settings = get_settings()
        media_dir = Path(settings.media_dir)
        models_dir = Path(settings.models_dir)
        resolved_media = resolve_project_media(project, media_dir)
        # ``font_path`` is never one of the overrides a caller may send (see
        # ``app.mcp.argspec.CaptionStyleArgs``); ``font_id`` is, and the path is
        # derived from it here so the captions carry the same server-resolved
        # pairing every other text element does (R-33).
        overrides = dict(style_overrides or {})
        font_id = overrides.pop("font_id", None)
        overrides.setdefault("pos", DEFAULT_CAPTION_POS)
        style = TextStyle(
            font_path=_resolved_font_path(session, font_id, owner_id=owner_id),
            font_id=font_id,
            **overrides,
        )
        updated, added, language = add_auto_captions(
            project,
            resolved_media=resolved_media,
            media_dir=media_dir,
            models_dir=models_dir,
            asr_model_size=settings.asr_model_size,
            asr_device=settings.asr_device,
            asr_compute_type=settings.asr_compute_type,
            style=style,
            max_chars_per_line=max_chars_per_line,
        )
        # Record the detected language in the project's opaque metadata (key
        # ``captions_lang``) so a later translation pass can read it back. Folded
        # into this same snapshot — no extra version bump, one journaled op.
        if language is not None:
            meta = metadata_to_dict(updated.metadata)
            meta["captions_lang"] = language
            updated = dataclasses.replace(updated, metadata=metadata_from_dict(meta))
        save_project(
            session, updated, owner_id=owner_id, op_name="auto_captions",
            op_args={
                "captions_added": added,
                "max_chars_per_line": max_chars_per_line,
                "language": language,
            },
        )
        return {**_summary(updated), "captions_added": added, "language": language}
    except EditorError as exc:
        return {"error": str(exc)}


# --- clip event detection ("moments of meaning") ----------------------------

def _analysis_cache_dir(media_dir: Path, source: str) -> Path:
    """A per-source download directory under ``media_dir/analysis_cache/``.
    The name is a digest of ``source`` — stable across runs (so a re-analysis
    reuses the download) and hex-only (so no caller input reaches the path)."""
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:_CACHE_KEY_CHARS]
    return media_dir / ANALYSIS_CACHE_SUBDIR / digest


def _resolve_analysis_source(
    session: Session, media_dir: Path, source: str | None, project_id: str | None,
    media_id: str | None, *, owner_id: int,
) -> Path:
    """The local file to analyse, for whichever of the two addressing modes the
    caller used. Raises ``EditorError`` on an unusable/ambiguous request."""
    if (source is None) == (media_id is None):
        raise EditorError("pass either source or project_id+media_id")
    if source is not None:
        resolved = resolve_media_source(source, _analysis_cache_dir(media_dir, source))
        if resolved is None:
            raise EditorError(f"Could not resolve media source {source!r}")
        return resolved
    if project_id is None:
        raise EditorError("project_id is required alongside media_id")
    project = _require_project(session, project_id, owner_id=owner_id)
    return resolve_asset_media(_find_asset(project, media_id), media_dir)


def analyze_media(
    session: Session,
    source: str | None = None,
    project_id: str | None = None,
    media_id: str | None = None,
    *,
    max_events: int = MAX_EVENTS,
    store_scope: Callable | None = None,
) -> dict:
    """Detect the "moments of meaning" in one media file and write keyframe
    strips for them. Deterministic (motion + audio math, no LLM); reads only —
    the project snapshot and its version are untouched.

    Two addressing modes, exactly one of which must be used:

    - ``source``: a local path or an http(s) URL, fetched into a per-source
      cache under ``media_dir/analysis_cache/``. For agents and the
      annotation pipeline, which hold a URL and no editor project.
    - ``project_id`` + ``media_id``: an asset already registered on a project
      (already downloaded by ``add_media``). For the web editor.

    Returns the ``"version": 2`` analysis contract: ``version, duration_sec,
    fps, events, cuts, cut_count, clip_type, payoff_at, keyframes, agent``.
    Each event is ``{t, motion_z, audio_z, combined, score, kind}``.

    RANK ON ``score``, NEVER ON ``combined``. ``score`` is the normalised
    strength that selected the event and the only cross-event comparable
    field; ``combined`` is the raw geometric mean and is legitimately ~0 on a
    solo-detected event (seen OR heard, not both at once — v2 finds these,
    v1 did not). ``keyframes`` are ``/ui/media/analysis/...`` URLs. The
    ``agent`` block is all-nulls — only a live vision session ever fills it in.
    """
    owner_id = get_principal().user_id
    try:
        media_dir = Path(get_settings().media_dir)
        # Resolve FIRST, hash second: the store is keyed by a digest of the
        # media's bytes, and a URL has no bytes to hash until the download has
        # landed. Resolving first is also what makes a URL and its downloaded
        # local copy collapse onto ONE stored record instead of two.
        path = _resolve_analysis_source(
            session, media_dir, source, project_id, media_id, owner_id=owner_id
        )
        outcome = analyze_with_cache(
            path,
            media_dir=media_dir,
            session_scope=_store_scope(session, store_scope),
            owner_id=owner_id,
            max_events=max_events,
        )
        # The payload, with ``keyframes`` rendered back into URLs —
        # ``from_cache``/``content_hash`` stay off the wire because the
        # the external annotation client reads this
        # documented contract. Step 10 moved STORAGE to basenames; this
        # contract still carries URLs, so the rendering happens here rather
        # than letting a storage change leak into an external consumer.
        return payload_with_keyframe_urls(outcome.data)
    except EditorError as exc:
        return {"error": str(exc)}


@contextmanager
def _session_as_scope(session: Session):
    """Present an already-open ``Session`` as a scope the store can use.

    The MCP/CLI path arrives with a live session for the whole tool call, so
    the cache read/write joins that transaction rather than opening a second
    connection to the same SQLite file — which under WAL would be a needless
    writer contending with its own caller.
    """
    yield session
    session.commit()


def _store_scope(session: Session | None, store_scope: Callable | None) -> Callable:
    """Which scope the analysis store should use, in precedence order:

    1. an explicitly injected ``store_scope`` — the HTTP path, which
       deliberately calls this tool with NO session so the decode holds no
       connection, and the tests, which point at a temp database;
    2. the caller's own open session (MCP/CLI);
    3. the process-wide scope, as a last resort.
    """
    if store_scope is not None:
        return store_scope
    if session is not None:
        return lambda: _session_as_scope(session)
    return default_session_scope


# --- voiceover (TTS over the auto-captions) ---------------------------------

def _caption_elements(project: EditorProject) -> list[TextElement]:
    """Every ``role=="caption"`` text element, in timeline order."""
    captions = [
        element
        for track in project.tracks
        for element in track.elements
        if isinstance(element, TextElement) and element.role == "caption"
    ]
    return sorted(captions, key=lambda element: element.start_time)


def _resolve_voice_name(project: EditorProject, voice: str | None) -> str:
    """The explicit ``voice``, or the default for the project's detected
    ``captions_lang``. An unknown/missing language with no explicit voice is an
    error (we will not guess a voice for an unrecognized language)."""
    if voice is not None:
        return voice
    lang = metadata_to_dict(project.metadata).get("captions_lang")
    resolved = DEFAULT_VOICE_BY_LANG.get(lang)
    if resolved is None:
        raise EditorError(
            f"no default TTS voice for captions language {lang!r}; pass an "
            "explicit voice (e.g. 'en_US-lessac-medium')"
        )
    return resolved


def _min_optional(*values: float | None) -> float | None:
    """The min of the non-None values, or None when all are None."""
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _find_audio_element(project: EditorProject, element_id: str) -> AudioElement | None:
    for track in project.tracks:
        for element in track.elements:
            if isinstance(element, AudioElement) and element.id == element_id:
                return element
    return None


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping/adjacent ``(start, stop)`` spans so the ducking
    envelope built from them stays monotonic and sane."""
    merged: list[tuple[float, float]] = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1] + _INTERVAL_EPSILON:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def _duck_keyframes(
    music: AudioElement, intervals: list[tuple[float, float]]
) -> tuple[VolumeKeyframe, ...]:
    """Volume keyframes that dip ``music`` to ``DUCK_FACTOR`` of its own volume
    during each (already-merged) voiceover interval, ramping over
    ``DUCK_RAMP_SEC`` on each side. Times are LOCAL to the music element's own
    ``start_time`` and clamped to its ``[0, duration]`` window (see
    ``VolumeKeyframe``). A plain dict keyed by time keeps the points de-duped
    and, once sorted, strictly monotonic."""
    base = music.volume
    dip = DUCK_FACTOR * base
    duration = float(music.duration)
    points: dict[float, float] = {0.0: base, duration: base}
    for start, stop in intervals:
        local_start = start - music.start_time
        local_stop = stop - music.start_time
        if local_stop <= 0 or local_start >= duration:
            continue
        for time, volume in (
            (local_start - DUCK_RAMP_SEC, base),
            (local_start, dip),
            (local_stop, dip),
            (local_stop + DUCK_RAMP_SEC, base),
        ):
            points[min(max(time, 0.0), duration)] = volume
    return tuple(
        VolumeKeyframe(time=time, volume=volume)
        for time, volume in sorted(points.items())
    )


def generate_voiceover(
    session: Session,
    project_id: str,
    *,
    voice: str | None = None,
    duck_music: bool = True,
) -> dict:
    """Synthesize a local TTS voiceover for every auto-caption and lay it on a
    dedicated new audio track, ducking any existing music bed underneath.

    Collects ``role=="caption"`` text elements in timeline order (run
    ``auto_captions`` first — none is an error). The voice defaults by the
    project's ``captions_lang`` (``ru`` / ``en``); any other language needs an
    explicit ``voice``. Each caption is synthesized to
    ``{media_dir}/voiceover/{project_id}/{caption_id}.wav`` fitted to its
    available slot (the shorter of its own on-screen duration and the gap to the
    next caption), placed at the caption's ``start_time`` at full volume on ONE
    new audio track. A wav that still overruns the gap to the next caption is
    clamped to fit (audio overlap is barred per track); ``fitted`` counts every
    caption squeezed by either the synth refit or that clamp.

    ``duck_music`` (default True) dips each pre-existing music bed to
    ``DUCK_FACTOR`` of its volume during the voiceover with ``DUCK_RAMP_SEC``
    ramps. All of this is ONE journaled op. Returns the summary plus
    ``voiceover_added`` / ``fitted`` / ``track_id``."""
    owner_id = get_principal().user_id
    try:
        project = _require_project(session, project_id, owner_id=owner_id)
        captions = _caption_elements(project)
        if not captions:
            raise EditorError(
                "no caption text elements found — run auto_captions first"
            )
        voice_name = _resolve_voice_name(project, voice)
        settings = get_settings()
        media_dir = Path(settings.media_dir)
        models_dir = Path(settings.models_dir)
        voice_path = ensure_voice(voice_name, models_dir / TTS_VOICES_SUBDIR)
        out_dir = media_dir / VOICEOVER_SUBDIR / project_id

        # Snapshot the music beds present BEFORE we add the voiceover track, so
        # ducking touches only pre-existing beds — never the voiceover itself.
        pre_existing_audio_ids = [
            element.id
            for track in project.tracks
            for element in track.elements
            if isinstance(element, AudioElement)
        ]

        updated, track_id = mutations.add_audio_track(project)

        fitted = 0
        intervals: list[tuple[float, float]] = []
        count = len(captions)
        for index, caption in enumerate(captions):
            gap = (
                captions[index + 1].start_time - caption.start_time
                if index + 1 < count
                else None
            )
            target = _min_optional(caption.duration, gap)
            out_wav = out_dir / f"{caption.id}.wav"
            _wav, wav_duration, synth_fitted = synthesize_caption(
                caption.content, voice_path, out_wav, target_sec=target
            )
            place_duration = wav_duration
            clamped = False
            if gap is not None and wav_duration > gap:
                place_duration = gap
                clamped = True
            if synth_fitted or clamped:
                fitted += 1
            updated, media_id = mutations.add_media(
                updated, source=str(out_wav), duration_sec=wav_duration
            )
            updated, _element_id = mutations.add_audio_clip(
                updated,
                media_id=media_id,
                track_id=track_id,
                start_time=caption.start_time,
                duration=place_duration,
                volume=1.0,
                gain_db=0.0,
            )
            intervals.append((caption.start_time, caption.start_time + place_duration))

        if duck_music and pre_existing_audio_ids and intervals:
            merged = _merge_intervals(intervals)
            for audio_id in pre_existing_audio_ids:
                music = _find_audio_element(updated, audio_id)
                if music is None:
                    continue
                updated = mutations.set_audio_envelope(
                    updated, audio_id, volume_keyframes=_duck_keyframes(music, merged)
                )

        save_project(
            session,
            updated,
            owner_id=owner_id,
            op_name="generate_voiceover",
            op_args={
                "voice": voice_name,
                "count": count,
                "fitted": fitted,
                "duck_music": duck_music,
            },
        )
        return {
            **_summary(updated),
            "voiceover_added": count,
            "fitted": fitted,
            "track_id": track_id,
        }
    except EditorError as exc:
        return {"error": str(exc)}


# --- layers -----------------------------------------------------------------

def _validate_text_placement(pos: str, pos_x: float | None, pos_y: float | None) -> str | None:
    """pos is a named preset; pos_x/pos_y override an axis with a normalized
    0..1 fraction of the frame — reject anything outside that contract."""
    if pos not in TEXT_POSITIONS:
        return f"pos must be one of {list(TEXT_POSITIONS)}, got {pos!r}"
    for name, value in (("pos_x", pos_x), ("pos_y", pos_y)):
        if value is not None and not 0.0 <= value <= 1.0:
            return f"{name} must be a 0..1 fraction of the frame, got {value!r}"
    return None


def add_text(
    session: Session,
    project_id: str,
    content: str,
    start: float,
    duration: float | None = None,
    *,
    pos: str = "center",
    size: int = 64,
    color: str = "#FFFFFF",
    pos_x: float | None = None,
    pos_y: float | None = None,
    role: str | None = None,
    font_id: str | None = None,
) -> dict:
    """Add a text overlay (hook title / caption / board slot). ``duration`` None
    = persistent header: hold from ``start`` to the end of the timeline.

    ``font_id`` picks the typeface: omitted/None = the service's configured
    font, otherwise the id of a ``.ttf``/``.otf`` on YOUR media-library shelf
    (``GET /api/media``). A path is never accepted — the server resolves the id
    to one, which is what keeps this field from being a file-existence oracle
    (R-33). An id that is not one of your own fonts is refused.

    Placement: ``pos`` is a named preset (``top``/``center``/``bottom``, always
    horizontally centred). ``pos_x``/``pos_y`` override an axis with a
    normalized 0..1 fraction of the frame anchored at the text's top-left
    corner. ``\\n`` in ``content`` forces a line break.

    ``role`` optionally marks the element's origin (e.g. ``"caption"``);
    ``None`` = unmarked.

    Returns the summary plus the new element_id.
    """
    owner_id = get_principal().user_id
    try:
        project = _require_project(session, project_id, owner_id=owner_id)
        placement_error = _validate_text_placement(pos, pos_x, pos_y)
        if placement_error:
            return {"error": placement_error}
        style = TextStyle(
            font_path=_resolved_font_path(session, font_id, owner_id=owner_id),
            font_id=font_id,
            size=size,
            color=color,
            pos=pos,
            pos_x=pos_x,
            pos_y=pos_y,
        )
        updated, element_id = mutations.add_text(
            project, content=content, start_time=start, duration=duration,
            style=style, role=role,
        )
        save_project(
            session, updated, owner_id=owner_id, op_name="add_text",
            op_args={
                "content": content, "start": start, "duration": duration,
                "role": role, "font_id": font_id,
            },
        )
        return {**_summary(updated), "element_id": element_id}
    except EditorError as exc:
        return {"error": str(exc)}


def update_text(
    session: Session,
    project_id: str,
    element_id: str,
    *,
    content: str | None = None,
    start_time: float | None = None,
    duration: float | None | object = mutations.UNSET,
    size: int | None = None,
    color: str | None = None,
    pos: str | None = None,
    pos_x: float | None | object = mutations.UNSET,
    pos_y: float | None | object = mutations.UNSET,
    font_id: str | None | object = mutations.UNSET,
) -> dict:
    """Partial in-place text edit (id stable) — see ``mutations.update_text``.
    Omitted args keep the current value; explicit ``null`` clears where that is
    meaningful (``duration`` → persistent, ``pos_x``/``pos_y`` → preset,
    ``font_id`` → the service's configured font).

    ``font_id`` is the id of a ``.ttf``/``.otf`` on YOUR media-library shelf.
    The server resolves it to a path; a path is never accepted from a caller
    (R-33), and an id that is not one of your own fonts is refused."""
    owner_id = get_principal().user_id
    if pos is not None and pos not in TEXT_POSITIONS:
        return {"error": f"pos must be one of {list(TEXT_POSITIONS)}, got {pos!r}"}
    for name, value in (("pos_x", pos_x), ("pos_y", pos_y)):
        if value is not mutations.UNSET and value is not None and not 0.0 <= value <= 1.0:
            return {"error": f"{name} must be a 0..1 fraction of the frame, got {value!r}"}
    journaled = {
        "element_id": element_id,
        "content": content,
        "start_time": start_time,
        "size": size,
        "color": color,
        "pos": pos,
    }
    # Resolved OUTSIDE the mutation lambda so an unusable font refuses before
    # anything is written, and paired with the id it came from — the mutation
    # layer is pure and cannot look one up (see ``mutations.update_text``).
    try:
        font_path = (
            None if font_id is mutations.UNSET
            else _resolved_font_path(session, font_id, owner_id=owner_id)
        )
    except EditorError as exc:
        return {"error": str(exc)}
    return _edit(
        session,
        project_id,
        lambda p: mutations.update_text(
            p,
            element_id,
            content=content,
            start_time=start_time,
            duration=duration,
            size=size,
            color=color,
            pos=pos,
            pos_x=pos_x,
            pos_y=pos_y,
            font_id=font_id,
            font_path=font_path,
        ),
        owner_id=owner_id,
        op_name="update_text",
        op_args={k: v for k, v in journaled.items() if v is not None},
    )


def update_texts(
    session: Session,
    project_id: str,
    *,
    updates: list[dict],
    reason: str | None = None,
) -> dict:
    """Atomic batch content-edit of many text elements in ONE journaled op —
    see ``mutations.update_texts``. Each item is
    ``{"element_id": str, "content": str}``; an unknown id or non-text element
    fails the WHOLE batch (nothing applied). The journal stays lean: only the
    edit ``count`` and an optional ``reason`` are recorded — the new text lives
    in the snapshot, not the op log."""
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda p: mutations.update_texts(p, updates=updates),
        owner_id=owner_id,
        op_name="update_texts",
        op_args={"count": len(updates), "reason": reason},
    )


def update_overlay(
    session: Session,
    project_id: str,
    element_id: str,
    *,
    start_time: float | None = None,
    duration: float | None = None,
    x: float | None = None,
    y: float | None = None,
    w: float | None = None,
    h: float | None = None,
    opacity: float | None = None,
) -> dict:
    """Partial in-place overlay edit (id stable) — the canvas-drag path; see
    ``mutations.update_overlay``. Bounds mirror ``_validate_overlay_rect`` but
    only for the values actually provided."""
    owner_id = get_principal().user_id
    for name, value in (("x", x), ("y", y), ("w", w), ("h", h), ("opacity", opacity)):
        if value is not None and not 0.0 <= value <= 1.0:
            return {"error": f"{name} must be a 0..1 fraction of the frame, got {value!r}"}
    if (w is not None and w <= 0.0) or (h is not None and h <= 0.0):
        return {"error": f"w and h must be positive, got w={w!r} h={h!r}"}
    provided = {
        "element_id": element_id,
        "start_time": start_time,
        "duration": duration,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "opacity": opacity,
    }

    def _mutate(p: EditorProject) -> EditorProject:
        # Only when duration is actually being changed: growing an overlay past
        # its source end makes it VANISH, not freeze (see
        # ``_validate_overlay_duration``). The asset is resolved through the
        # ELEMENT's own media_id, which is the only thing tying this edit to a
        # source. ``_edit`` turns the raised EditorError into {"error": ...}.
        if duration is not None:
            _validate_overlay_duration(_find_asset(p, _find_overlay(p, element_id).media_id), duration)
        return mutations.update_overlay(
            p,
            element_id,
            start_time=start_time,
            duration=duration,
            x=x,
            y=y,
            w=w,
            h=h,
            opacity=opacity,
        )

    return _edit(
        session,
        project_id,
        _mutate,
        owner_id=owner_id,
        op_name="update_overlay",
        op_args={k: v for k, v in provided.items() if v is not None},
    )


def add_music(
    session: Session,
    project_id: str,
    source: str,
    *,
    volume: float = 0.15,
    start: float = 0.0,
    duration: float | None = None,
) -> dict:
    """Register ``source`` as the music bed and place it on the audio track.
    ``duration`` defaults to the video-track total (so music spans the edit).
    Returns the summary plus the media_id and the new element_id."""
    owner_id = get_principal().user_id
    try:
        project = _require_project(session, project_id, owner_id=owner_id)
        resolved_duration = duration if duration is not None else probe_duration_sec(source)
        with_asset, media_id = mutations.add_media(project, source=source, duration_sec=resolved_duration)
        music_duration = duration if duration is not None else _video_track_total(with_asset)
        if not music_duration:
            return {"error": "cannot infer music duration; add clips first or pass duration"}
        updated, element_id = mutations.add_music(
            with_asset, media_id=media_id, start_time=start, duration=music_duration, volume=volume
        )
        save_project(
            session, updated, owner_id=owner_id, op_name="add_music",
            op_args={"source": source, "start": start},
        )
        return {**_summary(updated), "media_id": media_id, "element_id": element_id}
    except EditorError as exc:
        return {"error": str(exc)}


def add_transition(
    session: Session,
    project_id: str,
    to_element: str,
    *,
    kind: str = "dissolve",
    duration: float = 0.5,
) -> dict:
    """Cross-dissolve from ``to_element``'s immediate predecessor on the same
    video track into it (see ``mutations.add_transition`` /
    ``model.TransitionSpec`` for the exact overlap semantics — ``duration``
    consumes the last ``duration`` seconds of the predecessor and the first
    ``duration`` seconds of ``to_element``, dissolving between them). No
    ``from_element`` argument: the predecessor is derived from the pooled,
    start_time-ordered video track, so an agent cannot name a non-adjacent
    pair by mistake. MLT-engine only — rendering on the ffmpeg engine raises.
    """
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda p: mutations.add_transition(p, to_element, kind=kind, duration=duration),
        owner_id=owner_id,
        op_name="add_transition",
        op_args={"to_element": to_element, "kind": kind, "duration": duration},
    )


def remove_transition(session: Session, project_id: str, element_id: str) -> dict:
    """Clear ``element_id``'s ``transition_in`` — the off switch for
    ``add_transition`` (see ``mutations.remove_transition``)."""
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda p: mutations.remove_transition(p, element_id),
        owner_id=owner_id,
        op_name="remove_transition",
        op_args={"element_id": element_id},
    )


def set_keyframes(
    session: Session,
    project_id: str,
    element_id: str,
    *,
    keyframes: list[dict],
) -> dict:
    """Replace a clip's animated-transform keyframes. Each item in
    ``keyframes`` is a dict with a required ``time`` (seconds relative to the
    CLIP's own start, 0 = its first frame) and optional ``scale``/``pos_x``/
    ``pos_y``/``opacity``/``rotation`` (see ``model.TransformKeyframe`` for
    defaults). An empty list clears them (falls back to the clip's static
    transform). MLT-engine only."""
    owner_id = get_principal().user_id
    try:
        parsed = tuple(TransformKeyframe(**kf) for kf in keyframes)
    except TypeError as exc:
        return {"error": f"invalid keyframe: {exc}"}
    return _edit(
        session,
        project_id,
        lambda p: mutations.set_keyframes(p, element_id, keyframes=parsed),
        owner_id=owner_id,
        op_name="set_keyframes",
        op_args={"element_id": element_id, "keyframes": keyframes},
    )


def set_color(
    session: Session,
    project_id: str,
    element_id: str,
    *,
    brightness: float | None = None,
    contrast: float | None = None,
    saturation: float | None = None,
    gamma: float | None = None,
) -> dict:
    """Update a clip's color grade (MLT ``avfilter.eq``). Only provided
    fields change. MLT-engine only."""
    owner_id = get_principal().user_id
    return _edit(
        session,
        project_id,
        lambda p: mutations.set_color(
            p, element_id, brightness=brightness, contrast=contrast,
            saturation=saturation, gamma=gamma,
        ),
        owner_id=owner_id,
        op_name="set_color",
        op_args={
            "element_id": element_id, "brightness": brightness, "contrast": contrast,
            "saturation": saturation, "gamma": gamma,
        },
    )


def _validate_overlay_rect(x: float, y: float, w: float, h: float, opacity: float) -> str | None:
    """``x``/``y``/``w``/``h``/``opacity`` are normalized [0,1] fractions of
    the frame (mirrors ``_validate_text_placement``'s pos_x/pos_y contract);
    ``w``/``h`` must additionally be positive — a zero/negative-sized
    overlay is not a valid rect, not merely an unusual one."""
    for name, value in (("x", x), ("y", y), ("w", w), ("h", h), ("opacity", opacity)):
        if not 0.0 <= value <= 1.0:
            return f"{name} must be a 0..1 fraction of the frame, got {value!r}"
    if w <= 0.0 or h <= 0.0:
        return f"w and h must be positive, got w={w!r} h={h!r}"
    return None


# Absorbs ffprobe rounding so a "same length" overlay does not trip its own guard.
_OVERLAY_DURATION_EPSILON = 0.05


def _validate_overlay_duration(asset: MediaAsset, duration: float) -> None:
    """An overlay longer than its source does not freeze on its last frame —
    it DISAPPEARS (measured 2026-07-20: a 1.2s sticker in a 2.5s slot renders
    0 sticker pixels from 1.6s on, despite eof="pause" in the producer XML).
    Reject rather than silently ship a gap. Assets with an unknown duration
    (still images probe to None) are exempt — they hold indefinitely."""
    if asset.duration_sec is not None and duration > asset.duration_sec + _OVERLAY_DURATION_EPSILON:
        raise EditorError(
            f"Overlay duration {duration}s exceeds its source's "
            f"{asset.duration_sec}s — it would vanish partway through. "
            "Shorten the overlay or use a longer source."
        )


def _find_overlay(project: EditorProject, element_id: str) -> OverlayElement:
    for track in project.tracks:
        for element in track.elements:
            if isinstance(element, OverlayElement) and element.id == element_id:
                return element
    raise EditorError(f"Element {element_id!r} is not an overlay")


def add_overlay(
    session: Session,
    project_id: str,
    media_id: str,
    *,
    start_time: float | None = None,
    duration: float,
    x: float,
    y: float,
    w: float,
    h: float,
    opacity: float = 1.0,
) -> dict:
    """Place a PiP/sticker overlay (image or video, registered via
    ``editor_add_media`` like any other source) on the ``overlay`` track.
    ``x``/``y``/``w``/``h`` are normalized [0,1] frame fractions (top-left
    anchored); ``start_time`` None auto-appends after the last overlay.
    MLT-engine only. Returns the summary plus the new element_id."""
    owner_id = get_principal().user_id
    try:
        project = _require_project(session, project_id, owner_id=owner_id)
        asset = _find_asset(project, media_id)
        placement_error = _validate_overlay_rect(x, y, w, h, opacity)
        if placement_error:
            return {"error": placement_error}
        _validate_overlay_duration(asset, duration)
        updated, element_id = mutations.add_overlay(
            project, media_id=media_id, start_time=start_time, duration=duration,
            x=x, y=y, w=w, h=h, opacity=opacity,
        )
        save_project(
            session, updated, owner_id=owner_id, op_name="add_overlay",
            op_args={"media_id": media_id, "start_time": start_time, "duration": duration},
        )
        return {**_summary(updated), "element_id": element_id}
    except EditorError as exc:
        return {"error": str(exc)}


# --- verify / export --------------------------------------------------------

def render_preview(session: Session, project_id: str, at_time: float = 0.0) -> dict:
    """Render a single PNG frame of the timeline at ``at_time`` so the agent can
    'see' its edit before export. Writes under ``media_dir/previews/`` and
    returns the file path. Render/validation failures surface as
    ``{"error": ...}``."""
    owner_id = get_principal().user_id
    try:
        project = _require_project(session, project_id, owner_id=owner_id)
        settings = get_settings()
        media_dir = Path(settings.media_dir)
        output_path = (
            media_dir / PREVIEWS_SUBDIR / f"{project_id}_{int(at_time * _MS_PER_SEC)}.png"
        )
        # Font paths are re-derived from their ids here rather than trusted as
        # stored: a throwaway copy, exactly like ``export``'s aspect override.
        project = _with_resolved_fonts(session, project, owner_id=owner_id)
        path = render_preview_frame(
            project, media_dir=media_dir, output_path=output_path, at_time=at_time,
            font_path=settings.font_path,
        )
        return {"preview_path": str(path), "at_time": at_time}
    except (EditorError, subprocess.SubprocessError, OSError) as exc:
        return {"error": str(exc)}


def validate(session: Session, project_id: str, *, profile: str = "structural") -> dict:
    """Boundary validation. ``profile="structural"`` (default) checks only for
    a broken/desynchronized render; ``profile="shorts"`` additionally enforces
    this project's own duration/clip-count bounds and audio-coverage rules."""
    owner_id = get_principal().user_id
    project = load_project(session, project_id, owner_id=owner_id)
    if project is None:
        return {"error": f"Unknown project id: {project_id!r}"}
    errors = validate_project(project, profile=profile)
    return {"errors": errors, "ok": not errors}


def export(
    session: Session,
    project_id: str,
    *,
    profile: str = "structural",
    preset: str = DEFAULT_EXPORT_PRESET,
) -> dict:
    """Validate the project and, when valid, render it to an mp4 file
    synchronously (no separate async render task — export IS the render in
    this standalone product). Returns ``{ok, output_path, cover_path,
    version, validation_errors}``.

    ``preset`` selects encoder knobs + an optional resolution override from
    ``app.config.EXPORT_PRESETS`` — ``"shorts_1080"`` (default) reproduces
    the pre-phase-5 defaults byte-for-byte. A resolution override never
    touches the STORED project: the render call uses a throwaway
    ``dataclasses.replace`` copy with ``aspect_w``/``aspect_h`` swapped, so
    every position/size expressed as a fraction of the frame (text
    ``pos_x``/``pos_y``, crop-zoom, blur-fill rects) scales proportionally —
    exactly what a smaller preview should do.

    When ``project.cover_time`` is set, export additionally renders a
    full-res PNG cover frame beside the mp4 (reusing the same machinery as
    ``render_preview_frame``) and returns its path as ``cover_path`` (``None``
    otherwise) — a cover-frame failure does NOT fail the export itself (the
    video already rendered successfully), it only leaves ``cover_path: null``
    with the reason folded into ``validation_errors`` for visibility.

    The EXPORTING lock is only held for the duration of this one subprocess
    call, not across an async task boundary, so a second export attempt while
    one is in flight is refused; once EXPORTED, further saves are refused
    until ``editor_reopen_project`` is called.

    **Transaction discipline:** every ``set_status`` here is COMMITTED
    immediately, in its own short transaction, on the CALLER's session —
    never held open across the render. Holding the write lock through a
    multi-minute ffmpeg run is what turned every concurrent login/save into
    ``database is locked`` (SQLite has one writer), and an UNcommitted
    EXPORTING was invisible to other sessions, so the "second export refused"
    guard above never actually fired mid-render. The cost is honest: if the
    process dies mid-render the project stays EXPORTING (recoverable via
    ``editor_reopen_project``) instead of silently rolling back to DRAFT.
    """
    owner_id = get_principal().user_id
    project = load_project(session, project_id, owner_id=owner_id)
    if project is None:
        return {"error": f"Unknown project id: {project_id!r}"}
    preset_config = EXPORT_PRESETS.get(preset)
    if preset_config is None:
        return {"error": f"Unknown export preset {preset!r}; must be one of {sorted(EXPORT_PRESETS)}"}
    already_locked = locked_status(session, project_id, owner_id=owner_id)
    if already_locked is not None:
        return {
            "error": f"project {project_id!r} is already {already_locked.value}; "
                     "call editor_reopen_project to edit and re-export"
        }
    errors = validate_project(project, profile=profile)
    if errors:
        return {"ok": False, "output_path": None, "cover_path": None, "validation_errors": errors}

    set_status(session, project_id, ProjectStatus.EXPORTING, owner_id=owner_id)
    session.commit()  # release the write lock BEFORE ffmpeg; see docstring
    settings = get_settings()
    media_dir = Path(settings.media_dir)
    suffix = "" if preset == DEFAULT_EXPORT_PRESET else f"_{preset}"
    output_path = media_dir / EXPORTS_SUBDIR / f"{project_id}_{project.version}{suffix}.mp4"

    # Every text element's font path re-derived from its id before the graph is
    # built (R-33): what is in the row is a cached RESULT, and the renderer must
    # only ever open a path this server resolved. A font the owner has since
    # deleted refuses the export here rather than quietly rendering in another
    # typeface — the same refusal-over-fallback rule the rasterizer follows.
    try:
        resolved_project = _with_resolved_fonts(session, project, owner_id=owner_id)
    except EditorError as exc:
        set_status(session, project_id, ProjectStatus.DRAFT, owner_id=owner_id)
        session.commit()
        return {"ok": False, "output_path": None, "cover_path": None,
                "validation_errors": [str(exc)]}
    render_project = resolved_project
    if preset_config["width"] is not None and preset_config["height"] is not None:
        render_project = dataclasses.replace(
            resolved_project,
            aspect_w=preset_config["width"],
            aspect_h=preset_config["height"],
        )
    try:
        render_project_file(
            render_project,
            media_dir=media_dir,
            output_path=output_path,
            font_path=settings.font_path,
            crf=preset_config["crf"],
            preset=preset_config["preset"],
            engine=settings.render_engine,
            melt_binary=settings.melt_binary,
        )
    except (EditorError, subprocess.SubprocessError, OSError) as exc:
        set_status(session, project_id, ProjectStatus.DRAFT, owner_id=owner_id)  # failed render: don't strand it locked
        session.commit()
        return {"ok": False, "output_path": None, "cover_path": None, "validation_errors": [str(exc)]}
    except Exception:
        # EXPORTING is already durable (committed above), so the caller's
        # session_scope rollback can no longer undo it — an unexpected
        # exception must reset DRAFT itself or the project stays stranded
        # locked. Unlike the expected render failures above this re-raises:
        # it is a bug, not a bad render.
        set_status(session, project_id, ProjectStatus.DRAFT, owner_id=owner_id)
        session.commit()
        raise
    set_status(session, project_id, ProjectStatus.EXPORTED, owner_id=owner_id)
    session.commit()  # durable before the cover render below — another ffmpeg run

    cover_path, cover_errors = _render_cover_if_requested(
        # The font-resolved copy, but NOT the preset-resized one: a cover
        # represents the project, not one particular export.
        resolved_project,
        media_dir=media_dir, output_path=output_path, font_path=settings.font_path,
    )
    return {
        "ok": True,
        "output_path": str(output_path),
        "cover_path": cover_path,
        "version": project.version,
        "validation_errors": cover_errors,
    }


def _render_cover_if_requested(
    project: EditorProject, *, media_dir: Path, output_path: Path, font_path: str,
) -> tuple[str | None, list[str]]:
    """Render the project's own ``cover_time`` frame (full-res, project's
    OWN aspect — never the export preset's overridden resolution, a cover
    image represents the project, not one particular export) beside
    ``output_path``. Returns ``(cover_path_or_None, errors)``; a failure here
    never fails the export itself — the video already rendered."""
    if project.cover_time is None:
        return None, []
    cover_path = output_path.parent / f"{output_path.stem}_cover.png"
    try:
        render_preview_frame(
            project, media_dir=media_dir, output_path=cover_path,
            at_time=project.cover_time, font_path=font_path,
        )
    except (EditorError, subprocess.SubprocessError, OSError) as exc:
        return None, [f"cover frame render failed: {exc}"]
    return str(cover_path), []
