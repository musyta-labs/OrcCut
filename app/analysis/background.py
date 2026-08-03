"""Fire-and-forget clip analysis, so adding a clip never waits on a decode.

Adding a clip to a track must stay an in-memory timeline edit. Analysis is
~2s of CPU on an already-local file and can be minutes when the media still
has to be downloaded — blocking ``add_clip`` on that would make the editor
feel broken for a result the UI does not need in that instant.

Two layers of de-duplication, because both races are real:

1. **The dedup key**, ``(project_id, media_id)`` — two clips of the same asset
   added back to back queue ONE job, not two. WHERE that key is held depends on
   the shared-state backend (Gate 3 Step 6):

   * ``memory`` (default, and the whole SQLite self-host path) — a per-process
     ``_inflight`` set, exactly as before: process-local is global on one
     replica, so behaviour is unchanged bit-for-bit.
   * ``database`` — the ``analysis_jobs`` table
     (``app.db.repositories.analysis_jobs``), whose unique key dedups across
     EVERY replica and whose claim (``FOR UPDATE SKIP LOCKED`` on Postgres) hands
     each job to exactly one replica. This closes the multi-replica hole the old
     ``_inflight`` conceded: replica B can now see the job replica A is running,
     so it neither runs it twice nor under-reports ``is_pending``. The web side
     only ENQUEUES (an INSERT); the claim+run is the worker step, kept in this
     process's pool today but factored so it can move to a standalone worker
     (see ``run_claimed_job`` / ``_schedule_via_queue``).

2. **Per-content-hash lock**, inside ``app.analysis.service`` — catches what
   this layer structurally cannot see, since the same media in two different
   projects (or under two different asset ids) is two distinct keys here but
   one hash there. It also serialises against a foreground
   ``editor_analyze_media`` call, which never passes through this module.

Failure is contained on purpose: a job that raises logs with full media and
project context and writes NOTHING. No record is stored, so the media is
simply un-analysed and the next ``add_clip`` (or an explicit analyse call)
retries it. A half-written or plausible-but-wrong record would be far worse
than a missing one — the UI can show "not analysed yet"; it cannot detect that
markers were computed from a truncated download.
"""
from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.analysis.scheduler import FairScheduler
from app.analysis.service import analyze_with_cache
from app.common.logging import get_logger
from app.common.shared_state import DATABASE_BACKEND, get_slot_store
from app.config import get_settings
from app.db.repositories import analysis_jobs as jobs
from app.db.repositories.analysis_jobs import ClaimedJob
from app.db.repositories.projects import load_project
from app.editor.render import resolve_asset_media

logger = get_logger(__name__)

# Small on purpose. Analysis is CPU-bound ffmpeg work in a container sized for
# one render at a time; more workers would contend with an in-progress export
# rather than finishing sooner.
MAX_WORKERS = 2

# This replica's claim identity — stable for the process lifetime, so a job it
# claims in the ``analysis_jobs`` table is attributable to it and a DIFFERENT
# replica's stale claim is distinguishable. Diagnostic only: a claim's validity
# is decided by its heartbeat age, not by matching this id.
_REPLICA_ID = uuid.uuid4().hex

_executor: ThreadPoolExecutor | None = None
_inflight: set[tuple[str, str]] = set()
_guard = threading.Lock()
_scheduler: FairScheduler | None = None


def _use_db_queue() -> bool:
    """Whether the shared, cross-replica ``analysis_jobs`` broker is active.

    Tied to the SAME config switch as every other shared-state primitive
    (Gate 3 Step 4): ``memory`` keeps the per-process ``_inflight`` path
    unchanged, ``database`` routes dedup/pending/claim through the table."""
    return get_settings().shared_state_backend == DATABASE_BACKEND


def _get_executor() -> ThreadPoolExecutor:
    """Created lazily so merely importing this module (which the MCP tools do)
    never spawns threads — matters for the CLI twins and the test suite."""
    global _executor
    with _guard:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=MAX_WORKERS, thread_name_prefix="analysis"
            )
        return _executor


def get_scheduler() -> FairScheduler:
    """The process-wide fair scheduler that fronts the shared pool.

    Round-robins across owners and caps each owner at its ``max_concurrent_jobs``
    quota, so one tenant with a deep queue cannot starve another (see
    ``app.analysis.scheduler``). Created lazily for the same import-time reason
    as the executor. The executor is resolved through ``_get_executor`` on every
    dispatch, so a test that monkeypatches it is still honoured."""
    global _scheduler
    with _guard:
        if _scheduler is None:
            # Resolve the executor through a late-binding lookup rather than a
            # captured reference, so the pool stays lazy AND a test that
            # monkeypatches ``_get_executor`` is still honoured on every dispatch.
            _scheduler = FairScheduler(
                executor_factory=lambda: _get_executor(),
                max_workers=MAX_WORKERS,
                # The config-selected shared slot store backs the per-owner
                # ceiling: an in-process counter on ``memory`` (unchanged), a
                # shared table on ``database`` so the ceiling holds across
                # replicas (see app.analysis.scheduler).
                slot_store=get_slot_store(),
            )
        return _scheduler


# Mirrors app.db.repositories.quotas.DEFAULT_MAX_CONCURRENT_JOBS. Duplicated
# on purpose: the quota repo is closed core the public extract (E-01) does not
# ship, and this module must know the ceiling without it. A drift between the
# two would only widen/narrow the SINGLE-user default, never tenant isolation.
_DEFAULT_MAX_CONCURRENT_JOBS = 3


def _owner_ceiling(owner_id: int, session_scope: Callable) -> int:
    """The owner's ``max_concurrent_jobs`` quota, or the default when it cannot
    be read. Never raises — a quota lookup failure must degrade to the default
    ceiling, not sink a fire-and-forget schedule (see ``schedule_media_analysis``).

    The quota repo is imported lazily and its absence is not an error: the
    public extract ships without quota tables, and there the default IS the
    ceiling — one user, no tenants to protect from each other.
    """
    try:
        from app.db.repositories.quotas import get_quota
    except ImportError:
        logger.debug("quota repo absent (single-user build) — default ceiling %d",
                     _DEFAULT_MAX_CONCURRENT_JOBS)
        return _DEFAULT_MAX_CONCURRENT_JOBS
    try:
        with session_scope() as session:
            quota = get_quota(session, owner_id)
        if quota is not None:
            return quota.max_concurrent_jobs
    except Exception:
        logger.exception(
            "could not read concurrency quota for owner=%s — defaulting to %d",
            owner_id, _DEFAULT_MAX_CONCURRENT_JOBS,
        )
    return _DEFAULT_MAX_CONCURRENT_JOBS


def is_pending(project_id: str, media_id: str) -> bool:
    """Whether a background analysis for this asset is queued or running — the
    'pending' status the HTTP read endpoint reports.

    On the ``database`` backend this reads the shared ``analysis_jobs`` table, so
    a job running on ANOTHER replica reports pending here too — the exact
    cross-replica lie the per-process ``_inflight`` set used to tell. Never
    raises: a queue-read failure degrades to 'not pending' (the same 'absent'
    the read endpoint already shows for un-analysed media), never a 500 on a
    status read."""
    if _use_db_queue():
        return _is_pending_via_queue(project_id, media_id)
    with _guard:
        return (project_id, media_id) in _inflight


def _is_pending_via_queue(project_id: str, media_id: str) -> bool:
    """``is_pending`` against the shared table. Uses the process's own session
    factory — this is called from a request thread that has a real database,
    unlike the pool-thread worker which is handed a ``session_scope``."""
    from app.db.base import session_scope

    try:
        with session_scope() as session:
            return jobs.is_live(session, project_id=project_id, media_id=media_id)
    except Exception:
        logger.exception(
            "could not read job-queue pending state for project=%s media=%s — "
            "reporting not-pending",
            project_id, media_id,
        )
        return False


def schedule_media_analysis(
    *,
    project_id: str,
    media_id: str,
    owner_id: int,
    media_dir: Path,
    session_scope: Callable,
    replica_id: str | None = None,
) -> bool:
    """Queue an analysis for one asset. Returns whether a job was actually
    scheduled (``False`` = disabled by config, or already in flight/claimed
    elsewhere).

    ``owner_id`` is passed in as a plain argument and carried down to
    ``load_project`` by hand. It deliberately does NOT come from
    ``app.auth.principal.get_principal()`` inside the worker:
    ``ThreadPoolExecutor.submit`` does not copy contextvars into the pool
    thread, so the principal bound by the request that scheduled this job is
    simply absent here and ``get_principal()`` would raise. Binding one inside
    the job would be worse still — a bare ``current_principal.set()`` survives
    the job and leaks into whatever tenant's job reuses the thread next (see
    ``bind_principal``'s docstring). The owner is resolved on the SCHEDULING
    side, where a principal genuinely is bound, and travels as data.

    ``replica_id`` overrides this process's claim identity — only tests pass it,
    to stand up two independent "replicas" against one database. Production
    leaves it ``None`` and the process-wide ``_REPLICA_ID`` is used.

    NEVER raises — the whole body is guarded, not just the ``submit``. This is
    called from inside ``add_clip``'s success path, where the timeline edit is
    already saved and journaled, so ANY failure here (unreadable settings, a
    shut-down executor, anything) must degrade to "no analysis was scheduled"
    rather than turning a successful edit into an error the caller has to
    interpret. Auto-analysis is a side effect, never a precondition.
    """
    try:
        if not get_settings().auto_analyze_on_add:
            return False
        if _use_db_queue():
            return _schedule_via_queue(
                project_id=project_id, media_id=media_id, owner_id=owner_id,
                media_dir=Path(media_dir), session_scope=session_scope,
                replica_id=replica_id or _REPLICA_ID,
            )
        return _schedule_in_process(
            project_id=project_id, media_id=media_id, owner_id=owner_id,
            media_dir=Path(media_dir), session_scope=session_scope,
        )
    except Exception:
        logger.exception(
            "could not schedule background analysis for project=%s media=%s — "
            "the clip was still added; analysis stays available on demand",
            project_id, media_id,
        )
        return False


def _schedule_in_process(
    *, project_id: str, media_id: str, owner_id: int, media_dir: Path,
    session_scope: Callable,
) -> bool:
    """The default (``memory``) path: dedup in the per-process ``_inflight`` set,
    dispatch through the in-process fair scheduler. Unchanged from Gate 2."""
    key = (project_id, media_id)
    with _guard:
        if key in _inflight:
            logger.debug("analysis already in flight for %s/%s", project_id, media_id)
            return False
        _inflight.add(key)
    try:
        # Fair, per-tenant-bounded dispatch instead of a raw pool submit: the
        # ceiling comes from the OWNER's quota so one tenant's deep queue can
        # neither monopolise the pool nor starve another (see get_scheduler).
        ceiling = _owner_ceiling(owner_id, session_scope)
        get_scheduler().submit(
            owner_id=owner_id,
            ceiling=ceiling,
            fn=_run_analysis,
            args=(project_id, media_id, owner_id, media_dir, session_scope),
        )
        return True
    except Exception:
        with _guard:
            _inflight.discard(key)
        raise


def _schedule_via_queue(
    *, project_id: str, media_id: str, owner_id: int, media_dir: Path,
    session_scope: Callable, replica_id: str,
) -> bool:
    """The ``database`` path: enqueue into the shared ``analysis_jobs`` table,
    then claim the next runnable job and dispatch it on this replica's pool.

    Enqueue and claim are the two halves the operator's broker model splits — a
    web request only ENQUEUES (the INSERT), and a worker CLAIMS. They are kept
    together here so the enqueuing replica also drains the queue, preserving
    today's "the job runs promptly on the machine that scheduled it" latency;
    the split (``run_claimed_job`` is a standalone entry) is what lets the claim
    loop move to a dedicated worker later without touching this call site.

    Returns ``True`` only when THIS call claimed a job to run. A duplicate
    schedule of an already-running key — on this replica or another — enqueues
    nothing new and finds nothing claimable, and returns ``False``: that is the
    cross-replica dedup. If the just-enqueued job was claimed by another replica
    between the two steps, this call may instead claim a DIFFERENT pending job
    (it is a worker draining a shared queue) or ``None`` — either way no job is
    dropped and none runs twice.
    """
    with session_scope() as session:
        jobs.enqueue(session, project_id=project_id, media_id=media_id, owner_id=owner_id)
    with session_scope() as session:
        claimed = jobs.claim_next(session, replica_id=replica_id)
    if claimed is None:
        logger.debug(
            "nothing claimable for this replica after enqueuing %s/%s "
            "(already running elsewhere, or claimed by another replica)",
            project_id, media_id,
        )
        return False
    ceiling = _owner_ceiling(claimed.owner_id, session_scope)
    get_scheduler().submit(
        owner_id=claimed.owner_id,
        ceiling=ceiling,
        fn=run_claimed_job,
        args=(claimed, media_dir, session_scope),
    )
    return True


def run_claimed_job(
    claimed: ClaimedJob, media_dir: Path, session_scope: Callable
) -> None:
    """Run one claimed job, then remove its row so the key frees for retry.

    The standalone worker entry point (the database path's ``_run_analysis``):
    given a job already claimed out of ``analysis_jobs``, it resolves the asset,
    analyses it, and — on success OR failure — deletes the row (``complete``),
    mirroring the ``_inflight.discard`` the in-process path does in its own
    ``finally``. Catches everything for the same reason ``_run_analysis`` does:
    it runs on a pool thread whose exception nobody would see."""
    try:
        path = _resolve_asset_path(
            claimed.project_id, claimed.media_id, claimed.owner_id, media_dir,
            session_scope,
        )
        if path is not None:
            outcome = analyze_with_cache(
                path, media_dir=media_dir, session_scope=session_scope,
                owner_id=claimed.owner_id,
            )
            logger.info(
                "background analysis %s for project=%s media=%s (sha256=%s…)",
                "reused cached result" if outcome.from_cache else "computed and stored",
                claimed.project_id, claimed.media_id, outcome.content_hash[:12],
            )
    except Exception:
        logger.exception(
            "background analysis FAILED for project=%s media=%s — nothing stored, "
            "the job row is cleared so the next add_clip retries",
            claimed.project_id, claimed.media_id,
        )
    finally:
        _complete_job(claimed, session_scope)


def _complete_job(claimed: ClaimedJob, session_scope: Callable) -> None:
    """Delete the finished job's row, guarded so a completion failure cannot
    escape the pool thread. A row left behind would merely be reclaimed as stale
    later — a delayed retry, not corruption."""
    try:
        with session_scope() as session:
            jobs.complete(session, project_id=claimed.project_id, media_id=claimed.media_id)
    except Exception:
        logger.exception(
            "could not clear completed job row for project=%s media=%s — it will "
            "be reclaimed as stale",
            claimed.project_id, claimed.media_id,
        )


def _run_analysis(
    project_id: str, media_id: str, owner_id: int, media_dir: Path, session_scope: Callable
) -> None:
    """The worker body. Catches everything by design (see module docstring):
    it runs on a pool thread whose exception nobody would ever see, so an
    uncaught error would be a genuinely silent failure."""
    try:
        path = _resolve_asset_path(
            project_id, media_id, owner_id, media_dir, session_scope
        )
        if path is None:
            return
        outcome = analyze_with_cache(
            path, media_dir=media_dir, session_scope=session_scope, owner_id=owner_id
        )
        logger.info(
            "background analysis %s for project=%s media=%s (sha256=%s…)",
            "reused cached result" if outcome.from_cache else "computed and stored",
            project_id, media_id, outcome.content_hash[:12],
        )
    except Exception:
        logger.exception(
            "background analysis FAILED for project=%s media=%s — nothing stored, "
            "the media stays un-analysed and the next add_clip will retry",
            project_id, media_id,
        )
    finally:
        with _guard:
            _inflight.discard((project_id, media_id))


def _resolve_asset_path(
    project_id: str, media_id: str, owner_id: int, media_dir: Path, session_scope: Callable
) -> Path | None:
    """The asset's local file, downloading it if needed.

    The session is opened only to read the project and closed again before the
    download/decode — see the module docstring in ``app.analysis.service``.
    """
    with session_scope() as session:
        project = load_project(session, project_id, owner_id=owner_id)
        if project is None:
            logger.warning(
                "background analysis skipped: unknown project %s", project_id
            )
            return None
        asset = next((a for a in project.assets if a.id == media_id), None)
        if asset is None:
            logger.warning(
                "background analysis skipped: unknown media %s on project %s",
                media_id, project_id,
            )
            return None
    return resolve_asset_media(asset, media_dir)
