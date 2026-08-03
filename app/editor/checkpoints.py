"""The "end of activity" half of the project version history.

A checkpoint is minted on exactly two occasions. The explicit one is a request
(``POST /projects/{id}/versions``). This module is the other one: a periodic
background pass that finds projects nobody has touched for
``Settings.project_checkpoint_idle_minutes`` and freezes them where they came
to rest.

WHY A SWEEP AND NOT A TIMER ON THE REQUEST. The obvious alternative is to arm a
timer when a mutation lands and checkpoint when it fires. That fails at exactly
the moment it matters: the last mutation of a session is followed by the user
closing the tab, and a per-request timer lives in the process that served that
request — a deploy, a crash, or a replica scaling down between the mutation and
the timer loses the checkpoint that the whole feature exists to take. A sweep
reads the fact from the database (``ProjectRow.updated_at``), so ANY replica,
including one started after the editing stopped, can complete the work.

MULTI-REPLICA: no distributed lock, and none is needed. Two replicas that pick
the same idle project both try to write a checkpoint for the same
(project, mutation number); the unique index makes one INSERT win and
``project_checkpoints.create_checkpoint`` reports the other as "already there".
Redundant work, never a duplicate row and never an error.

The thread mechanics mirror ``app.mcp.retention``'s worker deliberately —
module-level singleton, ``threading.Event`` as an interruptible sleep, failures
contained inside one pass — so this codebase has ONE background-worker shape to
understand rather than two. The interval is much shorter (see
``Settings.checkpoint_sweep_interval_sec``): retention deletes things, where
lateness is free, while this captures state a user is waiting to see.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import CheckpointTrigger
from app.db.repositories import project_checkpoints as checkpoints_repo

logger = logging.getLogger(__name__)

# How many idle projects one pass may checkpoint. Bounds a single transaction
# so a backlog — a fleet that was down for a day, or the very first sweep after
# this feature ships — cannot turn one pass into a multi-minute write. The
# remainder is picked up on the next interval, which for a 60 s interval is a
# negligible delay for state that has already been idle for ten minutes.
CHECKPOINT_SWEEP_BATCH = 100

_worker_thread: threading.Thread | None = None
_stop_event = threading.Event()
_worker_lock = threading.Lock()


@dataclass(frozen=True)
class SweepReport:
    """What one pass did. ``skipped`` counts projects that were selected but
    already carried a checkpoint for their current version by the time the
    write ran — the visible footprint of another replica having got there
    first, which is expected rather than exceptional."""

    considered: int
    created: int
    skipped: int


def run_checkpoint_sweep(session, *, idle_minutes: int, limit: int) -> SweepReport:
    """One pass, on an open session, as a plain function so it can be called
    directly by a test or an operator without a thread anywhere near it.

    The caller commits ``session``. Each checkpoint goes through the same
    ``create_checkpoint`` the explicit route uses — there is no second policy
    for "what counts as already checkpointed", which is what keeps the dedup
    rule honest between the two triggers.
    """
    project_ids = checkpoints_repo.find_idle_project_ids(
        session, idle_minutes=idle_minutes, limit=limit
    )
    created = 0
    skipped = 0
    for project_id in project_ids:
        owner_id = checkpoints_repo.owner_of(session, project_id)
        if owner_id is None:
            # The project was deleted between the query and this line. Not an
            # error: there is nothing left to checkpoint.
            skipped += 1
            continue
        result = checkpoints_repo.create_checkpoint(
            session, project_id, owner_id=owner_id, trigger=CheckpointTrigger.IDLE
        )
        if result is not None and result.created:
            created += 1
        else:
            skipped += 1
    return SweepReport(considered=len(project_ids), created=created, skipped=skipped)


def _run_one_sweep(idle_minutes: int, limit: int) -> None:
    """One pass in its own session with failures CONTAINED — a sweep that
    raises must not kill the worker thread, because there would be no one left
    to retry it on the next interval."""
    try:
        with session_scope() as session:
            report = run_checkpoint_sweep(
                session, idle_minutes=idle_minutes, limit=limit
            )
        if report.created:
            logger.info(
                "checkpoint sweep: %d created, %d skipped, %d considered",
                report.created, report.skipped, report.considered,
            )
    except Exception:
        logger.exception("checkpoint sweep failed; retrying next interval")


def _worker_loop(interval_sec: float, idle_minutes: int, limit: int) -> None:
    # ``Event.wait`` doubles as an interruptible sleep: a set ``_stop_event``
    # returns True immediately and ends the loop instead of blocking a full
    # interval on shutdown.
    while not _stop_event.wait(interval_sec):
        _run_one_sweep(idle_minutes, limit)


def start_checkpoint_worker() -> None:
    """Start the periodic checkpoint sweep once, idempotently. A no-op when
    ``Settings.checkpoint_sweep_enabled`` is False, or when a worker is already
    running — a repeat call (an app assembled more than once in one process)
    never spawns a second thread."""
    global _worker_thread
    settings = get_settings()
    if not settings.checkpoint_sweep_enabled:
        return
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _stop_event.clear()
        _worker_thread = threading.Thread(
            target=_worker_loop,
            args=(
                settings.checkpoint_sweep_interval_sec,
                settings.project_checkpoint_idle_minutes,
                CHECKPOINT_SWEEP_BATCH,
            ),
            name="checkpoint-worker",
            daemon=True,
        )
        _worker_thread.start()


def stop_checkpoint_worker(*, timeout: float = 5.0) -> None:
    """Signal the sweep to stop and wait up to ``timeout`` seconds for its
    current sleep/pass to unwind. Safe to call when nothing is running. Tests
    use it in teardown so a started worker never bleeds a live thread into the
    next test."""
    global _worker_thread
    with _worker_lock:
        thread = _worker_thread
        _worker_thread = None
    if thread is None:
        return
    _stop_event.set()
    thread.join(timeout=timeout)
