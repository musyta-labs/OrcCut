"""Fair, per-tenant-bounded dispatch over a single shared worker pool.

The analysis pool is deliberately tiny (``background.MAX_WORKERS``): analysis is
CPU-bound ffmpeg work in a container sized for one render at a time. A raw
``ThreadPoolExecutor`` serves that pool strictly FIFO, which is exactly the
multi-tenant failure this module fixes: one owner that queues 50 clips at once
fills the FIFO queue, and every other tenant's job waits behind all 50 — a
single tenant monopolises the shared pool.

This scheduler sits IN FRONT of the executor and changes two things:

1. **Per-owner ceiling** — an owner never holds more than ``ceiling`` slots at
   once (sourced from that owner's ``max_concurrent_jobs`` quota), so a deep
   queue cannot convert into a deep share of the running pool.
2. **Round-robin dispatch** — when a slot frees, the next job is picked by
   rotating across owners rather than by arrival time. A newly-arriving tenant
   waits only for a free SLOT (bounded by the pool size), never for another
   tenant's backlog to drain.

Only up to ``max_workers`` jobs are ever handed to the executor at once;
everything else waits in per-owner queues here, so a blocked/queued job never
occupies a pool thread.

Gate 3 Step 6 — the per-owner ceiling is now enforced through a ``SlotStore``
(``app.common.shared_state``), NOT a per-process ``dict[owner, count]``. On the
default ``memory`` backend that store is an in-process counter, so the ceiling
behaves exactly as the old ``_active`` dict did — bit-for-bit on the SQLite
self-host path. On the ``database`` backend the SAME code acquires the slot in a
shared table, so "at most ``ceiling`` concurrent jobs for this owner" holds
ACROSS replicas rather than per-process: the two-replicas-each-run-ceiling
under-count this module's docstring used to concede is closed. The store is
injectable so tests drive it in isolation; ``background.get_scheduler`` wires in
the process-wide ``get_slot_store()`` that config selects.

The pool bound (``_running_total`` vs ``_max_workers``) stays a plain local
counter: it caps THIS process's own thread pool, which is inherently local — a
replica cannot hand work to another replica's threads — so there is nothing to
share there.
"""
from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Any

from app.common.shared_state import SlotLease, SlotStore
from app.common.shared_state_memory import InProcessSlotStore

# A queued unit of work: the function and its positional arguments.
_Job = tuple[Callable[..., Any], tuple[Any, ...]]


class FairScheduler:
    """Round-robin, per-owner-capped front end to a shared executor.

    Not a data model — a coordination primitive — so it owns mutable state by
    design; the immutability convention that governs the domain objects does
    not apply to the queues and counters here. All state is guarded by a single
    lock; ``_dispatch`` is only ever entered while holding it.
    """

    def __init__(
        self,
        *,
        executor_factory: Callable[[], Any],
        max_workers: int,
        slot_store: SlotStore | None = None,
    ) -> None:
        # Looked up through a factory (not a captured instance) so a test that
        # monkeypatches the executor is still honoured, and so the real pool
        # stays lazily created.
        self._executor_factory = executor_factory
        self._max_workers = max_workers
        # The per-owner ceiling lives here. Defaults to a FRESH in-process store
        # (isolated, and exactly the old ``dict[owner, count]`` behaviour) so a
        # unit test constructing a scheduler gets no shared state; the process
        # wires in the config-selected ``get_slot_store()`` for the real,
        # possibly cross-replica, ceiling. See module docstring.
        self._slot_store: SlotStore = slot_store if slot_store is not None else InProcessSlotStore()
        self._lock = threading.Lock()
        # owner_id -> queued jobs (FIFO within a single owner)
        self._pending: dict[int, deque[_Job]] = {}
        # owners that currently have queued work, in round-robin order
        self._order: deque[int] = deque()
        # owner_id -> its current per-owner ceiling (latest value wins)
        self._ceiling: dict[int, int] = {}
        # total jobs currently handed to the executor (<= max_workers)
        self._running_total = 0

    def submit(
        self,
        *,
        owner_id: int,
        ceiling: int,
        fn: Callable[..., Any],
        args: tuple[Any, ...] = (),
    ) -> None:
        """Enqueue one job for ``owner_id`` and dispatch what the pool can take.

        ``ceiling`` is the owner's maximum concurrent jobs; the most recent
        value provided wins, so a quota change is picked up on the next submit.
        """
        with self._lock:
            self._ceiling[owner_id] = ceiling
            queue = self._pending.get(owner_id)
            if queue is None:
                queue = deque()
                self._pending[owner_id] = queue
            was_empty = not queue
            queue.append((fn, args))
            # Register the owner in the round-robin ring exactly once while it
            # has pending work.
            if was_empty and owner_id not in self._order:
                self._order.append(owner_id)
            self._dispatch()

    def _acquire_dispatchable(self) -> tuple[int | None, SlotLease | None]:
        """The frontmost owner in the ring that has queued work AND could take a
        ceiling slot, together with the slot lease just taken for it — or
        ``(None, None)`` when nothing may run right now.

        Taking the slot IS the ceiling test: ``acquire`` returns a lease only
        while the owner holds fewer than ``ceiling`` slots, and refuses (``None``)
        at the boundary in one atomic step. Only the owner actually chosen has a
        slot taken, so a refused owner leaks nothing."""
        for owner_id in self._order:
            lease = self._slot_store.acquire(str(owner_id), limit=self._ceiling[owner_id])
            if lease is not None:
                return owner_id, lease
        return None, None

    def _dispatch(self) -> None:
        """Fill free pool slots, round-robin across owners. Caller holds lock."""
        while self._running_total < self._max_workers:
            owner_id, lease = self._acquire_dispatchable()
            if owner_id is None or lease is None:
                return
            fn, args = self._pending[owner_id].popleft()
            self._running_total += 1
            # Rotate this owner to the BACK of the ring so the next dispatch
            # prefers a different tenant — this is what bounds another owner's
            # wait to a slot rather than to this owner's backlog.
            self._order.remove(owner_id)
            if self._pending[owner_id]:
                self._order.append(owner_id)
            else:
                del self._pending[owner_id]
            self._executor_factory().submit(self._run, owner_id, lease, fn, args)

    def _run(
        self,
        owner_id: int,
        lease: SlotLease,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
    ) -> None:
        """Executor entry point: run the job, then free its slot and pump the
        queue. The job's own errors are the worker's contract to contain (see
        ``background._run_analysis``); this wrapper only guarantees the ceiling
        slot is released and the next job dispatched no matter how the job ends."""
        try:
            fn(*args)
        finally:
            with self._lock:
                self._slot_store.release(lease)
                self._running_total -= 1
                self._dispatch()

    # --- introspection / test-support ---------------------------------------

    def active_count(self, owner_id: int) -> int:
        """Jobs of ``owner_id`` currently running on the pool (its live ceiling
        slots). On the ``database`` backend this is the CLUSTER-wide count."""
        return self._slot_store.active_count(str(owner_id))

    def running_total(self) -> int:
        """Total jobs currently running on the pool across all owners."""
        with self._lock:
            return self._running_total

    def reset(self) -> None:
        """Drop all queued work and counters.

        For test isolation only: a test that hands the scheduler a fake executor
        which never runs the job leaves a slot marked active forever, which
        would leak into the next test. Never call this in production — it does
        not cancel jobs already on the real pool.
        """
        with self._lock:
            self._pending.clear()
            self._order.clear()
            self._ceiling.clear()
            self._slot_store.reset()
            self._running_total = 0
