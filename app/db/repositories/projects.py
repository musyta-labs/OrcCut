"""Persistence for editor timeline projects.

Business logic goes through these helpers, not raw sessions. A project is
stored as its serialized dict on ``ProjectRow.data``, keyed by the project's
uuid (``project_id``). ``save_project`` upserts by that uuid AND appends one
row to the append-only operation journal in the same transaction.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.models import OperationLogRow, ProjectRow, ProjectStatus
from app.editor.errors import EditorError
from app.editor.model import EditorProject
from app.editor.serialization import project_from_dict, project_to_dict

# Statuses in which editing the project is dangerous: EXPORTING means a render
# subprocess may be reading this exact snapshot right now; EXPORTED means it
# already shipped. There is no FAILED status in this model — export is
# synchronous (in the same tool call), so a failed render resets the project
# straight back to DRAFT instead of stranding it locked (see app.mcp.tools.export).
_LOCKED_STATUSES = frozenset({ProjectStatus.EXPORTING, ProjectStatus.EXPORTED})


def locked_status(session: Session, project_id: str, *, owner_id: int) -> ProjectStatus | None:
    """The status of the persisted row, if that status blocks further edits —
    otherwise None (unlinked/unknown project, or an editable DRAFT).

    A project belonging to ANOTHER owner reads as None here, exactly like an
    unknown id: this function answers "may I edit this", and for a project that
    is not yours the answer is not "it is locked" — it is "there is no such
    project of yours". Leaking the difference would turn this into an oracle
    for the existence and export state of other tenants' projects.
    """
    row = session.execute(
        select(ProjectRow).where(
            ProjectRow.project_id == project_id, ProjectRow.owner_id == owner_id
        )
    ).scalar_one_or_none()
    if row is None or row.status not in _LOCKED_STATUSES:
        return None
    return ProjectStatus(row.status)


def save_project(
    session: Session, project: EditorProject, *, owner_id: int, op_name: str, op_args: dict
) -> ProjectRow:
    """Upsert the snapshot AND append one journal row, in the same transaction.

    Refuses the write when this project is already EXPORTING/EXPORTED — see
    ``locked_status``. Call ``reopen_project`` first if the edit is deliberate.

    **The lock check and the write are ONE statement.** This used to be a
    read (``locked_status``), then a second read, then a write — three
    statements with the precondition established by the first two. Adding the
    owner predicate turned that into a genuine race: "confirm the caller owns
    this and it is unlocked" followed by "write it" is a TOCTOU window, and
    what fits in that window is another tenant's export starting, or a
    concurrent edit to a project whose ownership was checked a statement ago.
    Under SQLite it was previously masked by WAL effectively serialising
    writers, which is a property of today's deployment, not of this code.

    ``SELECT ... FOR UPDATE`` is NOT the fix here and is not used: SQLAlchemy
    compiles it to nothing on SQLite, so it would read as a lock while being a
    no-op — worse than the honest race. The conditional ``UPDATE ... WHERE
    project_id = ? AND owner_id = ? AND status NOT IN (locked)`` below carries
    every precondition into the write itself, so the database evaluates them
    atomically with it, on any backend.
    """
    data = project_to_dict(project)
    updated = session.execute(
        update(ProjectRow)
        .where(
            ProjectRow.project_id == project.id,
            ProjectRow.owner_id == owner_id,
            ProjectRow.status.not_in([s.value for s in _LOCKED_STATUSES]),
        )
        .values(version=project.version, data=data)
    ).rowcount

    if updated == 0:
        # Nothing was written. Three different reasons, and they need telling
        # apart, so re-read to classify — a read that is now merely diagnostic
        # and can no longer affect the write it explains.
        _raise_for_failed_save(session, project.id, owner_id=owner_id)
        row = ProjectRow(
            project_id=project.id, owner_id=owner_id, version=project.version, data=data
        )
        session.add(row)
    else:
        row = session.execute(
            select(ProjectRow).where(ProjectRow.project_id == project.id)
        ).scalar_one()

    session.add(OperationLogRow(
        project_id=project.id, op_name=op_name, op_args=op_args, version_after=project.version,
    ))
    session.flush()
    return row


def _raise_for_failed_save(session: Session, project_id: str, *, owner_id: int) -> None:
    """Explain a zero-row ``UPDATE`` in ``save_project``, or return so the
    caller can INSERT.

    Returns quietly only for "no such project at all", which is the ordinary
    first-save case. A project owned by SOMEONE ELSE raises the same
    ``EditorError`` as a locked one would for its owner — deliberately NOT a
    distinct "not yours" message, which would confirm that the id exists and
    belongs to another tenant.
    """
    row = session.execute(
        select(ProjectRow).where(ProjectRow.project_id == project_id)
    ).scalar_one_or_none()
    if row is None:
        return
    if row.owner_id != owner_id:
        raise EditorError(f"project {project_id!r} not found")
    raise EditorError(
        f"project {project_id!r} is {ProjectStatus(row.status).value}; refusing to save "
        "further edits — call editor_reopen_project first if this edit is deliberate"
    )


@dataclass(frozen=True)
class ProjectRecord:
    """Row-level project view for the web viewer: ``status``/``created_at``/
    ``updated_at`` are real ``ProjectRow`` columns that ``list_projects``/
    ``load_project`` below discard (they only return the deserialized
    ``EditorProject``, which has no notion of status or wall-clock time)."""

    project_id: str
    status: ProjectStatus
    version: int
    created_at: datetime
    updated_at: datetime
    project: EditorProject


def _record_from_row(row: ProjectRow) -> ProjectRecord:
    return ProjectRecord(
        project_id=row.project_id,
        status=ProjectStatus(row.status),
        version=row.version,
        created_at=row.created_at,
        updated_at=row.updated_at,
        project=project_from_dict(row.data),
    )


def list_project_records(session: Session, *, owner_id: int) -> list[ProjectRecord]:
    """This owner's projects with their row-level status/timestamps, newest
    first — the web viewer's data source for the project list page."""
    rows = session.execute(
        select(ProjectRow)
        .where(ProjectRow.owner_id == owner_id)
        .order_by(ProjectRow.created_at.desc())
    ).scalars()
    return [_record_from_row(row) for row in rows]


def get_project_record(
    session: Session, project_id: str, *, owner_id: int
) -> ProjectRecord | None:
    """A single project's row-level view (status/timestamps + snapshot), or
    None when the id is unknown TO THIS OWNER — the web viewer's data source
    for the detail and history pages.

    Another tenant's project is reported as absent rather than forbidden, so
    callers that map None to 404 (which is all of them) cannot be used to probe
    which project ids exist.
    """
    row = session.execute(
        select(ProjectRow).where(
            ProjectRow.project_id == project_id, ProjectRow.owner_id == owner_id
        )
    ).scalar_one_or_none()
    return None if row is None else _record_from_row(row)


def load_project(session: Session, project_id: str, *, owner_id: int) -> EditorProject | None:
    """Load one of this owner's projects by its uuid, or None when absent."""
    row = session.execute(
        select(ProjectRow).where(
            ProjectRow.project_id == project_id, ProjectRow.owner_id == owner_id
        )
    ).scalar_one_or_none()
    return None if row is None else project_from_dict(row.data)


def list_projects(
    session: Session, metadata_filter: dict[str, str] | None = None, *, owner_id: int
) -> list[EditorProject]:
    """This owner's projects, newest first, optionally filtered to those whose
    ``metadata`` is a superset of ``metadata_filter``. The metadata filter is
    applied in Python after loading — fine at this MVP's scale (a project list
    per client is not going to be thousands of rows); querying inside the JSON
    blob via SQLite's ``json_extract`` is a documented follow-up, not needed
    now. The OWNER predicate is applied in SQL, not in Python: it is the one
    filter whose correctness is a security property, and a row that does not
    belong to the caller must never be loaded into this process at all."""
    rows = session.execute(
        select(ProjectRow)
        .where(ProjectRow.owner_id == owner_id)
        .order_by(ProjectRow.created_at.desc())
    ).scalars()
    projects = [project_from_dict(row.data) for row in rows]
    if not metadata_filter:
        return projects
    wanted = metadata_filter.items()
    return [p for p in projects if wanted <= dict(p.metadata).items()]


def get_history(session: Session, project_id: str, *, owner_id: int) -> list[dict]:
    """The full append-only op journal for one of this owner's projects,
    oldest first — empty for a project that is not theirs.

    ``operations`` carries no ``owner_id`` of its own: it references its
    project by a plain string, deliberately (see ``app/db/models.py``), so
    ownership is derived through the project with an EXISTS, not duplicated
    onto the journal where it could drift out of agreement.
    """
    owned = (
        select(ProjectRow.id)
        .where(
            ProjectRow.project_id == OperationLogRow.project_id,
            ProjectRow.owner_id == owner_id,
        )
        .exists()
    )
    rows = session.execute(
        select(OperationLogRow)
        .where(OperationLogRow.project_id == project_id, owned)
        .order_by(OperationLogRow.version_after)
    ).scalars()
    return [
        {"op_name": r.op_name, "op_args": r.op_args, "version": r.version_after,
         "created_at": r.created_at.isoformat()}
        for r in rows
    ]


def set_status(session: Session, project_id: str, status: ProjectStatus, *, owner_id: int) -> None:
    """Set one of this owner's projects to ``status``. A no-op for an unknown
    id or another tenant's project — the same silence the unknown-id case
    already had."""
    session.execute(
        update(ProjectRow)
        .where(ProjectRow.project_id == project_id, ProjectRow.owner_id == owner_id)
        .values(status=status.value)
    )
    session.flush()


def reopen_project(session: Session, project_id: str, *, owner_id: int) -> None:
    """Generalizes 'a failed/finished export is still recoverable' into an
    explicit agent action: unlock an EXPORTED project for further edits."""
    set_status(session, project_id, ProjectStatus.DRAFT, owner_id=owner_id)
