"""Persistence for timeline error-annotations (the operator review pass).

Deliberately a separate, tiny repository — NOT part of
``app.db.repositories.projects`` — because annotations live on a different
lifecycle from the timeline snapshot:

* They are placed on ALREADY-EXPORTED projects (review happens after the
  render), so they must bypass ``save_project``, which refuses writes to an
  EXPORTED project and bumps the timeline version.
* They never touch the timeline blob or the project version — an annotation is
  a note pinned to a point in time, not an edit.

Input is validated at this boundary (severity whitelist, non-empty note); a
bad value raises ``ValueError`` for the caller (MCP tool / web route) to map to
a clear ``{"error": ...}`` or HTTP 422.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.models import AnnotationRow, ProjectRow

# The three severities an operator may assign, worst-first (also the order the
# web select renders them in). A blocker stops publish; major/minor are
# advisory. Kept as a whitelist so a typo becomes a clear error, never a row.
VALID_SEVERITIES: tuple[str, ...] = ("blocker", "major", "minor")

DEFAULT_SEVERITY = "major"


@dataclass(frozen=True)
class Annotation:
    """Immutable view of one annotation row for tools and templates (mirrors
    ``projects.ProjectRecord`` — the ORM row never leaves this module)."""

    id: int
    project_id: str
    time_sec: float
    element_id: str | None
    severity: str
    note: str
    author_user_id: int
    resolved: bool
    created_at: datetime | None


def _to_view(row: AnnotationRow) -> Annotation:
    return Annotation(
        id=row.id,
        project_id=row.project_id,
        time_sec=row.time_sec,
        element_id=row.element_id,
        severity=row.severity,
        note=row.note,
        author_user_id=row.author_user_id,
        resolved=row.resolved,
        created_at=row.created_at,
    )


def _owned_project(project_id: str, owner_id: int):
    """An EXISTS over ``projects`` — "this annotation's project belongs to
    ``owner_id``".

    Ownership is derived through the project rather than stored on the
    annotation. ``AnnotationRow.author_user_id`` records WHO WROTE a marker,
    which is a different question: a project's owner sees every marker pinned
    to their project, including ones written by someone else, and never sees a
    marker on a project that is not theirs even if they wrote it. Copying an
    ``owner_id`` onto this table would create a second answer to the ownership
    question, free to drift from the first.
    """
    return (
        select(ProjectRow.id)
        .where(ProjectRow.project_id == project_id, ProjectRow.owner_id == owner_id)
        .exists()
    )


def add_annotation(
    session: Session,
    *,
    project_id: str,
    time_sec: float,
    note: str,
    owner_id: int,
    author_user_id: int,
    severity: str = DEFAULT_SEVERITY,
    element_id: str | None = None,
) -> Annotation:
    """Pin a new error-marker to ``project_id`` at ``time_sec``. Bypasses the
    ``save_project`` EXPORTED lock on purpose (see module docstring). Validates
    ``severity`` against ``VALID_SEVERITIES`` and rejects an empty ``note``.

    Raises ``ValueError`` when ``project_id`` is not a project of ``owner_id``
    — including when it does not exist at all, which is the same message on
    purpose. Without this check the table would accept markers pinned to
    another tenant's project: the row would then be invisible to its writer
    (``list_annotations`` filters by the project's owner) and visible to that
    tenant, which is both a leak and a way to write into someone else's review.

    ``author_user_id`` is passed separately from ``owner_id`` because they are
    genuinely different facts, and will diverge as soon as a project is shared.
    """
    if severity not in VALID_SEVERITIES:
        raise ValueError(
            f"severity must be one of {list(VALID_SEVERITIES)}, got {severity!r}"
        )
    clean_note = note.strip()
    if not clean_note:
        raise ValueError("note must not be empty")
    if not session.execute(select(_owned_project(project_id, owner_id))).scalar():
        raise ValueError(f"project {project_id!r} not found")
    row = AnnotationRow(
        project_id=project_id,
        time_sec=time_sec,
        element_id=element_id,
        severity=severity,
        note=clean_note,
        author_user_id=author_user_id,
        resolved=False,
    )
    session.add(row)
    session.flush()
    session.refresh(row)  # populate the server-default created_at
    return _to_view(row)


def list_annotations(
    session: Session, project_id: str, *, owner_id: int, include_resolved: bool = False
) -> list[Annotation]:
    """A project's annotations, oldest first, for the project's OWNER — empty
    for anyone else. Resolved markers are hidden by default — pass
    ``include_resolved=True`` to get the full history."""
    query = select(AnnotationRow).where(
        AnnotationRow.project_id == project_id, _owned_project(project_id, owner_id)
    )
    if not include_resolved:
        query = query.where(AnnotationRow.resolved.is_(False))
    rows = session.execute(query.order_by(AnnotationRow.created_at, AnnotationRow.id)).scalars()
    return [_to_view(row) for row in rows]


def resolve_annotation(
    session: Session, annotation_id: int, *, project_id: str, owner_id: int
) -> bool:
    """Mark one annotation resolved. Returns True if a row was updated, False
    when the id is unknown, belongs to a DIFFERENT project, or belongs to a
    project this caller does not own (idempotent — resolving an
    already-resolved or absent marker is not an error).

    **``project_id`` is required, and adding it is the whole fix — a check
    inside the old signature could not have been reached.** This used to take
    ``annotation_id`` alone and select on that global auto-increment key,
    looking at no project at all. Over HTTP that made
    ``POST /api/projects/{anything}/ops`` with
    ``{"op": "resolve_annotation", "args": {"annotation_id": N}}`` resolve any
    marker by guessing a small integer, with the path's project never
    consulted: ``_invoke_tool`` (``app/web/api.py``) injects ``project_id``
    only into tools whose signature declares it. So the parameter is what
    makes the injection happen in the first place; validating inside the old
    shape would have had nothing to validate against.

    Gate 1 Step 7 narrowed this to the caller's own tenant. That left the
    within-tenant hole — one of your projects' markers resolvable from another
    of your projects' endpoints — which is what this closes.
    """
    updated = session.execute(
        update(AnnotationRow)
        .where(
            AnnotationRow.id == annotation_id,
            AnnotationRow.project_id == project_id,
            _owned_project(project_id, owner_id),
        )
        .values(resolved=True)
    ).rowcount
    session.flush()
    return updated > 0
