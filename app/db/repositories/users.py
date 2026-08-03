"""The single-user edition's account storage: one bootstrap owner, no
passwords, no login.

The cloud edition hashes human-chosen passwords with argon2id and resolves
callers from sessions and tokens. None of that exists here — the ``users``
table survives only because every project row carries ``owner_id`` (the same
owner-scoped repository layer as the cloud, deliberately), and that column
needs exactly one row to point at. ``password_hash`` stores an impossible
marker: there is no verifier for it to ever match against.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import UserRow, UserStatus

BOOTSTRAP_EMAIL = "bootstrap@localhost"

# Not a hash of anything. No login path exists in this edition, and if one is
# ever added it must refuse rows carrying this marker rather than compare it.
_UNUSABLE_PASSWORD_MARKER = "*single-user-no-login*"


def normalize_email(email: str) -> str:
    """Lowercased, stripped — same canonical form the cloud edition stores,
    so a database created here stays valid there."""
    return email.strip().lower()


def find_by_email(session: Session, email: str) -> UserRow | None:
    return session.execute(
        select(UserRow).where(UserRow.email == normalize_email(email))
    ).scalar_one_or_none()


def bootstrap_user_id(session: Session) -> int | None:
    """The lowest-id ACTIVE account, or ``None`` when there is none yet."""
    return session.execute(
        select(UserRow.id)
        .where(UserRow.status == UserStatus.ACTIVE.value)
        .order_by(UserRow.id)
        .limit(1)
    ).scalar_one_or_none()


def ensure_bootstrap_user(session: Session) -> int:
    """``bootstrap_user_id``, creating the account when none exists yet."""
    existing = bootstrap_user_id(session)
    if existing is not None:
        return existing
    row = UserRow(
        email=BOOTSTRAP_EMAIL,
        password_hash=_UNUSABLE_PASSWORD_MARKER,
        email_confirmed=True,
        status=UserStatus.ACTIVE.value,
    )
    session.add(row)
    session.flush()
    return row.id
