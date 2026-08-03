"""project ownership: owner_id and annotation author_user_id

Gate 1 Step 7. Gives every project a tenant and every annotation a real
author, so the repositories can filter on ownership instead of returning
the whole table to whoever asks.

**Hand-written, not the raw --autogenerate output.** Autogenerate produced
the right SCHEMA and an upgrade that cannot run: it emits
``add_column(..., nullable=False)``, which SQLite rejects outright on a
non-empty table ("Cannot add a NOT NULL column with default value NULL"),
and ``create_foreign_key`` outside a batch block, which SQLite cannot do at
all. Both only show up against a database with rows in it — an empty-database
test would have passed and shipped a migration that bricks every real
deployment. The three-phase shape below (add nullable → backfill → tighten
inside a batch) is the standard way round it.

**Who existing rows are assigned to.** The lowest-id ACTIVE account, matching
``app.db.repositories.users.bootstrap_user_id`` — on a real deployment that is
the operator's own account, created first, which keeps them the owner of their
own data. Only when no account exists at all is one created here, and it gets
a ``password_hash`` no password can ever verify against (see below): the
migration must never mint a working credential.

``annotations.author`` is dropped rather than migrated. It was a VARCHAR
defaulting to the literal "operator" that no caller ever set to anything else,
so there is nothing in it to preserve — see ``app/db/models.py``.

Revision ID: 81e1445feaa6
Revises: d461a3e6aaad
Create Date: 2026-07-21 16:15:13.484246

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '81e1445feaa6'
down_revision: Union[str, Sequence[str], None] = 'd461a3e6aaad'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Kept as literals rather than imported from app.db.repositories.users: a
# migration describes the schema as it was AT THIS REVISION, and importing
# today's application code would make an old revision's behaviour change every
# time that module does. BOOTSTRAP_EMAIL there and this string must agree —
# they are cross-referenced in both directions.
_BOOTSTRAP_EMAIL = "bootstrap@localhost"

# Deliberately not a valid argon2 encoded hash. ``argon2-cffi`` raises
# ``Argon2Error`` when it cannot parse the stored string, which
# ``app.db.repositories.users.verify_password`` catches and turns into False —
# so this account cannot be logged into with ANY password, including an empty
# one. A placeholder that argon2 could parse would be a working credential
# committed to source; a NULL would make the column's NOT NULL the only thing
# standing between an empty string and a login.
_UNUSABLE_PASSWORD_HASH = "!unusable"


def _bootstrap_owner_id(bind) -> int | None:
    """The account to attribute pre-existing rows to, or ``None`` when there
    are no such rows and therefore nothing to attribute."""
    pending = bind.execute(
        sa.text(
            "SELECT (SELECT COUNT(*) FROM projects) + (SELECT COUNT(*) FROM annotations)"
        )
    ).scalar_one()
    if not pending:
        return None
    existing = bind.execute(
        sa.text("SELECT id FROM users WHERE status = 'active' ORDER BY id LIMIT 1")
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    bind.execute(
        sa.text(
            "INSERT INTO users (email, password_hash, email_confirmed, status) "
            "VALUES (:email, :hash, 0, 'active')"
        ),
        {"email": _BOOTSTRAP_EMAIL, "hash": _UNUSABLE_PASSWORD_HASH},
    )
    return bind.execute(
        sa.text("SELECT id FROM users WHERE email = :email"), {"email": _BOOTSTRAP_EMAIL}
    ).scalar_one()


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()

    # Phase 1 — add both columns NULLABLE, which SQLite allows on a table that
    # already holds rows.
    op.add_column('projects', sa.Column('owner_id', sa.Integer(), nullable=True))
    op.add_column('annotations', sa.Column('author_user_id', sa.Integer(), nullable=True))

    # Phase 2 — backfill. Skipped entirely on an empty database, where there is
    # no row to own and so no reason to create an account.
    owner_id = _bootstrap_owner_id(bind)
    if owner_id is not None:
        bind.execute(
            sa.text("UPDATE projects SET owner_id = :owner WHERE owner_id IS NULL"),
            {"owner": owner_id},
        )
        bind.execute(
            sa.text(
                "UPDATE annotations SET author_user_id = :owner WHERE author_user_id IS NULL"
            ),
            {"owner": owner_id},
        )

    # Phase 3 — tighten to NOT NULL and add the foreign keys. Both need
    # batch_alter_table on SQLite, which recreates the table around the change.
    with op.batch_alter_table('projects') as batch:
        batch.alter_column('owner_id', existing_type=sa.Integer(), nullable=False)
        batch.create_foreign_key('fk_projects_owner_id_users', 'users', ['owner_id'], ['id'])
    op.create_index(op.f('ix_projects_owner_id'), 'projects', ['owner_id'], unique=False)
    op.create_index(
        'ix_projects_owner_id_updated_at', 'projects', ['owner_id', 'updated_at'], unique=False
    )

    with op.batch_alter_table('annotations') as batch:
        batch.alter_column('author_user_id', existing_type=sa.Integer(), nullable=False)
        batch.create_foreign_key(
            'fk_annotations_author_user_id_users', 'users', ['author_user_id'], ['id']
        )
        batch.drop_column('author')
    op.create_index(
        op.f('ix_annotations_author_user_id'), 'annotations', ['author_user_id'], unique=False
    )


def downgrade() -> None:
    """Downgrade schema.

    Restores ``annotations.author`` at its old default so the column is
    populated for existing rows, then drops the ownership columns. Attribution
    added since the upgrade is LOST here — every row comes back as "operator",
    because that string is all the old column could hold. Ownership itself is
    lost too: after this runs, every project is visible to every caller again.
    Do not treat this as a routine rollback.
    """
    op.drop_index(op.f('ix_annotations_author_user_id'), table_name='annotations')
    with op.batch_alter_table('annotations') as batch:
        batch.add_column(
            sa.Column(
                'author', sa.VARCHAR(length=64), nullable=False, server_default='operator'
            )
        )
        batch.drop_constraint('fk_annotations_author_user_id_users', type_='foreignkey')
        batch.drop_column('author_user_id')

    op.drop_index('ix_projects_owner_id_updated_at', table_name='projects')
    op.drop_index(op.f('ix_projects_owner_id'), table_name='projects')
    with op.batch_alter_table('projects') as batch:
        batch.drop_constraint('fk_projects_owner_id_users', type_='foreignkey')
        batch.drop_column('owner_id')
