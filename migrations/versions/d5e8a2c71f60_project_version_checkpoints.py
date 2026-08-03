"""project version history: compressed checkpoints

The server-side VERSION HISTORY (``app.db.models.ProjectCheckpointRow``). Until
now the server kept exactly one snapshot per project — the latest, on
``projects.data`` — plus the append-only ``operations`` journal, which records
what was DONE and never the document each op produced. Going back therefore
required the CLIENT to still be holding the older document. This table is what
lets the server answer "give me this project as it was" on its own.

A CHECKPOINT IS NOT A MUTATION, which is why ``version`` here is not a new
counter: it stores the MUTATION number (``projects.version``) the checkpoint
froze. Checkpoints are minted on an explicit request or after the editing goes
idle, so the numbers are SPARSE (3, 7, 12) and far fewer than the mutations.

``data_gz`` is BLOB/BYTEA rather than a JSON column because the payload is
zlib-compressed UTF-8 JSON: consecutive checkpoints are near-duplicates, the
history has no TTL, and nothing ever queries inside a snapshot (every read is
"give me version N, whole"). See the model docstring for the full argument.

The UNIQUE index on (project_id, version) is load-bearing, not hygiene: it is
what makes two replicas that simultaneously decide to checkpoint the same state
produce ONE row, with the loser's INSERT failing and being reported as
"already there" instead of writing a duplicate.

No foreign key on ``project_id``, mirroring ``operations`` and ``annotations``:
the project is addressed by its uuid everywhere in this codebase and the
deletion cascade is explicit (``app.db.repositories.accounts.delete_account``)
rather than delegated to a backend that may not enforce FKs at all (SQLite does
not, by default).

``create_table``/``drop_table`` are natively supported by both SQLite and
PostgreSQL, so no batch/table-recreate is needed and ``downgrade`` is exact —
it drops a table nothing else references, losing only the history itself.

Revision ID: d5e8a2c71f60
Revises: c8b1d7f45a92
Create Date: 2026-07-27 20:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd5e8a2c71f60'
down_revision: Union[str, Sequence[str], None] = 'c8b1d7f45a92'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Must stay equal to app.db.models.CheckpointTrigger.MANUAL and that enum's
# server_default on ProjectCheckpointRow.trigger. Inlined rather than imported
# because a migration describes the schema AT THIS REVISION and must not change
# when application code does.
_DEFAULT_TRIGGER = 'manual'


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'project_checkpoints',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('project_id', sa.String(length=64), nullable=False),
        # The MUTATION number this checkpoint froze — projects.version, not a
        # counter of its own.
        sa.Column('version', sa.Integer(), nullable=False),
        # zlib over UTF-8 JSON. BLOB on SQLite, BYTEA on PostgreSQL.
        sa.Column('data_gz', sa.LargeBinary(), nullable=False),
        sa.Column(
            'trigger', sa.String(length=16), nullable=False,
            server_default=sa.text(f"'{_DEFAULT_TRIGGER}'"),
        ),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    # ONE checkpoint per (project, mutation number). Also the only index the
    # table needs: leading with project_id serves both "this project's
    # checkpoints" and "this project at version N".
    op.create_index(
        'ix_project_checkpoints_project_id_version',
        'project_checkpoints', ['project_id', 'version'], unique=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        'ix_project_checkpoints_project_id_version',
        table_name='project_checkpoints',
    )
    op.drop_table('project_checkpoints')
