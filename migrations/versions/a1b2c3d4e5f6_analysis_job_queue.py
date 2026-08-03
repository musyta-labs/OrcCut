"""analysis job queue

Gate 3 Step 6. One new table (``analysis_jobs``) that is the durable, shared
broker replacing ``app.analysis.background._inflight``: dedup, pending-state and
the "which job runs next" claim all move from a per-process set into a table
every replica reads, so they hold across replicas. See ``app.db.job_models`` for
the column rationale and ``app.db.repositories.analysis_jobs`` for the
enqueue/claim/complete logic (``INSERT ... ON CONFLICT`` dedup, ``FOR UPDATE
SKIP LOCKED`` claim on PostgreSQL / serialised transaction on SQLite).

A pure addition — a single ``CREATE TABLE`` that touches no existing table and
takes no lock on live data, so it is safe to apply to a running database (the
operator constraint for this wave: no blocking ALTER). A deployment that stays
single-replica on the ``memory`` shared-state backend gets the table and simply
never reads it — ``_inflight`` remains the truth there, unchanged bit-for-bit.

Revision ID: a1b2c3d4e5f6
Revises: e7c2f9a4b6d1
Create Date: 2026-07-23 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'e7c2f9a4b6d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'analysis_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('project_id', sa.String(length=64), nullable=False),
        sa.Column('media_id', sa.String(length=64), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('claimed_by', sa.String(length=64), nullable=True),
        sa.Column('heartbeat_at', sa.Float(), nullable=True),
        sa.Column(
            'created_at', sa.DateTime(timezone=True),
            server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False,
        ),
        sa.PrimaryKeyConstraint('id'),
        # At most one LIVE job per asset, cluster-wide — the cross-replica dedup
        # the ON CONFLICT enqueue infers against (terminal rows are deleted, so
        # the key frees for a later re-analysis).
        sa.UniqueConstraint('project_id', 'media_id', name='uq_analysis_jobs_key'),
    )
    op.create_index(
        op.f('ix_analysis_jobs_owner_id'), 'analysis_jobs', ['owner_id'], unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_analysis_jobs_owner_id'), table_name='analysis_jobs')
    op.drop_table('analysis_jobs')
