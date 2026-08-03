"""shared-state accounting tables: slot leases + sliding-window hits

Gate 3 Step 4. Two new tables backing the ``database`` shared-state backend
(``app.common.shared_state_db``) so the sliding-window, inc-with-limit and
TTL-lease accounting is shared across replicas instead of per-process. See
``app.db.state_models`` for the column rationale.

Pure additions — two ``CREATE TABLE``s that touch no existing table and take no
lock on live data, so this is safe to apply to a running database (the operator
constraint for this step: no blocking ALTER). The default ``memory`` backend
never reads these tables, so a deployment that stays single-replica gets the
tables and ignores them.

``expires_at`` is a FLOAT (epoch seconds), not a timestamp: every comparison
against it is a relative deadline evaluated numerically, identical on SQLite and
PostgreSQL with no timezone round-trip — the mismatch ``AwareDateTime`` exists to
handle for the audit columns is deliberately avoided here.

Revision ID: e7c2f9a4b6d1
Revises: f3d9c1b2a7e4
Create Date: 2026-07-23 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e7c2f9a4b6d1'
down_revision: Union[str, Sequence[str], None] = 'f3d9c1b2a7e4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'state_slot_leases',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('slot_key', sa.String(length=255), nullable=False),
        sa.Column('lease_token', sa.String(length=64), nullable=False),
        sa.Column('expires_at', sa.Float(), nullable=True),
        sa.Column(
            'created_at', sa.DateTime(timezone=True),
            server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False,
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_state_slot_leases_slot_key'), 'state_slot_leases',
        ['slot_key'], unique=False,
    )
    # UNIQUE: a release targets exactly one lease by its token, and the second
    # release of the same token must match nothing (idempotent).
    op.create_index(
        op.f('ix_state_slot_leases_lease_token'), 'state_slot_leases',
        ['lease_token'], unique=True,
    )

    op.create_table(
        'state_window_hits',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('window_key', sa.String(length=255), nullable=False),
        sa.Column('expires_at', sa.Float(), nullable=False),
        sa.Column(
            'created_at', sa.DateTime(timezone=True),
            server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False,
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_state_window_hits_window_key'), 'state_window_hits',
        ['window_key'], unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_state_window_hits_window_key'), table_name='state_window_hits')
    op.drop_table('state_window_hits')
    op.drop_index(op.f('ix_state_slot_leases_lease_token'), table_name='state_slot_leases')
    op.drop_index(op.f('ix_state_slot_leases_slot_key'), table_name='state_slot_leases')
    op.drop_table('state_slot_leases')
