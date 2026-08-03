"""quotas: per-owner limits + storage usage counter

Gate 2 Step 3. One ``quotas`` row per owner carrying the account's resource
limits AND its live ``storage_bytes_used`` counter (see ``QuotaRow`` for why a
separate table rather than columns on ``users``).

One data step beyond the schema, mirroring the ``analysis_access`` seeding of
the previous revision: a quota row is BACKFILLED for every existing account,
so a deployment that already has tenants meters them from the first byte after
this upgrade instead of only from the moment each is next touched. A fresh
install has no users and nothing to backfill.

The default limits below are inlined literals (Alembic runs without importing
app code beyond env.py) and MUST stay equal to ``app.db.repositories.quotas``'
DEFAULT_* constants and ``app.db.models.QuotaRow``'s ``server_default``s.

Revision ID: b7f2a1c9d3e4
Revises: c3959889aab8
Create Date: 2026-07-22 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7f2a1c9d3e4'
down_revision: Union[str, Sequence[str], None] = 'c3959889aab8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Kept in lockstep with app.db.repositories.quotas.DEFAULT_* and
# app.db.models.QuotaRow.server_default — a divergence would let a backfilled
# account and a Step-7-registered account disagree on the same "default".
_DEFAULT_STORAGE_BYTES_LIMIT = 5 * 1024 ** 3  # 5 GiB
_DEFAULT_MAX_CONCURRENT_JOBS = 3


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'quotas',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column(
            'storage_bytes_limit', sa.BigInteger(), nullable=False,
            server_default=sa.text(str(_DEFAULT_STORAGE_BYTES_LIMIT)),
        ),
        sa.Column(
            'storage_bytes_used', sa.BigInteger(), nullable=False,
            server_default=sa.text('0'),
        ),
        sa.Column(
            'max_concurrent_jobs', sa.Integer(), nullable=False,
            server_default=sa.text(str(_DEFAULT_MAX_CONCURRENT_JOBS)),
        ),
        sa.Column('cpu_minutes_limit', sa.Integer(), nullable=True),
        sa.Column(
            'created_at', sa.DateTime(timezone=True),
            server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False,
        ),
        sa.Column(
            'updated_at', sa.DateTime(timezone=True),
            server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False,
        ),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    # One quota per owner: a single UNIQUE index carries both the constraint
    # and the lookup (QuotaRow.owner_id is unique=True, index=True) — matching
    # how ``users.email`` is expressed, not a separate named UniqueConstraint.
    op.create_index(op.f('ix_quotas_owner_id'), 'quotas', ['owner_id'], unique=True)
    _backfill_existing_users()


def _backfill_existing_users() -> None:
    """Give every existing account a default quota row.

    One INSERT..SELECT, portable across SQLite and PostgreSQL: the column
    server_defaults supply the limits and the zeroed counter, so only
    ``owner_id`` is projected. Guarded by NOT EXISTS so a re-run (or a partial
    upgrade retried) never inserts a second row for the same owner.
    """
    bind = op.get_bind()
    bind.execute(
        sa.text(
            "INSERT INTO quotas (owner_id) "
            "SELECT id FROM users u "
            "WHERE NOT EXISTS (SELECT 1 FROM quotas q WHERE q.owner_id = u.id)"
        )
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_quotas_owner_id'), table_name='quotas')
    op.drop_table('quotas')
