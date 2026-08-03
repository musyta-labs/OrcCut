"""media library: per-user file shelf + its own quota columns

The user MEDIA LIBRARY — a per-account set of files that exists outside any
project (see ``app.db.models.MediaLibraryFileRow``). Two schema changes in one
revision because they are one feature and must not be able to land apart: the
table would be unusable without its budget, and the budget would be dead weight
without the table.

The two ``quotas`` columns are added with a ``server_default`` on purpose,
NOT as nullable columns backfilled afterwards: an existing account must come
out of this upgrade with a REAL 500 MB limit, and a NULL limit would make the
accountant's ``used + n <= limit`` guard unsatisfiable — every library upload
would be rejected for every pre-existing tenant, with no error explaining why.
The server_default supplies the value for every existing row in the same ALTER,
so there is no separate UPDATE and no window in which the column is NULL.

The default limits below are inlined literals (Alembic runs without importing
app code beyond env.py) and MUST stay equal to
``app.db.repositories.quotas.DEFAULT_MEDIA_LIBRARY_BYTES_LIMIT`` and
``app.db.models.QuotaRow``'s ``server_default``s.

Revision ID: f4a7c2e91b83
Revises: a1b2c3d4e5f6
Create Date: 2026-07-27 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f4a7c2e91b83'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Kept in lockstep with app.db.repositories.quotas.DEFAULT_MEDIA_LIBRARY_BYTES_LIMIT
# and app.db.models.QuotaRow.media_bytes_limit's server_default — a divergence
# would let a migrated account and a freshly-registered one disagree on the
# same "default".
_DEFAULT_MEDIA_LIBRARY_BYTES_LIMIT = 500 * 1024 ** 2  # 500 MB


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'media_library_files',
        # A uuid hex, not an autoincrementing integer: this id appears in URLs
        # and in the storage key, where a sequential value would advertise how
        # many files exist and invite id enumeration.
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('filename', sa.String(length=120), nullable=False),
        sa.Column('size_bytes', sa.BigInteger(), nullable=False),
        sa.Column('content_type', sa.String(length=128), nullable=True),
        sa.Column('storage_key', sa.String(length=512), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_media_library_files_owner_id'),
        'media_library_files', ['owner_id'], unique=False,
    )
    # UNIQUE: two rows must never address the same object, or one row's delete
    # would silently destroy the other's file.
    op.create_index(
        op.f('ix_media_library_files_storage_key'),
        'media_library_files', ['storage_key'], unique=True,
    )
    # The library screen's one query: this owner's files, newest first.
    op.create_index(
        'ix_media_library_files_owner_id_created_at',
        'media_library_files', ['owner_id', 'created_at'], unique=False,
    )

    # Plain ADD COLUMN, not a batch/recreate: adding a column with a CONSTANT
    # default is natively supported by both SQLite and PostgreSQL, and a table
    # recreate would needlessly rewrite the hot quota counters.
    op.add_column(
        'quotas',
        sa.Column(
            'media_bytes_limit', sa.BigInteger(), nullable=False,
            server_default=sa.text(str(_DEFAULT_MEDIA_LIBRARY_BYTES_LIMIT)),
        ),
    )
    op.add_column(
        'quotas',
        sa.Column(
            'media_bytes_used', sa.BigInteger(), nullable=False,
            server_default=sa.text('0'),
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('quotas', 'media_bytes_used')
    op.drop_column('quotas', 'media_bytes_limit')
    op.drop_index(
        'ix_media_library_files_owner_id_created_at', table_name='media_library_files'
    )
    op.drop_index(
        op.f('ix_media_library_files_storage_key'), table_name='media_library_files'
    )
    op.drop_index(
        op.f('ix_media_library_files_owner_id'), table_name='media_library_files'
    )
    op.drop_table('media_library_files')
