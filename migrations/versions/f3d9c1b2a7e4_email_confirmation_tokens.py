"""email confirmation tokens

Gate 2 Step 7: schema for the one-time, short-lived, emailed token that flips
``users.email_confirmed`` from the ``False`` public registration writes to
``True`` (``app.db.models.EmailConfirmationTokenRow``). Structurally identical
to ``password_reset_tokens`` (revision ``1e4ed6161bfb``) — only its hash is
stored; the plaintext is emailed to the account holder exactly once and is
never persisted.

Uses ``sa.DateTime(timezone=True)`` for its aware-datetime columns, not
``app.db.types.AwareDateTime`` directly: that type's ``impl`` IS
``DateTime(timezone=True)`` with no DDL of its own, matching the emitted-DDL
convention every prior migration for such columns already established.

Reviewed by hand against ``app/db/models.py`` before being committed, same
process as every migration since the baseline.

Revision ID: f3d9c1b2a7e4
Revises: 8239bf618215
Create Date: 2026-07-22 15:20:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f3d9c1b2a7e4'
down_revision: Union[str, Sequence[str], None] = '8239bf618215'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('email_confirmation_tokens',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('used_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_email_confirmation_tokens_token_hash'), 'email_confirmation_tokens', ['token_hash'], unique=True)
    op.create_index(op.f('ix_email_confirmation_tokens_user_id'), 'email_confirmation_tokens', ['user_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_email_confirmation_tokens_user_id'), table_name='email_confirmation_tokens')
    op.drop_index(op.f('ix_email_confirmation_tokens_token_hash'), table_name='email_confirmation_tokens')
    op.drop_table('email_confirmation_tokens')
