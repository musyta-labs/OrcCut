"""sessions and password reset tokens

Gate 2 Step 8: schema for two new pieces of account self-service.

``sessions`` backs a per-session server-side kill-switch for browser cookie
sessions (``app.db.models.SessionRow``). Before this table, a cookie session
died only by expiry or by disabling ``users.status`` account-wide — there was
no way to invalidate ONE browser's session without also killing every other
session of the same account. ``session_id`` is carried inside the signed
cookie alongside ``user_id`` (``app.auth.sessions``); revoking is setting
``revoked_at`` on the one row that session_id names.

``password_reset_tokens`` stores only the SHA-256 hash of a one-time,
short-lived, emailed reset token (``app.db.models.PasswordResetTokenRow``),
mirroring ``api_tokens.token_hash`` — the plaintext is shown to the account
holder (via email) exactly once and is never persisted.

Both tables use ``sa.DateTime(timezone=True)`` for their aware-datetime
columns, not ``app.db.types.AwareDateTime`` directly: that type's ``impl`` IS
``DateTime(timezone=True)`` with no DDL of its own (see that module's
docstring), and the existing ``api_tokens``/``d461a3e6aaad`` migration
already established this as the emitted-DDL convention for such columns.

Generated via ``alembic revision --autogenerate`` against a database already
at the ``c3959889aab8`` head; reviewed by hand against ``app/db/models.py``
before being committed, same process as every migration since the baseline.

Revision ID: 1e4ed6161bfb
Revises: c3959889aab8
Create Date: 2026-07-22 13:51:10.123093

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1e4ed6161bfb'
down_revision: Union[str, Sequence[str], None] = 'c3959889aab8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('password_reset_tokens',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('used_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_password_reset_tokens_token_hash'), 'password_reset_tokens', ['token_hash'], unique=True)
    op.create_index(op.f('ix_password_reset_tokens_user_id'), 'password_reset_tokens', ['user_id'], unique=False)
    op.create_table('sessions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('session_id', sa.String(length=64), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_sessions_session_id'), 'sessions', ['session_id'], unique=True)
    op.create_index(op.f('ix_sessions_user_id'), 'sessions', ['user_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_sessions_user_id'), table_name='sessions')
    op.drop_index(op.f('ix_sessions_session_id'), table_name='sessions')
    op.drop_table('sessions')
    op.drop_index(op.f('ix_password_reset_tokens_user_id'), table_name='password_reset_tokens')
    op.drop_index(op.f('ix_password_reset_tokens_token_hash'), table_name='password_reset_tokens')
    op.drop_table('password_reset_tokens')
