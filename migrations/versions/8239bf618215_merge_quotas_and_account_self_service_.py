"""merge quotas and account-self-service heads

Revision ID: 8239bf618215
Revises: 1e4ed6161bfb, b7f2a1c9d3e4
Create Date: 2026-07-22 14:37:06.381193

"""
from typing import Sequence, Union



# revision identifiers, used by Alembic.
revision: str = '8239bf618215'
down_revision: Union[str, Sequence[str], None] = ('1e4ed6161bfb', 'b7f2a1c9d3e4')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
