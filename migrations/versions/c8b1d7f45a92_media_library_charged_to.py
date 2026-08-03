"""media library: record WHICH budget paid for each shelved file

``media_library_files`` rows are no longer all paid for by the same counter. A
file uploaded into a PROJECT is charged once to ``quotas.storage_bytes_used``
and then shelved here without a second charge (see
``app.web.api._register_stored_file``), while a file uploaded to the library
itself is charged to ``quotas.media_bytes_used``. The shared
``DELETE /api/media/{id}`` serves both kinds of row and had no way to tell them
apart, so it refunded the library budget unconditionally — crediting bytes that
budget had never been charged, and letting a repeated
{upload into a project → delete from the shelf} floor ``media_bytes_used`` at
zero and retire the 500 MB library limit. This column is what the refund reads.

DEFAULT ``'media'`` FOR EVERY EXISTING ROW, and that is not a fallback but the
truth: until the project ingest started writing here, the library's own upload
and the from-export copy were the ONLY writers, and both charge
``media_bytes_used``. A nullable column backfilled afterwards would leave a
window in which a delete could not answer "who paid", so the value is supplied
by the same ALTER that adds the column.

Plain ``add_column`` with a CONSTANT ``server_default``, exactly as
``f4a7c2e91b83`` adds the two quota columns: both SQLite and PostgreSQL support
that natively in a single ``ALTER TABLE``, so no batch/table-recreate is needed
(and a recreate of this table would needlessly rewrite every owner's shelf).
``downgrade`` is a plain ``drop_column`` for the same reason — SQLite has
supported ``ALTER TABLE ... DROP COLUMN`` since 3.35, which is what the
revision below this one already relies on.

The literal ``'media'`` mirrors ``app.db.models.MediaChargeBudget.MEDIA`` and
that enum's ``server_default`` on ``MediaLibraryFileRow.charged_to``; it is
inlined rather than imported because a migration describes the schema as it was
AT THIS REVISION and must not change when application code does.

Revision ID: c8b1d7f45a92
Revises: f4a7c2e91b83
Create Date: 2026-07-27 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8b1d7f45a92'
down_revision: Union[str, Sequence[str], None] = 'f4a7c2e91b83'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Must stay equal to app.db.models.MediaChargeBudget.MEDIA — a divergence would
# make every migrated row's refund go to a budget the application does not
# recognise (which app.web.media_library._release_for then has to guess at).
_DEFAULT_CHARGED_TO = 'media'


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'media_library_files',
        sa.Column(
            'charged_to', sa.String(length=16), nullable=False,
            server_default=sa.text(f"'{_DEFAULT_CHARGED_TO}'"),
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Lossy on purpose and safe to be so: without the column the delete route
    # reverts to refunding the library budget for every row, which is exactly
    # the behaviour of the revision below this one.
    op.drop_column('media_library_files', 'charged_to')
