"""Alembic wiring: config resolution, baseline-revision lookup, and
pre-Alembic database adoption logic.

Split out of ``app/db/base.py`` deliberately: that module is about the
engine/session lifecycle (connection pooling, WAL) — a different reason to
change than "how do we classify and migrate a database's schema state",
which is what lives here. ``init_db()`` in ``app/db/base.py`` stays thin and
calls into this module.

The four names below (``REPO_ROOT``, ``alembic_config``,
``baseline_revision``, ``MigrationState``, ``migration_state``) are
deliberately NOT underscore-prefixed, unlike most internal helpers in this
codebase: ``tests/test_migrations.py`` exercises them directly as a
white-box contract on the adoption logic, not only through ``init_db()``'s
end-to-end behavior — see that test module's own docstring.
"""
from __future__ import annotations

import enum
import os
import tempfile
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import MetaData, create_engine, inspect

from app.db.base import Base

# Repo root, resolved from this file's own location rather than the process's
# cwd: app/db/migrations.py -> app/db -> app -> repo root, where alembic.ini
# and migrations/ live. A container can be started from any working
# directory, so `Path("alembic.ini")` (relative to cwd) would be wrong; this
# is not. It is ALSO where the Docker image must physically place these two
# paths — see docker/Dockerfile's `COPY alembic.ini ./` / `COPY migrations
# ./migrations` alongside `COPY app ./app`. Without those, this resolves to
# a real path on disk that simply doesn't exist in the image, and every
# `init_db()` call fails at container start, on every database state.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def alembic_config() -> Config:
    """Build an Alembic ``Config`` anchored to ``REPO_ROOT``, not cwd.
    ``alembic.ini``'s own ``script_location = %(here)s/migrations`` also
    resolves relative to the ini file's path, so this is correct regardless
    of where the ini is passed from."""
    return Config(str(REPO_ROOT / "alembic.ini"))


def baseline_revision() -> str:
    """The single root revision (``down_revision is None``) in
    ``migrations/versions/`` — the one that stamps today's four pre-Alembic
    tables without altering them. Resolved from the migration scripts
    themselves, not hardcoded, so it can't silently drift if that file is
    ever renamed."""
    roots = ScriptDirectory.from_config(alembic_config()).get_bases()
    if len(roots) != 1:
        raise RuntimeError(
            "Expected exactly one baseline (root) revision in "
            f"migrations/versions/, found {roots!r} — adoption of a "
            "pre-Alembic database is ambiguous."
        )
    return roots[0]


class MigrationState(enum.StrEnum):
    """How a database relates to Alembic, decided BEFORE ``init_db()`` picks
    an action — see ``migration_state``."""

    EMPTY = "empty"
    UNVERSIONED_EXISTING_SCHEMA = "unversioned_existing_schema"
    UNVERSIONED_PARTIAL_BASELINE = "unversioned_partial_baseline"
    VERSIONED = "versioned"


# The exact table set the BASELINE revision (5ed7cccf2291) creates — fixed
# forever, unlike ``Base.metadata.tables``, which grows with every later
# revision (Gate 1 Step 3 added ``users``/``api_tokens`` this way). A
# genuinely pre-Alembic database — built by the OLD
# ``Base.metadata.create_all`` path, before Gate 1 Step 2 replaced it in
# ``init_db()`` — can only ever contain exactly this table set: it predates
# every table added by a later revision by construction, and after Step 2
# shipped, ``create_all`` is never called again in the production path, so
# no NEW unversioned database can appear with a larger schema either. Kept
# as a literal constant (not derived from ``Base.metadata``, which reflects
# TODAY's models) deliberately — comparing against today's growing metadata
# here is exactly the bug this constant fixes, see ``migration_state``'s own
# regression: a real legacy 4-table database on this codebase version used
# to raise ``RuntimeError`` (misread as a "partial" schema, because
# ``Base.metadata`` had already grown to 6 tables) instead of being adopted.
_BASELINE_TABLES = frozenset({"projects", "operations", "annotations", "media_analyses"})


def migration_state(database_url: str) -> MigrationState:
    """Classify a database for ``init_db()``'s adoption logic.

    - ``EMPTY``: no application tables exist — a normal ``upgrade head``
      creates everything from scratch.
    - ``UNVERSIONED_EXISTING_SCHEMA``: exactly the BASELINE's four tables
      exist (``_BASELINE_TABLES``) but ``alembic_version`` does not — this
      database predates Alembic (built by the old
      ``Base.metadata.create_all`` path). Running ``upgrade head`` against it
      directly re-issues the baseline's ``CREATE TABLE`` statements and fails
      with "table ... already exists" — it must be STAMPED at the baseline
      revision instead, never re-created.
    - ``UNVERSIONED_PARTIAL_BASELINE``: a strict, non-empty SUBSET of the
      baseline's tables exists, with no ``alembic_version``. This is a normal
      historical database, not a damaged one: before Alembic, the schema grew
      one table at a time through ``create_all`` (``app/db/models.py:12-15``),
      so a deployment from before ``annotations``/``media_analyses`` existed
      simply has fewer than four. It is adopted by CREATING the missing
      baseline tables and then stamping the baseline
      (``_create_missing_baseline_tables``).

      This case used to raise, and that was a blocking defect: a real
      deployment database (observed in a live
      volume, dated 2026-07-16, holding live project and operation rows) has
      exactly ``projects`` and ``operations``, so ``init_db()`` raised at
      container start, before the port was bound. Every deployment of that
      era was unbootable.
    - ``VERSIONED``: ``alembic_version`` already exists — a normal
      ``upgrade head`` applies whatever is pending.

    Raises ``RuntimeError`` when unversioned tables exist that are NOT a
    subset of the baseline — i.e. a table some LATER revision owns. A
    pre-Alembic database predates the baseline by definition, so it cannot
    contain those; stamping such a database at the baseline would record
    revisions that never ran, and the next ``upgrade head`` would collide
    with the table it is about to create.
    """
    # Deferred, not top-of-module: app.db.models imports Base FROM
    # app.db.base, and this module also imports Base from app.db.base — a
    # module-level import of app.db.models here works today, but the
    # deferred form is kept anyway because it must happen HERE regardless of
    # import order. Base.metadata.tables is only populated once
    # app.db.models has been imported somewhere, and relying on that having
    # already happened as a side effect of the CALLER's own import graph is
    # exactly the fragility that once silently misclassified an existing
    # schema as EMPTY in an otherwise-correct standalone script (no
    # application code had imported app.db.models yet), sending it down the
    # "create everything" path and reproducing the "table ... already
    # exists" failure this function exists to prevent.
    from app.db import models  # noqa: F401

    inspection_engine = create_engine(database_url)
    try:
        existing_tables = set(inspect(inspection_engine).get_table_names())
    finally:
        inspection_engine.dispose()

    if "alembic_version" in existing_tables:
        return MigrationState.VERSIONED

    app_tables = set(Base.metadata.tables.keys())
    present = app_tables & existing_tables
    if not present:
        return MigrationState.EMPTY
    if present == _BASELINE_TABLES:
        return MigrationState.UNVERSIONED_EXISTING_SCHEMA
    if present < _BASELINE_TABLES:
        return MigrationState.UNVERSIONED_PARTIAL_BASELINE
    raise RuntimeError(
        f"Database has an unversioned schema holding tables outside the "
        f"baseline (found {sorted(present)}, baseline is "
        f"{sorted(_BASELINE_TABLES)}) with no alembic_version table. A "
        "genuinely pre-Alembic database predates the baseline, so its tables "
        "are necessarily a subset of it; a table a LATER revision owns cannot "
        "be explained that way. init_db() will not guess — inspect and fix "
        "the database manually before restarting."
    )


def upgrade_to_head(database_url: str) -> None:
    """Bring ``database_url`` to ``head``, adopting a pre-Alembic database by
    stamping it at the baseline revision first if needed. The one function
    ``app.db.base.init_db()`` delegates to — see that function's docstring
    for the known N-instance startup race this does NOT solve."""
    cfg = alembic_config()
    state = migration_state(database_url)
    if state == MigrationState.UNVERSIONED_PARTIAL_BASELINE:
        _create_missing_baseline_tables(database_url)
    if state in (
        MigrationState.UNVERSIONED_EXISTING_SCHEMA,
        MigrationState.UNVERSIONED_PARTIAL_BASELINE,
    ):
        command.stamp(cfg, baseline_revision())
    command.upgrade(cfg, "head")


def _create_missing_baseline_tables(database_url: str) -> None:
    """Bring a partially-built pre-Alembic database up to the FULL baseline,
    so it can then be stamped at the baseline revision like any other.

    The missing tables are built **as the BASELINE revision defines them**, not
    as ``Base.metadata`` defines them today. Those two drifted apart the moment
    a later revision altered one of the four: Gate 1 Step 7 added
    ``projects.owner_id`` and replaced ``annotations.author`` with
    ``annotations.author_user_id``, so a ``Base.metadata.create_all`` here
    produced tables that ALREADY carried Step 7's columns, and the very next
    ``upgrade head`` — which still has Step 7's revision pending — died with
    "duplicate column name: author_user_id".

    That failure is not theoretical. The database this branch exists for (found
    in a live deployment volume, holding only
    ``projects`` and ``operations``) hits exactly this path, and the error
    lands inside ``init_db()`` at container start, before the port is bound —
    the same shape of unbootable deployment Step 2 and Step 3 each shipped
    once. It is the same root cause both earlier times had: reconstructing a
    HISTORICAL schema from TODAY's growing metadata. ``_BASELINE_TABLES`` fixed
    the table-name half of it; this fixes the column half.

    The baseline's own DDL is recovered by running that revision into a
    throwaway database and reflecting the result, rather than being duplicated
    as literal ``Column`` definitions here — a second copy would be one more
    thing that can drift from the revision it claims to mirror.

    Stamping without this step would be the dangerous move: it would record
    that the baseline revision had run when two of its four tables did not
    exist, and nothing afterwards would ever create them.
    """
    engine = create_engine(database_url)
    try:
        missing = _BASELINE_TABLES - set(inspect(engine).get_table_names())
        if not missing:
            return
        baseline_metadata = _reflect_baseline_schema()
        for name in sorted(missing):
            baseline_metadata.tables[name].create(engine)
    finally:
        engine.dispose()


def _reflect_baseline_schema() -> MetaData:
    """The baseline revision's schema, as a reflected ``MetaData``.

    Built by running ONLY that revision into a scratch database and reading
    the result back, so it describes the four tables exactly as they were when
    Alembic was adopted — frozen, and immune to every later revision.
    """
    with tempfile.TemporaryDirectory() as scratch_dir:
        scratch_url = f"sqlite:///{Path(scratch_dir) / 'baseline.db'}"
        cfg = alembic_config()
        # env.py resolves the url from get_settings(), not from this Config
        # object, so the env var is the only lever that actually redirects it.
        previous = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = scratch_url
        try:
            from app.config import get_settings

            get_settings.cache_clear()
            command.upgrade(cfg, baseline_revision())
        finally:
            if previous is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = previous
            get_settings.cache_clear()
        scratch_engine = create_engine(scratch_url)
        try:
            metadata = MetaData()
            metadata.reflect(bind=scratch_engine, only=sorted(_BASELINE_TABLES))
            return metadata
        finally:
            scratch_engine.dispose()
