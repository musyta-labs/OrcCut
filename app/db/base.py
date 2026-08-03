"""Database engine + session factory for the standalone editor service.

Engine construction is DIALECT-AWARE (Gate 3 Step 1): the SQLite self-host path
and a PostgreSQL multi-replica deployment need different ``create_engine``
wiring, and mixing them breaks both. ``check_same_thread`` and
``PRAGMA journal_mode=WAL`` are SQLite-only — psycopg rejects the former as an
unknown kwarg and the latter is not valid SQL on PostgreSQL — while a real
connection pool (pre-ping + sizing) only matters for a server backend. The
decision hangs off the URL's backend name (``_is_sqlite``), read WITHOUT
opening a connection or importing a driver, so a SQLite build never needs
psycopg present.

Schema migration logic (Alembic config, baseline-revision lookup, pre-Alembic
database adoption) lives in ``app.db.migrations`` — a separate reason to
change from the engine/session lifecycle here. ``init_db()`` below is a thin
delegate to keep that split visible: this file owns "how do we talk to the
database", not "what state is the schema in".
"""
from __future__ import annotations

import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Connection, Engine, create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import Settings, get_settings


class Base(DeclarativeBase):
    pass


def _is_sqlite(database_url: str) -> bool:
    """Whether ``database_url`` targets SQLite, decided from its backend name
    alone — no connection is opened and no DBAPI driver is imported, so this is
    safe to call on a SQLite-only install with no psycopg present."""
    return make_url(database_url).get_backend_name() == "sqlite"


def _engine_options(database_url: str, settings: Settings) -> dict:
    """The ``create_engine`` keyword arguments for ``database_url``.

    Common to every backend: ``hide_parameters`` keeps bound parameters OUT of
    exception text — a unique-violation IntegrityError
    otherwise stringifies the offending row (including a user's email), and any
    centralized error logger or 500 body would leak it. The DBAPI still gets
    the params; only ``str(exc)`` is redacted.

    SQLite-only: ``connect_args={"check_same_thread": False}`` — FastMCP may
    dispatch a tool call off the thread that created the engine, and SQLite's
    default same-thread check would otherwise raise. It is a SQLite DBAPI arg
    and must NEVER be sent to psycopg.

    Non-SQLite: a real connection pool — ``pool_pre_ping`` validates a
    connection before use (a proxy may have silently dropped it), and
    ``pool_size``/``max_overflow``/``pool_recycle`` bound and refresh the
    persistent connections one replica holds. SQLite gets none of these: its
    file access is process-local and a server-connection pool is meaningless
    there, so today's single-replica behaviour is left exactly as it was.
    """
    options: dict = {"hide_parameters": True}
    if _is_sqlite(database_url):
        options["connect_args"] = {"check_same_thread": False}
        return options
    options["pool_pre_ping"] = True
    options["pool_size"] = settings.db_pool_size
    options["max_overflow"] = settings.db_max_overflow
    options["pool_recycle"] = settings.db_pool_recycle
    return options


# How long a SQLite writer waits for the single write lock before giving up
# with ``database is locked``. The sqlite3 default (5 s) predates any real
# concurrency here; with tens of request threads sharing one file, a login or
# save colliding with another writer's short transaction must WAIT, not 500.
# 30 s is far above any legitimate write transaction in this app (export
# commits its status changes in their own short transactions — see
# ``app.mcp.tools.export``), so a wait this long only ever happens when
# something is genuinely wrong, and failing then is correct.
SQLITE_BUSY_TIMEOUT_MS = 30_000


def _build_engine(database_url: str, settings: Settings) -> Engine:
    """Create the engine for ``database_url`` and register any dialect-specific
    connect hooks. Both PRAGMAs are SQLite-only and are registered ONLY for a
    SQLite URL — firing them on every psycopg connection would raise."""
    engine = create_engine(database_url, **_engine_options(database_url, settings))
    if _is_sqlite(database_url):
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _record):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    return engine


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    return _build_engine(settings.database_url, settings)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# A fixed, arbitrary key in Postgres's session-advisory-lock keyspace (one
# process-wide bigint namespace shared by every ``pg_advisory_lock`` call on
# the same server, for any reason — see
# https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS).
# Derived by CRC32 of a descriptive name rather than hand-picked, so its
# origin is legible, and fixed permanently once chosen: two DIFFERENT keys
# here across a code change would let two replicas each believe they hold
# "the" migration lock at once and defeat the whole point.
_MIGRATION_ADVISORY_LOCK_KEY = zlib.crc32(b"mcpcut:schema-migration")


def _is_postgres_url(database_url: str) -> bool:
    """True when ``database_url`` targets Postgres under any driver
    (``postgresql://``, ``postgresql+psycopg://``, ...) or the legacy
    ``postgres://`` scheme some hosts (e.g. Heroku-style ``DATABASE_URL``
    values) still hand out — SQLAlchemy's own parser reports THAT one's
    backend name as the literal ``"postgres"``, not ``"postgresql"``, so
    both are checked explicitly rather than trusting a single string.

    Resolved via SQLAlchemy's own URL parser rather than a hand-rolled
    prefix check, so a driver suffix this codebase hasn't seen yet still
    classifies correctly.
    """
    return make_url(database_url).get_backend_name() in {"postgresql", "postgres"}


def _acquire_advisory_lock(conn: Connection) -> None:
    """Block until this connection holds the migration's session-level
    Postgres advisory lock.

    Split out of ``_migrate_with_advisory_lock`` as its own top-level
    function — not inlined SQL — so a test can monkeypatch this (and
    ``_release_advisory_lock``) onto a real ``threading.Lock`` and prove the
    SERIALIZATION behavior of ``init_db()`` without a real Postgres server
    (see ``tests/test_migrations.py``; Gate 3 Step 3 adds an optional CI run
    against real Postgres that exercises this SQL for real).
    """
    conn.execute(text("SELECT pg_advisory_lock(:key)"), {"key": _MIGRATION_ADVISORY_LOCK_KEY})


def _release_advisory_lock(conn: Connection) -> None:
    """Release the lock ``_acquire_advisory_lock`` took. Always called from a
    ``finally`` so a replica whose migration itself fails still frees the
    lock for the next one to try, rather than wedging every other replica
    forever."""
    conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": _MIGRATION_ADVISORY_LOCK_KEY})


def _migrate_with_advisory_lock(database_url: str) -> None:
    """Serialize ``upgrade_to_head`` across every replica racing the same
    cold Postgres database via a session-level ``pg_advisory_lock``.

    This is the fix for the race ``init_db``'s docstring used to only
    document: N replicas each observing "not yet migrated" and each
    attempting ``CREATE TABLE``/``stamp``, with N-1 losing and exiting
    non-zero. Here only one replica's connection ever holds the lock at a
    time — every other caller blocks INSIDE ``pg_advisory_lock`` until it is
    released, then runs its own ``upgrade_to_head`` against a database the
    first replica has already brought to ``head``; Alembic's ``upgrade``
    against a database already at ``head`` is a no-op, so the losers do not
    re-attempt the DDL that used to make them the "N-1 that raise".

    A dedicated engine/connection here (not ``get_engine()``) is
    deliberate: the lock is SESSION-scoped in Postgres — it must be held on
    one specific DBAPI connection for the whole migration and released on
    that same connection, never on one a pool could hand to another caller
    mid-lock.
    """
    from app.db.migrations import upgrade_to_head

    # Postgres-only — init_db returns early for SQLite one branch above, so
    # this bare engine deliberately has no WAL/busy_timeout pragma listeners;
    # adding them "for consistency" would raise on psycopg.
    engine = create_engine(database_url)
    try:
        with engine.connect() as conn:
            _acquire_advisory_lock(conn)
            try:
                upgrade_to_head(database_url)
            finally:
                _release_advisory_lock(conn)
    finally:
        engine.dispose()


def init_db() -> None:
    """Bring the schema to ``head`` via Alembic migrations (delegates to
    ``app.db.migrations.upgrade_to_head`` — see that module for the config
    resolution and pre-Alembic adoption logic).

    Replaces the old ``Base.metadata.create_all`` (which only ever created
    MISSING tables and could never evolve an existing one — see
    ``migrations/versions/`` for the full history now).

    **Concurrent startup (Gate 3 Step 2).** On Postgres, N replicas calling
    this at once against a cold/unversioned database are serialized through
    ``_migrate_with_advisory_lock``: one replica migrates, the rest queue on
    ``pg_advisory_lock`` and return once the schema is already at ``head``.
    This closes the race an earlier version of this docstring only
    documented — see ``_migrate_with_advisory_lock`` for the mechanism, and
    ``tests/test_migrations.py`` for the concurrent-replica regression test
    (SQLite-backed with the lock primitives mocked onto a real
    ``threading.Lock``, since no real Postgres runs in this suite; Gate 3
    Step 3 adds an optional CI job that exercises the real
    ``pg_advisory_lock`` SQL).

    ``RUN_MIGRATIONS_ON_STARTUP`` (``Settings.run_migrations_on_startup``,
    default ``True``) lets a Postgres-mode replica skip running the startup
    migration in its OWN process entirely — e.g. every replica except a
    single designated migrator sets this ``False``, so they never even
    attempt the advisory-lock wait. It is read ONLY on the Postgres branch:
    SQLite is always a single process against its own file, so there is no
    "which replica migrates" question to opt out of, and honoring the flag
    there would risk leaving the ONE process able to reach ``head`` unable
    to.

    On SQLite, behavior is unchanged from before this Step: ``init_db()``
    calls ``upgrade_to_head`` directly, no lock, no flag — ``docker-compose.yml``
    runs exactly one instance against this database, so there is nothing to
    serialize.
    """
    from app.db.migrations import upgrade_to_head

    settings = get_settings()
    database_url = settings.database_url

    if not _is_postgres_url(database_url):
        upgrade_to_head(database_url)
        return

    if not settings.run_migrations_on_startup:
        return

    _migrate_with_advisory_lock(database_url)
