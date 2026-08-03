import logging
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# app.db.models must be imported (not just app.db.base) so every mapped
# table registers on Base.metadata before autogenerate/compare inspects it —
# importing app.db.base alone leaves the metadata empty.
from app.config import get_settings
from app.db.base import Base
from app.db import models  # noqa: F401  (populates Base.metadata as a side effect)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging — but ONLY when nothing has
# configured logging yet, i.e. the `alembic` CLI case. The app runs
# `upgrade head` programmatically at STARTUP (app.db.base.init_db) AFTER
# app.common.logging.setup_logging, and the ini's logging section must not
# be allowed to clobber that: fileConfig's disable_existing_loggers default
# of True permanently disabled every already-imported app logger, and even
# with it False the ini's root level=WARNING silently ate every app INFO
# line (app.email.sender's "email not sent" among them) for the life of the
# process.
if config.config_file_name is not None and not logging.getLogger().handlers:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# The database URL always comes from the application's own settings
# (env vars / .env via app.config.get_settings), never from alembic.ini —
# so a single source of truth drives both the app and its migrations, and a
# container started from a different cwd or env still resolves correctly.
config.set_main_option("sqlalchemy.url", get_settings().database_url)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
