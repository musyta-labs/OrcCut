"""SQLite-backed persistence tables for the standalone editor.

Six tables:

- ``ProjectRow`` — the latest-snapshot-plus-lock-status row.
- ``OperationLogRow`` — the append-only journal (the PRD's "own storage +
  journaled operation history").
- ``ProjectCheckpointRow`` — the server-side VERSION HISTORY: one compressed
  full snapshot per checkpoint. Distinct from the journal above, which records
  what was done rather than the states it passed through.
- ``AnnotationRow`` — operator error-markers pinned to an exported timeline.
- ``MediaAnalysisRow`` — persisted clip event detection, keyed by media
  CONTENT rather than by project or clip.
- ``UserRow`` — a registered account (Gate 1 Step 3: schema + token minting
  only, no request-time identity resolution yet — that is Step 4).
- ``ApiTokenRow`` — a minted API token's HASH, never the plaintext.
- ``MediaLibraryFileRow`` — one file on a user's personal media library shelf,
  owned by the ACCOUNT rather than by any project.

Schema changes go through Alembic (``migrations/versions/``, driven by
``app.db.base.init_db``), not ``Base.metadata.create_all`` — the baseline
revision stamps exactly the four original tables, and every future addition,
including a column on an existing table, is its own revision from here on.
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer,
    LargeBinary, String, Text, UniqueConstraint, func, text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.types import AwareDateTime

# Register the Gate 3 Step 4 shared-state tables on Base.metadata as a side
# effect of importing this module. Every metadata consumer (Alembic's env.py,
# tests' create_all, reflection) imports app.db.models; importing the state
# tables here means they are always present in the metadata without each of
# those sites having to know about a second module.
from app.db import state_models  # noqa: E402,F401

# Same side-effect import for the Gate 3 Step 6 durable analysis-job queue
# table (``app.db.job_models``): every metadata consumer imports app.db.models,
# so registering it here keeps Alembic's env.py, tests' create_all and
# reflection all aware of ``analysis_jobs``.
from app.db import job_models  # noqa: E402,F401


def utc_now() -> datetime:
    """Python-side UTC ``default``/``onupdate`` for ``AwareDateTime`` columns.

    Used where ``func.now()`` is too coarse: SQLite renders it as
    CURRENT_TIMESTAMP, which has one-second resolution, so a burst of rows
    written inside the same second becomes unorderable and an UPDATE inside the
    same second leaves ``updated_at`` looking unchanged.
    """
    return datetime.now(timezone.utc)


class ProjectStatus(enum.StrEnum):
    DRAFT = "draft"
    EXPORTING = "exporting"
    EXPORTED = "exported"


class MediaChargeBudget(enum.StrEnum):
    """WHICH quota counter paid for one media-library row's bytes.

    A library row is no longer proof that ``media_bytes_used`` was charged.
    A file uploaded INTO A PROJECT is charged to ``storage_bytes_used`` once
    and then shelved on the owner's library screen without a second charge
    (see ``app.web.api._register_stored_file`` for that bargain) — so the row
    and the counter that paid for it can differ, and only the row knows which.

    Without this the shared ``DELETE /api/media/{id}`` had to guess, and it
    guessed ``media``: deleting a project-charged row credited a budget that
    never paid, so repeating {upload into a project → delete from the shelf}
    drove ``media_bytes_used`` to its floor of zero and retired the 500 MB
    library limit entirely. The refund now follows the charge, per row.

    ``media`` is the safe default for every row written before this column
    existed: the shelf was the ONLY writer then, so every historical row was
    genuinely paid for out of the library budget.
    """

    MEDIA = "media"
    STORAGE = "storage"


class ProjectRow(Base):
    """Latest snapshot of one EditorProject, keyed by its uuid."""

    __tablename__ = "projects"

    __table_args__ = (
        # The web viewer and the MCP list tool both ask the same question —
        # "this owner's projects, newest first" — and that is now a two-column
        # question. Leading with owner_id makes the index usable for the
        # equality predicate alone as well, which every single-project lookup
        # also needs.
        Index("ix_projects_owner_id_updated_at", "owner_id", "updated_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # The tenant this project belongs to. NOT NULL with no default and no
    # server_default on purpose: a project with no owner is not a valid state,
    # and a default would let an INSERT that forgot the owner succeed quietly
    # — attributing someone's work to whoever the default named. Every write
    # path derives this from the request's Principal, never from an argument
    # the caller controls (see app.mcp.tools).
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(16), default=ProjectStatus.DRAFT.value)
    version: Mapped[int] = mapped_column(Integer, default=1)
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class OperationLogRow(Base):
    """Append-only journal: one row per mutation, kept forever (operator
    decision 2026-07-16 — 'pruning would blind the history view')."""

    __tablename__ = "operations"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64), index=True)
    op_name: Mapped[str] = mapped_column(String(64))
    op_args: Mapped[dict] = mapped_column(JSON, default=dict)
    version_after: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CheckpointTrigger(enum.StrEnum):
    """WHY a checkpoint exists — the only two occasions that mint one.

    ``MANUAL`` — the operator asked for it (``POST /projects/{id}/versions``).
    ``IDLE`` — the project stopped being edited and the background sweep
    (``app.editor.checkpoints``) froze where it came to rest.

    Recorded rather than inferred because the two mean different things to a
    reader of the history: a manual point is someone saying "this state
    matters", an idle one is only "nothing happened after this for a while".
    """

    MANUAL = "manual"
    IDLE = "idle"


class ProjectCheckpointRow(Base):
    """One project's FULL serialized state at ONE CHECKPOINT — the server-side
    version history.

    A CHECKPOINT IS NOT A MUTATION. ``ProjectRow.version`` counts mutations and
    bumps on every single op — nudging a clip two pixels is a mutation — and
    treating each of those as a "version" would bury the three states a person
    actually cares about under two hundred they never chose. A checkpoint is
    minted on exactly two occasions (see ``CheckpointTrigger``), so there are
    far fewer checkpoints than mutations, and nothing writes one from
    ``save_project``.

    ``version`` HERE IS THE MUTATION NUMBER the checkpoint froze, not a
    separate counter. That is deliberate: a third, independent numbering would
    show the user a "version 3" that appears nowhere in the operation journal
    and nowhere on the project row. The consequence is that checkpoint numbers
    are SPARSE — 3, 7, 12 — and nothing may assume they are consecutive.

    ``ProjectRow`` keeps only the latest snapshot and ``OperationLogRow`` keeps
    only what was DONE (op name + args), never the document each op produced.
    Between them there was no way for the server to hand back an earlier state:
    rolling back required the CLIENT to still be holding the older document.
    This table is the missing half — the journal says what happened, these rows
    say what the project WAS.

    LIFETIME: none of its own. A checkpoint dies when the project does (see
    ``app.db.repositories.accounts.delete_account``). Deliberately NOT given a
    TTL and deliberately not mentioned in ``app.mcp.retention``: the owner's
    rule is "history lives with the project", and a retention sweep that aged
    checkpoints out would silently punch holes in the middle of a history whose
    whole value is being complete.

    THE BLOB IS COMPRESSED (zlib over UTF-8 JSON), which is why ``data_gz`` is
    ``LargeBinary`` rather than the ``JSON`` column every other snapshot-shaped
    row in this module uses. Three facts make that the right trade:

    - Consecutive checkpoints are near-duplicates. A checkpoint taken ten
      minutes after the previous one differs in a handful of floats and is
      otherwise byte-identical, and each one is stored in full.
    - The history is UNBOUNDED — no TTL, no cap on checkpoints per project — so
      the multiplier on that duplication is "however long the project lives".
      Timeline JSON is highly repetitive text (long key names, uuids, repeated
      structure) and compresses roughly 5-10x, turning a table that would grow
      in tens of megabytes per busy project into one that grows in single-digit
      megabytes.
    - Nothing queries INSIDE a checkpoint. Every read is "give me the
      checkpoint at version N, whole" (``get_checkpoint``) or "list the
      checkpoints" (``list_checkpoints``, which reads the metadata columns and
      never the blob), so the queryability a ``JSON`` column would buy —
      ``json_extract``, GIN indexes — has no consumer to serve and would be
      paid for on every row.

    Encoding and decoding live in ONE place,
    ``app.db.repositories.project_checkpoints``; nothing else may touch this
    column, or the compression format becomes a second contract to keep.

    ``project_id`` is a plain string reference, not a foreign key — mirroring
    ``OperationLogRow`` and ``AnnotationRow``, and for the same reason: the
    project is addressed by its uuid everywhere in this codebase, and the
    deletion cascade is explicit (``delete_account``) rather than delegated to
    a database that may or may not enforce FKs (SQLite does not, by default).
    """

    __tablename__ = "project_checkpoints"

    __table_args__ = (
        # ONE checkpoint per (project, mutation number), spelled as a named
        # UNIQUE INDEX rather than an anonymous UNIQUE constraint so the ORM
        # metadata and the Alembic revision describe the identical object (the
        # same choice ``media_library_files.storage_key`` documents).
        #
        # THE UNIQUENESS IS LOAD-BEARING, not hygiene. Deduplication ("this
        # state is already checkpointed") is checked with a SELECT, and between
        # that SELECT and the INSERT sits a window that two API replicas — or a
        # replica and a background sweep — can both be inside. The constraint
        # is what makes the loser's INSERT fail instead of writing a duplicate;
        # ``app.db.repositories.project_checkpoints.create_checkpoint`` catches
        # that failure and reports "already existed", so a race is a no-op
        # rather than an error.
        #
        # It is also the only index this table needs: leading with
        # ``project_id`` makes it serve the equality-only lookup
        # ("this project's checkpoints") as well as the two-column one
        # ("this project at version N").
        Index(
            "ix_project_checkpoints_project_id_version",
            "project_id", "version",
            unique=True,
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64))
    # The MUTATION number (``ProjectRow.version``) this checkpoint froze — see
    # the class docstring. Sparse, never a counter of its own.
    version: Mapped[int] = mapped_column(Integer)
    # zlib-compressed UTF-8 JSON of ``project_to_dict`` — see the class
    # docstring for why this is bytes and not a JSON column.
    data_gz: Mapped[bytes] = mapped_column(LargeBinary)
    # Stored as a plain ``String``, not a native enum, for the same reason
    # ``MediaLibraryFileRow.charged_to`` is: SQLite has no ENUM type, and a
    # CHECK constraint would make adding a third trigger a table rewrite.
    trigger: Mapped[str] = mapped_column(
        String(16),
        default=CheckpointTrigger.MANUAL.value,
        server_default=text(f"'{CheckpointTrigger.MANUAL.value}'"),
    )
    # Python-side UTC default, like ``MediaLibraryFileRow``: SQLite's
    # CURRENT_TIMESTAMP has one-SECOND resolution, which is too coarse for a
    # list whose whole job is showing WHEN each checkpoint happened.
    # ``AwareDateTime`` guarantees the value read back is tz-aware UTC on every
    # dialect, so the ISO-8601 the API emits always carries its offset.
    created_at: Mapped[datetime] = mapped_column(AwareDateTime, default=utc_now)


class AnnotationRow(Base):
    """One operator error-marker pinned to a point on a project's exported
    timeline. Deliberately decoupled from ``ProjectRow``: annotations are
    placed on ALREADY-EXPORTED projects (review happens after the render), so
    they must NOT go through ``save_project`` — which refuses writes to an
    EXPORTED project and would also bump the timeline version. They never
    touch the timeline blob or the project version; ``project_id`` is a plain
    string reference, not a foreign key, mirroring ``OperationLogRow``."""

    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64), index=True)
    time_sec: Mapped[float] = mapped_column(Float)
    element_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    severity: Mapped[str] = mapped_column(String(16))
    note: Mapped[str] = mapped_column(Text)
    # Replaced the old ``author`` string column in Gate 1 Step 7. That column
    # defaulted to the literal "operator" and no caller ever passed anything
    # else, so it carried no information at all — while being the only record
    # of who wrote a marker. A real FK carries the attribution the string
    # pretended to. Ownership for ISOLATION is still derived through the
    # project (``project_id``), not from this column: an annotation is visible
    # to whoever owns the project it is pinned to, not only to its author.
    author_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MediaAnalysisRow(Base):
    """One persisted clip-event-detection result, keyed by the sha256 of the
    media's BYTES (``app.analysis.digest.content_digest``).

    Keyed by media content, NOT by clip or project, deliberately: analysis is a
    property of the source file, not of a particular placement on a track.
    Keying by clip would re-analyse the same video when it is added twice or to
    another project, and would let markers drift apart between placements.

    ``content_hash`` is UNIQUE — exactly one row per media, superseded in place
    by a re-analysis rather than appended to (unlike ``OperationLogRow``, whose
    whole job is history). ``version`` is the ``ANALYSIS_VERSION`` contract the
    payload was computed under, stored alongside so a stored result from an
    older or newer contract is recomputed instead of returned.

    This row stays SHARED across tenants on purpose (Gate 1 Step 10): the
    result is a property of the bytes, so two tenants holding the same file
    legitimately read the same numbers, and that sharing is the whole saving.
    What is NOT shared is the right to fetch the keyframe strips it names —
    that is ``AnalysisAccessRow``.
    """

    __tablename__ = "media_analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    version: Mapped[int] = mapped_column(Integer)
    # The uuid4 hex every keyframe strip of this analysis is named with:
    # "{analysis_id}_overview.png", "{analysis_id}_t03.50.png"
    # (app/analysis/analyze.py::_write_keyframes). Stored so serving one strip
    # can be traced back to the bytes it came from WITHOUT parsing payloads or
    # keeping a second name->hash table: the filename prefix IS the key.
    #
    # Nullable because rows written before Step 10 have strips on disk whose
    # id was never recorded; those simply fail the ownership check and 404
    # until re-analysed, which is the safe direction to fail.
    analysis_id: Mapped[str | None] = mapped_column(String(32), index=True, default=None)
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AnalysisAccessRow(Base):
    """One tenant's right to see the artifacts of one analysed media.

    A row is written whenever a tenant analyses media — on a cache MISS and,
    just as importantly, on a cache HIT: a hit is exactly the moment a second
    tenant legitimately arrives at bytes someone else already analysed. Without
    the hit case the second tenant would read the shared payload and then 404
    on every strip it names.

    "Has seen these bytes", not "owns them": the same media can be legitimately
    held by any number of tenants, so this is deliberately many-to-many and
    additive. Nothing revokes a row today — the media a tenant already
    downloaded is not un-seen by deleting a project — and that stays a
    conscious gap rather than an oversight.
    """

    __tablename__ = "analysis_access"

    __table_args__ = (
        # One row per (tenant, media). The serve-path check is
        # "does a row exist for this user and this hash", so the unique
        # constraint doubles as the index that answers it.
        UniqueConstraint("user_id", "content_hash", name="uq_analysis_access_user_hash"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserStatus(enum.StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"


class UserRow(Base):
    """One registered account. Storage only — no login endpoint, no session,
    no request-time identity resolution exists yet (Gate 1 Step 4 adds
    that). ``password_hash`` is argon2id, produced by
    ``app.db.repositories.users.create_user``; nothing in this module ever
    sees or stores the plaintext password."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    # 320 = RFC 5321's own max total address length (64 local-part + '@' +
    # 255 domain), not an arbitrary round number.
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    email_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(16), default=UserStatus.ACTIVE.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class QuotaRow(Base):
    """One tenant's resource quota AND its live usage counter (Gate 2 Step 3).

    One row per owner (``owner_id`` UNIQUE) — a quota is a property of the
    account, so a separate table keyed by ``owner_id`` rather than columns on
    ``users``. Two reasons for the split:

    - ``storage_bytes_used`` is a HOT counter, rewritten on every upload,
      import and render. ``users`` is the identity/auth row read on the token
      verification hot path; wedging a counter that changes on every byte of
      ingress into it would make each accounting write contend with auth
      reads for the same row. Keeping the counter in its own table isolates
      that churn.
    - Step 4 (per-tenant concurrency) and a future CPU-minutes budget extend
      THIS table (``max_concurrent_jobs`` is already here for Step 4 to read),
      not the account row.

    ``storage_bytes_limit``/``storage_bytes_used`` are ``BigInteger``: a
    storage budget is measured in gigabytes, well past ``Integer``'s ~2.1 GB
    ceiling on PostgreSQL. ``BigInteger`` is dialect-neutral — INTEGER on
    SQLite (already 64-bit), BIGINT on PostgreSQL — so the counter stays
    countable on the SQLite self-host path.

    ``cpu_minutes_limit`` is nullable: NULL means "unmetered" (no CPU budget
    assigned). It is defined now so Step 4/9 can populate it without another
    migration; nothing charges it in this step.

    ``media_bytes_used``/``media_bytes_limit`` are a SECOND, independent pair
    of counters, for the user MEDIA LIBRARY (``MediaLibraryFileRow``) — the
    per-account file shelf that lives outside any project. They are their own
    columns rather than a share of ``storage_bytes_*`` because the two budgets
    answer different questions: project storage covers renders, previews and
    per-project ingest (5 GiB), while the library is a much smaller personal
    shelf (500 MB). Merging them would let a full library block an export, and
    would make "how much library space is left?" — a number the library screen
    shows on every render — unanswerable.
    """

    __tablename__ = "quotas"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), unique=True, index=True
    )
    # A new account's limits. Server defaults so a row inserted by a bare SQL
    # backfill (the migration) or a forgetful caller still lands sane values —
    # the same numbers the accountant's DEFAULT_* constants name in Python.
    storage_bytes_limit: Mapped[int] = mapped_column(
        BigInteger, server_default=text(str(5 * 1024 ** 3))  # 5 GiB
    )
    storage_bytes_used: Mapped[int] = mapped_column(
        BigInteger, server_default=text("0")
    )
    max_concurrent_jobs: Mapped[int] = mapped_column(
        Integer, server_default=text("3")
    )
    cpu_minutes_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # The media-library budget. Same ``BigInteger`` + ``server_default``
    # reasoning as the storage pair above: the migration that adds these
    # columns relies on the server_default to give every EXISTING quota row a
    # real limit instead of a NULL that no comparison would ever satisfy.
    media_bytes_limit: Mapped[int] = mapped_column(
        BigInteger, server_default=text(str(500 * 1024 ** 2))  # 500 MB
    )
    media_bytes_used: Mapped[int] = mapped_column(
        BigInteger, server_default=text("0")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ApiTokenRow(Base):
    """One minted API token's HASH — never the plaintext. The plaintext
    (``secrets.token_urlsafe(32)``, minted by
    ``app.db.repositories.tokens.mint_token``) is shown to the caller
    exactly once and is not recoverable from this row afterwards.

    ``token_hash`` is plain SHA-256 (stdlib, no salt), deliberately NOT
    argon2 like ``UserRow.password_hash``: the token is 32 cryptographically
    random bytes with full entropy already, so there is no low-entropy
    secret for a slow KDF to defend against, and hashing on every request's
    verification hot path (Gate 1 Step 4) would only add latency for no
    security benefit. ``scopes`` is a JSON array of strings, consumed
    directly as the MCP tool dictionary's ``required_scopes`` in Step 5
    without a split/join step.

    ``last_used_at`` is updated by a SEPARATE transaction
    (``app.db.repositories.tokens.touch_last_used``), never inside the
    verification read itself — seeing recent-but-not-live usage data is an
    acceptable trade for not writing to the token row on every authenticated
    request.

    ``last_used_at``/``expires_at``/``revoked_at`` use ``AwareDateTime``
    (``app.db.types``), not plain ``DateTime(timezone=True)``: these three
    are the columns compared against ``datetime.now(timezone.utc)`` on the
    verification hot path (``find_active_token``), and SQLite silently
    returns a NAIVE value from a ``DateTime(timezone=True)`` column on
    round-trip — see that module's docstring. Same DDL either way, no
    migration needed.
    """

    __tablename__ = "api_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    scopes: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_used_at: Mapped[datetime | None] = mapped_column(AwareDateTime, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(AwareDateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(AwareDateTime, nullable=True)


class SessionRow(Base):
    """One server-side record backing a browser cookie session — Gate 2
    Step 8. The signed cookie (``app.auth.sessions``) carries this row's
    ``session_id`` alongside ``user_id``; the cookie alone used to BE the
    entire session (no lookup table at all, see that module's old
    docstring), which meant the only kill-switches were expiry and
    ``UserRow.status`` — logging out one browser could not, and cannot
    still, invalidate any OTHER session of the same account. This row is
    what makes a PER-SESSION kill-switch possible: revoking it (setting
    ``revoked_at``) invalidates exactly the one cookie that named this
    ``session_id``, independent of every other still-active session.

    ``session_id`` is stored in plain text, unlike ``api_tokens.token_hash``:
    it is not a bearer credential by itself — forging a request that carries
    it also requires a valid ``SESSION_SECRET_KEY``-signed cookie, which is
    what actually proves possession. A leaked row value alone opens nothing.
    """

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    revoked_at: Mapped[datetime | None] = mapped_column(AwareDateTime, nullable=True)


class PasswordResetTokenRow(Base):
    """One one-time password-reset token's HASH — Gate 2 Step 8. Mirrors
    ``ApiTokenRow``: the plaintext (``secrets.token_urlsafe(32)``, minted by
    ``app.db.repositories.password_resets.create_reset_token``) is emailed to
    the account holder exactly once and is never itself persisted, only its
    SHA-256 hash. Unsalted SHA-256 is correct here for the same reason it is
    on ``ApiTokenRow.token_hash``: the plaintext already has full
    cryptographic entropy, so there is no low-entropy secret for a slow KDF
    to defend against.

    ``expires_at`` is short-lived by design (``RESET_TOKEN_TTL``, currently 1
    hour) — this credential travels over email, a channel this application
    does not fully control, unlike a session cookie or a CLI-minted API
    token. ``used_at`` makes the token single-use: set once by
    ``consume_reset_token`` when a reset actually succeeds, so a stolen but
    already-used token cannot reset the password a second time.
    """

    __tablename__ = "password_reset_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(AwareDateTime)
    used_at: Mapped[datetime | None] = mapped_column(AwareDateTime, nullable=True)


class EmailConfirmationTokenRow(Base):
    """One one-time email-confirmation token's HASH — Gate 2 Step 7.

    Structurally identical to ``PasswordResetTokenRow`` and minted the same
    way (``app.db.repositories.email_confirmations.create_confirmation_token``):
    the plaintext (``secrets.token_urlsafe(32)``) is emailed to the address a
    new account claims, exactly once, and only its SHA-256 hash is persisted.
    Unsalted SHA-256 is correct here for the same reason it is on
    ``PasswordResetTokenRow`` and ``ApiTokenRow`` — the plaintext already has
    full cryptographic entropy, so a slow KDF would defend against nothing.

    This is the credential that flips ``UserRow.email_confirmed`` from the
    ``False`` public registration writes to ``True`` (Gate 2 Step 7): until
    it is redeemed, the account cannot obtain a working browser session (see
    ``app.auth.resolve._is_usable``). ``expires_at`` is longer-lived than a
    reset token (``CONFIRM_TOKEN_TTL``) because confirming a brand-new
    account is a less time-critical, once-per-account action than recovering
    a live account's password. ``used_at`` makes it single-use, set by
    ``consume_confirmation_token`` once confirmation succeeds.
    """

    __tablename__ = "email_confirmation_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(AwareDateTime)
    used_at: Mapped[datetime | None] = mapped_column(AwareDateTime, nullable=True)


class MediaLibraryFileRow(Base):
    """One file on a user's personal MEDIA LIBRARY shelf.

    The library is deliberately NOT project-scoped: a user uploads a file once
    and reuses it across projects, so this row references its owner and
    nothing else. Project ingest (``uploads/<uuid>/...``) stays exactly as it
    was — this is a second, parallel shelf, not a replacement for it.

    ``storage_key`` is the ``ArtifactStore`` address of the bytes and is
    IMMUTABLE: it is minted once as ``library/<owner_id>/<id>/<name>`` and is
    never rewritten, not even by a rename. A rename moves ``filename`` (the
    DISPLAY name) alone, so renaming costs one UPDATE instead of a copy-and-
    delete against the object store — and a rename can therefore never strand
    the bytes it was supposed to move. UNIQUE so two rows can never claim the
    same object and have one's delete silently destroy the other's file.

    ``id`` is a String uuid rather than an autoincrementing integer: it appears
    in URLs (``/api/media/<id>/download``) and in the storage key, and a
    sequential id there would advertise how many files exist and invite
    enumeration of ids belonging to other tenants (which the owner predicate
    would refuse, but which should not be guessable in the first place).

    ``size_bytes`` is the accounting record for whichever counter paid: the
    counters are DB counters, never a store scan (see
    ``app.db.repositories.quotas``), so this column is the AMOUNT a delete
    refunds — and ``charged_to`` names the BUDGET it refunds it to.
    ``BigInteger`` for the same dialect-neutral reason the quota columns use it.

    ``charged_to`` exists because a shelf row does not imply a library charge:
    a project upload charges ``storage_bytes_used`` once and is shelved here
    uncharged, so a delete that assumed ``media_bytes_used`` would credit a
    budget that never paid (see ``MediaChargeBudget``). Stored as a plain
    ``String`` rather than a native enum: SQLite has no ENUM type, and a CHECK
    constraint would make adding a third budget a table rewrite. Written from
    ``MediaChargeBudget``, read back through
    ``app.web.media_library._release_for``, which treats anything unrecognised
    as the library budget — the pre-column default.
    """

    __tablename__ = "media_library_files"

    __table_args__ = (
        # The library screen asks exactly one question — "this owner's files,
        # newest first" — so the index leads with owner_id (also serving the
        # equality-only lookup that every by-id fetch performs).
        Index("ix_media_library_files_owner_id_created_at", "owner_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    # Display name only. Sanitized at every write boundary
    # (``app.media.uploads.sanitize_filename``), bounded by the same
    # MAX_FILENAME_CHARS the on-disk path length is bounded by.
    filename: Mapped[str] = mapped_column(String(120))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    # Both a Python ``default`` and a ``server_default``: the first is what an
    # ORM insert that does not name the budget lands on, the second is what
    # gives every PRE-EXISTING row a real value in the same ALTER that adds the
    # column (a NULL here would make the refund lookup ambiguous forever).
    charged_to: Mapped[str] = mapped_column(
        String(16),
        default=MediaChargeBudget.MEDIA.value,
        server_default=text(f"'{MediaChargeBudget.MEDIA.value}'"),
    )
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # ``index=True`` alongside ``unique=True``: that spells the constraint as a
    # named UNIQUE INDEX (``ix_media_library_files_storage_key``) rather than an
    # anonymous UNIQUE constraint, so the ORM metadata and the Alembic revision
    # describe the identical object — matching how ``quotas.owner_id`` does it.
    storage_key: Mapped[str] = mapped_column(String(512), unique=True, index=True)
    # Python-side UTC defaults, not ``func.now()``: SQLite's CURRENT_TIMESTAMP
    # has one-SECOND resolution, which is too coarse both for "newest first"
    # ordering of a burst of uploads and for a client that watches
    # ``updated_at`` to notice a rename. ``AwareDateTime`` then guarantees the
    # value read back is tz-aware UTC on every dialect, so the ISO-8601 the
    # API emits always carries its offset.
    created_at: Mapped[datetime] = mapped_column(AwareDateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        AwareDateTime, default=utc_now, onupdate=utc_now
    )
