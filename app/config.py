"""Settings for the standalone editor service.

No ``config.yaml`` split — nothing else
in this MVP needs YAML-level behavior config yet; env vars + ``.env`` are
enough for a single-service product.
"""
from __future__ import annotations

import ipaddress
import logging
from functools import lru_cache

from pydantic import BaseModel, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

IpNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


def parse_ip_networks(value: str) -> tuple[IpNetwork, ...]:
    """Parse a comma-separated CIDR list ("198.18.0.0/15, 10.1.0.0/16").

    ``strict=False`` so a host address stands in for its /32 (``198.18.0.110``
    means that one address). An unparseable entry raises ``ValueError``: a
    typo in an egress allowlist must fail loudly at startup, never silently
    widen or narrow what the SSRF guard lets through.
    """
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    return tuple(ipaddress.ip_network(entry, strict=False) for entry in entries)

# Valid values for Settings.render_engine.
RENDER_ENGINES = ("ffmpeg", "mlt")

# Valid values for Settings.shared_state_backend (Gate 3 Step 4). "memory" is
# the in-process implementation — the correct and only choice on the SQLite
# self-host / single-replica path, where process-local counters ARE the global
# truth. "database" backs the same primitives with rows in the application
# database (Postgres in a multi-replica deployment), so the sliding-window,
# inc-with-limit and TTL-lease accounting is shared across every replica rather
# than each enforcing its own fraction of the limit.
SHARED_STATE_BACKENDS = ("memory", "database")
DEFAULT_SHARED_STATE_BACKEND = "memory"

# Export presets (``editor_export(preset=...)`` / ``app.editor.render``).
# ``width``/``height`` of ``None`` means "use the project's own aspect_w/
# aspect_h" — a preset only overrides resolution when it deliberately
# targets a DIFFERENT size than the project's own (today, only
# "preview_720": a fixed 9:16 quick-look size, so it only makes sense for a
# 9:16-oriented project — a client rendering another aspect should use
# "shorts_1080"/"master" instead, which never override).
# "shorts_1080" reproduces today's pre-phase-5 defaults byte-for-byte
# (crf=20, preset="medium", no resolution override) — it is also the default
# preset, so an existing caller that never passes ``preset=`` sees no change.
EXPORT_PRESETS: dict[str, dict[str, int | str | None]] = {
    "shorts_1080": {"crf": 20, "preset": "medium", "width": None, "height": None},
    "preview_720": {"crf": 26, "preset": "fast", "width": 720, "height": 1280},
    "master": {"crf": 16, "preset": "slow", "width": None, "height": None},
}
DEFAULT_EXPORT_PRESET = "shorts_1080"


class FrozenModel(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = "sqlite:///./data/editor.db"
    # Connection-pool sizing for NON-SQLite backends (Gate 3 Step 1). These are
    # inert on the SQLite self-host path — ``app.db.base`` only threads them
    # into ``create_engine`` when the URL is not SQLite, because SQLite's file
    # access is process-local and does not benefit from a server-connection
    # pool (and ``check_same_thread`` covers its threading need instead). On
    # PostgreSQL they bound the persistent connections one replica holds:
    # ``db_pool_size`` kept-open connections, up to ``db_max_overflow`` extra
    # under burst, each recycled after ``db_pool_recycle`` seconds so a
    # connection a proxy silently dropped is not reused. ``pool_pre_ping`` (set
    # unconditionally for non-SQLite in base.py) additionally validates a
    # connection before handing it out.
    # Sized against the ~40-thread request pool one replica runs (Starlette
    # threadpool + MCP tool offload share it): 5+10 produced pool_timeout
    # errors long before the threads were busy. 20+20 keeps the ceiling above
    # any realistic concurrent-request burst while the heavy-op limiter (see
    # ``max_global_heavy_ops``) keeps the long-running renders off the pool.
    db_pool_size: int = 20
    db_max_overflow: int = 20
    db_pool_recycle: int = 1800  # 30 minutes
    # Process-wide ceiling on concurrent HEAVY operations (ffmpeg/melt
    # renders, whisper transcription, TTS synthesis) across ALL tenants —
    # the per-owner ``max_concurrent_jobs`` quota bounds one tenant, but 100
    # tenants x 3 slots each would still mean 300 parallel ffmpeg runs on one
    # host. Sized for a small shared server: renders are CPU-bound, so more
    # concurrent slots than ~cores/2 only trades throughput for thrash.
    max_global_heavy_ops: int = 4
    # --- G1-01: the per-tenant DAILY spend ceiling (app.web.free_tier) ------
    # ``max_global_heavy_ops`` and ``max_concurrent_jobs`` bound how much runs
    # AT ONCE; nothing bounded how much runs PER DAY, so three renders in
    # parallel round the clock passed every limit this server had. These do.
    #
    # ZERO MEANS NO CEILING, and that is the default on purpose: an existing
    # single-tenant self-host must upgrade into identical behaviour rather than
    # discover at 3am that its half-hour source no longer imports. The public
    # cloud turns them on by configuration (10 / 100 / 600s / 1080p) — see
    # ``_warn_if_daily_quota_missing_in_public_deployment`` below, which makes
    # a public deployment that forgot to say so complain at startup.
    #
    # Both counters key on OWNER, never on token or IP: a human and the agent
    # they connected to the same account share one budget, or the account has
    # two.
    free_daily_heavy_ops: int = 0
    # Previews are a tenth of a heavy unit, expressed as their own ten-times
    # larger budget — see ``app.web.concurrency.PREVIEW_OPS`` for why the price
    # is a second counter rather than a fraction of the first.
    free_daily_preview_ops: int = 0
    # Source ceilings, checked where ffprobe metadata is already in hand.
    # Without them the daily count bounds nothing useful: ten renders of a
    # 4K hour is still ten times the 15-minute subprocess timeout.
    free_source_max_duration_sec: float = 0.0
    # A PIXEL budget rather than a height, mirroring ``max_media_pixels``: this
    # editor's own output is 9:16, where "1080p" is 1080x1920 and a height
    # check would read the wrong side of the frame. 1920*1080 = 2073600.
    free_source_max_pixels: int = 0
    # --- R-19/R-20: the DISK half of the spend ceiling ----------------------
    # G1-01 above bounds how much COMPUTE a tenant may spend per day. These two
    # bound the BYTES, which the daily counter says nothing about: an import
    # costs one unit whether it fetches 3 MB or 2 GB, and the fetched bytes then
    # sit on the volume for up to 54 hours (a 48h TTL plus one 6h sweep). Both
    # default to OFF for the same reason every G1-01 ceiling does — an existing
    # self-host must upgrade into today's behaviour, not into a new refusal —
    # and both are flagged at startup when a public deployment leaves them off
    # (see ``_warn_if_disk_ceilings_missing_in_public_deployment`` below).
    #
    # R-19: per-tenant ceiling, in BYTES, on the yt-dlp download cache
    # ({media_dir}/clips/<asset_id>/, app.web.free_tier). Deliberately a
    # SEPARATE ceiling rather than bytes charged into the 5 GiB storage quota:
    # that quota is the meter a person sees and manages (delete a file, get the
    # space back), while the cache is neither created deliberately nor deletable
    # by them — it appears when they paste a link and disappears on a TTL. Folded
    # into one number, the meter would move on its own and offer nothing to
    # delete, which is the worst kind of refusal. 0 = no ceiling.
    free_clip_cache_bytes_limit: int = 0
    # R-20: refuse new imports and renders once the volume holding ``media_dir``
    # has less than this much free space (app.storage.capacity). A ceiling per
    # tenant does not bound the HOST: N tenants each inside their limits still
    # add up, and a burst inside the six-hour window between retention sweeps
    # fills the volume. Failing at admission is the difference between one
    # honest refusal and a write that dies halfway through — which for a render
    # means a wasted 15-minute subprocess, and for the operator means every
    # tenant failing at once. 0 = the check is off.
    min_free_disk_bytes: int = 0
    # Uvicorn capacity bounds (wired in app.mcp.server.main). Past
    # ``http_limit_concurrency`` simultaneous connections uvicorn answers 503
    # immediately instead of queueing forever — under saturation a fast
    # honest refusal beats every client timing out. ``http_timeout_keep_alive``
    # closes idle keep-alive connections (browsers hold several per tab).
    http_limit_concurrency: int = 256
    http_timeout_keep_alive: int = 5
    media_dir: str = "media"
    # Gate 3 Step 10 (pre-baked weights): where ASR/TTS MODEL weights live —
    # deliberately its OWN path, never a subdirectory of ``media_dir``. The
    # two are different kinds of thing: ``media_dir`` holds per-project USER
    # artifacts (exports/previews/uploads/voiceover clips — Gate 3 Step 8's
    # object-storage candidates), while ``models_dir`` holds a handful of
    # shared, content-addressed model files that never vary per project and
    # that a deployment may legitimately want to bake into the image and
    # mount read-only (so ``media_dir`` alone can become read-only/object-
    # backed without also breaking model loading). Defaults to a writable,
    # download-on-first-use directory — today's behavior, unchanged — see
    # ``app.captions.transcribe``/``app.tts.synthesize`` for the read-only
    # (pre-baked) path, which is auto-detected from this directory's
    # permissions, not a separate flag.
    models_dir: str = "models"
    font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    mcp_host: str = "0.0.0.0"
    # Deliberately different from the companion client's default port so the
    # two servers can run side by side on one machine without a collision.
    mcp_port: int = 8100
    mcp_transport: str = "streamable-http"
    log_level: str = "INFO"
    # Render engine switch: "ffmpeg" (today's hand-built filter_complex graph)
    # or "mlt" (XML -> melt subprocess, phase 3; default since phase 4). The
    # parity gate passed live on comp 9 (2026-07-17, see
    # the phase-3 engine report) and every wave-1 CapCut
    # feature (transitions/keyframes/color/overlays) is MLT-only — the ffmpeg
    # path stays as a fallback and now guards itself (see
    # ``app.editor.render._render_via_ffmpeg``) against a project that uses
    # any of them.
    render_engine: str = "mlt"
    # Explicit encoder knobs so both engines can be told to encode identically
    # (parity harness compares them).
    video_crf: int = 20
    video_preset: str = "medium"
    melt_binary: str = "melt"
    # Auto-captions (app/captions/transcribe.py): faster-whisper model size/
    # device/compute type. "small"/"int8"/"cpu" measured ~460MB model, ~1GiB
    # RSS working set — inside the 4-8 GiB envelope. By default the model
    # downloads on first use to ``{models_dir}/asr_models``, a writable
    # volume. It CAN also
    # be baked into the image ahead of time (``docker/Dockerfile``'s
    # ``model-prebake`` stage) and mounted read-only — ``transcribe.py``
    # detects a non-writable ``models_dir`` and reads the pre-baked weights
    # straight off it instead of ever attempting a download.
    asr_model_size: str = "small"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    # Browser upload (app/media/uploads.py). The cap is enforced while
    # streaming to disk, NOT from the Content-Length header — a client
    # controls that header and a lying one must still be stopped.
    max_upload_mb: int = 2048
    # Pathological-media guard (app/media/guards.py). A byte-size cap alone is
    # insufficient: a small, valid-looking file can DECODE into an enormous
    # number of gigantic frames (an absurd resolution) or carry an absurd
    # number of streams — either one turns the very next ffmpeg decode
    # (analysis, export, preview) into an OOM or a multi-hour hang no
    # per-process subprocess timeout alone prevents. These bounds are checked
    # against ffprobe's OWN metadata, before any decode is attempted.
    #
    # 3840x2160 (4K UHD) — generous headroom above any real input this 9:16
    # short-form editor's own presets ever target (see EXPORT_PRESETS, all
    # <= 1080x1920), while still rejecting a crafted stream with an absurd
    # claimed resolution.
    max_media_pixels: int = 3840 * 2160
    # 1 hour — this editor composes short-form clips; nothing legitimate it
    # ingests runs anywhere near this long.
    max_media_duration_sec: float = 60 * 60
    # A real clip carries a small handful of audio/video streams; this is
    # headroom for e.g. multiple audio tracks, not an invitation to expect
    # dozens.
    max_media_streams: int = 16
    # GIPHY search (app/media/giphy.py). Empty = the GIFs tab reports 503
    # "not configured" instead of failing obscurely; the key never leaves the
    # server — the browser only ever talks to our /api/gifs/search proxy.
    giphy_api_key: str = ""
    # Auto-analysis (app/analysis/background.py): analyse a clip's media in the
    # background when it is added to a track, so markers are ready without
    # anyone asking. Only fires for media with no CURRENT-version stored
    # analysis — a cache hit costs one content hash and no decode. Set false to
    # keep analysis strictly on-demand (the test suite does this by default so
    # ordinary timeline tests never spawn ffmpeg threads).
    auto_analyze_on_add: bool = True
    # Cookie-session signing key (app/auth/sessions.py, itsdangerous). MUST be
    # set via SESSION_SECRET_KEY in any real deployment — unlike
    # giphy_api_key's safe empty-string "feature disabled" default, an empty
    # signing key here is never "disabled", because there is nothing that
    # would make forging a session impossible without a real secret; it is
    # deliberately left empty only so importing this module never requires
    # an env var, and app/auth/sessions.py refuses outright to sign or verify
    # anything while it is empty (see that module's docstring).
    session_secret_key: str = ""
    # Whether the session cookie carries the Secure flag (Gate 1 Step 6,
    # app/web/auth.py). Defaults to False ONLY because local development is
    # plain http, where a Secure cookie is never sent back at all: the login
    # would appear to succeed and every subsequent request would arrive
    # anonymous — a confusing failure, not a safe one. Any deployment
    # reachable over https MUST set SESSION_COOKIE_SECURE=true, or the
    # session credential travels in clear text on the first plain-http
    # request anything makes to it.
    session_cookie_secure: bool = False
    # Outbound email via Resend (app/email/sender.py). BOTH must be set for
    # real delivery: the API key alone says nothing about which verified
    # sender address to stamp on the mail, so a half-configured pair stays on
    # the LoggingEmailSender fallback — same "empty means the feature quietly
    # does nothing" posture as giphy_api_key, and deliberately NOT
    # session_secret_key's refuse-outright posture (see app/email/sender.py's
    # module docstring for why a broken provider must never block
    # registration or password reset). EMAIL_FROM must be an address on a
    # domain verified in the operator's Resend account.
    #
    # For a REAL public launch both are effectively REQUIRED, not optional:
    # email confirmation is mandatory before login (app.web.auth.login), so
    # the LoggingEmailSender fallback silently strands every new registrant
    # at "created, e-mail not confirmed" forever — nobody can ever log in.
    # EMAIL_FROM's domain must also carry the SPF/DKIM DNS records Resend's
    # dashboard requires to consider it "verified"; without them Resend
    # rejects or spam-folds the send, which looks identical to this being
    # unset from the outside. See
    # ``_warn_if_email_delivery_missing_in_public_deployment`` below for the
    # startup signal that catches an operator who deploys without either.
    resend_api_key: str = ""
    email_from: str = ""
    # G1-02: closes SELF-SERVICE registration without touching anything else.
    # Read at BOTH the GET and the POST of /auth/register (app/web/register.py,
    # app/web/auth_pages.py) — a closed registration must not even show the
    # form, or a visitor fills it in only to be refused on submit. Defaults to
    # True: an operator who never sets REGISTRATION_ENABLED keeps today's
    # open-door behavior, the same "unset env var changes nothing" posture as
    # mcp_auth_enabled/retention_enabled/checkpoint_sweep_enabled above.
    # Deliberately narrow in scope: login, email confirmation and password
    # reset (request + confirm) do NOT read this flag — closing the door to
    # NEW signups must never accidentally lock out people already inside.
    # Before this field existed there was no such switch at all (verified: no
    # flag anywhere closed registration) — see G1-02 in
    # the release roadmap.
    registration_enabled: bool = True
    # Absolute origin stamped into the links inside confirmation and
    # password-reset emails (app/web/register.py, app/web/password_reset.py).
    # The server cannot derive this from a request — the email is read far
    # from any request context, and a reverse proxy would hide the public
    # scheme/host anyway. The default matches docker-compose's loopback
    # publish so local testing yields clickable links out of the box.
    public_base_url: str = "http://127.0.0.1:8100"
    # Operator contact shown on the /legal pages (app/web/legal_pages.py).
    # Empty renders the documents with a visible "not yet configured" gap
    # rather than failing — the pages must stay reachable regardless. That
    # tolerance is for local/dev use only: a REAL public deployment must set
    # this, because "contact address not yet configured" on a live ToS/privacy/
    # complaints page is not an acceptable end state for a service with real
    # users. See ``_warn_if_legal_contact_missing_in_public_deployment`` below
    # for the loud (non-fatal) startup signal that catches an operator who
    # forgot it.
    legal_contact_email: str = ""
    # Reverse-proxy trust for client IPs (app/mcp/server.py -> uvicorn).
    # Empty (the default) = X-Forwarded-For is NEVER trusted: the TCP peer is
    # the client, which is correct for direct serving and spoof-proof — a
    # caller cannot forge someone else's IP into the rate-limit key
    # (app/web/ratelimit.py keys pre-auth throttles on client IP). Behind a
    # reverse proxy the peer is the PROXY for every request, so all clients
    # would share one bucket (the limit silently becomes global) — set this
    # to the proxy's IP(s), comma-separated ("*" trusts every peer; only ever
    # correct when the app is unreachable except through the proxy), and
    # uvicorn's own proxy-headers middleware restores the real client IP.
    proxy_trusted_ips: str = ""
    # SSRF egress escape hatch (app/net/egress.py). Comma-separated CIDRs that
    # the egress guard treats as fetchable even though they are non-public.
    #
    # Empty by default, and it must stay empty in any real deployment: every
    # entry here is a hole punched straight through the SSRF guard that stops
    # an untrusted tenant's URL from reaching loopback, the container's own
    # private network, or the cloud metadata endpoint.
    #
    # The one case it exists for is a DEVELOPER MACHINE behind a VPN/proxy in
    # "fake-ip" DNS mode (Clash/Surge/mihomo and friends), where public
    # hostnames deliberately resolve into 198.18.0.0/15 — a synthetic address
    # the proxy intercepts. The guard cannot tell that apart from a genuine
    # non-public target and refuses the fetch, so YouTube imports fail locally
    # with "resolves to non-public address 198.18.x.y". Setting
    # EGRESS_ALLOWED_NETWORKS=198.18.0.0/15 unblocks exactly that range and
    # nothing else — loopback, RFC1918 and 169.254.169.254 stay blocked.
    egress_allowed_networks: str = ""
    # MCP bearer authentication (Gate 1 Step 5, app/auth/verifier.py).
    #
    # ON by default, and that is the point: this server's whole reason for
    # Gate 1 is that /mcp used to accept every caller. A default of "off"
    # would mean any deployment that forgets one env var is silently the old
    # open server again — the failure mode would be invisible, because an
    # open server behaves exactly like a working one.
    mcp_auth_enabled: bool = True
    # The base URL clients actually reach this server on. It becomes the
    # RFC 9728 resource identifier and the base of the well-known metadata
    # path, so it MUST match what the client typed, not what the process
    # binds: mcp_host is 0.0.0.0 here, which is a bind address and not a
    # resource identity. Behind a proxy or on a real domain this has to be
    # set explicitly via MCP_PUBLIC_URL; the localhost default is only right
    # for the local docker-compose case (which publishes on 127.0.0.1).
    mcp_public_url: str = ""
    # Gate 3 Step 2 (app/db/base.py's ``init_db``): whether THIS process
    # should run the startup migration at all when ``database_url`` is
    # Postgres. Defaults True so a deployment that never sets this env var
    # behaves exactly as before — every replica migrates itself (serialized
    # by ``pg_advisory_lock``, see ``init_db``). Set False on replicas that
    # must never attempt DDL against a cold volume themselves — e.g. a
    # designated "migrator" replica runs with this True (or unset) and every
    # other replica runs with it False, so a slow/locked-out advisory-lock
    # wait never happens on the ones serving traffic. Ignored on SQLite: a
    # single-process SQLite deployment has no "which replica migrates"
    # question, so ``init_db`` always migrates there regardless of this flag
    # — a False value that silently skipped SQLite's migration would leave
    # the one process with no other way to reach ``head``.
    run_migrations_on_startup: bool = True
    # Gate 3 Step 4: which implementation backs the shared-state primitives
    # (sliding-window rate limiting, per-owner slot ceilings, TTL leases).
    # Defaults to "memory" so the single-replica SQLite self-host path — the
    # blessed default deployment — keeps today's exact in-process behaviour and
    # introduces no database dependency for accounting. A multi-replica Postgres
    # deployment sets SHARED_STATE_BACKEND=database so every replica shares one
    # count instead of enforcing "replicas × limit". See SHARED_STATE_BACKENDS.
    shared_state_backend: str = DEFAULT_SHARED_STATE_BACKEND

    # Gate 3 Step 8 (app/storage): where media artifacts live. "local" (the
    # default) keeps today's behavior byte-for-byte — every export/preview/
    # analysis-strip/upload/voiceover is a file under ``media_dir``, and the
    # self-host single-node path never touches an object store. "s3" points
    # the SAME artifacts at an S3-compatible bucket (MinIO in the bundled
    # compose stack; the same client reaches AWS S3 or R2 by changing only
    # ``s3_endpoint``), for a multi-replica deployment that must share one
    # durable store instead of a node-local disk. The default is load-bearing:
    # a deployment that sets no ARTIFACT_BACKEND behaves exactly as before.
    artifact_backend: str = "local"
    # S3 connection — read only when artifact_backend == "s3". Empty endpoint/
    # key/secret with the s3 backend selected is an operator error, not a
    # silent fall-back to local (see app/storage/factory.py). ``s3_secure``
    # is the https switch (True for AWS/R2, typically False for a plain-http
    # in-cluster MinIO). ``s3_endpoint`` is host[:port] WITHOUT a scheme —
    # minio-py takes the scheme from ``s3_secure``.
    s3_endpoint: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = "editor-media"
    s3_secure: bool = True
    s3_region: str = ""
    # Lifetime of a presigned GET used to serve an artifact by redirect. Short:
    # the browser follows the redirect immediately, so the signed capability
    # need not outlive the click by much.
    s3_presign_expiry_sec: int = 900
    # Gate 3 Step 9 (app/mcp/retention.py): whether THIS process starts the
    # periodic background retention sweep on launch. Defaults True for the
    # same reason mcp_auth_enabled does — a deployment that never sets this
    # env var gets bounded disk growth for exports/clips/editor_text (and
    # everything app.mcp.maintenance_cli already knew how to clean)
    # automatically, rather than silently depending on an operator cron that
    # was never actually scheduled. Set False on a process that must never
    # touch the media dir/store itself (mirrors run_migrations_on_startup's
    # per-replica opt-out shape) — every OTHER replica keeps sweeping, since
    # the sweep is safe to run redundantly (see that module's docstring).
    retention_enabled: bool = True
    # How often the background sweep runs, in seconds. 6 hours: media
    # artifacts do not need minute-level pruning, and each pass does real
    # filesystem/DB work (a directory listing + mtime stat per candidate),
    # so this stays sparse enough to never meaningfully compete with an
    # in-flight render or analysis job, while still keeping stale artifacts
    # from lingering for days between passes.
    retention_interval_sec: int = 6 * 60 * 60
    # --- project version history (app/editor/checkpoints.py) ----------------
    # Whether the background CHECKPOINT sweep runs. Separate from
    # retention_enabled and defaulted True for the same reason: a deployment
    # that sets no env var still gets the "end of activity" half of the version
    # history, rather than one that only ever records what someone remembered
    # to freeze by hand. Set False on a replica that must not write to the DB
    # on a timer — every other replica keeps sweeping, and the sweep is safe to
    # run redundantly (the unique index on (project, version) makes a
    # simultaneous duplicate a no-op, see ProjectCheckpointRow).
    checkpoint_sweep_enabled: bool = True
    # How long a project must go WITHOUT a mutation before the sweep freezes it
    # as a version, in minutes.
    #
    # This is the whole reason a checkpoint is not a mutation. Every op bumps
    # ProjectRow.version, so "one version per change" would make nudging a clip
    # two pixels a version and bury the states a person actually cares about.
    # A checkpoint marks where the editing came to REST instead.
    #
    # 10 minutes: longer than any pause inside real editing (thinking, watching
    # a preview back, resolving media) so a single session does not fragment
    # into a dozen versions, short enough that closing the tab and coming back
    # after lunch finds the work already frozen.
    project_checkpoint_idle_minutes: int = 10
    # How often that sweep runs, in seconds. 60 s — two orders of magnitude
    # shorter than retention_interval_sec (6 h), and deliberately so: retention
    # deletes artifacts, where being a few hours late costs nothing, while this
    # sweep CAPTURES state a user is waiting to see in their history. A pass is
    # one indexed query bounded by CHECKPOINT_SWEEP_BATCH plus a write only for
    # projects that actually went idle, so on an idle fleet it costs a single
    # query a minute.
    checkpoint_sweep_interval_sec: int = 60
    # TTL, in days, for an ORPHANED editor_text raster (app/mcp/retention.py)
    # before it is swept — orphaned meaning no ProjectRow still claims it.
    # This is now the ONLY category it governs: exports moved to their own
    # export_ttl_hours below, and clips/ has always had clip_cache_ttl_hours.
    # Longer than maintenance_cli.DEFAULT_STALE_DAYS (7) deliberately: unlike
    # a disposable preview frame, a text raster is a render INPUT reused on
    # every re-render of the same project — it deserves a longer grace
    # period, and the sweep never removes one that is still owned regardless
    # of age (see app.mcp.retention's module docstring).
    retention_artifact_ttl_days: int = 30
    # TTL, in HOURS, for a finished export bundle — the rendered mp4, its
    # .mlt sidecar and its _cover.png (app/media/exports.py names the set;
    # app/mcp/retention.py sweeps it). Age-based and liveness-blind: a live
    # project's export expires exactly like a deleted project's.
    #
    # WHY THE DELIVERABLE HAS THE SHORTEST TTL OF ANYTHING HERE. An export is
    # a DELIVERY, not something this service stores on the user's behalf.
    # What is stored is the project: the timeline snapshot in ProjectRow plus
    # the append-only OperationLogRow journal, neither of which any retention
    # sweep touches. Re-running editor_export on the current version
    # reproduces the file, so the bytes on disk carry no state the user would
    # lose. The user takes the delivery away either by downloading it or by
    # copying it into their own media library (POST /api/media/from-export),
    # and BOTH of those paths delete the server copy the moment the hand-off
    # is confirmed — see app.web.media_library and the release route in
    # app.web.api. This TTL therefore only ever fires for the one case those
    # two cannot cover: someone who never chose, because they closed the tab
    # or lost the network.
    #
    # 24 hours: long enough that coming back the next morning still finds the
    # render waiting, short enough that an abandoned one stops occupying disk
    # within a day. Sweeps run every retention_interval_sec (6h), so the
    # effective lifetime is this value plus up to one interval.
    #
    # THE HONEST LIMIT: re-rendering assumes the sources are still fetchable.
    # A URL-sourced clip whose remote copy has since disappeared (and whose
    # clips/ cache has aged out) cannot be re-rendered, and nothing here
    # defends against that — it is an accepted risk, not an oversight.
    export_ttl_hours: int = 24
    # TTL, in HOURS, for the yt-dlp download cache
    # ({media_dir}/clips/<asset_id>/, app/media/downloader.py). Deliberately
    # its OWN setting rather than a share of retention_artifact_ttl_days,
    # because it is the one media category swept on AGE ALONE — a clip
    # directory is removed once it is older than this, whether or not a live
    # project's assets list still names that asset id.
    #
    # WHY AGE-BASED IS SAFE HERE AND NOWHERE ELSE. Every other swept category
    # is either irreplaceable or expensive to reproduce: an export is a
    # finished deliverable no code path can regenerate without re-rendering
    # the whole timeline, and an editor_text raster belongs to a specific
    # project's text element. A clip under clips/ is neither — it is a pure
    # CACHE of a remote URL the MediaAsset itself still carries, and
    # app.web.api's resolve_media re-downloads it on demand (identical
    # outtmpl, identical destination) the next time anything needs it.
    # Deleting one therefore loses no data, only the time to refetch it; an
    # orphan-only rule, by contrast, would pin every clip of every live
    # project on disk forever, which is exactly the unbounded growth this
    # setting exists to stop (a handful of long source videos is gigabytes).
    #
    # 48 hours: comfortably longer than a single editing session (so nobody
    # loses their cache mid-work and pays the re-download inside a render),
    # short enough that an abandoned import stops occupying disk within two
    # days. Sweeps happen every retention_interval_sec (6h), so the effective
    # lifetime is this value plus up to one interval.
    clip_cache_ttl_hours: int = 48

    # --- R-18 (minimum): optional error tracking (app.observability) --------
    # Empty means "feature quietly does nothing" — the same posture as
    # giphy_api_key above, not session_secret_key's refuse-outright one:
    # there is no security property an empty DSN would be pretending to
    # provide, only a service that reports its own errors instead of an
    # external one. ``app.observability.init_error_tracking`` never imports
    # ``sentry_sdk`` at all while this is unset, so a self-host that never
    # sets SENTRY_DSN pays no dependency cost and touches no network — see
    # that module's docstring for the lazy-import convention this mirrors
    # (faster-whisper/piper/yt-dlp/minio).
    sentry_dsn: str = ""
    # Tag stamped on every event sent to Sentry, read ONLY when sentry_dsn is
    # set. Defaults to "development" rather than "production": an operator who
    # points a real DSN at a local run while testing this feature must not
    # have every event silently filed under "production" in their Sentry
    # project by an unrelated missing env var.
    sentry_environment: str = "development"

    @property
    def max_upload_bytes(self) -> int:
        """``max_upload_mb`` in bytes — the ONE place that conversion happens,
        so every caller that needs a byte ceiling (browser upload, URL-import,
        and yt-dlp's own ``max_filesize``) reads the identical number rather
        than each re-deriving it."""
        return self.max_upload_mb * 1024 ** 2

    @property
    def mcp_base_url(self) -> str:
        """``mcp_public_url``, or the local-dev default, without a trailing
        slash."""
        return (self.mcp_public_url or f"http://localhost:{self.mcp_port}").rstrip("/")

    @property
    def mcp_resource_url(self) -> str:
        """The RFC 9728 resource identifier for the MCP endpoint — the
        ``mcp_base_url`` plus the ``/mcp`` path the SDK serves on."""
        return f"{self.mcp_base_url}/mcp"

    @property
    def mcp_issuer_url(self) -> str:
        """The token issuer this server advertises.

        It is THIS service, because this service mints its own opaque tokens
        (``app.db.repositories.tokens.mint_token``) and no OAuth authorization
        server exists anywhere in this codebase — the spec explicitly allows
        the AS to be "hosted with the resource server". The field is filled
        because the SDK's ``AuthSettings.issuer_url`` is required, and because
        ``claims["iss"]`` has to agree with it; it does NOT mean an OAuth
        authorization endpoint is being served. See ``app.auth.verifier``.
        """
        return self.mcp_base_url

    @field_validator("render_engine")
    @classmethod
    def _validate_render_engine(cls, value: str) -> str:
        if value not in RENDER_ENGINES:
            raise ValueError(
                f"render_engine must be one of {RENDER_ENGINES!r}, got {value!r}"
            )
        return value

    @field_validator("egress_allowed_networks")
    @classmethod
    def _validate_egress_allowed_networks(cls, value: str) -> str:
        try:
            parse_ip_networks(value)
        except ValueError as exc:
            raise ValueError(
                f"egress_allowed_networks must be a comma-separated CIDR list, "
                f"got {value!r}: {exc}"
            ) from exc
        return value

    @property
    def egress_allowed_ip_networks(self) -> tuple[IpNetwork, ...]:
        """``egress_allowed_networks`` parsed — see ``app.net.egress``."""
        return parse_ip_networks(self.egress_allowed_networks)

    @field_validator("shared_state_backend")
    @classmethod
    def _validate_shared_state_backend(cls, value: str) -> str:
        if value not in SHARED_STATE_BACKENDS:
            raise ValueError(
                f"shared_state_backend must be one of {SHARED_STATE_BACKENDS!r}, "
                f"got {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _warn_if_legal_contact_missing_in_public_deployment(self) -> "Settings":
        """Loud, non-fatal signal at startup when this looks like a public
        deployment but ``LEGAL_CONTACT_EMAIL`` was never set.

        ``session_cookie_secure`` is the flag CLAUDE.md and this module's own
        docstring already name as the production marker ("any deployment
        reachable over https MUST set SESSION_COOKIE_SECURE=true") — there is
        no separate "environment" setting in this codebase, so it is reused
        here rather than inventing a second one. A local/dev run (the default,
        ``session_cookie_secure=False``) is silent; only a deployment that has
        already opted into the public-facing cookie posture without also
        setting a contact address gets flagged.

        A WARNING, not a raised error: an operator who already deployed
        without this must not be locked out of their own server by an
        unrelated env var, and ``app.web.legal_pages``' own fallback still
        keeps the documents themselves reachable regardless (see that
        module's "contact address not yet configured" gap). Logged with the
        stdlib ``logging`` module directly rather than
        ``app.common.logging.get_logger`` — ``Settings`` is constructed before
        ``setup_logging`` runs (``get_settings()`` is evaluated as an argument
        to that very call in ``app.mcp.server``), and an unconfigured root
        logger still emits WARNING and above to stderr via Python's
        ``logging.lastResort`` handler, so the message is never silently lost.
        """
        if self.session_cookie_secure and not self.legal_contact_email:
            logging.getLogger(__name__).warning(
                "SESSION_COOKIE_SECURE is true (this looks like a public "
                "deployment) but LEGAL_CONTACT_EMAIL is not set — /legal/* "
                "pages will show 'contact address not yet configured' to real "
                "users. Set LEGAL_CONTACT_EMAIL before opening registration."
            )
        return self

    @model_validator(mode="after")
    def _warn_if_daily_quota_missing_in_public_deployment(self) -> "Settings":
        """Loud, non-fatal startup signal when this looks like a public
        deployment but the per-tenant DAILY ceiling was left off.

        Built deliberately in the image of
        ``_warn_if_legal_contact_missing_in_public_deployment`` above — same
        ``session_cookie_secure`` "this is a real deployment" marker, same
        WARNING-not-raise posture, same stdlib logger for the same reason —
        rather than as a second convention.

        What it protects against is specific: ``free_daily_heavy_ops`` defaults
        to 0 (no ceiling) so an existing self-host upgrades into unchanged
        behaviour, and that same default, shipped to a public cloud, means
        strangers render for free without bound and the bill is the operator's.
        A silent default cannot be the difference between those two outcomes.
        """
        if self.session_cookie_secure and self.free_daily_heavy_ops < 1:
            logging.getLogger(__name__).warning(
                "SESSION_COOKIE_SECURE is true (this looks like a public "
                "deployment) but FREE_DAILY_HEAVY_OPS is not set — every "
                "tenant can run unlimited renders, transcriptions and URL "
                "imports per day at your expense. Set FREE_DAILY_HEAVY_OPS "
                "(and FREE_DAILY_PREVIEW_OPS, FREE_SOURCE_MAX_DURATION_SEC, "
                "FREE_SOURCE_MAX_PIXELS) before opening registration."
            )
        return self

    @model_validator(mode="after")
    def _warn_if_disk_ceilings_missing_in_public_deployment(self) -> "Settings":
        """Loud, non-fatal startup signal when this looks like a public
        deployment but the DISK half of the spend ceiling (R-19/R-20) was left
        off.

        Built in the image of
        ``_warn_if_daily_quota_missing_in_public_deployment`` above — same
        ``session_cookie_secure`` marker, same WARNING-not-raise posture, same
        stdlib logger for the same reason — rather than as a second convention.

        Why it is a SEPARATE warning from the daily quota's: the two bound
        different resources and are configured independently, so a deployment
        that set ``FREE_DAILY_HEAVY_OPS`` and nothing else is the exact case
        this catches. A daily count says nothing about bytes — one import costs
        one unit whether it fetches 3 MB or 2 GB — and neither ceiling notices
        that the volume itself is nearly full.
        """
        if not self.session_cookie_secure:
            return self
        missing = []
        if self.free_clip_cache_bytes_limit < 1:
            missing.append("FREE_CLIP_CACHE_BYTES_LIMIT")
        if self.min_free_disk_bytes < 1:
            missing.append("MIN_FREE_DISK_BYTES")
        if missing:
            logging.getLogger(__name__).warning(
                "SESSION_COOKIE_SECURE is true (this looks like a public "
                "deployment) but %s is not set — a tenant's download cache is "
                "unbounded and/or nothing stops the volume filling up between "
                "retention sweeps, in which case writes fail halfway through "
                "instead of being refused. Set both before opening "
                "registration.",
                " and ".join(missing),
            )
        return self

    @model_validator(mode="after")
    def _warn_if_email_delivery_missing_in_public_deployment(self) -> "Settings":
        """Loud, non-fatal startup signal — built the same way as
        ``_warn_if_legal_contact_missing_in_public_deployment`` above, not by
        a separate convention: same ``session_cookie_secure`` "this looks
        like a public deployment" marker, same WARNING-not-raise posture
        (an operator must never be locked out of their own server by this),
        same stdlib ``logging`` call for the same reason (``Settings`` is
        constructed before ``setup_logging`` runs).

        Why this gap is worse than a missing legal contact: email
        confirmation is MANDATORY before login (``app.web.auth.login``), and
        ``app.email.sender.get_email_sender`` silently falls back to
        ``LoggingEmailSender`` — which only logs, never delivers — whenever
        EITHER ``resend_api_key`` or ``email_from`` is unset. That fallback
        exists so a misconfigured provider can never make registration
        itself fail (see that module's docstring); the cost is that the
        failure becomes invisible from outside — a fresh account is created,
        its confirmation mail only ever reaches the server log, and the
        human is stuck at "created, e-mail not confirmed" forever with
        nothing telling the operator anything is wrong. This warning is that
        signal, caught at startup instead of discovered from a support
        ticket after the first real registrant is already stranded.
        """
        if self.session_cookie_secure and not (self.resend_api_key and self.email_from):
            logging.getLogger(__name__).warning(
                "SESSION_COOKIE_SECURE is true (this looks like a public "
                "deployment) but RESEND_API_KEY and/or EMAIL_FROM is not set "
                "— confirmation emails will only be logged, never delivered, "
                "and nobody who registers will ever be able to confirm their "
                "account or log in. Set both, plus the SPF/DKIM DNS records "
                "for EMAIL_FROM's domain, before opening registration."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
