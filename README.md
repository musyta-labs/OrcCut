# Orccut

**A real video editor for AI agents, served over MCP.**

Orccut gives any MCP-capable agent (Claude Code, Claude Desktop, or anything
else that speaks the protocol) an actual editing model — not a wrapper around
one ffmpeg command. Projects are immutable snapshots with a journaled history
of named operations; rendering goes through FFmpeg or the MLT framework; a
deterministic CLI twin drives the same logic without a server.

## Two ways to use it

**☁️ Hosted (fastest — no install).** [orccut.com](https://orccut.com) runs
this editor as a service, with a browser editor on top: sign up, create an
API token, and point your agent at the cloud MCP endpoint:

```bash
claude mcp add --transport http orccut https://orccut.com/mcp \
  --header "Authorization: Bearer <your token>"
```

You and your agent then work on the **same** projects — the agent edits over
MCP, you review and tweak in the browser editor.

**🖥 Self-hosted (this repository).** The open, single-user core: run it on
your own machine, point your agent at it, keep everything local. No accounts,
no browser editor — just the MCP server, the editing engine and the CLI. The
rest of this README is about this option.

## What the agent gets

44 tools over one consistent model:

- **Projects & history** — create/list projects, journaled operations
  (`editor_get_history` shows every edit ever made), named versions with
  restore, annotations.
- **Timeline** — multiple video/audio tracks, clips with trim/split/move/
  resize, transforms with keyframes, transitions, canvas fit, color
  adjustments.
- **Text & graphics** — text overlays rasterized server-side, image/video
  overlays, covers.
- **Audio** — music beds, per-clip audio, volume envelopes.
- **Media in** — local paths or URLs (URL import goes through an SSRF egress
  guard; media is sanity-checked with ffprobe before it touches a timeline).
- **Understanding** — media analysis (scenes, audio peaks, keyframes),
  auto-captions via faster-whisper (`.[asr]` extra), local voiceover via
  piper (`.[tts]` extra, GPL — see THIRD_PARTY_NOTICES.md).
- **Out** — validation, preview renders, full exports with presets. Each
  export writes an `.mlt` sidecar you can open in Shotcut or Kdenlive.

Every mutation validates its arguments against the tool's real signature
(strings can't leak into numeric filter fields), returns a fresh snapshot
with a bumped version, and lands in the journal. Nothing edits in place.

## Prerequisites

- Python **3.12+**
- **FFmpeg** (`ffmpeg` + `ffprobe` on PATH)
- **MLT** (`melt` on PATH) — the default render engine. Alternatively set
  `RENDER_ENGINE=ffmpeg` and skip melt.
- A font for text rendering, e.g. Debian/Ubuntu:
  `apt install fonts-dejavu-core fonts-noto-color-emoji`
  (elsewhere, point `FONT_PATH` at any `.ttf`).

```bash
# Debian/Ubuntu, everything at once:
apt install ffmpeg melt fonts-dejavu-core fonts-noto-color-emoji
```

## Quickstart

```bash
git clone https://github.com/musyta-labs/orccut && cd orccut
python3 -m venv .venv && .venv/bin/pip install -e .

cp .env.example .env       # MCP_AUTH_ENABLED=false is already set there
.venv/bin/python -m app.mcp.server
```

The server starts on `http://127.0.0.1:8100` — the MCP endpoint is
`http://127.0.0.1:8100/mcp`. The database (SQLite) and media workspace are
created on first run.

Or with Docker (ffmpeg, melt and fonts included in the image):

```bash
docker build -t orccut .
docker run -p 127.0.0.1:8100:8100 -e MCP_TRANSPORT=streamable-http \
  -v orccut-data:/data orccut
```

> **Single-user by design.** This build has no accounts: whoever can reach
> the port is the operator, with full tool access including local file paths.
> Keep it bound to localhost, or put an authenticating reverse proxy in front.
> `MCP_AUTH_ENABLED=false` must be set explicitly — the server refuses to
> start otherwise, so an open port is always a decision you made.

### Connect an agent

Claude Code:

```bash
claude mcp add --transport http orccut http://127.0.0.1:8100/mcp
```

Any other MCP client, in its JSON config:

```json
{
  "mcpServers": {
    "orccut": { "type": "http", "url": "http://127.0.0.1:8100/mcp" }
  }
}
```

Then ask the agent for something real: *“make a 30-second cut of
~/videos/talk.mp4 with auto-captions and export it.”* The `skills/` directory
contains ready-made instructions you can hand to any agent — see
[skills/README.md](skills/README.md).

### Or drive it without a server

The CLI twin runs the same operations deterministically:

```bash
echo '{"op": "create_project", "args": {"metadata": {}}}' > /tmp/ops.json
.venv/bin/python -m app.mcp.cli /tmp/ops.json
```

## Configuration

Everything lives in environment variables (or `.env`); see
[.env.example](.env.example) for the full annotated list. The ones that
matter most:

| Variable | Default | What it does |
|---|---|---|
| `MCP_AUTH_ENABLED` | `true` | must be set to `false` in this build (see above) |
| `DATABASE_URL` | `sqlite:///./data/editor.db` | SQLite by default; Postgres via `.[postgres]` |
| `MEDIA_DIR` | `./media` | originals, exports, previews, caches |
| `MCP_HOST` / `MCP_PORT` | `0.0.0.0` / `8100` | bind address of the server |
| `RENDER_ENGINE` | `mlt` | `mlt` or `ffmpeg` |
| `FONT_PATH` | DejaVu Bold (Debian path) | font for text overlays and captions |
| `EXPORT_TTL_HOURS` / `CLIP_CACHE_TTL_HOURS` | `24` / `48` | retention sweep; `RETENTION_ENABLED=false` disables it |

**Retention is real:** a background daemon deletes exports after 24 h and
cached clips after 48 h by default. On a personal machine either download
your exports promptly or set `RETENTION_ENABLED=false`.

## How it's built

| Layer | Where | What |
|---|---|---|
| Timeline model | `app/editor/model.py` | frozen dataclasses; every edit returns a **new** project, `version + 1` |
| Mutations | `app/editor/mutations.py` | pure functions, one named journaled operation per gesture |
| Validation | `app/editor/validation.py` | structural checks + opt-in shorts profile |
| Render | `app/editor/render.py` | one contract, two engines: `mlt_graph.py` (default) and `ffmpeg_graph.py` |
| Analysis | `app/analysis/` | scenes, peaks, keyframes — feeds the agent's decisions |
| Storage | `app/db/` | SQLite/Postgres via SQLAlchemy + Alembic; append-only operation journal |
| MCP server | `app/mcp/server.py` | streamable-http or stdio; tool schemas derived from real signatures |
| CLI twin | `app/mcp/cli.py` | same operations, no server, deterministic |

Safety properties the codebase holds everywhere: no `shell=True` (argv lists
only), timeouts on every subprocess, atomic export writes, SSRF egress guard
on URL imports, argument validation derived from tool signatures.

## License

[MIT](LICENSE). Third-party obligations (FFmpeg/MLT installed by you, the
GPL `piper-tts` extra, fonts) are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
