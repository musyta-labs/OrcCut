# Third-party notices

This repository distributes **source code only** — no Docker images, no
bundled binaries, no font files. The notes below cover what the source
depends on and what an operator installs alongside it.

## Runtime tools installed by the operator (not distributed here)

- **FFmpeg** (LGPL-2.1+ / GPL-2+ depending on build) — all media probing,
  rendering and thumbnailing runs through `ffmpeg`/`ffprobe` subprocesses.
- **MLT framework / `melt`** (LGPL-2.1+) — the default render engine
  (`RENDER_ENGINE=mlt`). The `.mlt` sidecar written next to each export is a
  standard MLT XML project usable in Shotcut and Kdenlive.
- **Fonts** — text rendering needs a font file on disk
  (`FONT_PATH`, default `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`).
  Install e.g. `fonts-dejavu-core` (Bitstream Vera license) and
  `fonts-noto-color-emoji` (SIL OFL 1.1) from your distribution; their license
  texts ship with the distribution packages. This repository intentionally
  vendors no fonts so it carries no font-license obligations.

## Python dependencies (installed from PyPI by `pip`)

Declared in `pyproject.toml`; licenses per their own distributions, notably:

- `mcp` (MIT), `sqlalchemy` (MIT), `alembic` (MIT), `pydantic` /
  `pydantic-settings` (MIT), `starlette` (BSD-3), `uvicorn` (BSD-3),
  `httpx` (BSD-3), `numpy` (BSD-3), `Pillow` (MIT-CMU), `anyio` (MIT),
  `yt-dlp` (Unlicense).
- **Optional extra `.[tts]`: `piper-tts` is GPL-3.0.** It is imported lazily
  and only if you install the extra yourself. If you redistribute a build
  that includes piper, GPLv3 obligations apply to that distribution — that is
  the reason it is an opt-in extra rather than a base dependency.
- Optional extras: `faster-whisper` (MIT, `.[asr]`), `psycopg` (LGPL-3.0,
  `.[postgres]`), `minio` (Apache-2.0, `.[s3]`).

## Media services

URL import supports any source your `yt-dlp` install supports. Complying with
the terms of the platforms you import from — and holding the rights to the
content you process — is your responsibility as the operator.
