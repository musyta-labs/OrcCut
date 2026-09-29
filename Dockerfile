# Single-user Orccut in a container. Build: docker build -t orccut .
#
# Default command speaks MCP over stdio (what registry/catalog inspectors
# expect). For the HTTP transport, publish the port and override the
# transport:
#   docker run -p 127.0.0.1:8100:8100 -e MCP_TRANSPORT=streamable-http \
#     -v orccut-data:/data orccut
#
# The optional ASR/TTS extras are NOT installed here; piper-tts is GPL-3.0
# and stays a deliberate opt-in (see THIRD_PARTY_NOTICES.md).
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg melt fonts-dejavu-core fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/orccut
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md alembic.ini ./
COPY app ./app
COPY migrations ./migrations
RUN pip install --no-cache-dir . && mkdir -p /data

# Single-user by design: whoever can reach the transport is the operator.
# Keep the HTTP port bound to localhost (or behind an authenticating proxy).
ENV MCP_AUTH_ENABLED=false \
    MCP_TRANSPORT=stdio \
    DATABASE_URL=sqlite:////data/editor.db \
    MEDIA_DIR=/data/media \
    MODELS_DIR=/data/models

VOLUME /data
EXPOSE 8100
CMD ["python", "-m", "app.mcp.server"]
