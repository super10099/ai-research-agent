# ── Stage: production image ───────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System packages:
#   curl  — used by the ChromaDB healthcheck wait-script
#   build-essential, libgomp1 — required by some sentence-transformers deps on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary from the official image rather than pip-installing it.
# Avoids a pip round-trip and gives us the exact uv version pinned upstream.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ── Dependency installation ───────────────────────────────────────────────────
# Copy lock files BEFORE source so this layer is cached when only src/ changes.
# If pyproject.toml or uv.lock change, Docker invalidates from here downward.
COPY pyproject.toml uv.lock ./

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# --frozen: fail if uv.lock is out of date with pyproject.toml (no silent drift)
# --no-dev:  skip pytest/ruff/mypy — dev tools waste ~50 MB in production images
RUN uv sync --frozen --no-dev

# ── Embedding model pre-bake ──────────────────────────────────────────────────
# sentence-transformers downloads bge-large-en-v1.5 (~1.3 GB) on first import.
# Doing it here bakes the weights into a Docker layer so cold starts are fast.
# The layer is cached until the model name in config changes.
# HF_HOME controls the cache directory; we keep it inside /app so the layer
# is self-contained and doesn't depend on the host's ~/.cache.
ENV HF_HOME=/app/.hf_cache
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-large-en-v1.5'); \
print('Model pre-baked.')"

# ── Source code ───────────────────────────────────────────────────────────────
# This COPY is last: it's the layer that changes most often (every code edit).
# Placing it here means Docker reuses all the expensive layers above on rebuild.
COPY src/ ./src/

# Data directory — will be shadowed by a named volume in docker-compose,
# but needs to exist in the image so uvicorn can start even without a volume.
RUN mkdir -p /data/chroma

EXPOSE 8000

# Run as a non-root user — minimal attack surface.
RUN useradd --create-home appuser && chown -R appuser /app /data
USER appuser

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
