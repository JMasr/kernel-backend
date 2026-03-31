# ── Stage 1: install dependencies ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Copy uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install deps first (layer cache — only re-runs when lock file changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source and install the project itself
COPY src/ src/
RUN uv sync --frozen --no-dev

# ── Stage 2: production runtime ────────────────────────────────────────────────
FROM python:3.11-slim AS runner

# System dependencies:
#   ffmpeg     — audio/video processing (signing engine)
#   curl       — docker-compose healthcheck for backend-api
#   libgomp1   — OpenMP runtime required by numpy/scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Non-root user
RUN addgroup --system --gid 1001 appgroup \
 && adduser  --system --uid 1001 --ingroup appgroup appuser

# Copy virtual environment and source from builder
COPY --from=builder --chown=appuser:appgroup /app/.venv ./.venv
COPY --from=builder --chown=appuser:appgroup /app/src   ./src

# Copy runtime files
COPY --chown=appuser:appgroup main.py      ./main.py
COPY --chown=appuser:appgroup alembic/     ./alembic/
COPY --chown=appuser:appgroup alembic.ini  ./alembic.ini

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

# Default entrypoint — overridden in docker-compose per service:
#   backend-api:    uvicorn main:app (this default)
#   backend-worker: python -m arq kernel_backend.infrastructure.queue.worker.WorkerSettings
#   backend-migrate: alembic upgrade head
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
