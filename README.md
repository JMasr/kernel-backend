# kernel-backend

FastAPI backend for the Kernel Security SaaS platform. Provides cryptographic
signing and forensic watermarking for audio/video content, with multi-tenant
organization isolation and an ARQ-based background job queue.

---

## What it does

Organizations upload media files. The backend embeds an invisible, forensically
robust watermark — a cryptographic identity tied to the uploader — then returns
a signed manifest. Anyone can later submit a suspected copy to the public
verification endpoint; the system iterates over all registered organization
peppers to determine whether the watermark is present and identify its origin,
without exposing signing secrets.

---

## Tech stack

| Layer            | Technology                                                                   |
|------------------|------------------------------------------------------------------------------|
| Framework        | FastAPI 0.115 + Uvicorn                                                      |
| Language         | Python 3.11                                                                  |
| Package manager  | uv (lock-file based)                                                         |
| Database         | PostgreSQL via SQLAlchemy 2 async + asyncpg                                  |
| Migrations       | Alembic                                                                      |
| Job queue        | ARQ 0.26 (async Redis queue, Valkey-compatible)                              |
| Object store     | boto3 — MinIO locally, Cloudflare R2 in production                           |
| Auth             | Stack Auth (JWT verification via JWKS) + local admin JWT fallback            |
| Watermarking     | Custom engine: spread-spectrum + QIM + Reed-Solomon + psychoacoustic masking |
| Media processing | ffmpeg-python, librosa, scipy, opencv-headless                               |
| Email            | Resend                                                                       |
| Observability    | structlog + Sentry SDK                                                       |
| Testing          | pytest-asyncio, hypothesis, custom polygon test suite                        |
| Linting          | ruff                                                                         |

---

## Repository layout

```
kernel-backend/
├── src/kernel_backend/
│   ├── api/              FastAPI routers (one sub-package per domain)
│   │   ├── signing/      POST /sign — receive upload, enqueue job
│   │   ├── verification/ POST /verify — authenticated verification
│   │   ├── public/       POST /verify/public — unauthenticated verification
│   │   ├── downloads/    Presigned download URL generation
│   │   ├── identity/     Cryptographic identity management
│   │   ├── organizations/Organization CRUD + member management
│   │   ├── invitations/  Email-based invitation flow
│   │   ├── users/        User profile endpoints
│   │   ├── auth/         Local admin auth (Stack Auth is handled via middleware)
│   │   ├── content/      Signed content list
│   │   └── health/       /health/live + /health readiness checks
│   ├── core/
│   │   ├── domain/       Pure domain models (no I/O)
│   │   ├── ports/        Abstract interfaces (storage, queue, embedder…)
│   │   └── services/     Business logic (signing, verification, crypto)
│   ├── engine/
│   │   ├── audio/        Pilot-tone watermarking + fingerprinting
│   │   ├── video/        WID watermarking + fingerprinting
│   │   ├── codec/        Spread-spectrum, frequency hopping, Reed-Solomon
│   │   └── perceptual/   JND model + psychoacoustic masking (embed strength)
│   └── infrastructure/
│       ├── database/     SQLAlchemy models + repositories
│       ├── storage/      S3StorageAdapter (dual-client for presigned URLs)
│       ├── queue/        ARQ worker, job definitions, cleanup job
│       ├── media/        ffmpeg/librosa media service
│       └── email/        Resend adapter + Jinja2 templates
├── alembic/              Database migrations
├── tests/
│   ├── unit/             Fast tests, no I/O
│   ├── integration/      Real DB + Valkey + MinIO (Docker required)
│   ├── polygon/          End-to-end watermark quality tests with real media
│   └── boundary/         Import boundary enforcement
├── scripts/
│   ├── validate_env.sh   Validates secret lengths before service start
│   └── hooks/            Claude Code hooks (guard_immutable, run_module_tests)
├── Dockerfile
├── Makefile
└── pyproject.toml
```

---

## Architecture decisions

**Hexagonal architecture (ports & adapters)**
The core domain (`core/`) has zero dependencies on infrastructure. Storage,
queue, and media operations are defined as abstract `Port` interfaces and
injected at startup. This makes the engine and business logic fully unit-testable
without Docker or network access.

**ARQ job queue with shared signing_tmp volume**
The signing flow is intentionally split across two containers:
1. `backend-api` receives the upload, writes the file to `/signing` (a named
   Docker volume shared with the worker), and enqueues an ARQ job with the path.
2. `backend-worker` reads the file, runs the watermarking engine, stores the
   result in MinIO, and updates the job status in Valkey.

This keeps the API response latency independent of the watermarking duration
(which can be 30–300s for large video files). The `stop_grace_period: 380s`
on the worker container ensures in-flight jobs complete before the container
is replaced during a deploy.

**Dual boto3 client for presigned URLs**
The internal S3 client uses `S3_ENDPOINT_URL=http://minio:9000` for uploads and
reads. A separate `_presign_client` uses `S3_PUBLIC_ENDPOINT_URL=https://api.kernelsecurity.tech`
to generate URLs that browsers can reach. Nginx routes `/kernel-media/*` to MinIO
with `proxy_set_header Host $host` so MinIO can verify HMAC signatures on the
presigned URLs.

**Pepper-based watermark identity**
Each organization has a system-generated pepper (random secret) that seeds the
PRNG controlling watermark embedding positions and hopping pattern. Verification
iterates over all active organization peppers — this is intentional, enabling
public verification without knowing which organization signed the content.
The global `KERNEL_SYSTEM_PEPPER` seeds an additional layer; changing it
invalidates all previously signed content.

**Import boundaries**
`tests/boundary/test_import_boundaries.py` enforces that `core/` never imports
from `infrastructure/` or `api/`. Checked in CI. This prevents accidental
coupling that would make the domain untestable.

---

## Local development

### Prerequisites
- Python 3.11
- `uv` (`pip install uv` or `brew install uv`)
- Docker (for dependent services)
- ffmpeg system package

### Setup

```bash
# Install dependencies
uv sync

# Start dependent services (Valkey + MinIO + createbucket)
make services-up

# Copy and fill environment
cp .env.example .env
# Required: DATABASE_URL, REDIS_PASSWORD, MINIO_ROOT_PASSWORD, S3_ACCESS_KEY_ID,
#           S3_SECRET_ACCESS_KEY, KERNEL_SYSTEM_PEPPER

# Run migrations
make migrate

# Start dev server
make up      # uvicorn with --reload
# or
make worker  # ARQ worker process
```

### Makefile targets

| Target | What it does |
|--------|-------------|
| `make up` | FastAPI dev server (port 8000, hot reload) |
| `make worker` | ARQ worker |
| `make test` | Unit tests (excludes polygon) |
| `make integration` | Full integration suite (requires Docker services) |
| `make lint` | ruff check |
| `make format` | ruff format + ruff check --fix |
| `make typecheck` | mypy |
| `make migrate` | alembic upgrade head |
| `make migrate-create m="desc"` | New migration |
| `make services-up` | Start Valkey + MinIO via Docker Compose |
| `make services-up-dev` | Same + dev port overrides |
| `make validate-env` | Check secret lengths and forbidden defaults |

---

## Environment variables

See `.env.example` for the full reference with comments. Key variables:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL async URL (pooled, for the app) |
| `MIGRATION_DATABASE_URL` | PostgreSQL direct URL (Alembic only) |
| `KERNEL_SYSTEM_PEPPER` | Global watermark seed — changing invalidates all signed content |
| `REDIS_HOST` / `REDIS_PASSWORD` | Valkey connection |
| `S3_ENDPOINT_URL` | Internal MinIO URL (`http://minio:9000` in Docker) |
| `S3_PUBLIC_ENDPOINT_URL` | Public URL for presigned links (`https://api.kernelsecurity.tech`) |
| `STORAGE_BACKEND` | `s3` (production) or `local` (testing without MinIO) |
| `CORS_ORIGINS` | JSON array of allowed origins |

---

## API overview

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/sign` | Required | Upload media + enqueue signing job |
| `GET` | `/sign/{job_id}` | Required | Poll job status |
| `POST` | `/verify` | Required | Verify signed media (private) |
| `POST` | `/verify/public` | None | Verify any content (iterates all org peppers) |
| `GET` | `/downloads/{key}` | Required | Generate presigned download URL |
| `GET` | `/health/live` | None | Liveness probe |
| `GET` | `/health` | None | Readiness (DB + Valkey + worker) |

Full interactive docs available at `/docs` (Swagger UI) when running locally.

---

## CI/CD

On push to `master`:
1. GitHub Actions builds the Docker image (multi-stage, no dev deps)
2. Pushes `ghcr.io/jmasr/kernel-backend:latest` and `ghcr.io/jmasr/kernel-backend:sha-<sha>`
3. Sends `repository_dispatch: backend-image-pushed` to `JMasr/kernel-infra`
4. kernel-infra SSHes into the VPS and runs `deploy.sh`

The VPS pulls the pre-built image — no build step occurs on production.

See `JMasr/kernel-infra` for the full deployment architecture and design decisions.
