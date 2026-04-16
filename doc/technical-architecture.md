# Kernel Security Backend -- Technical Architecture

## Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI (async, Python 3.12+) |
| ORM | SQLAlchemy 2.0 (async, asyncpg driver) |
| Database | PostgreSQL (via Neon/PgBouncer) |
| Migrations | Alembic |
| Job Queue | ARQ (Redis/Valkey-backed) |
| Storage | S3-compatible (MinIO dev, Cloudflare R2 prod) |
| Media | FFmpeg, OpenCV, librosa |
| Crypto | cryptography (Ed25519), galois (Reed-Solomon), pywt (DWT) |
| Email | Resend API + Jinja2 templates |
| Auth | Stack Auth (OAuth), local HS256 JWT, API keys |
| Monitoring | Sentry (optional) |
| Container | Docker + docker-compose |

---

## Package Structure

```
src/kernel_backend/
├── api/                    # HTTP layer (routers, schemas, middleware)
│   ├── auth/               # POST /auth/login, /auth/refresh
│   ├── identity/           # POST /identity/generate, GET /me, DELETE /me
│   ├── signing/            # POST /sign, GET /sign/{job_id}
│   ├── verification/       # POST /verify
│   ├── public/             # POST /verify/public (unauthenticated)
│   ├── organizations/      # CRUD orgs, members, API keys
│   ├── invitations/        # Admin + public invitation endpoints
│   ├── content/            # List, delete, download signed content
│   ├── downloads/          # Presigned S3 URL generation
│   ├── users/              # User profile endpoints
│   ├── health/             # Liveness + readiness probes
│   └── middleware/
│       └── auth.py         # HybridAuthMiddleware (API key / JWT / Stack Auth)
│
├── core/                   # Pure domain logic (no I/O dependencies)
│   ├── domain/             # Dataclasses: Certificate, Invitation, Organization, DSP manifests, chunk
│   └── services/           # Business logic: signing, verification, crypto, invitations,
│                           # chunk_planner / chunk_worker / chunk_assembler (parallel video encode)
│
├── engine/                 # DSP algorithms (stateless, CPU-bound)
│   ├── audio/              # wid_beacon.py (DSSS+DWT), fingerprint.py (log-mel)
│   ├── video/              # wid_watermark.py (QIM+DCT), fingerprint.py (2D-DCT)
│   ├── codec/              # reed_solomon.py, spread_spectrum.py, hopping.py
│   └── perceptual/         # psychoacoustic.py, jnd_model.py
│
├── infrastructure/         # I/O adapters (ports & adapters pattern)
│   ├── database/           # models.py (ORM), session.py, repositories.py
│   ├── storage/            # s3_storage.py, local_storage.py
│   ├── media/              # media_service.py (FFmpeg/OpenCV wrapper)
│   ├── queue/              # worker.py, jobs.py, redis_pool.py
│   └── email/              # resend_adapter.py, templates/
│
├── config.py               # Pydantic BaseSettings (env vars)
└── __init__.py
```

**Dependency rule:** `api` -> `core` -> `engine`. Infrastructure adapters are injected at startup. `core` and `engine` have zero I/O imports.

---

## Application Lifecycle

**Entry point:** `main.py`

```
create_app()
├── lifespan() context manager
│   ├── Startup
│   │   ├── Create SQLAlchemy async engine (NullPool, SSL)
│   │   ├── Create async session factory
│   │   ├── Initialize storage adapter (S3 or local)
│   │   ├── Create Redis connection pool
│   │   ├── Create shared httpx.AsyncClient (Stack Auth calls)
│   │   └── Store all in app.state
│   └── Shutdown
│       ├── Close httpx client
│       ├── Dispose DB engine
│       └── Close Redis pool
├── Register middleware
│   ├── CORSMiddleware (configurable origins)
│   ├── HybridAuthMiddleware (see Auth section)
│   └── AccessLogMiddleware (structured request logging)
└── Mount routers (12 route groups)
```

---

## Authentication & Authorization

### HybridAuthMiddleware (`api/middleware/auth.py`)

Runs on every request except public paths. Detection order:

1. **API Key** -- Header matches `krnl_<hex>` pattern -> SHA-256 hash lookup in `api_keys` table.
2. **Local Admin JWT** -- `jwt.decode(token, JWT_SECRET, "HS256")` succeeds -> admin session.
3. **Stack Auth Session** -- HTTP call to Stack Auth API to validate token -> user session (cached 90s, max 2000 entries).

**Public paths** (skip auth): `/health`, `/download/`, `/invitations/accept/`, `/verify/public`, `/auth/login`, `/auth/refresh`.

**Request state set by middleware:**

| Field | Type | Set By |
|-------|------|--------|
| `request.state.user_id` | `str` | JWT / Stack Auth |
| `request.state.email` | `str` | JWT / Stack Auth |
| `request.state.org_id` | `UUID` | All methods (looked up from DB) |
| `request.state.is_admin` | `bool` | JWT (true), API key (false), Stack Auth (from DB role) |
| `request.state.auth_type` | `str` | `"api_key"`, `"local_jwt"`, `"neon_auth"` |

---

## Database Schema

### Core Tables

```
organizations
├── id: UUID (PK)
├── name: str
├── pepper_v1: str (hex, 32 bytes) -- seeds all watermark PRNG
├── current_pepper_version: int = 1
└── created_at: datetime

organization_members
├── id: UUID (PK)
├── org_id: UUID (FK -> organizations)
├── user_id: str
├── role: str ("admin" | "member")
├── created_at: datetime
└── UNIQUE(org_id, user_id)

api_keys
├── id: UUID (PK)
├── org_id: UUID (FK -> organizations)
├── key_hash: str (SHA-256, unique)
├── key_prefix: str (first 12 chars)
├── name: str | null
├── is_active: bool
├── last_used_at: datetime | null
└── created_at: datetime

identities
├── id: UUID (PK)
├── author_id: str (unique, indexed)
├── name: str
├── institution: str
├── public_key_pem: str
├── org_id: UUID (FK -> organizations)
└── created_at: datetime

invitations
├── id: UUID (PK)
├── token: UUID (unique, indexed)
├── email: str (indexed)
├── org_id: UUID (FK -> organizations)
├── status: str ("pending" | "accepted" | "expired")
├── expires_at: datetime
├── accepted_at: datetime | null
└── created_at: datetime
```

### Content Registry Tables

```
videos
├── id: UUID (PK)
├── content_id: str (unique, indexed)
├── author_id: str (indexed)
├── author_public_key: str
├── content_hash_sha256: bytes
├── storage_key: str               -- "signed/{content_id}/output{ext}"
├── is_signed: bool
├── manifest_json: str             -- RFC 8785 canonical JSON
├── manifest_signature: bytes      -- 64-byte Ed25519
├── active_signals_json: str       -- ["wid_audio", "fingerprint_audio", ...]
├── rs_n: int
├── embedding_params: JSON
├── output_encoding_params: JSON
├── status: str ("VALID" | "REVOKED")
├── org_id: UUID (FK -> organizations)
└── created_at: datetime

video_segments
├── id: UUID (PK)
├── content_id: str (FK -> videos.content_id)
├── segment_index: int
├── start_time_s: float
├── end_time_s: float
├── rs_codeword: bytes
└── created_at: datetime

audio_fingerprints
├── id: UUID (PK)
├── content_id: str (FK -> videos.content_id)
├── time_offset_ms: int
├── hash_hex: str (16 chars = 64 bits)
├── is_original: bool
└── created_at: datetime
```

### Transparency Log Tables

```
transparency_log_entries
├── id: UUID (PK)
├── content_id: str
├── author_id: str
├── entry_hash: str
├── leaf_index: int
├── manifest_json: str
└── created_at: datetime

transparency_log_roots
├── id: UUID (PK)
├── tree_size: int
├── root_hash: str
└── created_at: datetime
```

**Connection setup:** NullPool (PgBouncer manages pooling), SSL enabled, statement cache disabled.

---

## API Endpoints

### Authentication

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/auth/login` | Public | Local admin login -> HS256 JWT (8h TTL) |
| `POST` | `/auth/refresh` | Public | Refresh JWT before expiry |

### Identity

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/identity/generate` | Required | Generate Ed25519 keypair (409 if exists) |
| `GET` | `/identity/me` | Required | Get public certificate |
| `DELETE` | `/identity/me` | Required | Delete keypair (allows regeneration) |

### Signing

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/sign` | Required | Upload media + certificate + private key -> enqueue job (202) |
| `GET` | `/sign/{job_id}` | Required | Poll job status (queued/in_progress/complete/failed) |

**POST /sign** accepts multipart form:
- `file`: Media binary (max 2 GB)
- `certificate_json`: JSON from `/identity/generate`
- `private_key_pem`: Private key (never stored)

### Verification

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/verify` | Required | Verify media using caller's org pepper |
| `POST` | `/verify/public` | Public | Verify media trying all org peppers |

### Organizations

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/organizations` | Admin | Create organization |
| `GET` | `/organizations` | Admin | List all organizations (paginated) |
| `POST` | `/organizations/{id}/api-keys` | Org Admin | Generate API key (shown once) |
| `GET` | `/organizations/users/{user_id}/organization` | Public | Lookup user's org |

### Invitations

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/admin/invitations` | Admin | Create + send invitation email |
| `GET` | `/admin/invitations` | Admin | List invitations (paginated, filterable) |
| `GET` | `/invitations/accept/{token}` | Public | Validate invitation token |
| `POST` | `/invitations/accept/{token}` | Required | Accept invitation, join org |

### Content

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/content` | Required | List signed content (org-scoped, paginated) |
| `GET` | `/content/{id}/download` | Required | Get presigned S3 download URL (1h) |
| `DELETE` | `/content/{id}` | Required | Delete content (members own, admins any) |

### Health

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/health/live` | Public | Liveness probe |
| `GET` | `/health/ready` | Public | Readiness probe (DB + Redis) |

---

## Signing Pipeline

### Sequence

```
Client                  API Server              ARQ Worker              S3 / DB
  |                         |                       |                      |
  |-- POST /sign ---------->|                       |                      |
  |   (file, cert, privkey) |                       |                      |
  |                         |-- save to /signing -->|                      |
  |                         |-- enqueue job ------->|                      |
  |<-- 202 {job_id} -------|                       |                      |
  |                         |                       |                      |
  |                         |                  [CPU subprocess]            |
  |                         |                       |-- probe media        |
  |                         |                       |-- extract fingerprints|
  |                         |                       |-- derive content_id  |
  |                         |                       |-- sign manifest      |
  |                         |                       |-- derive WID (HKDF)  |
  |                         |                       |-- embed WID symbols  |
  |                         |                       |-- encode output      |
  |                         |                       |                      |
  |                         |                  [async I/O]                 |
  |                         |                       |-- upload to S3 ----->|
  |                         |                       |-- save to DB ------->|
  |                         |                       |-- send email         |
  |                         |                       |-- update Redis status|
  |                         |                       |                      |
  |-- GET /sign/{job_id} -->|                       |                      |
  |<-- {status, result} ----|                       |                      |
```

### CPU Phase (`_sign_sync` in `signing_service.py`)

Runs in a `ProcessPoolExecutor` to avoid blocking the async event loop.

1. **Probe** -- FFprobe detects streams (audio/video), duration, bitrate, fps, sample rate.
2. **Route** -- Calls `_sign_audio_cpu()`, `_sign_video_cpu()`, or `_sign_av_cpu()` based on detected streams.
3. **Fingerprint** -- Extract perceptual fingerprints from raw media.
4. **Content ID** -- `SHA256(private_key_pem + author_id)[:32]` -- deterministic per signer.
5. **Manifest** -- Build canonical JSON manifest (RFC 8785), sign with Ed25519.
6. **WID** -- `HKDF-SHA256(ikm=signature, info=content_id)` -> 16 bytes (128 bits).
7. **Reed-Solomon** -- Encode 16-byte WID into *n* RS symbols (n = f(duration)).
8. **Embed** -- Distribute RS symbols across media segments:
   - Audio: DSSS modulation into DWT detail band (db4, level 2).
   - Video: QIM into 4x4 DCT AC coefficients on Y plane.
9. **Encode** -- Write output via FFmpeg (H.264 + AAC, adaptive CRF/bitrate).
   - **Sequential path** -- Single libx264 subprocess fed by the embed loop. Used when the
     planner returns a single chunk (short videos or `CHUNK_WORKERS<=1`).
   - **Chunked path** -- `chunk_planner.plan_chunks(rs_n, …, total_duration_s)` partitions
     the segment range into N spans (1-segment guard bands at boundaries). N
     `process_video_chunk()` workers run in an inner `ProcessPoolExecutor` inside the
     ARQ child process. Each worker decodes its time window with `ffmpeg -ss / -to`,
     embeds its slice of `rs_symbols` (guards remain unmodified), encodes via libx264
     with `-force_key_frames expr:gte(t,n_forced*5)` so segment boundaries are always
     keyframes, then trims with `-c copy`. `chunk_assembler.validate_chunks` checks
     duration drift (per-chunk tol 1.5s, total tol 3.0s — absorbs the trailing partial
     GOP that stream-copy drops) before `concatenate_chunks` joins them with the
     ffmpeg concat demuxer. WID derivation, RS encoding, and per-frame embed math are
     identical between the two paths.

### I/O Phase (`_persist_payload`)

1. Upload signed file to S3: `signed/{content_id}/output{ext}`.
2. Save `Video` record (manifest, signature, params).
3. Save `VideoSegment` records (per-segment RS codewords).
4. Save `AudioFingerprint` records.
5. Update Redis job status to `complete`.
6. Send completion email (fire-and-forget).

### Job Configuration (ARQ + nested pools)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `ARQ_MAX_JOBS` | 2 | Concurrent ARQ jobs per worker process |
| `SIGN_POOL_MAX_WORKERS` | 2 | Outer ProcessPoolExecutor (one CPU-bound `_sign_sync` per slot) |
| `SIGN_POOL_MAX_TASKS_PER_CHILD` | 50 | Recycle each pool child after N jobs to bound RSS |
| `CHUNK_WORKERS` | 4 | Inner pool for parallel video encode (one ffmpeg+libx264 per slot) |
| `CHUNK_GUARD_SEGMENTS` | 1 | Guard segments per chunk boundary (encoder warmup context) |
| `CHUNK_MIN_PAYLOAD_SEGMENTS` | 4 | Below this, planner collapses to a single chunk → sequential path |
| Job timeout | 360s (6 min) | |
| Result TTL | 3600s (1 hour) | |
| Graceful shutdown | 360s | |
| Cron: cleanup temp files | Every 30 min | |

Recommended 8 vCPU layout: `ARQ_MAX_JOBS=2`, `SIGN_POOL_MAX_WORKERS=1`,
`CHUNK_WORKERS=4` — saturates 8 cores during a single video encode and keeps
audio jobs from contending. See `.env.example`.

---

## Verification Pipeline

### Sequence

```
Client                  API Server                           DB
  |                         |                                 |
  |-- POST /verify -------->|                                 |
  |   (media file)          |                                 |
  |                         |                                 |
  |                    [Phase A: Fingerprint Lookup]           |
  |                         |-- extract fingerprints          |
  |                         |-- query registry (Hamming <10)->|
  |                         |<-- candidate (content_id) ------|
  |                         |                                 |
  |                    [Phase B: Cryptographic Auth]           |
  |                         |-- extract WID symbols           |
  |                         |-- RS decode (with erasures)     |
  |                         |-- compare WID == stored_WID     |
  |                         |-- verify Ed25519 signature      |
  |                         |                                 |
  |<-- VerificationResponse-|                                 |
  |    {verdict, reason}    |                                 |
```

### Verdict Logic

```python
if extracted_WID == stored_WID and ed25519_verify(manifest, signature, pubkey):
    verdict = "VERIFIED"
else:
    verdict = "RED"
    red_reason = <specific reason>
```

### Public Verification Pepper Strategy

`POST /verify/public` iterates over all organization peppers to find the one that produces a fingerprint match. This is slower (O(num_orgs * extraction_time)) but allows unauthenticated third-party verification.

---

## Watermarking Engine Components

### Audio WID Embedding (`engine/audio/wid_beacon.py`)

```
Input PCM (float32, 44.1 kHz)
    |
    v
DWT decompose (db4, level 2)
    |
    v
Select cD2 detail band (5.5-11 kHz)
    |
    v
Spread RS symbol bits via DSSS (32 chips/bit)
    |
    v
Add spread signal to cD2 (SNR: -20 dB audio-only, -18 dB AV)
    |
    v
DWT reconstruct -> watermarked PCM
```

### Video WID Embedding (`engine/video/wid_watermark.py`)

```
Input frame (BGR uint8)
    |
    v
BGR -> YCrCb -> extract Y (float32)
    |
    v
Select 4x4 blocks (128 per segment, HMAC-seeded positions)
    |
    v
4x4 DCT on each block
    |
    v
QIM on AC coefficients {(0,1),(1,0)} + optional {(1,1),(0,2)}
    step = qim_step_base (default 64.0, JND-adaptive in [44, 128])
    fallback (JND disabled / tests): QIM_STEP_WID = 48.0
    |
    v
Inverse DCT -> clip to uint8
    |
    v
Write Y back -> YCrCb -> BGR
```

### Reed-Solomon Codec (`engine/codec/reed_solomon.py`)

- Parameters: RS(n, k=16), n in [17, 255].
- `encode(wid_bytes, n)` -> n symbols distributed across segments.
- `decode_with_erasures(codeword, erasure_positions)` -> original 16 bytes.
- Tolerates up to `n - k` erasures.

### Hopping (`engine/codec/hopping.py`)

HMAC-SHA256 seeded by `(content_id, pepper, [public_key])` produces deterministic but unpredictable:
- Audio: DWT band configurations per segment.
- Video: Block positions and DCT coefficient selections per segment.

---

## Infrastructure Adapters

### Storage (`infrastructure/storage/`)

**S3 adapter** uses a dual-client pattern:
- **Internal client** (`S3_ENDPOINT_URL`): Used for uploads and reads (e.g., `http://minio:9000` in Docker).
- **Presign client** (`S3_PUBLIC_ENDPOINT_URL`): Used to generate presigned download URLs (e.g., `https://api.kernelsecurity.tech`).

**Local adapter** uses filesystem paths with HMAC-signed download tokens.

### Media Service (`infrastructure/media/media_service.py`)

Wraps FFmpeg and provides:
- `probe(path)` -> `MediaProfile` (streams, duration, bitrate, fps, sample rate).
- `iter_audio_segments()` -> lazy PCM float32 stream (mono, 44.1 kHz).
- `iter_video_frames()` -> lazy BGR numpy array stream.
- `encode_audio_from_pcm()` -> write watermarked audio (AAC or WAV).
- `write_video_frames()` -> write watermarked video (H.264/yuvj420p).

### Email (`infrastructure/email/`)

Resend API via `resend` Python package. Templates:
- `invitation.html` -- Org invitation link.
- `job_complete.html` -- Signing job finished notification.
- `org_created.html` -- Organization creation confirmation.
- `org_deleted.html` -- Organization deletion notification.

All sends are fire-and-forget (failures logged, never block the operation).

### Redis / Job Queue (`infrastructure/queue/`)

- `redis_pool.py` -- Connection factory with TLS support.
- `worker.py` -- ARQ worker settings (functions, cron jobs, concurrency).
- `jobs.py` -- `process_sign_job()` entry point, progress tracking via Redis keys.

Job status stored as: `job:{job_id}:status` -> JSON `{status, progress, result, error}`.

---

## Configuration (`config.py`)

Pydantic `BaseSettings` loading from environment variables. Critical groups:

**Database:**
- `DATABASE_URL` -- Async PostgreSQL (pooled endpoint).
- `MIGRATION_DATABASE_URL` -- Direct PostgreSQL (Alembic).

**Security:**
- `KERNEL_SYSTEM_PEPPER` -- 64-char hex, global watermark seed. **Changing invalidates all existing content.**
- `JWT_SECRET` -- HS256 signing key for local admin tokens.
- `STORAGE_HMAC_SECRET` -- HMAC key for presigned URLs. Validated non-placeholder in production.

**Infrastructure:**
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, `REDIS_SSL` -- Valkey/Redis connection.
- `STORAGE_BACKEND` (`"local"` | `"s3"`) -- Selects storage adapter.
- `S3_ENDPOINT_URL`, `S3_PUBLIC_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_BUCKET_NAME`.

**Auth:**
- `NEON_AUTH_URL`, `NEON_AUTH_API_KEY`, `NEON_AUTH_PUBLISHABLE_KEY`, `NEON_AUTH_SECRET_SERVER_KEY` -- Stack Auth.
- `ADMIN_EMAIL`, `ADMIN_PASS` -- Local admin fallback.

**Email:**
- `RESEND_API_KEY`, `RESEND_FROM_EMAIL`, `FRONTEND_BASE_URL`.

---

## Deployment

### Docker Compose (`docker-compose.yml`)

Services:
- **api** -- FastAPI app (uvicorn).
- **worker** -- ARQ worker (same image, different entrypoint).
- **postgres** -- Database.
- **redis/valkey** -- Job queue + caching.
- **minio** -- S3-compatible storage (dev).

Shared volumes: `/signing` temp directory between api and worker for media handoff.

### Health Checks

- `GET /health/live` -- Always 200 (process alive).
- `GET /health/ready` -- Checks DB connection + Redis ping.

### CI/CD

GitHub Actions workflow builds and pushes Docker image to GHCR.
