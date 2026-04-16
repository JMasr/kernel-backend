# Kernel Backend — Architecture

FastAPI service that signs multimedia files with cryptographic watermarks and verifies their authenticity. The system embeds invisible acoustic and visual marks derived from the author's private key, then recovers and validates those marks during verification without storing the private key.

---

## Module Structure

The project follows hexagonal architecture. The core domain never imports infrastructure code.

```
src/kernel_backend/
├── core/                        # Pure business logic — no infra imports
│   ├── domain/                  # Data models: identity, watermark, manifest, verification, chunk
│   ├── services/                # Orchestration: signing_service, verification_service, crypto_service,
│   │                            # chunk_planner, chunk_worker, chunk_assembler
│   └── ports/                   # Abstract interfaces: RegistryPort, StoragePort, MediaPort
├── engine/                      # Signal processing (DSP)
│   ├── audio/                   # DWT-DSSS audio watermark (wid_beacon.py)
│   ├── video/                   # DCT-QIM video watermark (wid_watermark.py)
│   ├── codec/                   # Reed-Solomon codec, spread-spectrum, hopping
│   └── perceptual/              # Psychoacoustic and JND masking models
├── infrastructure/              # Concrete adapters
│   ├── database/                # SQLAlchemy models and repositories
│   ├── storage/                 # S3/MinIO and local filesystem adapters
│   ├── queue/                   # ARQ job queue (jobs.py, worker.py, cleanup_job.py)
│   └── media/                   # FFmpeg wrapper (MediaService)
└── api/                         # FastAPI routers and middleware
    ├── auth/                    # JWT and API key authentication
    ├── identity/                # Keypair generation
    ├── signing/                 # POST /sign + job polling
    ├── verification/            # POST /verify
    └── organizations/           # Multi-tenant management
```

---

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/identity/generate` | JWT | Generate Ed25519 keypair; private key returned once |
| GET | `/identity/me` | JWT | Retrieve public certificate |
| POST | `/sign` | JWT or API key | Enqueue a signing job, returns `job_id` |
| GET | `/sign/{job_id}` | None | Poll job status and progress (0–100) |
| POST | `/verify` | None | Submit file for verification, returns verdict |
| POST | `/verify/public` | None | Same as above, public endpoint |
| GET | `/content/{id}/download` | JWT | Get presigned S3 download URL |
| POST | `/organizations` | Admin | Create org and set pepper |
| POST | `/invitations` | Admin | Invite user to org |

---

## Signing Flow

### Summary

The client uploads a media file together with the author's public certificate and their Ed25519 private key. The API enqueues the job; the worker processes it in a subprocess (CPU-intensive DSP), then the async loop persists the result to storage and database.

```
POST /sign
  (file + certificate_json + private_key_pem)
        │
        ▼
  Validate: duration, media type, auth
        │
        ▼
  Enqueue job → Redis (ARQ)  ←── returns 202 + job_id immediately
        │
        ▼ (background worker)
  _sign_sync()  [ProcessPoolExecutor — CPU-bound]
  ├── Probe media type (audio / video / AV)
  ├── Extract perceptual fingerprints (streaming)
  ├── Hash file content (SHA-256)
  ├── Build CryptographicManifest
  │     └── content_id, content_hash, fingerprints, author, public_key, created_at
  ├── Sign manifest with Ed25519 private key  →  64-byte signature
  ├── Derive WID: HKDF-SHA256(signature, content_id)  →  16-byte WatermarkID
  ├── Encode WID with Reed-Solomon  →  n codeword symbols (n = min(segments, 255))
  ├── Plan segment hopping (HMAC-seeded, deterministic from pepper + org_id)
  └── Embed RS symbols into each segment  (audio: DWT-DSSS / video: DCT-QIM)
        │   Video encode runs sequentially or in N parallel chunks (see below).
        │
        ▼
  _persist_payload()  [async I/O — back in event loop]
  ├── Upload signed file to S3  →  key: signed/{content_id}/output.ext
  ├── Write VideoEntry to database (manifest_json, manifest_signature, rs_n)
  └── Write fingerprint segments to database
```

### Audio Watermark — DWT-DSSS

- **Transform:** Discrete Wavelet Transform (db4 wavelet, level 2)
- **Target band:** cD2 detail coefficients — 5.5–11 kHz range
- **Embedding:** Each RS symbol (8 bits) spread as a chip stream (`chips_per_bit = 32`) via Direct-Sequence Spread Spectrum
- **Masking:** MPEG-1 psychoacoustic model with silence gating; energy conserved via `1/√n` scaling across bands
- **Target SNR:** –20 dB (audio-only) or –14 dB (AV files, more compression-resilient)
- **Key file:** `engine/audio/wid_beacon.py`

### Video Watermark — DCT-QIM

- **Transform:** 4×4 DCT on luma (Y) channel of selected macroblocks
- **Target coefficients:** AC positions `(0,1)` and `(1,0)` always; optionally `(1,1)` and `(0,2)`
- **Embedding:** Quantization Index Modulation (QIM) with adaptive step (`qim_step_base = 64.0`, min 44, max 128, dark-content boost to 80), tuned to survive H.264 CRF ≤ 28. The non-adaptive fallback `QIM_STEP_WID = 48.0` is reserved for tests and the JND-disabled path.
- **Block selection:** Deterministic HMAC-seeded per segment and frame using pepper (hopping scheme)
- **JND adaptation:** Per-block luminance scaling (Chou–Li model) reduces visible artefacts; production always passes JND params via `algorithm_router.py` (speech / music / silence routing).
- **Key file:** `engine/video/wid_watermark.py`

### Chunked Parallel Video Encode

Long video encodes (decode → embed → libx264) are the dominant signing cost. The pipeline can split this stage across N worker processes when the source has enough segments.

```
plan_chunks(rs_n, segment_s=5.0, n_workers=CHUNK_WORKERS,
            guard_segments=1, total_duration_s=profile.duration_s)
        │
        ▼
ChunkManifest (chunks=N, payload spans + 1-segment guard bands)
        │
        ▼
ProcessPoolExecutor(max_workers=N)  ── inside the ARQ worker child
        │   one process_video_chunk() per ChunkSpec
        │   ├── ffmpeg -ss decode_start_s -to decode_end_s ... | libx264
        │   │     encoder pinned to a keyframe every VIDEO_SEGMENT_S=5s
        │   │     so trim with -c copy is exact at segment boundaries
        │   ├── for each segment in window:
        │   │     payload range  → embed_video_frame_yuvj420_planes(rs_symbol)
        │   │     guard range    → frame_to_yuvj420_planes (passthrough warmup)
        │   └── ffmpeg -ss payload_start_s -t expected_payload_duration_s
        │         -c copy  → chunk_{id}.mp4
        ▼
validate_chunks(manifest, results, media.probe)
        │   presence, success, no duplicate ids, file non-empty,
        │   per-chunk + total duration drift within tolerance
        ▼
concatenate_chunks(results, signed_path)
        │   ffmpeg -f concat -safe 0 -i list.txt -c copy → signed.mp4
        ▼
cleanup_chunks(results)  (best-effort)
```

- `total_chunks` is reduced automatically if any chunk would fall below `CHUNK_MIN_PAYLOAD_SEGMENTS`; below that floor the planner returns a single chunk and `signing_service` falls through to the existing sequential encode.
- `total_duration_s` is threaded through so the trailing chunk's expected payload duration is clamped to the real source length — `extract_video_hashes` counts segments with ceil rounding, so the last fingerprint frequently represents only a fractional segment.
- `validate_chunks` tolerances (`_PER_CHUNK_DURATION_TOL_S = 1.5`, `_TOTAL_DURATION_TOL_S = 3.0`) absorb the keyframe-rounding loss that `-c copy` incurs at the trailing partial GOP. Catastrophic drops (missing chunk, empty output) are caught earlier by dedicated checks.
- WID derivation, RS encoding, and per-frame embedding logic are byte-identical to the sequential path; chunking only parallelizes the ffmpeg + libx264 work.
- Audio sub-pipeline of AV files stays sequential — only the video encode is chunked.
- Tuning lives in `Settings` (`CHUNK_WORKERS`, `CHUNK_GUARD_SEGMENTS`, `CHUNK_MIN_PAYLOAD_SEGMENTS`); see `.env.example` for the recommended `ARQ_MAX_JOBS=2` × `CHUNK_WORKERS=4` layout on an 8 vCPU host.

### Reed-Solomon Code

| Parameter | Value |
|-----------|-------|
| Symbol alphabet | GF(2^8), 256 values |
| Data length k | 16 bytes (WID) |
| Total length n | min(total_segments, 255), adaptive |
| Error correction | ⌊(n − k) / 2⌋ symbol errors |
| Erasure threshold (audio) | Z-score < 1.0 → erasure |
| Erasure threshold (video) | Agreement ratio < 0.52 → erasure |

### Cryptographic Secrets

| Secret | Type | Purpose |
|--------|------|---------|
| Ed25519 private key | Per-author, client-held | Signs the manifest; derives WID |
| Ed25519 public key | Per-author, stored in DB | Verifies manifest signature at verification time |
| WID | 16 bytes, derived | `HKDF-SHA256(ikm=signature, salt=content_id)` — unique per file |
| KERNEL_SYSTEM_PEPPER | 32 bytes, system-wide | HMAC seed for hopping and fingerprint keying. **Never rotate.** |
| Org pepper (`pepper_v1`) | 32 bytes, per-org | Cryptographic isolation between tenants |

The private key is returned to the client once on generation and never persisted server-side. The WID is deterministic only for the holder of the private key that produced the original signature.

---

## Verification Flow

Verification is a two-phase pipeline. Phase A identifies the candidate via perceptual fingerprints. Phase B extracts the embedded WID and validates the cryptographic chain.

```
POST /verify  (file)
        │
        ▼
  Probe media type (audio / video / AV)
        │
        ▼
  PHASE A — Candidate lookup
  ├── Extract perceptual fingerprints (streaming)
  ├── Query DB: match_fingerprints(hashes, max_hamming=10)
  └── No match → RED (CANDIDATE_NOT_FOUND)
        │
        ▼
  PHASE B — Cryptographic validation
  ├── Iterate segments, extract RS symbol per segment
  │     Audio: DSSS correlation (Z-score threshold)
  │     Video: QIM majority voting across frames (agreement threshold)
  ├── RS decode(symbols + erasures)  →  extracted_wid
  ├── Compare extracted_wid == stored_wid
  │     Mismatch → RED (WID_MISMATCH — content tampered)
  ├── Ed25519.verify(manifest_json, stored_signature, author_public_key)
  │     Failure → RED (SIGNATURE_INVALID — manifest altered)
  └── All pass → VERIFIED
        │
        ▼
  Return VerificationResult
  { verdict: "VERIFIED" | "RED", reason?, author, org, content_id,
    audio_verdict?, video_verdict? }
```

### Fingerprint Matching

- **Audio:** 2-second segments — log-mel + DCT + keyed random projection → 64-bit hash
- **Video:** 5-second segments — grayscale DCT + keyed random projection → 64-bit hash
- Hamming distance threshold: ≤ 10 (robustness to minor re-encoding)
- Purpose: candidate lookup only, not authentication

---

## Identity and Keypair System

```
POST /identity/generate  (name, institution)
        │
        ▼
  Generate Ed25519 keypair on server
  ├── Store Certificate in DB: author_id, public_key_pem, org_id
  └── Return { private_key_pem, public_key_pem, author_id, ... }
      ── private key returned ONCE, never stored server-side ──
```

- The frontend downloads the PEM and stores it in browser localStorage.
- On every signing operation the private key is sent inside the multipart request as a form field, used in-memory to sign the manifest, and not persisted.
- Public key is embedded in every signed file's manifest, allowing future verification without a server call.

---

## Worker Architecture (ARQ)

ARQ is used instead of Celery — it is fully async and runs on top of Redis.

```
backend-api  ──enqueue──→  Redis (Valkey)  ←──poll──  backend-worker
                                                              │
                                          outer ProcessPoolExecutor
                                          (SIGN_POOL_MAX_WORKERS, default 2)
                                                              │
                                                       _sign_sync()
                                                    (CPU-bound DSP, picklable)
                                                              │
                                          inner ProcessPoolExecutor (only for video encode)
                                          (CHUNK_WORKERS, default 4) — chunked path
                                                              │
                                              back to async loop
                                                       _persist_payload()
                                                    (S3 + DB I/O)
```

- `ARQ_MAX_JOBS = 2`, `job_timeout = 360s`, `graceful_shutdown_timeout = 360s`
- The outer pool (`SIGN_POOL_MAX_WORKERS`) caps concurrent `_sign_sync` invocations across ARQ jobs; each child is recycled after `SIGN_POOL_MAX_TASKS_PER_CHILD` jobs to bound RSS growth.
- The inner chunk pool is created lazily inside the child process for the video encode stage only — Linux fork makes the nested pool safe; no `set_start_method` calls anywhere.
- Recommended 8 vCPU layout: `ARQ_MAX_JOBS=2`, `SIGN_POOL_MAX_WORKERS=1`, `CHUNK_WORKERS=4` (saturate 8 cores when one job is encoding) — see `.env.example`.
- A cron job (`cleanup_job.py`) runs every 30 minutes to delete orphaned `/tmp` files from crashed workers
- Job progress (0–100%) is tracked in Redis under `job:{job_id}:status`

---

## Storage Pattern

```
S3/MinIO key layout:
  signed/{content_id}/{filename}_signed.{ext}

Presigned URL generation:
  _presign_client uses S3_PUBLIC_ENDPOINT_URL (external hostname)
  Falls back to S3_ENDPOINT_URL (internal, Docker) if not set
  TTL: 1 hour
```

Production uses Cloudflare R2 (S3-compatible, no egress charges). Dev uses MinIO.

---

## Data Models (key tables)

| Table | Purpose |
|-------|---------|
| `identities` | `author_id`, `public_key_pem`, `org_id` |
| `videos` | `content_id`, `manifest_json`, `manifest_signature`, `rs_n`, `embedding_params` |
| `audio_fingerprints` | `content_id`, `time_offset_ms`, `hash_hex` |
| `video_segments` | `content_id`, `segment_index`, `start_time_s`, `end_time_s` |
| `organizations` | `org_id`, `name`, `pepper_v1`, `current_pepper_version` |
| `api_keys` | `key_hash`, `org_id` |

---

## Key Dependencies

| Library | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `sqlalchemy` + `asyncpg` | Async ORM for PostgreSQL |
| `arq` | Async Redis job queue |
| `cryptography` | Ed25519 signing, HKDF-SHA256 |
| `rfc8785` | Canonical JSON serialization for manifest |
| `numpy` / `scipy` | Numerical DSP operations |
| `pywt` | Discrete Wavelet Transform (audio) |
| `cv2` (OpenCV) | Video frame processing, DCT |
| `reedsolo` | Reed-Solomon GF(2^8) codec |
| `boto3` | S3/R2 object storage client |
| `ffmpeg-python` | Media encode/decode streaming |
