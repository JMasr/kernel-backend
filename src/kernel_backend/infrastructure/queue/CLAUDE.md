# CLAUDE.md — infrastructure/queue/

## Responsibility

ARQ job queue wiring. Defines how CPU-heavy signing and verification jobs
are enqueued, executed, and polled for status.

## Key design decisions

**CPU-bound work must run in ProcessPoolExecutor.**
ARQ runs on asyncio. Signing a video is 30–120 seconds of CPU work.
Blocking the event loop will starve all other jobs and requests.

```python
# jobs.py — correct pattern
async def process_sign_job(ctx: dict, content_id: str, input_path: str):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        ctx["process_pool"],   # outer ProcessPoolExecutor passed via WorkerSettings.ctx
        _sign_cpu_work,        # top-level function (picklable)
        content_id,
        input_path,
    )
    return result
```

**Nested ProcessPoolExecutor for chunked video encode.**
Inside the outer pool's child process, `_sign_video_cpu` / `_sign_av_cpu` may
create a *second* `ProcessPoolExecutor(max_workers=CHUNK_WORKERS)` for the
parallel encode stage (`core/services/chunk_worker.py`). Linux fork makes this
safe; the codebase never calls `multiprocessing.set_start_method`. The chunked
path runs only when `chunk_planner.plan_chunks` returns more than one chunk —
short videos fall through to the existing sequential encode loop.

```
ARQ async loop
   └── outer ProcessPoolExecutor(SIGN_POOL_MAX_WORKERS, default 2)
         └── _sign_sync()                          ← CPU phase
              └── inner ProcessPoolExecutor(CHUNK_WORKERS, default 4)
                    └── process_video_chunk() × N  ← parallel ffmpeg+libx264
```

Pool children are recycled after `SIGN_POOL_MAX_TASKS_PER_CHILD` jobs (default
50) to bound RSS growth from cv2 / ffmpeg-python long-lived state.

**Jobs must be idempotent.**
ARQ requeues jobs if a worker dies mid-execution. The sign job must be safe
to run twice for the same `content_id` (check if already processed before writing to DB).

**Two-phase signing architecture (Phase 7.2):**

`_sign_sync()` (subprocess) calls `_sign_*_cpu()` and returns a `RawSigningPayload` dict —
no async I/O, no `_NullRegistry`. `process_sign_job` (parent async loop) calls
`_persist_payload(payload, storage, registry)` after `run_in_executor` returns:

```python
# jobs.py — correct CPU/IO split
def _sign_sync(media_path, cert_data, private_key_pem, pepper, org_id) -> dict:
    # Routes to _sign_audio_cpu / _sign_video_cpu / _sign_av_cpu based on media type
    # Returns RawSigningPayload — no storage, no DB, no asyncio.run()
    ...

async def process_sign_job(ctx, ...):
    payload = await loop.run_in_executor(process_pool, _sign_sync, ...)
    # I/O phase with real adapters from ctx
    await _persist_payload(payload, ctx["storage"], ctx["registry"])
```

**Status polling — Redis key takes precedence over ARQ (Phase 7.1):**

`process_sign_job` writes a Redis key `job:{job_id}:status` (TTL 3600 s) at each
milestone so the API endpoint can return a `progress` integer (0–100):

```python
# jobs.py — progress tracking pattern
async def _set_job_status(redis, job_id: str, status: dict) -> None:
    await redis.set(f"job:{job_id}:status", json.dumps(status), ex=3600)

# In process_sign_job:
await _set_job_status(redis, job_id, {"status": "processing", "progress": 0})
# ... CPU work in subprocess ...
await _set_job_status(redis, job_id, {"status": "processing", "progress": 20})
# ... I/O phase in parent loop ...
await _set_job_status(redis, job_id, {"status": "completed", "progress": 100, "result": ...})
```

`ctx["redis"]` and `ctx["job_id"]` are injected automatically by ARQ.

**Fallback: ARQ native Job.info() (no progress):**
```python
from arq.jobs import Job, JobStatus
job = Job(job_id=job_id, redis=redis_pool)
status = await job.status()    # JobStatus enum
info   = await job.info()      # includes result when complete
```

The `GET /sign/{job_id}` endpoint checks the Redis key first; only falls back to
ARQ's native status if the key is absent (e.g. jobs enqueued before Phase 7.1).

## redis_pool.py

```python
from arq.connections import RedisSettings

REDIS_SETTINGS = RedisSettings(
    host="your-endpoint.upstash.io",
    port=6379,
    password="YOUR_PASSWORD",
    ssl=True,
    conn_timeout=10,
)
```
Upstash free tier: 10k commands/day, 256 MB. Sufficient for MVP.
Use the TCP/TLS endpoint — NOT the REST API endpoint.

## WorkerSettings (worker.py)

```python
class WorkerSettings:
    functions                 = [process_sign_job, health_check_job]
    redis_settings            = make_redis_settings(_settings)
    job_timeout               = 360                       # 6 min — covers large video signing
    graceful_shutdown_timeout = 360                       # must be <= compose stop_grace_period (380)
    max_jobs                  = _settings.ARQ_MAX_JOBS    # concurrent jobs per worker process
    keep_result               = 3600                      # 1 h
    cron_jobs                 = [cron(cleanup_signing_tmp, minute={0, 30}, run_at_startup=True)]
```

`on_startup` builds `ctx["process_pool"] =
ProcessPoolExecutor(SIGN_POOL_MAX_WORKERS, max_tasks_per_child=
SIGN_POOL_MAX_TASKS_PER_CHILD)`; this is the *outer* pool. The inner chunk pool
is created lazily inside the child by `signing_service` only when chunked encode
fires.

## Tuning recommendation

The recommended layout for an 8 vCPU host (see `.env.example`):

| Setting | Value | Effect |
|---------|-------|--------|
| `ARQ_MAX_JOBS` | 2 | Up to 2 concurrent jobs per worker container |
| `SIGN_POOL_MAX_WORKERS` | 1 | One CPU-bound `_sign_sync` slot per ARQ slot |
| `SIGN_POOL_MAX_TASKS_PER_CHILD` | 50 | Recycle child to bound RSS |
| `CHUNK_WORKERS` | 4 | Inner pool fans out video encode across 4 cores |
| `CHUNK_GUARD_SEGMENTS` | 1 | One segment of warmup context per chunk boundary |
| `CHUNK_MIN_PAYLOAD_SEGMENTS` | 4 | Below this, fall through to sequential encode |

A single video sign saturates 8 cores (4 chunks × ffmpeg + libx264). Two
concurrent jobs over-subscribe but the kernel scheduler handles it; lower
`CHUNK_WORKERS` if you want strict isolation.

## Validation

```bash
# Start worker in one terminal
make worker

# In another terminal, enqueue a test job
python -m pytest tests/unit/test_queue_jobs.py -v
```

Expected: job enqueues → status = "queued" → status = "in_progress" → status = "complete".
