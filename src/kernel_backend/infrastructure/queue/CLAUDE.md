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
        ctx["process_pool"],   # ProcessPoolExecutor passed via WorkerSettings.ctx
        _sign_cpu_work,        # top-level function (picklable)
        content_id,
        input_path,
    )
    return result
```

**Jobs must be idempotent.**
ARQ requeues jobs if a worker dies mid-execution. The sign job must be safe
to run twice for the same `content_id` (check if already processed before writing to DB).

**Status polling via Job.info():**
```python
from arq.jobs import Job, JobStatus
job = Job(job_id=job_id, redis=redis_pool)
status = await job.status()    # JobStatus enum
info   = await job.info()      # includes result when complete
```

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
    functions    = [process_sign_job]
    redis_settings = REDIS_SETTINGS
    job_timeout  = 180        # 3 min — covers 150 MB video worst case
    max_jobs     = 4          # concurrent jobs per worker process
    retry_jobs   = True
    ctx          = {}         # populated in on_startup
```

## Validation

```bash
# Start worker in one terminal
make worker

# In another terminal, enqueue a test job
python -m pytest tests/unit/test_queue_jobs.py -v
```

Expected: job enqueues → status = "queued" → status = "in_progress" → status = "complete".
