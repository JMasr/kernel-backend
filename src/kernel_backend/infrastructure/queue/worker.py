"""ARQ WorkerSettings. Redis wiring happens at startup via make_redis_settings()."""


class WorkerSettings:
    functions: list = []
    job_timeout: int = 180
    max_jobs: int = 4
