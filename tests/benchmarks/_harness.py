"""
Subprocess-side helpers used by benchmark workers.

Workers run in a fresh `multiprocessing.Process` (spawn context) so that
`resource.getrusage().ru_maxrss` reflects the worker's own peak RSS rather
than a value contaminated by previous tests in the same pytest process.

A worker is a top-level callable with signature `worker(q: Queue) -> None`
that performs its own setup, calls `measure(fn)` around the timed region,
and puts the resulting dict on `q`. On error it should put
`{"error": <str>}` so the parent can re-raise.
"""
from __future__ import annotations

import resource
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def measure(fn: Callable[[], T]) -> tuple[T, dict[str, Any]]:
    """Run ``fn`` and return its result plus a metrics dict.

    Metrics:
      - wall_s, cpu_user_s, cpu_sys_s
      - cpu_children_user_s, cpu_children_sys_s   (ffmpeg subprocesses)
      - peak_rss_kb                               (process high-water mark)

    Note: ``ru_maxrss`` is the high-water mark for the whole process
    lifetime, not a delta. For verify-style benchmarks where setup also
    signs, the reported peak reflects the higher of (sign setup, measured
    op) — usually sign-dominated.
    """
    pre_self = resource.getrusage(resource.RUSAGE_SELF)
    pre_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    pre_wall = time.perf_counter()

    result = fn()

    post_wall = time.perf_counter()
    post_self = resource.getrusage(resource.RUSAGE_SELF)
    post_children = resource.getrusage(resource.RUSAGE_CHILDREN)

    return result, {
        "wall_s": post_wall - pre_wall,
        "cpu_user_s": post_self.ru_utime - pre_self.ru_utime,
        "cpu_sys_s": post_self.ru_stime - pre_self.ru_stime,
        "cpu_children_user_s": post_children.ru_utime - pre_children.ru_utime,
        "cpu_children_sys_s": post_children.ru_stime - pre_children.ru_stime,
        "peak_rss_kb": int(post_self.ru_maxrss),
    }
