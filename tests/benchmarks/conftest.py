"""
Benchmark harness for the kernel-backend perf suite.

Each benchmark runs in a fresh ``multiprocessing.Process`` (spawn context)
so that ``ru_maxrss`` reports the worker's own peak RSS instead of a
process-lifetime high-water mark contaminated by earlier tests.

Workers are top-level functions in test modules (not closures — they must
be picklable under spawn). They use ``tests.benchmarks._harness.measure``
to wrap the timed region and put a metrics dict on the queue the harness
provides.

Output:
  - One JSON blob per run at ``tests/benchmarks/results/<UTC-iso>-<sha>.json``
    (override path with ``--benchmark-json=PATH``).
  - A printed summary table at session end.
"""
from __future__ import annotations

import datetime as _dt
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

_RESULTS_DIR = Path(__file__).parent / "results"
_SUBPROCESS_TIMEOUT_S = 600.0  # 10 min — sign_av on a 22 MB clip can be slow on cold cache


@dataclass
class BenchmarkRecord:
    name: str
    wall_s: float
    cpu_user_s: float
    cpu_sys_s: float
    cpu_children_user_s: float
    cpu_children_sys_s: float
    peak_rss_kb: int
    input_bytes: int
    extra: dict[str, Any] = field(default_factory=dict)


class BenchmarkHarness:
    def __init__(self, records: list[BenchmarkRecord]) -> None:
        self._records = records
        self._ctx = mp.get_context("spawn")

    def run(
        self,
        name: str,
        worker: Callable[..., None],
        *worker_args: Any,
    ) -> BenchmarkRecord:
        """Run ``worker(queue, *worker_args)`` in a fresh subprocess.

        ``worker`` must be a top-level function (picklable) that puts a
        single dict on the queue with keys produced by ``measure`` plus
        ``input_bytes`` and optional ``extra``. On error it should put
        ``{"error": traceback_str}``.
        """
        q: mp.Queue = self._ctx.Queue()
        proc = self._ctx.Process(target=worker, args=(q, *worker_args))
        proc.start()
        try:
            payload = q.get(timeout=_SUBPROCESS_TIMEOUT_S)
        except Exception:
            proc.kill()
            proc.join()
            raise
        proc.join(timeout=10.0)
        if proc.exitcode != 0:
            raise RuntimeError(
                f"benchmark {name!r} subprocess exited with {proc.exitcode}; payload={payload!r}"
            )
        if "error" in payload:
            raise RuntimeError(f"benchmark {name!r} worker raised:\n{payload['error']}")

        record = BenchmarkRecord(
            name=name,
            wall_s=float(payload["wall_s"]),
            cpu_user_s=float(payload["cpu_user_s"]),
            cpu_sys_s=float(payload["cpu_sys_s"]),
            cpu_children_user_s=float(payload["cpu_children_user_s"]),
            cpu_children_sys_s=float(payload["cpu_children_sys_s"]),
            peak_rss_kb=int(payload["peak_rss_kb"]),
            input_bytes=int(payload.get("input_bytes", 0)),
            extra=dict(payload.get("extra", {})),
        )
        self._records.append(record)
        return record


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--benchmark-json",
        action="store",
        default=None,
        help="Write benchmark results as JSON to this path (overrides default tests/benchmarks/results/<run>.json).",
    )


# Records are stashed on the session config so sessionfinish can drain them
# regardless of which benchmark fixtures resolved first.
_SESSION_RECORDS_KEY = "_kernel_benchmark_records"


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    setattr(session.config, _SESSION_RECORDS_KEY, [])


@pytest.fixture
def benchmark(request: pytest.FixtureRequest) -> BenchmarkHarness:
    records: list[BenchmarkRecord] = getattr(request.session.config, _SESSION_RECORDS_KEY)
    return BenchmarkHarness(records)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    records: list[BenchmarkRecord] = getattr(session.config, _SESSION_RECORDS_KEY, [])
    if not records:
        return

    run_id = _make_run_id()
    out_path_opt = session.config.getoption("--benchmark-json")
    if out_path_opt:
        out_path = Path(out_path_opt)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _RESULTS_DIR / f"{run_id}.json"

    payload = {
        "run_id": run_id,
        "host": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "records": [asdict(r) for r in records],
    }
    out_path.write_text(json.dumps(payload, indent=2))

    _print_table(records, out_path)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_run_id() -> str:
    stamp = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = _git_short_sha()
    return f"{stamp}-{sha}"


def _git_short_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or "nogit"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def _print_table(records: list[BenchmarkRecord], out_path: Path) -> None:
    headers = ("name", "wall_s", "cpu_user", "cpu_sys", "ffmpeg_user", "peak_rss_MB", "in_MB")
    rows = [
        (
            r.name,
            f"{r.wall_s:.3f}",
            f"{r.cpu_user_s:.3f}",
            f"{r.cpu_sys_s:.3f}",
            f"{r.cpu_children_user_s:.3f}",
            f"{r.peak_rss_kb / 1024:.1f}",
            f"{r.input_bytes / (1024 * 1024):.2f}",
        )
        for r in records
    ]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    sep = "  "
    line = sep.join(h.ljust(w) for h, w in zip(headers, widths))

    print(file=sys.stderr)
    print("── benchmark results ─────────────────────────────────────────────", file=sys.stderr)
    print(line, file=sys.stderr)
    print(sep.join("-" * w for w in widths), file=sys.stderr)
    for row in rows:
        print(sep.join(c.ljust(w) for c, w in zip(row, widths)), file=sys.stderr)
    print(f"\nwrote {out_path}", file=sys.stderr)
