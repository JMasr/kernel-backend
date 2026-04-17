"""Integration tests for ``process_sign_job`` phase-boundary logging.

These tests stub out the CPU work, storage, and registry so the signing
pipeline does not actually run — what we're asserting is the *observability
contract*: every job emits a fixed sequence of structured events that carry
the job/request/trace correlation ids, and a failing validation produces a
``sign.validation.failed`` + ``sign.job.failed`` pair with the same ids.

If the observability contract regresses (a log line is dropped, a field is
renamed, contextvars leak across jobs), these tests fail — which is the
whole point of Phase B.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import structlog
from structlog.testing import LogCapture

from kernel_backend.core.domain.media import MediaProfile
from kernel_backend.infrastructure.queue.jobs import process_sign_job


@pytest.fixture
def cap_logs() -> "list[dict[str, Any]]":
    """Install a structlog pipeline that preserves ``merge_contextvars`` — so
    the captured events carry ``job_id`` / ``request_id`` / ``trace_id`` —
    and appends each event to a list the test can assert on.

    ``structlog.testing.capture_logs`` would also work but *replaces* the
    whole processor chain, which drops contextvars. We want them kept.
    """
    prior = structlog.get_config()
    capture = LogCapture()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            capture,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    try:
        yield capture.entries
    finally:
        structlog.configure(**prior)


# A minimal, valid certificate_json payload. ``process_sign_job`` only cares
# that json.loads succeeds and that the five required keys are present.
_CERT_JSON = (
    '{"author_id": "author-xyz",'
    '"name": "Test Author",'
    '"institution": "Test Inst",'
    '"public_key_pem": "stub-pub-key",'
    '"created_at": "2026-04-17T00:00:00Z"}'
)


def _profile(*, has_video: bool, has_audio: bool, duration_s: float = 120.0) -> MediaProfile:
    return MediaProfile(
        has_video=has_video,
        has_audio=has_audio,
        width=1280 if has_video else 0,
        height=720 if has_video else 0,
        fps=30.0 if has_video else 0.0,
        duration_s=duration_s,
        sample_rate=44100 if has_audio else 0,
    )


def _make_ctx(*, with_pool: bool = False, with_redis: bool = True) -> dict[str, Any]:
    """Build a minimal ARQ ``ctx`` dict. ``process_pool=None`` routes the job
    through the in-process fallback, which is what we want to exercise — the
    subprocess path can't be captured by ``capture_logs``."""
    ctx: dict[str, Any] = {
        "job_id": "arq-job-42",
        "storage": object(),
        "registry": object(),
        "pepper": b"\x00" * 32,
        "process_pool": None if not with_pool else object(),
        "email_adapter": None,
    }
    if with_redis:
        ctx["redis"] = AsyncMock()
    return ctx


def _events_of(cap_logs: list[dict[str, Any]], *, level: str | None = None) -> list[dict[str, Any]]:
    return [r for r in cap_logs if level is None or r.get("log_level") == level]


@pytest.fixture(autouse=True)
def _clear_contextvars():
    """Prevent one test's contextvar binding from leaking into the next."""
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()


@pytest.mark.asyncio
async def test_happy_path_emits_five_phase_boundaries_in_order(
    tmp_path: Path, cap_logs: list[dict[str, Any]]
) -> None:
    """Audio-only job through the in-process branch must emit the full sequence:

    ``sign.job.start`` → ``sign.validation.start`` → ``sign.validation.end`` →
    ``sign.job.complete``

    and every event must carry ``job_id``, ``request_id``, ``trace_id``, ``org_id``.
    """
    media_file = tmp_path / "clip.aac"
    media_file.write_bytes(b"fake")

    profile = _profile(has_video=False, has_audio=True, duration_s=90.0)

    fake_result = type(
        "R",
        (),
        {
            "content_id": "ct-001",
            "signed_media_key": "signed/ct-001.aac",
            "active_signals": ["pilot"],
            "rs_n": 1,
        },
    )()

    with patch(
        "kernel_backend.infrastructure.queue.jobs._validate_local_media",
        AsyncMock(return_value=(str(media_file), profile)),
    ), patch(
        "kernel_backend.infrastructure.queue.jobs.sign_audio",
        AsyncMock(return_value=fake_result),
    ):
        result = await process_sign_job(
            ctx=_make_ctx(with_pool=False),
            media_path=str(media_file),
            certificate_json=_CERT_JSON,
            private_key_pem="-----BEGIN STUB-----\nAA\n-----END STUB-----",
            org_id="11111111-1111-4111-8111-111111111111",
            original_filename="clip.aac",
            request_id="rid-abc",
            trace_id="tid-xyz",
        )

    assert result["content_id"] == "ct-001"

    events = [r["event"] for r in cap_logs]
    # sign.cpu.* only fires on the process-pool branch; in-process skips it.
    for required in ("sign.job.start", "sign.validation.start", "sign.validation.end", "sign.job.complete"):
        assert required in events, f"missing phase log {required!r} in {events!r}"

    # Ordering: start → validation.start → validation.end → complete
    assert events.index("sign.job.start") < events.index("sign.validation.start")
    assert events.index("sign.validation.start") < events.index("sign.validation.end")
    assert events.index("sign.validation.end") < events.index("sign.job.complete")

    # Every captured event must carry the full correlation quartet. ``capture_logs``
    # sees contextvars as top-level keys on each event dict.
    for rec in cap_logs:
        assert rec.get("job_id") == "arq-job-42", rec
        assert rec.get("request_id") == "rid-abc", rec
        assert rec.get("trace_id") == "tid-xyz", rec
        assert rec.get("org_id") == "11111111-1111-4111-8111-111111111111", rec


@pytest.mark.asyncio
async def test_validation_failure_emits_failed_logs_and_reraises(
    tmp_path: Path, cap_logs: list[dict[str, Any]]
) -> None:
    """Validation ``ValueError`` must surface as both ``sign.validation.failed``
    and ``sign.job.failed`` (the outer catch-all), and re-raise so ARQ marks
    the job failed."""
    media_file = tmp_path / "too_long.mp4"
    media_file.write_bytes(b"fake")

    with patch(
        "kernel_backend.infrastructure.queue.jobs._validate_local_media",
        AsyncMock(side_effect=ValueError("File is too long")),
    ):
        with pytest.raises(ValueError, match="too long"):
            await process_sign_job(
                ctx=_make_ctx(with_pool=False),
                media_path=str(media_file),
                certificate_json=_CERT_JSON,
                private_key_pem="stub",
                org_id="22222222-2222-4222-8222-222222222222",
                original_filename="too_long.mp4",
                request_id="rid-fail",
                trace_id=None,
            )

    events = [r["event"] for r in cap_logs]
    assert "sign.job.start" in events
    assert "sign.validation.start" in events
    assert "sign.validation.failed" in events
    assert "sign.job.failed" in events
    # sign.job.complete must NOT appear
    assert "sign.job.complete" not in events

    failed = next(r for r in cap_logs if r["event"] == "sign.job.failed")
    assert failed["log_level"] == "error"
    assert failed["exc_type"] == "ValueError"
    assert "too long" in failed["error"]
    assert failed.get("request_id") == "rid-fail"
    assert failed.get("job_id") == "arq-job-42"


@pytest.mark.asyncio
async def test_contextvars_cleared_between_jobs(tmp_path: Path) -> None:
    """The ``finally`` in ``process_sign_job`` clears contextvars so a later
    log line in the same worker (e.g. cleanup_job) does not inherit the last
    job's ``job_id`` / ``request_id``."""
    media_file = tmp_path / "ok.aac"
    media_file.write_bytes(b"fake")
    profile = _profile(has_video=False, has_audio=True, duration_s=90.0)
    fake = type("R", (), {"content_id": "c", "signed_media_key": "k", "active_signals": [], "rs_n": 0})()

    with patch(
        "kernel_backend.infrastructure.queue.jobs._validate_local_media",
        AsyncMock(return_value=(str(media_file), profile)),
    ), patch(
        "kernel_backend.infrastructure.queue.jobs.sign_audio",
        AsyncMock(return_value=fake),
    ):
        await process_sign_job(
            ctx=_make_ctx(with_pool=False),
            media_path=str(media_file),
            certificate_json=_CERT_JSON,
            private_key_pem="stub",
            org_id="11111111-1111-4111-8111-111111111111",
            request_id="rid-1",
            trace_id="tid-1",
        )

    # After the job returns, nothing should remain bound.
    merged = structlog.contextvars.get_merged_contextvars(structlog.get_logger())
    assert merged == {}, f"contextvars leaked after job: {merged!r}"
