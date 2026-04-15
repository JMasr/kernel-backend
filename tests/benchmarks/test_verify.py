"""
Verify-side benchmarks — measures the multi-pepper public verify path.

Production orgs carry 5–50 peppers per verify call; 10 is the midpoint and
the value used here. Two scenarios:

  * ``test_verify_match_1_of_10`` — correct pepper at the END of the list
    (worst-case ordering for a successful match).
  * ``test_verify_miss_10_peppers`` — no match; all peppers are wrong
    (full pepper-loop cost; RED verdict).

Note on RSS: each worker also signs once during setup. ``ru_maxrss`` is a
process-lifetime high-water mark, so the reported peak RSS reflects the
higher of (sign setup, verify) — usually sign-dominated. v1 accepts this
limitation; a clean-room two-stage harness would resolve it.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import tempfile
import traceback
from pathlib import Path

import pytest


def _worker_verify_match_1_of_10(q: mp.Queue) -> None:
    try:
        from kernel_backend.core.services.signing_service import sign_audio
        from kernel_backend.core.services.verification_service import VerificationService
        from kernel_backend.infrastructure.media.media_service import MediaService

        from tests.benchmarks._fixtures import (
            BENCH_PEPPER,
            make_keypair_and_cert,
            make_synthetic_audio_120s,
            random_pepper,
        )
        from tests.benchmarks._harness import measure
        from tests.helpers.fakes import FakeRegistry, FakeStorage
        from tests.helpers.signing_defaults import DEFAULT_AUDIO_PARAMS

        # ── Setup (NOT timed; counts toward ru_maxrss) ──────────────────
        audio_path = make_synthetic_audio_120s()
        private_pem, _public_pem, cert = make_keypair_and_cert()
        storage = FakeStorage()
        registry = FakeRegistry()
        media = MediaService()

        sign_result = asyncio.run(
            sign_audio(
                media_path=audio_path,
                certificate=cert,
                private_key_pem=private_pem,
                storage=storage,
                registry=registry,
                pepper=BENCH_PEPPER,
                media=media,
                audio_params=DEFAULT_AUDIO_PARAMS,
            )
        )

        signed_bytes = asyncio.run(storage.get(sign_result.signed_media_key))
        signed_path = Path(tempfile.mkdtemp(prefix="bench_signed_")) / "signed.wav"
        signed_path.write_bytes(signed_bytes)

        peppers = [random_pepper() for _ in range(9)] + [BENCH_PEPPER]
        service = VerificationService()

        # ── Measured ─────────────────────────────────────────────────────
        result, metrics = measure(
            lambda: asyncio.run(
                service.verify_public(
                    media_path=signed_path,
                    media=media,
                    storage=storage,
                    registry=registry,
                    peppers=peppers,
                )
            )
        )
        metrics["input_bytes"] = signed_path.stat().st_size
        metrics["extra"] = {
            "n_peppers": len(peppers),
            "match_position": len(peppers) - 1,  # worst case
            "verdict": str(result.verdict),
        }
        q.put(metrics)
    except Exception:
        q.put({"error": traceback.format_exc()})


def _worker_verify_miss_10_peppers(q: mp.Queue) -> None:
    try:
        from kernel_backend.core.services.signing_service import sign_audio
        from kernel_backend.core.services.verification_service import VerificationService
        from kernel_backend.infrastructure.media.media_service import MediaService

        from tests.benchmarks._fixtures import (
            BENCH_PEPPER,
            make_keypair_and_cert,
            make_synthetic_audio_120s,
            random_pepper,
        )
        from tests.benchmarks._harness import measure
        from tests.helpers.fakes import FakeRegistry, FakeStorage
        from tests.helpers.signing_defaults import DEFAULT_AUDIO_PARAMS

        audio_path = make_synthetic_audio_120s()
        private_pem, _public_pem, cert = make_keypair_and_cert()
        storage = FakeStorage()
        registry = FakeRegistry()
        media = MediaService()

        sign_result = asyncio.run(
            sign_audio(
                media_path=audio_path,
                certificate=cert,
                private_key_pem=private_pem,
                storage=storage,
                registry=registry,
                pepper=BENCH_PEPPER,
                media=media,
                audio_params=DEFAULT_AUDIO_PARAMS,
            )
        )

        signed_bytes = asyncio.run(storage.get(sign_result.signed_media_key))
        signed_path = Path(tempfile.mkdtemp(prefix="bench_signed_")) / "signed.wav"
        signed_path.write_bytes(signed_bytes)

        peppers = [random_pepper() for _ in range(10)]  # all wrong
        service = VerificationService()

        result, metrics = measure(
            lambda: asyncio.run(
                service.verify_public(
                    media_path=signed_path,
                    media=media,
                    storage=storage,
                    registry=registry,
                    peppers=peppers,
                )
            )
        )
        metrics["input_bytes"] = signed_path.stat().st_size
        metrics["extra"] = {
            "n_peppers": len(peppers),
            "match_position": None,
            "verdict": str(result.verdict),
        }
        q.put(metrics)
    except Exception:
        q.put({"error": traceback.format_exc()})


@pytest.mark.benchmark
def test_verify_match_1_of_10(benchmark) -> None:
    record = benchmark.run("verify_match_1_of_10", _worker_verify_match_1_of_10)
    assert record.wall_s > 0.0


@pytest.mark.benchmark
def test_verify_miss_10_peppers(benchmark) -> None:
    record = benchmark.run("verify_miss_10_peppers", _worker_verify_miss_10_peppers)
    assert record.wall_s > 0.0
