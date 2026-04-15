"""
Sign-side benchmarks.

Each test runs in its own subprocess (see conftest.BenchmarkHarness) so
``ru_maxrss`` reflects only that benchmark's peak RSS. ffmpeg subprocess
CPU is captured via ``RUSAGE_CHILDREN`` from inside the worker.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import traceback

import pytest


def _worker_sign_audio_120s(q: mp.Queue) -> None:
    try:
        from kernel_backend.core.services.signing_service import sign_audio
        from kernel_backend.infrastructure.media.media_service import MediaService

        from tests.benchmarks._fixtures import (
            BENCH_PEPPER,
            make_keypair_and_cert,
            make_synthetic_audio_120s,
        )
        from tests.benchmarks._harness import measure
        from tests.helpers.fakes import FakeRegistry, FakeStorage
        from tests.helpers.signing_defaults import DEFAULT_AUDIO_PARAMS

        audio_path = make_synthetic_audio_120s()
        private_pem, _public_pem, cert = make_keypair_and_cert()

        _, metrics = measure(
            lambda: asyncio.run(
                sign_audio(
                    media_path=audio_path,
                    certificate=cert,
                    private_key_pem=private_pem,
                    storage=FakeStorage(),
                    registry=FakeRegistry(),
                    pepper=BENCH_PEPPER,
                    media=MediaService(),
                    audio_params=DEFAULT_AUDIO_PARAMS,
                )
            )
        )
        metrics["input_bytes"] = audio_path.stat().st_size
        metrics["extra"] = {"input_seconds": 120.0, "media": "audio_wav_mono_44100"}
        q.put(metrics)
    except Exception:
        q.put({"error": traceback.format_exc()})


def _worker_sign_av_speech(q: mp.Queue) -> None:
    try:
        from kernel_backend.core.services.signing_service import sign_av
        from kernel_backend.infrastructure.media.media_service import MediaService

        from tests.benchmarks._fixtures import (
            BENCH_PEPPER,
            SPEECH_AV_PATH,
            make_keypair_and_cert,
        )
        from tests.benchmarks._harness import measure
        from tests.helpers.fakes import FakeRegistry, FakeStorage
        from tests.helpers.signing_defaults import DEFAULT_AV_AUDIO_PARAMS

        if not SPEECH_AV_PATH.exists():
            q.put({"error": f"missing fixture: {SPEECH_AV_PATH}"})
            return

        private_pem, _public_pem, cert = make_keypair_and_cert()

        _, metrics = measure(
            lambda: asyncio.run(
                sign_av(
                    media_path=SPEECH_AV_PATH,
                    certificate=cert,
                    private_key_pem=private_pem,
                    storage=FakeStorage(),
                    registry=FakeRegistry(),
                    pepper=BENCH_PEPPER,
                    media=MediaService(),
                    audio_params=DEFAULT_AV_AUDIO_PARAMS,
                )
            )
        )
        metrics["input_bytes"] = SPEECH_AV_PATH.stat().st_size
        metrics["extra"] = {"media": "av_mp4_speech_22mb"}
        q.put(metrics)
    except Exception:
        q.put({"error": traceback.format_exc()})


@pytest.mark.benchmark
def test_sign_audio_120s(benchmark) -> None:
    record = benchmark.run("sign_audio_120s", _worker_sign_audio_120s)
    assert record.wall_s > 0.0
    assert record.peak_rss_kb > 0


@pytest.mark.benchmark
def test_sign_av_speech(benchmark) -> None:
    record = benchmark.run("sign_av_speech", _worker_sign_av_speech)
    assert record.wall_s > 0.0
    assert record.peak_rss_kb > 0
