"""
Pipeline tests for sign -> verify roundtrip at domain level.

Tests the full signing and verification pipeline with real DSP, real services,
and no mocking. If signing params change (e.g. chips_per_bit, dwt_levels), the
verify side will immediately fail because the WID is embedded at one rate and
extracted at another — no mock hides this desync.

Replaces the old sign-only smoke tests (test_sign_audio_smoke.py,
test_sign_video_smoke.py, test_sign_av_smoke.py) and mock-heavy verification
tests (test_verification_service.py, test_verify_av_service.py).

Edge cases from those files are migrated here; verdict logic is in
test_verdict_logic.py.

Run:
    uv run python -m pytest tests/unit/test_pipeline_sign_verify.py -v
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.domain.verification import (
    AVVerificationResult,
    RedReason,
    VerificationResult,
    Verdict,
)
from kernel_backend.core.domain.watermark import EmbeddingParams, embedding_params_to_dict
from kernel_backend.core.services.crypto_service import derive_wid, generate_keypair
from kernel_backend.core.services.signing_service import (
    sign_audio,
    sign_av,
    sign_video,
)
from kernel_backend.core.services.verification_service import VerificationService
from kernel_backend.infrastructure.media.media_service import MediaService
from tests.helpers.fakes import FakeRegistry, FakeStorage
from tests.helpers.signing_defaults import DEFAULT_AUDIO_PARAMS, DEFAULT_EMBEDDING_PARAMS

PEPPER = b"pipeline-test-pepper-padded-32b!"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_cert(public_key_pem: str) -> Certificate:
    return Certificate(
        author_id="pipeline-test-author",
        name="Pipeline Test Author",
        institution="Test Org",
        public_key_pem=public_key_pem,
        created_at="2026-01-01T00:00:00+00:00",
    )


# ── Module-scoped FFmpeg fixtures (generated once) ───────────────────────────


@pytest.fixture(scope="module")
def synthetic_audio_120s(tmp_path_factory) -> Path:
    """120s broadband noise WAV — 60 segments x 2s, well above RS minimum 17."""
    tmp = tmp_path_factory.mktemp("pipeline_audio")
    out = tmp / "audio_120s.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anoisesrc=duration=120:sample_rate=44100",
            "-ac", "1",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture(scope="module")
def synthetic_video_120s(tmp_path_factory) -> Path:
    """120s video with noise-enriched testsrc, CRF 0 (lossless), no audio."""
    tmp = tmp_path_factory.mktemp("pipeline_video")
    out = tmp / "video_120s.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "testsrc=duration=120:size=320x240:rate=15,noise=c0s=100:allf=t",
            "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
            "-an",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture(scope="module")
def synthetic_av_120s(tmp_path_factory) -> Path:
    """120s AV: noise-enriched testsrc video + anoisesrc audio, CRF 0."""
    tmp = tmp_path_factory.mktemp("pipeline_av")
    out = tmp / "av_120s.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i",
            "testsrc=duration=120:size=320x240:rate=15,noise=c0s=100:allf=t",
            "-f", "lavfi", "-i", "anoisesrc=duration=120:sample_rate=44100",
            "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
            "-c:a", "aac", "-ar", "44100", "-ac", "1",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


# ── Pipeline roundtrip tests ─────────────────────────────────────────────────


@pytest.mark.slow
async def test_audio_sign_verify_roundtrip(synthetic_audio_120s: Path) -> None:
    """sign_audio -> verify_audio -> VERIFIED, wid_match, signature_valid."""
    private_pem, public_pem = generate_keypair()
    cert = _make_cert(public_pem)
    storage = FakeStorage()
    registry = FakeRegistry()
    media = MediaService()

    sign_result = await sign_audio(
        media_path=synthetic_audio_120s,
        certificate=cert,
        private_key_pem=private_pem,
        storage=storage,
        registry=registry,
        pepper=PEPPER,
        media=media,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        signed_path = Path(tmp.name)
    try:
        signed_bytes = await storage.get(sign_result.signed_media_key)
        signed_path.write_bytes(signed_bytes)

        service = VerificationService()
        result: VerificationResult = await service.verify_audio(
            media_path=signed_path,
            media=media,
            storage=storage,
            registry=registry,
            pepper=PEPPER,
        )
    finally:
        signed_path.unlink(missing_ok=True)

    assert result.verdict == Verdict.VERIFIED, (
        f"Expected VERIFIED but got {result.verdict} (reason={result.red_reason})"
    )
    assert result.wid_match is True
    assert result.signature_valid is True


@pytest.mark.slow
async def test_video_sign_verify_roundtrip(synthetic_video_120s: Path) -> None:
    """sign_video -> verify -> VERIFIED, wid_match, signature_valid."""
    private_pem, public_pem = generate_keypair()
    cert = _make_cert(public_pem)
    storage = FakeStorage()
    registry = FakeRegistry()
    media = MediaService()

    sign_result = await sign_video(
        media_path=synthetic_video_120s,
        certificate=cert,
        private_key_pem=private_pem,
        storage=storage,
        registry=registry,
        pepper=PEPPER,
        media=media,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        signed_path = Path(tmp.name)
    try:
        signed_bytes = await storage.get(sign_result.signed_media_key)
        signed_path.write_bytes(signed_bytes)

        service = VerificationService()
        result: VerificationResult = await service.verify(
            media_path=signed_path,
            media=media,
            storage=storage,
            registry=registry,
            pepper=PEPPER,
        )
    finally:
        signed_path.unlink(missing_ok=True)

    assert result.verdict == Verdict.VERIFIED, (
        f"Expected VERIFIED but got {result.verdict} (reason={result.red_reason})"
    )
    assert result.wid_match is True
    assert result.signature_valid is True


@pytest.mark.slow
async def test_av_sign_verify_roundtrip(synthetic_av_120s: Path) -> None:
    """sign_av -> verify_av -> VERIFIED on both channels."""
    private_pem, public_pem = generate_keypair()
    cert = _make_cert(public_pem)
    storage = FakeStorage()
    registry = FakeRegistry()
    media = MediaService()

    sign_result = await sign_av(
        media_path=synthetic_av_120s,
        certificate=cert,
        private_key_pem=private_pem,
        storage=storage,
        registry=registry,
        pepper=PEPPER,
        media=media,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        signed_path = Path(tmp.name)
    try:
        signed_bytes = await storage.get(sign_result.signed_media_key)
        signed_path.write_bytes(signed_bytes)

        service = VerificationService()
        result: AVVerificationResult = await service.verify_av(
            media_path=signed_path,
            media=media,
            storage=storage,
            registry=registry,
            pepper=PEPPER,
        )
    finally:
        signed_path.unlink(missing_ok=True)

    assert result.verdict == Verdict.VERIFIED, (
        f"Expected VERIFIED but got {result.verdict} "
        f"(reason={result.red_reason}, "
        f"audio={result.audio_verdict}, video={result.video_verdict})"
    )
    assert result.wid_match is True
    assert result.signature_valid is True
    assert result.audio_verdict == Verdict.VERIFIED
    assert result.video_verdict == Verdict.VERIFIED


# ── Negative pipeline tests (security invariants) ────────────────────────────


async def test_audio_unsigned_returns_red(synthetic_audio_120s: Path) -> None:
    """Unsigned audio -> RED / CANDIDATE_NOT_FOUND."""
    service = VerificationService()
    result = await service.verify_audio(
        media_path=synthetic_audio_120s,
        media=MediaService(),
        storage=FakeStorage(),
        registry=FakeRegistry(),  # empty — no signed content
        pepper=PEPPER,
    )
    assert result.verdict == Verdict.RED
    assert result.red_reason == RedReason.CANDIDATE_NOT_FOUND


async def test_video_unsigned_returns_red(synthetic_video_120s: Path) -> None:
    """Unsigned video -> RED / CANDIDATE_NOT_FOUND."""
    service = VerificationService()
    result = await service.verify(
        media_path=synthetic_video_120s,
        media=MediaService(),
        storage=FakeStorage(),
        registry=FakeRegistry(),
        pepper=PEPPER,
    )
    assert result.verdict == Verdict.RED
    assert result.red_reason == RedReason.CANDIDATE_NOT_FOUND


async def test_av_unsigned_returns_red(synthetic_av_120s: Path) -> None:
    """Unsigned AV -> RED / CANDIDATE_NOT_FOUND."""
    service = VerificationService()
    result = await service.verify_av(
        media_path=synthetic_av_120s,
        media=MediaService(),
        storage=FakeStorage(),
        registry=FakeRegistry(),
        pepper=PEPPER,
    )
    assert result.verdict == Verdict.RED
    assert result.red_reason == RedReason.CANDIDATE_NOT_FOUND


# ── Contract & edge-case tests (migrated from removed smoke files) ───────────


@pytest.mark.slow
async def test_stored_params_match_defaults(synthetic_audio_120s: Path) -> None:
    """Signing stores embedding_params in the registry entry — must match current defaults."""
    private_pem, public_pem = generate_keypair()
    cert = _make_cert(public_pem)
    registry = FakeRegistry()

    result = await sign_audio(
        media_path=synthetic_audio_120s,
        certificate=cert,
        private_key_pem=private_pem,
        storage=FakeStorage(),
        registry=registry,
        pepper=PEPPER,
        media=MediaService(),
    )

    entry = registry._videos[result.content_id]
    # Audio-only signing stores video=None (only audio params are relevant)
    expected = EmbeddingParams(audio=DEFAULT_AUDIO_PARAMS, video=None)
    assert entry.embedding_params == expected


@pytest.mark.slow
async def test_no_pilot_in_active_signals(synthetic_audio_120s: Path) -> None:
    """Pilot embedding was removed — active_signals must not contain pilot_audio."""
    private_pem, public_pem = generate_keypair()
    cert = _make_cert(public_pem)

    result = await sign_audio(
        media_path=synthetic_audio_120s,
        certificate=cert,
        private_key_pem=private_pem,
        storage=FakeStorage(),
        registry=FakeRegistry(),
        pepper=PEPPER,
        media=MediaService(),
    )

    assert set(result.active_signals) == {"wid_audio", "fingerprint_audio"}
    assert "pilot_audio" not in result.active_signals


@pytest.mark.slow
async def test_av_shared_wid(synthetic_av_120s: Path) -> None:
    """sign_av uses a single WID for both audio and video channels."""
    private_pem, public_pem = generate_keypair()
    cert = _make_cert(public_pem)
    registry = FakeRegistry()

    result = await sign_av(
        media_path=synthetic_av_120s,
        certificate=cert,
        private_key_pem=private_pem,
        storage=FakeStorage(),
        registry=registry,
        pepper=PEPPER,
        media=MediaService(),
    )

    entry = registry._videos[result.content_id]
    rederived_wid = derive_wid(entry.manifest_signature, result.content_id)
    assert result.wid.data == rederived_wid.data
    assert len(registry._videos) == 1


async def test_short_audio_raises_value_error(tmp_path: Path) -> None:
    """sign_audio raises ValueError when audio < 17 segments (< 34 seconds)."""
    short = tmp_path / "short.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anoisesrc=duration=10:sample_rate=44100",
            "-ac", "1",
            str(short),
        ],
        check=True,
        capture_output=True,
    )
    private_pem, public_pem = generate_keypair()
    cert = _make_cert(public_pem)

    with pytest.raises(ValueError, match="Audio is too short"):
        await sign_audio(
            media_path=short,
            certificate=cert,
            private_key_pem=private_pem,
            storage=FakeStorage(),
            registry=FakeRegistry(),
            pepper=PEPPER,
            media=MediaService(),
        )


async def test_short_video_raises_value_error(tmp_path: Path) -> None:
    """sign_video raises ValueError when video < 17 segments (< 85 seconds)."""
    short = tmp_path / "short.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "testsrc=duration=30:size=320x240:rate=15,noise=c0s=100:allf=t",
            "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast", "-an",
            str(short),
        ],
        check=True,
        capture_output=True,
    )
    private_pem, public_pem = generate_keypair()
    cert = _make_cert(public_pem)

    with pytest.raises(ValueError, match="Video is too short"):
        await sign_video(
            media_path=short,
            certificate=cert,
            private_key_pem=private_pem,
            storage=FakeStorage(),
            registry=FakeRegistry(),
            pepper=PEPPER,
            media=MediaService(),
        )


async def test_audio_only_file_rejected_by_sign_video(tmp_path: Path) -> None:
    """sign_video raises ValueError on an audio-only file."""
    audio = tmp_path / "audio_only.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anoisesrc=duration=10:sample_rate=44100",
            "-ac", "1",
            str(audio),
        ],
        check=True,
        capture_output=True,
    )
    private_pem, public_pem = generate_keypair()
    cert = _make_cert(public_pem)

    with pytest.raises(ValueError, match="no video track"):
        await sign_video(
            media_path=audio,
            certificate=cert,
            private_key_pem=private_pem,
            storage=FakeStorage(),
            registry=FakeRegistry(),
            pepper=PEPPER,
            media=MediaService(),
        )
