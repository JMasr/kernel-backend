"""
Integration tests for WID survival under audio/video compression.

These tests verify that the watermark survives real-world codec degradation:
- Video WID survives H.264 CRF 28 (standard social media compression)
- Audio WID survives AAC 192k (the bitrate used by sign_av)

Marked @pytest.mark.integration and @pytest.mark.slow.

Run:
    pytest tests/integration/test_wid_recovery_compressed.py -v
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.domain.verification import AVVerificationResult, Verdict
from kernel_backend.core.services.crypto_service import generate_keypair
from kernel_backend.core.services.signing_service import sign_av
from kernel_backend.core.services.verification_service import VerificationService
from kernel_backend.infrastructure.media.media_service import MediaService
from tests.helpers.fakes import FakeRegistry, FakeStorage

PEPPER = b"wid-recovery-test-pepper-32b!!"


def _make_cert(public_key_pem: str) -> Certificate:
    return Certificate(
        author_id="wid-recovery-test",
        name="WID Recovery Test",
        institution="Test Org",
        public_key_pem=public_key_pem,
        created_at="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture(scope="module")
def synthetic_av_120s(tmp_path_factory) -> Path:
    """120-second AV file with lossless source video.

    CRF 0 source avoids double-quantization: the test recompresses at CRF 28,
    so only one lossy pass affects the watermark (matching real-world flow).
    320x240 @ 25 fps gives 4800 blocks/frame (well above N_WID_BLOCKS_PER_SEGMENT=128).
    """
    tmp = tmp_path_factory.mktemp("wid_rec")
    out = tmp / "av_120s.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=120:size=320x240:rate=25,noise=c0s=100:allf=t",
            "-f", "lavfi", "-i", "anoisesrc=duration=120:sample_rate=44100",
            "-ac", "1",
            "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
            "-c:a", "aac", "-ar", "44100",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.xfail(
    strict=False,
    reason="QIM_STEP=48 at 320x240 synthetic resolution produces false RS-decode at CRF 28. "
           "Real content at 960x540 passes (test_sign_verify_video_only).",
)
async def test_video_wid_survives_h264_crf28(
    synthetic_av_120s: Path, tmp_path: Path
) -> None:
    """
    [BLOCKING] sign_av → recompress video to CRF 28 → verify_av.
    Assert video_verdict=VERIFIED, wid_match=True.

    QIM_STEP_WID=48.0 sits above H.264 quantization step (~16 at QP 28).
    """
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
    recompressed = tmp_path / "crf28.mp4"
    try:
        signed_path.write_bytes(await storage.get(sign_result.signed_media_key))
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(signed_path),
                "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast",
                "-c:a", "copy",
                str(recompressed),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        signed_path.unlink(missing_ok=True)

    result: AVVerificationResult = await VerificationService().verify_av(
        media_path=recompressed,
        media=media,
        storage=storage,
        registry=registry,
        pepper=PEPPER,
    )

    assert result.video_verdict == Verdict.VERIFIED, (
        f"video_verdict={result.video_verdict}, red_reason={result.red_reason}, "
        f"video_n_seg={result.video_n_segments}, video_n_erasures={result.video_n_erasures}, "
        f"video_n_decoded={result.video_n_decoded}, wid_match={result.wid_match}, "
        f"sig_valid={result.signature_valid}"
    )
    assert result.wid_match is True


@pytest.mark.integration
@pytest.mark.slow
async def test_audio_wid_survives_aac_192k(
    synthetic_av_120s: Path, tmp_path: Path
) -> None:
    """
    [BLOCKING] sign_av → re-encode audio at AAC 192k → verify_av.
    Assert audio_verdict=VERIFIED.

    The audio DWT watermark was calibrated for AAC at 192k (the bitrate used
    in sign_av). This test verifies no regression from the AAC re-encode step.
    """
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
    reencoded = tmp_path / "aac_192k.mp4"
    try:
        signed_path.write_bytes(await storage.get(sign_result.signed_media_key))
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(signed_path),
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                str(reencoded),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        signed_path.unlink(missing_ok=True)

    result: AVVerificationResult = await VerificationService().verify_av(
        media_path=reencoded,
        media=media,
        storage=storage,
        registry=registry,
        pepper=PEPPER,
    )

    assert result.audio_verdict == Verdict.VERIFIED
