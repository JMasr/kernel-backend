"""
Integration test: adversarially tampered AV file returns RED.

Signs a synthetic 120s AV file, then applies strong luma-channel noise
(σ=80 per pixel) to corrupt the QIM-embedded DCT AC coefficients across
all video segments, re-encodes to H.264 CRF 18, and verifies the result.

Pixel noise σ=80 >> QIM_STEP_WID/2 = 32, so the noise is large enough to
flip the QIM grid decisions after DCT, destroying the WID signal.

Expected outcome: RED verdict for any reason
(VIDEO_WID_UNDECODABLE, VIDEO_WID_MISMATCH, or CANDIDATE_NOT_FOUND).
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

PEPPER = b"tamper-test-pepper-32-bytes-pad!"


def _make_cert(public_key_pem: str) -> Certificate:
    return Certificate(
        author_id="tamper-test-author",
        name="Tamper Test",
        institution="Test Org",
        public_key_pem=public_key_pem,
        created_at="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture(scope="module")
def synthetic_av_120s(tmp_path_factory) -> Path:
    """120-second AV file with noise on both video and audio channels."""
    tmp = tmp_path_factory.mktemp("tamper_av")
    out = tmp / "av_120s.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=120:size=320x240:rate=25,noise=c0s=50:allf=t",
            "-f", "lavfi", "-i", "anoisesrc=duration=120:sample_rate=44100",
            "-ac", "1",
            "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast",
            "-c:a", "aac", "-ar", "44100",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@pytest.mark.integration
@pytest.mark.slow
async def test_tampered_video_ac_noise_returns_red(
    synthetic_av_120s: Path, tmp_path: Path
) -> None:
    """
    [BLOCKING — security] sign_av → apply luma noise σ=80 across all video
    segments → re-encode CRF 18 → verify_av → assert RED verdict.

    Pixel-domain noise σ=80 corrupts QIM-embedded DCT AC coefficients because
    σ_dct ≈ σ = 80 >> QIM_STEP_WID/2 = 32, making the WID unrecoverable.
    The verdict is RED for any reason: VIDEO_WID_UNDECODABLE, VIDEO_WID_MISMATCH,
    or CANDIDATE_NOT_FOUND (all are valid security outcomes for a tampered file).
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
    tampered_path = tmp_path / "tampered_crf18.mp4"
    try:
        signed_path.write_bytes(await storage.get(sign_result.signed_media_key))

        # Apply strong luma noise (σ=80, temporal per-frame) to corrupt DCT-domain
        # QIM bits across all video segments, then re-encode at CRF 18.
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(signed_path),
                "-vf", "noise=c0s=80:c0f=t",
                "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast",
                "-c:a", "copy",
                str(tampered_path),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        signed_path.unlink(missing_ok=True)

    service = VerificationService()
    result: AVVerificationResult = await service.verify_av(
        media_path=tampered_path,
        media=media,
        storage=storage,
        registry=registry,
        pepper=PEPPER,
    )

    assert result.verdict == Verdict.RED, (
        f"Expected RED for tampered file but got {result.verdict} "
        f"(reason={result.red_reason})"
    )
