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
from kernel_backend.core.domain.watermark import SegmentFingerprint, VideoEntry
from kernel_backend.core.ports.registry import RegistryPort
from kernel_backend.core.ports.storage import StoragePort
from kernel_backend.core.services.crypto_service import generate_keypair
from kernel_backend.core.services.signing_service import sign_av
from kernel_backend.core.services.verification_service import VerificationService
from kernel_backend.infrastructure.media.media_service import MediaService

PEPPER = b"wid-recovery-test-pepper-32b!!"


class FakeStorage(StoragePort):
    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        self._store[key] = data

    async def get(self, key: str) -> bytes:
        return self._store[key]

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def presigned_upload_url(self, key: str, expires_in: int) -> str:
        return f"fake://{key}"

    async def presigned_download_url(self, key: str, expires_in: int) -> str:
        return f"fake://{key}"


class FakeRegistry(RegistryPort):
    def __init__(self) -> None:
        self._videos: dict[str, VideoEntry] = {}
        self._segments: dict[str, list[SegmentFingerprint]] = {}

    async def save_video(self, entry: VideoEntry) -> None:
        self._videos[entry.content_id] = entry

    async def get_by_content_id(self, content_id: str) -> VideoEntry | None:
        return self._videos.get(content_id)

    async def get_valid_candidates(self) -> list[VideoEntry]:
        return [e for e in self._videos.values() if e.status == "VALID"]

    async def save_segments(
        self, content_id: str, segments: list[SegmentFingerprint], is_original: bool
    ) -> None:
        existing = self._segments.get(content_id, [])
        self._segments[content_id] = existing + list(segments)

    async def match_fingerprints(
        self, hashes: list[str], max_hamming: int = 10
    ) -> list[VideoEntry]:
        from kernel_backend.engine.video.fingerprint import hamming_distance
        matches: set[str] = set()
        for query_hash in hashes:
            for content_id, stored_fps in self._segments.items():
                for sfp in stored_fps:
                    if hamming_distance(query_hash, sfp.hash_hex) <= max_hamming:
                        matches.add(content_id)
        return [self._videos[cid] for cid in matches if cid in self._videos]


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
    tmp = tmp_path_factory.mktemp("wid_rec")
    out = tmp / "av_120s.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=120:size=320x240:rate=25,noise=c0s=100:allf=t",
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
async def test_video_wid_survives_h264_crf28(
    synthetic_av_120s: Path, tmp_path: Path
) -> None:
    """
    [BLOCKING] sign_av → recompress video to CRF 28 → verify_av.
    Assert video_verdict=VERIFIED, wid_match=True.

    QIM_STEP_WID=64.0 sits well above H.264 quantization step (~16 at QP 28).
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
