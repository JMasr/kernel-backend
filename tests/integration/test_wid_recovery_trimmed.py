"""
Integration tests for WID recovery after content trimming.

Verifies that the Reed-Solomon erasure correction is strong enough to recover
the full WID after a signed audio file is trimmed at the end.

RS parameters: K=16, N=min(n_segments, 255), parity=N-K.
For a 120s audio file: N=60, parity=44.
Trimming to 60s removes 30 RS symbols — 30 erasures ≤ 44 parity → recoverable.

Key design invariant: trimming from the END is safe because DWT segment extraction
proceeds from time=0 and naturally marks out-of-bounds segments as erasures.
Trimming from the START shifts segment offsets and is NOT expected to work.

Run:
    pytest tests/integration/test_wid_recovery_trimmed.py -v
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.domain.verification import RedReason, VerificationResult, Verdict
from kernel_backend.core.domain.watermark import SegmentFingerprint, VideoEntry
from kernel_backend.core.ports.registry import RegistryPort
from kernel_backend.core.ports.storage import StoragePort
from kernel_backend.core.services.crypto_service import generate_keypair
from kernel_backend.core.services.signing_service import sign_audio
from kernel_backend.core.services.verification_service import VerificationService
from kernel_backend.infrastructure.media.media_service import MediaService

PEPPER = b"integration-test-pepper-padded!!"


# ── Fake infrastructure ────────────────────────────────────────────────────────


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
        from kernel_backend.engine.audio.fingerprint import hamming_distance
        matches: set[str] = set()
        for query_hash in hashes:
            for content_id, stored_fps in self._segments.items():
                for sfp in stored_fps:
                    if hamming_distance(query_hash, sfp.hash_hex) <= max_hamming:
                        matches.add(content_id)
        return [self._videos[cid] for cid in matches if cid in self._videos]


def _make_cert(public_key_pem: str) -> Certificate:
    return Certificate(
        author_id="integration-author",
        name="Integration Test Author",
        institution="Test Org",
        public_key_pem=public_key_pem,
        created_at="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture(scope="module")
def synthetic_audio_120s(tmp_path_factory) -> Path:
    """120-second broadband noise audio for trim tests."""
    tmp = tmp_path_factory.mktemp("integ_trim")
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


@pytest.mark.integration
@pytest.mark.slow
async def test_audio_wid_survives_trailing_trim_50_percent(
    synthetic_audio_120s: Path, tmp_path: Path
) -> None:
    """
    [BLOCKING] sign 120s audio → trim trailing 60s (keep first 60s) → verify.

    RS parameters for 120s:
      N = 60 segments, K = 16, parity = 44
    After trim to 60s:
      Extractable symbols ≈ 30 (segments 0–29 intact)
      Erasures ≈ 30 (segments 30–59 removed)
      30 erasures ≤ 44 parity → RS decode succeeds → VERIFIED
    """
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

    # Write signed bytes to temp file for trimming
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
        signed_path = Path(tmp_f.name)
    trimmed_path = tmp_path / "trimmed_60s.wav"
    try:
        signed_bytes = await storage.get(sign_result.signed_media_key)
        signed_path.write_bytes(signed_bytes)

        # Trim to first 60 seconds (keep segments 0–29, erase 30–59)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(signed_path),
                "-t", "60",
                "-c", "copy",
                str(trimmed_path),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        signed_path.unlink(missing_ok=True)

    service = VerificationService()
    result: VerificationResult = await service.verify_audio(
        media_path=trimmed_path,
        media=media,
        storage=storage,
        registry=registry,
        pepper=PEPPER,
    )

    assert result.verdict == Verdict.VERIFIED, (
        f"50% trailing trim: expected VERIFIED but got {result.verdict} "
        f"(reason={result.red_reason})"
    )
    assert result.wid_match is True


@pytest.mark.integration
@pytest.mark.slow
async def test_audio_wid_survives_trailing_trim_25_percent(
    synthetic_audio_120s: Path, tmp_path: Path
) -> None:
    """
    [BLOCKING] sign 120s audio → trim trailing 30s (keep first 90s) → verify.

    After trim to 90s:
      Extractable symbols ≈ 45, erasures ≈ 15
      15 erasures ≤ 44 parity → comfortably recoverable → VERIFIED
    """
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

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
        signed_path = Path(tmp_f.name)
    trimmed_path = tmp_path / "trimmed_90s.wav"
    try:
        signed_bytes = await storage.get(sign_result.signed_media_key)
        signed_path.write_bytes(signed_bytes)

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(signed_path),
                "-t", "90",
                "-c", "copy",
                str(trimmed_path),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        signed_path.unlink(missing_ok=True)

    service = VerificationService()
    result: VerificationResult = await service.verify_audio(
        media_path=trimmed_path,
        media=media,
        storage=storage,
        registry=registry,
        pepper=PEPPER,
    )

    assert result.verdict == Verdict.VERIFIED, (
        f"25% trailing trim: expected VERIFIED but got {result.verdict} "
        f"(reason={result.red_reason})"
    )
    assert result.wid_match is True
