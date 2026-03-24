from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.domain.watermark import SegmentFingerprint, VideoEntry
from kernel_backend.core.ports.registry import RegistryPort
from kernel_backend.core.ports.storage import StoragePort
from kernel_backend.core.services.crypto_service import derive_wid, generate_keypair
from kernel_backend.core.services.signing_service import sign_audio
from kernel_backend.infrastructure.media.media_service import MediaService

PEPPER = b"test-pepper-bytes-padded-to-32b!"


@pytest.fixture
def synthetic_audio(tmp_path: Path) -> Path:
    """40-second audio file at 44100 Hz encoded as AAC (20 segments × 2 s ≥ 17 needed)."""
    out = tmp_path / "test.aac"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "sine=frequency=200:duration=40",
            "-c:a", "aac",
            "-ar", "44100",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


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
        self._segments[content_id] = list(segments)

    async def match_fingerprints(
        self, hashes: list[str], max_hamming: int = 10
    ) -> list[VideoEntry]:
        return []


def _make_cert(public_key_pem: str) -> Certificate:
    return Certificate(
        author_id="test-author-id",
        name="Test Author",
        institution="Test Org",
        public_key_pem=public_key_pem,
        created_at="2026-01-01T00:00:00+00:00",
    )


async def test_signing_returns_valid_result(
    synthetic_audio: Path, tmp_path: Path
) -> None:
    """sign_audio returns a SigningResult with valid WID, signature, and UUID content_id."""
    private_pem, public_pem = generate_keypair()
    cert = _make_cert(public_pem)

    result = await sign_audio(
        media_path=synthetic_audio,
        certificate=cert,
        private_key_pem=private_pem,
        storage=FakeStorage(),
        registry=FakeRegistry(),
        pepper=PEPPER,
        media=MediaService(),
    )

    # WID is 16 bytes
    assert len(result.wid.data) == 16
    # signature is 64 bytes (Ed25519)
    assert len(result.signature) == 64
    # content_id is a valid UUID string (8-4-4-4-12 format)
    parts = result.content_id.split("-")
    assert len(parts) == 5
    assert [len(p) for p in parts] == [8, 4, 4, 4, 12]
    # signed_media_key contains content_id
    assert result.content_id in result.signed_media_key
    # active_signals are all three audio signals
    assert set(result.active_signals) == {"pilot_audio", "wid_audio", "fingerprint_audio"}


async def test_wid_is_deterministic(synthetic_audio: Path, tmp_path: Path) -> None:
    """WID is deterministically derived from signature and content_id (HKDF is deterministic)."""
    private_pem, public_pem = generate_keypair()
    cert = _make_cert(public_pem)

    result = await sign_audio(
        media_path=synthetic_audio,
        certificate=cert,
        private_key_pem=private_pem,
        storage=FakeStorage(),
        registry=FakeRegistry(),
        pepper=PEPPER,
        media=MediaService(),
    )

    # Re-derive WID from result's own signature + content_id — must match
    re_derived = derive_wid(result.signature, result.content_id)
    assert result.wid == re_derived


async def test_short_audio_raises_value_error(tmp_path: Path) -> None:
    """
    sign_audio raises ValueError when audio yields fewer than 17 fingerprint segments.

    With extract_hashes default overlap=0.5 and segment_duration=2 s, a 10 s clip
    produces only ~9 segments — well below the RS K=16 minimum of 17.
    """
    short = tmp_path / "short.aac"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "sine=frequency=200:duration=10",
            "-c:a", "aac",
            "-ar", "44100",
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


async def test_pilot_hash_formula(synthetic_audio: Path, tmp_path: Path) -> None:
    """pilot_hash_48 == int.from_bytes(sha256(content_id)[:6], 'big')."""
    private_pem, public_pem = generate_keypair()
    cert = _make_cert(public_pem)

    result = await sign_audio(
        media_path=synthetic_audio,
        certificate=cert,
        private_key_pem=private_pem,
        storage=FakeStorage(),
        registry=FakeRegistry(),
        pepper=PEPPER,
        media=MediaService(),
    )

    expected = int.from_bytes(
        hashlib.sha256(result.content_id.encode()).digest()[:6], "big"
    )
    assert result.pilot_hash_48 == expected
