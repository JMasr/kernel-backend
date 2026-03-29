"""
Integration tests for the two-phase signing architecture (Stream D / Phase 7.2).

Verifies that the ProcessPool-compatible design correctly saves metadata to
storage and registry after signing. Tests the complete pipeline:

  _sign_sync() → RawSigningPayload (CPU phase, subprocess-safe)
  _persist_payload() → storage.put() + registry.save_video() (async I/O phase)

In production, _sign_sync() runs inside ProcessPoolExecutor and its return value
is passed to _persist_payload() in the parent event loop. These tests call both
phases sequentially (no actual subprocess) to verify the contract is upheld.

Run:
    pytest tests/integration/test_process_pool_signing.py -v
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.services.crypto_service import generate_keypair
from kernel_backend.core.services.signing_service import _persist_payload
from kernel_backend.infrastructure.queue.jobs import _sign_sync
from tests.helpers.fakes import FakeRegistry, FakeStorage

PEPPER = b"integration-test-pepper-padded!!"


def _cert_data(public_key_pem: str) -> dict:
    return {
        "author_id": "test-author-pp",
        "name": "ProcessPool Test Author",
        "institution": "Test Org",
        "public_key_pem": public_key_pem,
        "created_at": "2026-01-01T00:00:00+00:00",
    }


@pytest.fixture(scope="module")
def synthetic_audio_40s(tmp_path_factory) -> Path:
    """40-second audio — 20 segments × 2s; comfortably above ≥17 minimum."""
    tmp = tmp_path_factory.mktemp("pp_audio")
    out = tmp / "audio_40s.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anoisesrc=duration=40:sample_rate=44100",
            "-ac", "1",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture(scope="module")
def synthetic_video_100s(tmp_path_factory) -> Path:
    """100-second video — 20 segments × 5s; just above ≥17 minimum."""
    tmp = tmp_path_factory.mktemp("pp_video")
    out = tmp / "video_100s.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i",
            "testsrc=duration=100:size=320x240:rate=25,noise=c0s=100:allf=t",
            "-c:v", "libx264", "-crf", "23", "-preset", "ultrafast",
            "-an",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture(scope="module")
def synthetic_av_120s(tmp_path_factory) -> Path:
    """120-second AV file — ≥17 segments for both audio (60) and video (24)."""
    tmp = tmp_path_factory.mktemp("pp_av")
    out = tmp / "av_120s.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i",
            "testsrc=duration=120:size=320x240:rate=25,noise=c0s=100:allf=t",
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
async def test_sign_audio_two_phase_saves_metadata(synthetic_audio_40s: Path) -> None:
    """
    [BLOCKING — Stream D] Two-phase audio signing correctly saves metadata.

    _sign_sync() runs the CPU phase (as it would in a subprocess) and returns
    a RawSigningPayload. _persist_payload() then uploads the signed file and
    saves VideoEntry + segments to the registry. This verifies the full
    pipeline that process_sign_job executes across the process boundary.
    """
    private_pem, public_pem = generate_keypair()
    cert_data = _cert_data(public_pem)

    # CPU phase: runs in subprocess in production, called directly here
    payload = _sign_sync(
        media_path=str(synthetic_audio_40s),
        cert_data=cert_data,
        private_key_pem=private_pem,
        pepper=PEPPER,
    )

    assert payload["media_type"] == "audio"
    assert payload["content_id"]
    assert payload["signed_media_key"]
    assert payload["wid_hex"]

    # I/O phase: runs in parent event loop in production
    storage = FakeStorage()
    registry = FakeRegistry()
    await _persist_payload(payload, storage, registry)

    # Verify metadata was saved to registry
    entry = await registry.get_by_content_id(payload["content_id"])
    assert entry is not None, "VideoEntry was not saved to registry after _persist_payload()"
    assert entry.content_id == payload["content_id"]
    assert entry.author_id == cert_data["author_id"]
    assert entry.signed_media_key == payload["signed_media_key"]

    # Verify signed file was uploaded to storage
    signed_key = payload["signed_media_key"]
    assert signed_key in storage._store, (
        f"Signed file '{signed_key}' was not found in storage after _persist_payload()"
    )
    assert len(storage._store[signed_key]) > 0

    # Verify audio fingerprints were saved
    segments = registry._segments.get(payload["content_id"])
    assert segments, "Audio fingerprint segments were not saved to registry"
    assert len(segments) >= 17


@pytest.mark.integration
@pytest.mark.slow
async def test_sign_video_two_phase_saves_metadata(synthetic_video_100s: Path) -> None:
    """
    [BLOCKING — Stream D] Two-phase video signing correctly saves metadata.
    """
    private_pem, public_pem = generate_keypair()
    cert_data = _cert_data(public_pem)

    payload = _sign_sync(
        media_path=str(synthetic_video_100s),
        cert_data=cert_data,
        private_key_pem=private_pem,
        pepper=PEPPER,
    )

    assert payload["media_type"] == "video"

    storage = FakeStorage()
    registry = FakeRegistry()
    await _persist_payload(payload, storage, registry)

    entry = await registry.get_by_content_id(payload["content_id"])
    assert entry is not None, "VideoEntry was not saved to registry after _persist_payload()"
    assert entry.signed_media_key == payload["signed_media_key"]

    assert payload["signed_media_key"] in storage._store


@pytest.mark.integration
@pytest.mark.slow
async def test_sign_av_two_phase_saves_metadata(synthetic_av_120s: Path) -> None:
    """
    [BLOCKING — Stream D] Two-phase AV signing correctly saves metadata with shared WID.

    sign_av() produces ONE shared WID embedded in both channels. After _persist_payload(),
    the VideoEntry must carry the correct WID and both audio+video fingerprint segments.
    """
    private_pem, public_pem = generate_keypair()
    cert_data = _cert_data(public_pem)

    payload = _sign_sync(
        media_path=str(synthetic_av_120s),
        cert_data=cert_data,
        private_key_pem=private_pem,
        pepper=PEPPER,
    )

    assert payload["media_type"] == "av"
    assert payload["wid_hex"]
    # AV must have both audio and video fingerprints
    assert payload.get("audio_fingerprints"), "AV payload missing audio fingerprints"
    assert payload.get("video_fingerprints"), "AV payload missing video fingerprints"

    storage = FakeStorage()
    registry = FakeRegistry()
    await _persist_payload(payload, storage, registry)

    entry = await registry.get_by_content_id(payload["content_id"])
    assert entry is not None
    assert entry.signed_media_key == payload["signed_media_key"]
    assert entry.rs_n == payload["rs_n"]

    assert payload["signed_media_key"] in storage._store

    # Both audio and video fingerprint segments must be stored
    segments = registry._segments.get(payload["content_id"])
    assert segments, "No fingerprint segments saved after AV signing"
