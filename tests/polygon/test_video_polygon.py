"""
Polygon video tests — Phase 2b + Phase 3.

All tests are marked @pytest.mark.polygon and are excluded from CI by default.
Video clips must be placed manually under data/video/ — see data/README.md.

Run:
    pytest tests/ -m "polygon" -v
    pytest tests/ -m "polygon" -k "video" -v

Sections:
  1. Metadata validation   — clip accessible on disk, manifest fields are non-zero
  2. Probe consistency     — MediaService.probe() agrees with manifest metadata
  3. Audio fingerprint     — extract_hashes() on the video's audio track
  4. Video fingerprint     — discriminates speech vs outside clips (Phase 3)
  5. Pilot tone            — embeds/detects on real video frames (Phase 3)
  6. WID agreement         — WID agreement under H.264 degradation (Phase 3)
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from kernel_backend.engine.audio.fingerprint import extract_hashes, hamming_distance
from kernel_backend.engine.video.fingerprint import (
    extract_hashes as extract_video_hashes,
    hamming_distance as video_hamming_distance,
)
from kernel_backend.engine.video.pilot_tone import embed_pilot, detect_pilot
from kernel_backend.engine.video.wid_watermark import embed_segment, extract_segment
from kernel_backend.infrastructure.media.media_service import MediaService
from tests.fixtures.polygon.registry import DatasetRegistry, VideoClip

KEY_A = b"author-public-key-material-AAA"
KEY_B = b"author-public-key-material-BBB"
PEPPER = b"system-pepper-bytes-padded-32b!"

# Seconds of audio decoded per clip for fingerprint tests.
# Keeps tests fast on very long clips (camping_01 is ~18 min).
_FINGERPRINT_WINDOW_S = 30


def _load_audio_window(clip: VideoClip, duration_s: float = _FINGERPRINT_WINDOW_S):
    """Decode the first `duration_s` seconds of audio from a video clip."""
    media = MediaService()
    target_sr = 44100
    samples_list = [chunk for _, chunk, _ in media.iter_audio_segments(clip.path, target_sample_rate=target_sr)]
    samples = np.concatenate(samples_list) if samples_list else np.array([], dtype=np.float32)
    return samples[: int(duration_s * target_sr)], target_sr


# ════════════════════════════════════════════════════════════════════════════════
# 1. METADATA VALIDATION
# One test per category — verifies the manifest fields and file existence.
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.polygon
def test_polygon_video_speech_metadata(video_speech_clip: VideoClip):
    """[POLYGON/BLOCKING] Speech clip: exists on disk, manifest fields non-zero."""
    assert video_speech_clip.exists(), f"{video_speech_clip.id}: file not found"
    assert video_speech_clip.duration_s > 0
    assert video_speech_clip.fps > 0
    w, h = video_speech_clip.resolution
    assert w > 0 and h > 0


@pytest.mark.polygon
def test_polygon_video_outside_metadata(video_outside_clip: VideoClip):
    """[POLYGON/BLOCKING] Outside clip: exists on disk, manifest fields non-zero."""
    assert video_outside_clip.exists(), f"{video_outside_clip.id}: file not found"
    assert video_outside_clip.duration_s > 0
    assert video_outside_clip.fps > 0
    w, h = video_outside_clip.resolution
    assert w > 0 and h > 0


@pytest.mark.polygon
def test_polygon_video_without_audio_metadata(video_without_audio_clip: VideoClip):
    """[POLYGON/BLOCKING] Without-audio clip: exists on disk, manifest fields non-zero."""
    assert video_without_audio_clip.exists(), f"{video_without_audio_clip.id}: file not found"
    assert video_without_audio_clip.duration_s > 0
    assert video_without_audio_clip.fps > 0
    w, h = video_without_audio_clip.resolution
    assert w > 0 and h > 0


@pytest.mark.polygon
def test_polygon_video_others_metadata(video_others_clip: VideoClip):
    """[POLYGON/INFO] Others clip: exists on disk, manifest fields non-zero."""
    assert video_others_clip.exists(), f"{video_others_clip.id}: file not found"
    assert video_others_clip.duration_s > 0
    assert video_others_clip.fps > 0
    w, h = video_others_clip.resolution
    assert w > 0 and h > 0


# ════════════════════════════════════════════════════════════════════════════════
# 2. PROBE CONSISTENCY
# MediaService.probe() must agree with what the manifest records.
# Tolerance: ±5% on duration, ±0.5 on fps, exact on resolution.
# ════════════════════════════════════════════════════════════════════════════════

def _assert_probe_matches(clip: VideoClip) -> None:
    """Probe the file and assert it matches the manifest within tolerances."""
    media = MediaService()
    profile = media.probe(clip.path)

    # Duration: within ±5 %
    tol = max(1.0, clip.duration_s * 0.05)
    assert abs(profile.duration_s - clip.duration_s) <= tol, (
        f"[{clip.id}] duration mismatch: manifest={clip.duration_s:.2f}s "
        f"probe={profile.duration_s:.2f}s"
    )

    # Resolution: exact
    assert (profile.width, profile.height) == clip.resolution, (
        f"[{clip.id}] resolution mismatch: manifest={clip.resolution} "
        f"probe=({profile.width},{profile.height})"
    )

    # FPS: within ±0.5
    assert abs(profile.fps - clip.fps) <= 0.5, (
        f"[{clip.id}] fps mismatch: manifest={clip.fps} probe={profile.fps:.2f}"
    )

    # has_audio consistency
    assert profile.has_audio == clip.has_audio, (
        f"[{clip.id}] has_audio mismatch: manifest={clip.has_audio} "
        f"probe={profile.has_audio}"
    )


@pytest.mark.polygon
def test_polygon_video_speech_probe(video_speech_clip: VideoClip):
    """[POLYGON/BLOCKING] Speech clip: probe output matches manifest."""
    _assert_probe_matches(video_speech_clip)


@pytest.mark.polygon
def test_polygon_video_outside_probe(video_outside_clip: VideoClip):
    """[POLYGON/BLOCKING] Outside clip: probe output matches manifest."""
    _assert_probe_matches(video_outside_clip)


@pytest.mark.polygon
def test_polygon_video_without_audio_probe(video_without_audio_clip: VideoClip):
    """[POLYGON/BLOCKING] Without-audio clip: probe output matches manifest."""
    _assert_probe_matches(video_without_audio_clip)


@pytest.mark.polygon
def test_polygon_video_others_probe(video_others_clip: VideoClip):
    """[POLYGON/INFO] Others clip: probe output matches manifest."""
    _assert_probe_matches(video_others_clip)


# ════════════════════════════════════════════════════════════════════════════════
# 3. AUDIO FINGERPRINT ON VIDEO AUDIO TRACKS
# Validates the audio pipeline on real video containers (not WAV files).
# Uses only the first _FINGERPRINT_WINDOW_S seconds per clip for speed.
# Skipped if clip has no audio track.
# ════════════════════════════════════════════════════════════════════════════════

def _skip_if_no_audio(clip: VideoClip) -> None:
    if not clip.has_audio:
        pytest.skip(f"{clip.id} has no audio track")


@pytest.mark.polygon
def test_polygon_video_speech_audio_fingerprint_deterministic(video_speech_clip: VideoClip):
    """[POLYGON/BLOCKING] Speech video audio: same window + same key → identical hashes."""
    _skip_if_no_audio(video_speech_clip)
    samples, sr = _load_audio_window(video_speech_clip)
    h1 = extract_hashes(samples, sr, KEY_A, PEPPER)
    h2 = extract_hashes(samples, sr, KEY_A, PEPPER)
    assert len(h1) > 0, f"[{video_speech_clip.id}] no fingerprint segments extracted"
    for a, b in zip(h1, h2):
        assert a.hash_hex == b.hash_hex, (
            f"[{video_speech_clip.id}] non-deterministic at {a.time_offset_ms} ms"
        )


@pytest.mark.polygon
def test_polygon_video_speech_audio_fingerprint_keyed(video_speech_clip: VideoClip):
    """[POLYGON/BLOCKING] Speech video audio: different key → different hashes."""
    _skip_if_no_audio(video_speech_clip)
    samples, sr = _load_audio_window(video_speech_clip)
    h_a = extract_hashes(samples, sr, KEY_A, PEPPER)
    h_b = extract_hashes(samples, sr, KEY_B, PEPPER)
    assert any(a.hash_hex != b.hash_hex for a, b in zip(h_a, h_b)), (
        f"[{video_speech_clip.id}] key material has no effect on video audio"
    )


@pytest.mark.polygon
def test_polygon_video_outside_audio_fingerprint_deterministic(video_outside_clip: VideoClip):
    """[POLYGON/BLOCKING] Outdoor video audio: same window → identical hashes."""
    _skip_if_no_audio(video_outside_clip)
    samples, sr = _load_audio_window(video_outside_clip)
    h1 = extract_hashes(samples, sr, KEY_A, PEPPER)
    h2 = extract_hashes(samples, sr, KEY_A, PEPPER)
    assert len(h1) > 0, f"[{video_outside_clip.id}] no fingerprint segments extracted"
    for a, b in zip(h1, h2):
        assert a.hash_hex == b.hash_hex, (
            f"[{video_outside_clip.id}] non-deterministic at {a.time_offset_ms} ms"
        )


@pytest.mark.polygon
def test_polygon_video_speech_vs_outside_discriminated(
    video_speech_clip: VideoClip,
    video_outside_clip: VideoClip,
):
    """
    [POLYGON/BLOCKING] Speech video vs outdoor video: audio fingerprints must be
    discriminated — ≥ 80% of aligned segment pairs at Hamming ≥ 20.
    """
    _skip_if_no_audio(video_speech_clip)
    _skip_if_no_audio(video_outside_clip)

    sa, sr_a = _load_audio_window(video_speech_clip)
    sb, sr_b = _load_audio_window(video_outside_clip)
    assert sr_a == sr_b, "sample rate mismatch after iter_audio_segments"

    h_a = extract_hashes(sa, sr_a, KEY_A, PEPPER)
    h_b = extract_hashes(sb, sr_a, KEY_A, PEPPER)
    n = min(len(h_a), len(h_b))
    if n == 0:
        pytest.skip("Not enough segments to discriminate")

    high_dist_rate = sum(
        1 for k in range(n)
        if hamming_distance(h_a[k].hash_hex, h_b[k].hash_hex) >= 20
    ) / n
    assert high_dist_rate >= 0.80, (
        f"[{video_speech_clip.id} vs {video_outside_clip.id}] "
        f"only {high_dist_rate:.0%} discriminated"
    )


# ════════════════════════════════════════════════════════════════════════════════
# 4. VIDEO FINGERPRINT DISCRIMINATION (Phase 3)
# Speech and outside clips must produce distinct fingerprints.
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.polygon
def test_polygon_video_fingerprint_speech_deterministic(video_speech_clip: VideoClip):
    """[POLYGON/BLOCKING] Same video file + same key → identical video fingerprint hashes."""
    h1 = extract_video_hashes(str(video_speech_clip.path), KEY_A, PEPPER)
    h2 = extract_video_hashes(str(video_speech_clip.path), KEY_A, PEPPER)
    assert len(h1) > 0, f"[{video_speech_clip.id}] no video fingerprint segments"
    for a, b in zip(h1, h2):
        assert a.hash_hex == b.hash_hex, (
            f"[{video_speech_clip.id}] non-deterministic at {a.time_offset_ms} ms"
        )


@pytest.mark.polygon
def test_polygon_video_fingerprint_speech_keyed(video_speech_clip: VideoClip):
    """[POLYGON/BLOCKING] Different key → different video hashes."""
    h_a = extract_video_hashes(str(video_speech_clip.path), KEY_A, PEPPER)
    h_b = extract_video_hashes(str(video_speech_clip.path), KEY_B, PEPPER)
    assert any(a.hash_hex != b.hash_hex for a, b in zip(h_a, h_b)), (
        f"[{video_speech_clip.id}] key has no effect on video fingerprint"
    )


@pytest.mark.polygon
def test_polygon_video_fingerprint_speech_vs_outside(
    video_speech_clip: VideoClip,
    video_outside_clip: VideoClip,
    polygon: DatasetRegistry,
):
    """
    [POLYGON/BLOCKING] Speech and outside clips must have Hamming > 20
    for >= 80% of aligned segment pairs.
    """
    h_s = extract_video_hashes(str(video_speech_clip.path), KEY_A, PEPPER)
    h_o = extract_video_hashes(str(video_outside_clip.path), KEY_A, PEPPER)
    n = min(len(h_s), len(h_o))
    if n == 0:
        pytest.skip("Not enough segments to discriminate")

    high_dist_rate = sum(
        1 for k in range(n)
        if video_hamming_distance(h_s[k].hash_hex, h_o[k].hash_hex) >= 20
    ) / n
    assert high_dist_rate >= 0.80, (
        f"[{video_speech_clip.id} vs {video_outside_clip.id}] "
        f"only {high_dist_rate:.0%} discriminated"
    )


# ════════════════════════════════════════════════════════════════════════════════
# 5. PILOT TONE ON REAL VIDEO FRAMES (Phase 3)
# ════════════════════════════════════════════════════════════════════════════════

_N_PILOT_FRAMES = 5  # number of frames to test pilot on


def _read_frames(clip: VideoClip, n: int = _N_PILOT_FRAMES) -> list[np.ndarray]:
    media = MediaService()
    frames, _ = media.read_video_frames(clip.path, start_frame=0, n_frames=n)
    return frames


@pytest.mark.polygon
def test_polygon_video_pilot_roundtrip(video_speech_clip: VideoClip):
    """
    [POLYGON/BLOCKING] Embed pilot on real video frame, detect on same frame.
    Agreement must be >= 0.75 (PILOT_AGREEMENT_THRESHOLD).
    """
    frames = _read_frames(video_speech_clip)
    assert len(frames) > 0, f"[{video_speech_clip.id}] could not read frames"

    content_id = "polygon-pilot-test-001"
    frame = frames[0]
    watermarked = embed_pilot(frame, content_id, PEPPER)
    det = detect_pilot(watermarked, content_id, PEPPER)
    assert det.detected, (
        f"[{video_speech_clip.id}] pilot not detected, agreement={det.agreement:.3f}"
    )


def _recompress_frames(frames, fps, crf, tmp_path):
    """Write frames to H.264 at given CRF and read back. Uses raw pipe to avoid double encoding."""
    import cv2

    h, w = frames[0].shape[:2]
    dst = tmp_path / f"recomp_crf{crf}.mp4"

    # Write raw frames as yuv4mpegpipe → H.264 in one pass via ffmpeg
    raw_data = b""
    for f in frames:
        yuv = cv2.cvtColor(f, cv2.COLOR_BGR2YUV_I420)
        raw_data += yuv.tobytes()

    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "rawvideo", "-pix_fmt", "yuv420p",
         "-s", f"{w}x{h}", "-r", str(fps),
         "-i", "pipe:0",
         "-vcodec", "libx264", "-crf", str(crf),
         "-preset", "fast", "-loglevel", "quiet",
         str(dst)],
        input=raw_data,
        check=True,
    )

    cap = cv2.VideoCapture(str(dst))
    result = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result.append(frame)
    cap.release()
    return result


@pytest.mark.polygon
@pytest.mark.parametrize("crf", [23, 28])
def test_polygon_video_pilot_h264(video_speech_clip: VideoClip, crf: int, tmp_path):
    """
    [POLYGON/BLOCKING] Pilot detected after H.264 recompression at CRF 23 and 28.
    Uses 30 frames with pilot embedded, recompresses, checks middle frame.
    """
    media = MediaService()
    frames, fps = media.read_video_frames(video_speech_clip.path, n_frames=30)
    assert len(frames) >= 10

    content_id = "polygon-pilot-h264-test"
    watermarked = [embed_pilot(f, content_id, PEPPER) for f in frames]

    recomp = _recompress_frames(watermarked, fps, crf, tmp_path)
    assert len(recomp) > 0, "no frames after recompression"

    # Check middle frame for best H.264 prediction
    mid = len(recomp) // 2
    det = detect_pilot(recomp[mid], content_id, PEPPER)
    assert det.detected, (
        f"[{video_speech_clip.id}] pilot not detected after CRF {crf}, "
        f"agreement={det.agreement:.3f}"
    )


@pytest.mark.polygon
@pytest.mark.xfail(strict=False, reason="CRF 35 pilot detection is informational")
def test_polygon_video_pilot_h264_crf35(video_speech_clip: VideoClip, tmp_path):
    """[POLYGON/INFO] Pilot detection after H.264 CRF 35 — may fail."""
    media = MediaService()
    frames, fps = media.read_video_frames(video_speech_clip.path, n_frames=30)
    assert len(frames) >= 10

    content_id = "polygon-pilot-h264-crf35"
    watermarked = [embed_pilot(f, content_id, PEPPER) for f in frames]

    recomp = _recompress_frames(watermarked, fps, crf=35, tmp_path=tmp_path)
    assert len(recomp) > 0

    mid = len(recomp) // 2
    det = detect_pilot(recomp[mid], content_id, PEPPER)
    assert det.detected, f"agreement={det.agreement:.3f}"


# ════════════════════════════════════════════════════════════════════════════════
# 6. WID AGREEMENT UNDER H.264 DEGRADATION (Phase 3)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.polygon
@pytest.mark.parametrize("crf", [23, 28])
def test_polygon_video_wid_h264(video_speech_clip: VideoClip, polygon: DatasetRegistry, crf: int, tmp_path):
    """
    [POLYGON/BLOCKING] WID agreement > threshold after H.264 recompression.
    Uses first 30 frames of speech_01.
    """
    import cv2

    media = MediaService()
    frames, fps = media.read_video_frames(video_speech_clip.path, n_frames=30)
    assert len(frames) >= 5, f"[{video_speech_clip.id}] not enough frames"

    content_id = "polygon-wid-h264-test"
    pubkey = "polygon-test-pubkey"
    symbol_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)

    embedded = embed_segment(frames, symbol_bits, content_id, pubkey, 0, PEPPER)

    # Clean extraction
    clean_result = extract_segment(embedded, content_id, pubkey, 0, PEPPER)

    # H.264 recompress
    h, w = embedded[0].shape[:2]
    src = tmp_path / "wid_src.mp4"
    dst = tmp_path / f"wid_crf{crf}.mp4"

    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in embedded:
        writer.write(f)
    writer.release()

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vcodec", "libx264",
         "-crf", str(crf), "-preset", "fast", "-loglevel", "quiet", str(dst)],
        check=True,
    )

    recomp_frames, _ = media.read_video_frames(Path(dst))
    recomp_result = extract_segment(recomp_frames, content_id, pubkey, 0, PEPPER)

    threshold = polygon.get_threshold("video.speech", f"h264_crf{crf}")
    min_agreement = threshold.get("min_agreement", 0.52)

    assert recomp_result.agreement > min_agreement, (
        f"[{video_speech_clip.id}] WID agreement after CRF {crf}: "
        f"{recomp_result.agreement:.3f} <= {min_agreement}"
    )


# ════════════════════════════════════════════════════════════════════════════════
# 7. FINGERPRINT STABILITY UNDER RESIZE (Phase 3, informational)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.polygon
@pytest.mark.xfail(strict=False, reason="resize stability informational in MVP")
def test_polygon_video_fingerprint_stability_resize(video_speech_clip: VideoClip, tmp_path):
    """
    [POLYGON/INFO] Same clip at original resolution vs 720p resize.
    Hamming <= 10 for >= 75% of aligned segments.
    """
    # Original fingerprint
    h_orig = extract_video_hashes(str(video_speech_clip.path), KEY_A, PEPPER)

    # Resize to 720p
    dst = tmp_path / "resized_720p.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_speech_clip.path),
         "-vf", "scale=1280:720", "-vcodec", "libx264", "-crf", "18",
         "-preset", "fast", "-loglevel", "quiet", str(dst)],
        check=True,
    )

    h_resized = extract_video_hashes(str(dst), KEY_A, PEPPER)
    n = min(len(h_orig), len(h_resized))
    if n == 0:
        pytest.skip("Not enough segments")

    stable_rate = sum(
        1 for k in range(n)
        if video_hamming_distance(h_orig[k].hash_hex, h_resized[k].hash_hex) <= 10
    ) / n
    assert stable_rate >= 0.75, f"only {stable_rate:.0%} stable under resize"


# ════════════════════════════════════════════════════════════════════════════════
# 8. VERIFICATION ROUNDTRIP (Phase 4)
# Full sign → verify on real clips.
# ════════════════════════════════════════════════════════════════════════════════

async def _sign_speech(video_speech_clip: VideoClip, pepper: bytes, tmp_path):
    """Helper: sign speech_01 and return (signed_path, content_id, registry, storage, media)."""
    import tempfile
    from kernel_backend.core.services.crypto_service import generate_keypair
    from kernel_backend.core.services.signing_service import sign_video
    from kernel_backend.core.domain.identity import Certificate
    from kernel_backend.infrastructure.media.media_service import MediaService
    from kernel_backend.infrastructure.storage.local_storage import LocalStorageAdapter
    from kernel_backend.infrastructure.database.repositories import VideoRepository
    from kernel_backend.infrastructure.database.models import Base
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

    priv_pem, pub_pem = generate_keypair()
    cert = Certificate(
        author_id="polygon-verify-author",
        name="Polygon Verify Test",
        institution="Test Suite",
        public_key_pem=pub_pem,
        created_at="2026-01-01T00:00:00+00:00",
    )
    media = MediaService()
    storage = LocalStorageAdapter(base_path=tmp_path / "storage")
    (tmp_path / "storage").mkdir(exist_ok=True)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        registry = VideoRepository(session=session)
        sign_result = await sign_video(
            media_path=video_speech_clip.path,
            certificate=cert,
            private_key_pem=priv_pem,
            storage=storage,
            registry=registry,
            pepper=pepper,
            media=media,
        )

    await engine.dispose()
    return sign_result, storage, priv_pem, pub_pem


@pytest.mark.polygon
@pytest.mark.slow
@pytest.mark.asyncio
async def test_polygon_verify_speech_roundtrip(video_speech_clip: VideoClip, polygon, tmp_path):
    """
    [BLOCKING] Full verification roundtrip on real speech_01.
    sign_video() → store → verify() → assert VERIFIED.
    """
    from kernel_backend.core.services.verification_service import VerificationService
    from kernel_backend.core.domain.verification import Verdict
    from kernel_backend.infrastructure.media.media_service import MediaService
    from kernel_backend.infrastructure.storage.local_storage import LocalStorageAdapter
    from kernel_backend.infrastructure.database.repositories import VideoRepository
    from kernel_backend.infrastructure.database.models import Base
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

    PEPPER = b"polygon-verify-pepper-padded32!"
    media = MediaService()

    from kernel_backend.core.services.crypto_service import generate_keypair
    from kernel_backend.core.services.signing_service import sign_video
    from kernel_backend.core.domain.identity import Certificate

    priv_pem, pub_pem = generate_keypair()
    cert = Certificate(
        author_id="polygon-roundtrip-author",
        name="Polygon Test",
        institution="Test Suite",
        public_key_pem=pub_pem,
        created_at="2026-01-01T00:00:00+00:00",
    )
    storage = LocalStorageAdapter(base_path=tmp_path / "storage")
    (tmp_path / "storage").mkdir(exist_ok=True)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        registry = VideoRepository(session=session)
        sign_result = await sign_video(
            media_path=video_speech_clip.path,
            certificate=cert,
            private_key_pem=priv_pem,
            storage=storage,
            registry=registry,
            pepper=PEPPER,
            media=media,
        )

        signed_bytes = await storage.get(sign_result.signed_media_key)
        signed_path = tmp_path / "signed.mp4"
        signed_path.write_bytes(signed_bytes)

        service = VerificationService()
        result = await service.verify(
            media_path=signed_path,
            media=media,
            storage=storage,
            registry=registry,
            pepper=PEPPER,
        )

    assert result.verdict == Verdict.VERIFIED, (
        f"[{video_speech_clip.id}] Expected VERIFIED, got {result.verdict} / {result.red_reason}"
    )
    assert result.n_segments_decoded > 0


@pytest.mark.polygon
@pytest.mark.slow
@pytest.mark.parametrize("crf", [23, 28])
@pytest.mark.asyncio
async def test_polygon_verify_h264_degradation(
    video_speech_clip: VideoClip, polygon, crf: int, tmp_path
):
    """
    [BLOCKING] Sign → H.264 recompress at CRF → verify → VERIFIED.
    Tests the real-world distribution scenario on actual video content.
    """
    from kernel_backend.core.services.verification_service import VerificationService
    from kernel_backend.core.domain.verification import Verdict
    from kernel_backend.infrastructure.media.media_service import MediaService
    from kernel_backend.infrastructure.storage.local_storage import LocalStorageAdapter
    from kernel_backend.infrastructure.database.repositories import VideoRepository
    from kernel_backend.infrastructure.database.models import Base
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from kernel_backend.core.services.crypto_service import generate_keypair
    from kernel_backend.core.services.signing_service import sign_video
    from kernel_backend.core.domain.identity import Certificate

    PEPPER = b"polygon-verify-pepper-padded32!"
    media = MediaService()

    priv_pem, pub_pem = generate_keypair()
    cert = Certificate(
        author_id=f"polygon-crf{crf}-author",
        name="CRF Test",
        institution="Test Suite",
        public_key_pem=pub_pem,
        created_at="2026-01-01T00:00:00+00:00",
    )
    storage = LocalStorageAdapter(base_path=tmp_path / "storage")
    (tmp_path / "storage").mkdir(exist_ok=True)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        registry = VideoRepository(session=session)
        sign_result = await sign_video(
            media_path=video_speech_clip.path,
            certificate=cert,
            private_key_pem=priv_pem,
            storage=storage,
            registry=registry,
            pepper=PEPPER,
            media=media,
        )

    signed_bytes = await storage.get(sign_result.signed_media_key)
    signed_path = tmp_path / "signed.mp4"
    signed_path.write_bytes(signed_bytes)

    recomp_path = tmp_path / f"recomp_crf{crf}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(signed_path),
         "-vcodec", "libx264", "-crf", str(crf), "-preset", "fast",
         "-loglevel", "quiet", str(recomp_path)],
        check=True,
    )

    async with factory() as session:
        registry = VideoRepository(session=session)
        service = VerificationService()
        result = await service.verify(
            media_path=recomp_path,
            media=media,
            storage=storage,
            registry=registry,
            pepper=PEPPER,
        )
    await engine.dispose()

    assert result.verdict == Verdict.VERIFIED, (
        f"[{video_speech_clip.id}] CRF {crf}: "
        f"Expected VERIFIED, got {result.verdict} / {result.red_reason}"
    )


@pytest.mark.polygon
@pytest.mark.slow
def test_polygon_verify_long_video_memory(video_outside_clip: VideoClip, polygon):
    """
    [BLOCKING] camping_01 (1058s) must be verified without OOM.
    Peak memory must stay < 500 MB — validates iter_video_segments lazy loading.
    camping_01 is unsigned, so verdict == RED(CANDIDATE_NOT_FOUND) is expected.
    Memory budget is the point of this test, not the verdict.
    """
    import asyncio
    import tracemalloc

    from kernel_backend.core.services.verification_service import VerificationService
    from kernel_backend.core.domain.verification import Verdict
    from kernel_backend.infrastructure.media.media_service import MediaService
    from kernel_backend.infrastructure.database.repositories import VideoRepository
    from kernel_backend.infrastructure.database.models import Base
    from kernel_backend.infrastructure.storage.local_storage import LocalStorageAdapter
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    import tempfile

    PEPPER = b"polygon-memory-pepper-padded32!!"
    media = MediaService()

    async def _verify():
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        storage = LocalStorageAdapter(base_path=Path(tempfile.mkdtemp()))
        async with factory() as session:
            registry = VideoRepository(session=session)
            service = VerificationService()
            result = await service.verify(
                media_path=video_outside_clip.path,
                media=media,
                storage=storage,
                registry=registry,
                pepper=PEPPER,
            )
        await engine.dispose()
        return result

    tracemalloc.start()
    result = asyncio.get_event_loop().run_until_complete(_verify())
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_bytes / 1024 / 1024
    assert peak_mb < 500, (
        f"Peak memory {peak_mb:.1f} MB exceeds 500 MB limit. "
        "iter_video_segments may be buffering the entire video."
    )
    # camping_01 is unsigned — expect RED, but memory must be bounded regardless
    assert result.verdict in (Verdict.VERIFIED, Verdict.RED), (
        f"Unexpected verdict: {result.verdict}"
    )
