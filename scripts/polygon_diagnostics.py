"""
scripts/polygon_diagnostics.py

Standalone diagnostic script for the watermark engine.
Measures actual metric values (not just pass/fail) on real polygon clips.

Sections:
  A. Audio fingerprint robustness (per-degradation pass rates)
  B. Video WID agreement vs H.264 compression (fixed vs JND-adaptive)
  C. Performance — sign_av timing on speech_01
  D. Memory — sign_av peak on dark_no_audio_01 (1080p)

Run:
    uv run python scripts/polygon_diagnostics.py
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = REPO_ROOT / "data"
MANIFEST_PATH = DATA_ROOT / "manifest.yaml"

PEPPER = b"system-pepper-bytes-padded-32b!"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")


def _psnr(orig: np.ndarray, modified: np.ndarray) -> float:
    mse = float(np.mean((orig.astype(np.float64) - modified.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def _recompress_frames(frames: list[np.ndarray], fps: float, crf: int, tmp_dir: Path) -> list[np.ndarray]:
    """Pipe-based H.264 recompression — no double lossy encoding."""
    import cv2
    h, w = frames[0].shape[:2]
    dst = tmp_dir / f"recomp_crf{crf}.mp4"
    raw = b"".join(cv2.cvtColor(f, cv2.COLOR_BGR2YUV_I420).tobytes() for f in frames)
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "rawvideo", "-pix_fmt", "yuv420p",
         "-s", f"{w}x{h}", "-r", str(int(fps)),
         "-i", "pipe:0",
         "-vcodec", "libx264", "-crf", str(crf),
         "-preset", "fast", "-loglevel", "quiet",
         str(dst)],
        input=raw, check=True,
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


# ── Section A: Audio Fingerprint Robustness ──────────────────────────────────

def section_a_audio_robustness() -> None:
    _print_section("A. Audio Fingerprint Robustness (speech clips)")

    from tests.fixtures.polygon.registry import DatasetRegistry
    from tests.fixtures.polygon.conftest import load_polygon_audio
    from tests.fixtures.audio_signals import (
        add_babble_noise, add_pink_noise,
        simulate_mp3_compression, simulate_voip_codec,
    )
    from kernel_backend.engine.audio.fingerprint import extract_hashes, hamming_distance

    if not MANIFEST_PATH.exists():
        print("  SKIP: manifest.yaml not found")
        return

    registry = DatasetRegistry()
    speech_clips = registry.get("audio.speech").available_audio()
    if not speech_clips:
        print("  SKIP: no speech clips available")
        return

    degradations = [
        ("clean",       lambda s, sr: s),
        ("babble@10dB", lambda s, sr: add_babble_noise(s, snr_db=10.0)),
        ("pink@20dB",   lambda s, sr: add_pink_noise(s, snr_db=20.0)),
        ("MP3@32kbps",  lambda s, sr: simulate_mp3_compression(s, sr, bitrate_kbps=32)),
        ("VoIP",        lambda s, sr: simulate_voip_codec(s, sr)),
    ]
    thresholds = {
        "clean":       1.00,
        "babble@10dB": 0.80,
        "pink@20dB":   0.85,
        "MP3@32kbps":  0.80,
        "VoIP":        0.70,
    }

    # Header
    col_w = 14
    header = f"{'Clip':<22}" + "".join(f"{d[0]:>{col_w}}" for d in degradations)
    print(header)
    print("-" * len(header))

    for clip in speech_clips:
        samples, sr = load_polygon_audio(clip)
        baseline = extract_hashes(samples, sr, key_material=PEPPER, pepper=PEPPER)
        row = f"{clip.id:<22}"
        for deg_name, deg_fn in degradations:
            degraded = deg_fn(samples, sr)
            dg_hashes = extract_hashes(degraded, sr, key_material=PEPPER, pepper=PEPPER)
            n = min(len(baseline), len(dg_hashes))
            if n == 0:
                row += f"{'N/A':>{col_w}}"
                continue
            pass_count = sum(
                1 for k in range(n)
                if hamming_distance(baseline[k].hash_hex, dg_hashes[k].hash_hex) <= 10
            )
            rate = pass_count / n
            thr = thresholds[deg_name]
            status = "✓" if rate >= thr else "✗"
            row += f"{rate:.0%} {status}".rjust(col_w)
        print(row)

    print()
    print("Threshold row:")
    thr_row = f"{'(threshold)':<22}" + "".join(
        f"{thresholds[d[0]]:.0%}".rjust(col_w) for d in degradations
    )
    print(thr_row)


# ── Section B: Video WID Agreement vs H.264 ──────────────────────────────────

def section_b_video_wid() -> None:
    _print_section("B. Video WID Agreement vs H.264 Compression")

    from tests.fixtures.polygon.registry import DatasetRegistry
    from kernel_backend.core.domain.watermark import VideoEmbeddingParams
    from kernel_backend.engine.video.wid_watermark import (
        WID_AGREEMENT_THRESHOLD,
        embed_video_frame,
        extract_segment,
    )
    from kernel_backend.infrastructure.media.media_service import MediaService

    if not MANIFEST_PATH.exists():
        print("  SKIP: manifest.yaml not found")
        return

    registry = DatasetRegistry()
    speech_clips = registry.get("video.speech").available_video()
    if not speech_clips:
        print("  SKIP: no video speech clips available")
        return

    clip = speech_clips[0]
    media = MediaService()
    frames, fps = media.read_video_frames(clip.path, n_frames=30)
    if not frames:
        print(f"  SKIP: could not read frames from {clip.id}")
        return

    content_id = "diag-wid-test"
    pubkey = "diag-pubkey"
    symbol_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)

    jnd_params = VideoEmbeddingParams(
        jnd_adaptive=True,
        qim_step_base=64.0,
        qim_step_min=44.0,
        qim_step_max=128.0,
        qim_quantize_to=4.0,
    )

    crfs = [0, 18, 23, 28, 35]

    print(f"  Clip: {clip.id}  ({clip.resolution[0]}×{clip.resolution[1]}, {fps:.0f} fps, 30 frames)")
    print(f"  Threshold: {WID_AGREEMENT_THRESHOLD}\n")

    modes = [
        ("Fixed (64.0)", False, None),
        ("JND Adaptive", True, jnd_params),
    ]

    col_w = 12
    header = f"{'Mode':<18}" + "".join(f"CRF {c:>2}".rjust(col_w) for c in crfs)
    print(header)
    print("-" * len(header))

    psnr_rows = []

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for mode_name, use_jnd, jp in modes:
            embedded = [
                embed_video_frame(
                    f, symbol_bits, content_id, pubkey, 0, PEPPER,
                    use_jnd_adaptive=use_jnd, jnd_params=jp,
                )
                for f in frames
            ]
            psnr_val = _psnr(frames[0], embedded[0])
            psnr_rows.append((mode_name, psnr_val))

            row = f"{mode_name:<18}"
            for crf in crfs:
                if crf == 0:
                    result = extract_segment(
                        embedded, content_id, pubkey, 0, PEPPER,
                        use_jnd_adaptive=use_jnd, jnd_params=jp,
                    )
                    agreement = result.agreement
                else:
                    recomp = _recompress_frames(embedded, fps, crf, tmp)
                    if not recomp:
                        row += f"{'ERR':>{col_w}}"
                        continue
                    result = extract_segment(
                        recomp, content_id, pubkey, 0, PEPPER,
                        use_jnd_adaptive=use_jnd, jnd_params=jp,
                    )
                    agreement = result.agreement
                status = "✓" if agreement >= WID_AGREEMENT_THRESHOLD else "✗"
                row += f"{agreement:.3f}{status}".rjust(col_w)
            print(row)

    print()
    print("PSNR (watermark distortion, pre-H.264):")
    for mode_name, psnr_val in psnr_rows:
        print(f"  {mode_name:<18}  {psnr_val:.1f} dB  {'✓' if psnr_val >= 38.0 else '✗'} (threshold: 38.0 dB)")


# ── Section C: Performance ────────────────────────────────────────────────────

async def section_c_performance() -> None:
    _print_section("C. Performance — sign_av on real clips")

    from tests.fixtures.polygon.registry import DatasetRegistry
    from kernel_backend.core.services.signing_service import sign_av
    from kernel_backend.core.domain.identity import Certificate
    from kernel_backend.core.services.crypto_service import generate_keypair
    from kernel_backend.infrastructure.media.media_service import MediaService
    from kernel_backend.infrastructure.storage.local_storage import LocalStorageAdapter
    from kernel_backend.infrastructure.database.repositories import VideoRepository
    from kernel_backend.infrastructure.database.models import Base
    from sqlalchemy.ext.asyncio import (
        create_async_engine, AsyncSession, async_sessionmaker,
    )

    if not MANIFEST_PATH.exists():
        print("  SKIP: manifest.yaml not found")
        return

    registry = DatasetRegistry()
    clips_to_test = []
    for speech in registry.get("video.speech").available_video():
        clips_to_test.append(("speech", speech))
    for without_audio in registry.get("video.without_audio").available_video():
        clips_to_test.append(("without_audio", without_audio))

    if not clips_to_test:
        print("  SKIP: no video clips available")
        return

    POLY_PEPPER = b"diag-perf-pepper-padded-to-32b!!"

    col = 28
    print(f"{'Clip':<22}{'Resolution':<14}{'Duration':>10}{'sign_av':>10}{'frames/s':>10}")
    print("-" * 70)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for cat, clip in clips_to_test:
            if not clip.has_audio:
                print(f"{clip.id:<22}{'—':<14}{'—':>10}  SKIP (no audio)")
                continue
            if clip.duration_s < 85:
                res_str = f"{clip.resolution[0]}×{clip.resolution[1]}"
                print(
                    f"{clip.id:<22}{res_str:<14}"
                    f"{clip.duration_s:>9.0f}s"
                    f"{'—':>9}  SKIP (< 85s min)"
                )
                continue

            priv_pem, pub_pem = generate_keypair()
            cert = Certificate(
                author_id="diag-perf-author",
                name="Diag Perf",
                institution="Test",
                public_key_pem=pub_pem,
                created_at="2026-01-01T00:00:00+00:00",
            )
            media = MediaService()
            storage_path = tmp / f"storage_{clip.id}"
            storage_path.mkdir()
            storage = LocalStorageAdapter(base_path=storage_path)

            engine = create_async_engine(
                "sqlite+aiosqlite:///:memory:",
                connect_args={"check_same_thread": False},
            )
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

            t0 = time.perf_counter()
            async with factory() as session:
                db_registry = VideoRepository(session=session)
                await sign_av(
                    media_path=clip.path,
                    certificate=cert,
                    private_key_pem=priv_pem,
                    storage=storage,
                    registry=db_registry,
                    pepper=POLY_PEPPER,
                    media=media,
                )
            elapsed = time.perf_counter() - t0
            await engine.dispose()

            res_str = f"{clip.resolution[0]}×{clip.resolution[1]}"
            total_frames = int(clip.fps * clip.duration_s)
            fps_rate = total_frames / elapsed
            print(
                f"{clip.id:<22}{res_str:<14}"
                f"{clip.duration_s:>9.0f}s"
                f"{elapsed:>9.1f}s"
                f"{fps_rate:>9.1f}"
            )


# ── Section D: Memory ─────────────────────────────────────────────────────────

async def section_d_memory() -> None:
    _print_section("D. Memory — sign_av peak on 1080p clip")

    from tests.fixtures.polygon.registry import DatasetRegistry
    from kernel_backend.core.services.signing_service import sign_av
    from kernel_backend.core.domain.identity import Certificate
    from kernel_backend.core.services.crypto_service import generate_keypair
    from kernel_backend.infrastructure.media.media_service import MediaService
    from kernel_backend.infrastructure.storage.local_storage import LocalStorageAdapter
    from kernel_backend.infrastructure.database.repositories import VideoRepository
    from kernel_backend.infrastructure.database.models import Base
    from sqlalchemy.ext.asyncio import (
        create_async_engine, AsyncSession, async_sessionmaker,
    )

    if not MANIFEST_PATH.exists():
        print("  SKIP: manifest.yaml not found")
        return

    registry = DatasetRegistry()
    clips = registry.get("video.without_audio").available_video()
    if not clips:
        print("  SKIP: no without_audio clips available")
        return

    POLY_PEPPER = b"diag-mem-pepper-padded-to-32bb!!"
    MEM_THRESHOLD_MB = 800.0

    print(f"  Threshold: < {MEM_THRESHOLD_MB:.0f} MB\n")
    print(f"  {'Clip':<22}{'Resolution':<14}{'Duration':>10}{'Peak MB':>10}{'Status':>10}")
    print("  " + "-" * 66)

    # Use speech clips (≥85s) for memory measurement — without_audio clips are 30s (too short)
    from tests.fixtures.polygon.registry import DatasetRegistry as _DR
    all_video = (
        _DR().get("video.speech").available_video()
        + _DR().get("video.without_audio").available_video()
    )
    long_clips = [c for c in all_video if c.has_audio and c.duration_s >= 85]
    if not long_clips:
        print("  SKIP: no clips ≥85s available for sign_av memory test")
        return

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for clip in long_clips:
            priv_pem, pub_pem = generate_keypair()
            cert = Certificate(
                author_id="diag-mem-author",
                name="Diag Mem",
                institution="Test",
                public_key_pem=pub_pem,
                created_at="2026-01-01T00:00:00+00:00",
            )
            media = MediaService()
            storage_path = tmp / f"mem_storage_{clip.id}"
            storage_path.mkdir()
            storage = LocalStorageAdapter(base_path=storage_path)

            engine = create_async_engine(
                "sqlite+aiosqlite:///:memory:",
                connect_args={"check_same_thread": False},
            )
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

            tracemalloc.start()
            async with factory() as session:
                db_registry = VideoRepository(session=session)
                await sign_av(
                    media_path=clip.path,
                    certificate=cert,
                    private_key_pem=priv_pem,
                    storage=storage,
                    registry=db_registry,
                    pepper=POLY_PEPPER,
                    media=media,
                )
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            await engine.dispose()

            peak_mb = peak_bytes / 1024 / 1024
            res_str = f"{clip.resolution[0]}×{clip.resolution[1]}"
            status = "✓ OK" if peak_mb < MEM_THRESHOLD_MB else "✗ OVER"
            print(
                f"  {clip.id:<22}{res_str:<14}"
                f"{clip.duration_s:>9.0f}s"
                f"{peak_mb:>9.1f}"
                f"{status:>10}"
            )


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 72)
    print("  Watermark Engine — Polygon Diagnostics")
    print("  Sprints 2 (Psychoacoustic) + 3 (JND Adaptive QIM)")
    print("=" * 72)

    section_a_audio_robustness()
    section_b_video_wid()
    await section_c_performance()
    await section_d_memory()

    print(f"\n{'═' * 72}")
    print("  Done.")
    print(f"{'═' * 72}\n")


if __name__ == "__main__":
    asyncio.run(main())
