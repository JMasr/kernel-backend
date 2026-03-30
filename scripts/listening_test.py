#!/usr/bin/env python3
"""
End-to-End Watermark Inspection Tool.

Reproduces the full production pipeline for each polygon clip:
  1. Sign with sign_audio / sign_video / sign_av
  2. Save the signed media file as the only output artifact
  3. Apply degradation conditions (in-memory temp files, not saved)
  4. Verify with VerificationService
  5. Report VERIFIED/RED verdict per clip per degradation condition

Supports custom watermark energy levels for imperceptibility/robustness
calibration via --audio-snr and --video-step flags.

Usage:
    uv run python scripts/listening_test.py --polygon              # audio only
    uv run python scripts/listening_test.py --polygon --all        # audio + video
    uv run python scripts/listening_test.py --polygon --video-only # video only
    uv run python scripts/listening_test.py --polygon --all --out-dir /tmp/inspect

    # Custom watermark energy (sweep calibration):
    uv run python scripts/listening_test.py --polygon --audio-snr -26.0
    uv run python scripts/listening_test.py --polygon --video-only --video-step 32.0
    uv run python scripts/listening_test.py --polygon --all --audio-snr -22.0 --video-step 40.0

Output: scripts/output/listening_test/
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.domain.verification import (
    AVVerificationResult,
    VerificationResult,
    Verdict,
)
from kernel_backend.core.domain.watermark import (
    AudioEmbeddingParams,
    SegmentFingerprint,
    VideoEmbeddingParams,
    VideoEntry,
)
from kernel_backend.core.ports.registry import RegistryPort
from kernel_backend.core.ports.storage import StorageKeyNotFoundError, StoragePort
from kernel_backend.core.services.crypto_service import generate_keypair
from kernel_backend.core.services.signing_service import (
    _DEFAULT_AUDIO_PARAMS,
    _DEFAULT_AV_AUDIO_PARAMS,
    _DEFAULT_VIDEO_PARAMS,
    sign_audio,
    sign_av,
    sign_video,
)
from kernel_backend.core.services.verification_service import VerificationService
from kernel_backend.infrastructure.media.media_service import MediaService


# ── In-memory infrastructure stubs ────────────────────────────────────────────


class InMemoryStorage(StoragePort):
    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        self._store[key] = data

    async def get(self, key: str) -> bytes:
        try:
            return self._store[key]
        except KeyError:
            raise StorageKeyNotFoundError(key)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def presigned_upload_url(self, key: str, expires_in: int) -> str:
        return f"mem://{key}"

    async def presigned_download_url(self, key: str, expires_in: int) -> str:
        return f"mem://{key}"


class InMemoryRegistry(RegistryPort):
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
        self, hashes: list[str], max_hamming: int = 10, org_id: UUID | None = None
    ) -> list[VideoEntry]:
        from kernel_backend.engine.audio.fingerprint import hamming_distance

        matches: set[str] = set()
        for query_hash in hashes:
            for content_id, stored_fps in self._segments.items():
                for sfp in stored_fps:
                    if hamming_distance(query_hash, sfp.hash_hex) <= max_hamming:
                        matches.add(content_id)
        return [self._videos[cid] for cid in matches if cid in self._videos]


# ── Infrastructure bootstrap ───────────────────────────────────────────────────


@dataclass
class Infra:
    private_pem: str
    certificate: Certificate
    storage: InMemoryStorage
    registry: InMemoryRegistry
    media: MediaService
    pepper: bytes


def bootstrap_infra() -> Infra:
    private_pem, public_pem = generate_keypair()
    certificate = Certificate(
        author_id="polygon-test-author",
        name="Polygon Test",
        institution="Kernel Security",
        public_key_pem=public_pem,
        created_at="2026-01-01T00:00:00+00:00",
    )
    return Infra(
        private_pem=private_pem,
        certificate=certificate,
        storage=InMemoryStorage(),
        registry=InMemoryRegistry(),
        media=MediaService(),
        pepper=os.urandom(32),
    )


# ── Polygon clip registry ──────────────────────────────────────────────────────

POLYGON_DIR = Path(__file__).resolve().parent.parent / "data"

POLYGON_AUDIO_CLIPS = {
    "libri_male":    POLYGON_DIR / "audio" / "speech" / "libri_male_01.wav",
    "libri_female":  POLYGON_DIR / "audio" / "speech" / "libri_female_01.wav",
    "choice_hiphop": POLYGON_DIR / "audio" / "speech" / "choice_hiphop_01.wav",
    "brahms_piano":  POLYGON_DIR / "audio" / "music"  / "brahms_piano_01.wav",
    "vibeace":       POLYGON_DIR / "audio" / "music"  / "vibeace_01.wav",
}

POLYGON_VIDEO_CLIPS = {
    "speech_01":        POLYGON_DIR / "video" / "speech"        / "speech.mp4",
    "camping_01":       POLYGON_DIR / "video" / "outdoor"       / "camping_01.mp4",
    "dark_no_audio_01": POLYGON_DIR / "video" / "without_audio" / "video_dark_no_audio.mp4",
    "show_01":          POLYGON_DIR / "video" / "others"        / "show.mp4",
}

AUDIO_MIN_S = 34.0   # 17 segments × 2 s
VIDEO_MIN_S = 85.0   # 17 segments × 5 s

# Degradation conditions
AUDIO_CONDITIONS = ["clean", "aac_128k", "aac_64k", "mp3_128k"]
VIDEO_CONDITIONS  = ["clean", "h264_crf23", "h264_crf28"]


# ── Degradation helpers ────────────────────────────────────────────────────────


def apply_audio_degradation(src: Path, condition: str, tmp_dir: str) -> Path:
    """Return path to degraded audio file. 'clean' returns src unchanged."""
    if condition == "clean":
        return src

    if condition in ("aac_128k", "aac_64k"):
        bitrate = "128k" if condition == "aac_128k" else "64k"
        out = Path(tmp_dir) / f"{src.stem}_{condition}.m4a"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-acodec", "aac", "-b:a", bitrate, str(out)],
            check=True, capture_output=True,
        )
        return out

    if condition == "mp3_128k":
        out = Path(tmp_dir) / f"{src.stem}_{condition}.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-acodec", "libmp3lame", "-b:a", "128k", str(out)],
            check=True, capture_output=True,
        )
        return out

    raise ValueError(f"Unknown audio condition: {condition}")


def apply_video_degradation(src: Path, condition: str, tmp_dir: str) -> Path:
    """Return path to degraded video file. 'clean' returns src unchanged."""
    if condition == "clean":
        return src

    out = Path(tmp_dir) / f"{src.stem}_{condition}.mp4"
    crf = "23" if condition == "h264_crf23" else "28"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-vcodec", "libx264", "-crf", crf,
         "-acodec", "copy",
         "-loglevel", "quiet",
         str(out)],
        check=True, capture_output=True,
    )
    return out


# ── Result dataclasses ─────────────────────────────────────────────────────────


@dataclass
class ConditionResult:
    condition: str
    verdict: str                   # "VERIFIED" / "RED" / "ERROR"
    red_reason: str | None = None
    wid_match: bool = False
    signature_valid: bool = False
    # Audio / video-only fields
    n_segments_total: int = 0
    n_segments_decoded: int = 0
    n_erasures: int = 0
    fingerprint_confidence: float = 0.0
    # AV per-channel fields
    audio_verdict: str | None = None
    video_verdict: str | None = None
    audio_n_segments: int = 0
    audio_n_decoded: int = 0
    audio_n_erasures: int = 0
    video_n_segments: int = 0
    video_n_decoded: int = 0
    video_n_erasures: int = 0
    verify_time_s: float = 0.0
    error: str | None = None


@dataclass
class ClipResult:
    name: str
    path: Path
    clip_type: str                 # "audio" / "video" / "av"
    duration_s: float
    signed_path: Path | None
    signed_size_mb: float
    sign_time_s: float
    sign_error: str | None
    conditions: list[ConditionResult] = field(default_factory=list)
    short_clip_warning: str | None = None


# ── Core E2E processing ────────────────────────────────────────────────────────


async def process_clip(
    name: str,
    path: Path,
    infra: Infra,
    output_dir: Path,
    audio_params: AudioEmbeddingParams | None = None,
    video_params: VideoEmbeddingParams | None = None,
) -> ClipResult:
    # Determine clip type
    profile = infra.media.probe(path)
    duration_s = profile.duration_s

    if profile.has_video and profile.has_audio:
        clip_type = "av"
    elif profile.has_video:
        clip_type = "video"
    else:
        clip_type = "audio"

    ext = "mp4" if clip_type in ("video", "av") else "wav"
    signed_path = output_dir / f"{name}_signed.{ext}"

    # Duration warning
    short_warning = None
    if clip_type == "audio" and duration_s < AUDIO_MIN_S:
        short_warning = (
            f"⚠  {duration_s:.1f} s < {AUDIO_MIN_S:.0f} s minimum"
            f" — WID RS recovery requires ≥17 segments"
        )
    elif clip_type in ("video", "av") and duration_s < VIDEO_MIN_S:
        short_warning = (
            f"⚠  {duration_s:.1f} s < {VIDEO_MIN_S:.0f} s minimum"
            f" — WID RS recovery requires ≥17 segments"
        )

    # ── Sign ──────────────────────────────────────────────────────────────────
    print(f"  [{name}] signing ({clip_type}, {duration_s:.1f} s)...")
    t0 = time.perf_counter()
    sign_error = None
    result = None
    try:
        if clip_type == "av":
            result = await sign_av(
                path, infra.certificate, infra.private_pem,
                infra.storage, infra.registry, infra.pepper, infra.media,
                audio_params=audio_params,
                video_params=video_params,
            )
        elif clip_type == "video":
            result = await sign_video(
                path, infra.certificate, infra.private_pem,
                infra.storage, infra.registry, infra.pepper, infra.media,
                video_params=video_params,
            )
        else:
            result = await sign_audio(
                path, infra.certificate, infra.private_pem,
                infra.storage, infra.registry, infra.pepper, infra.media,
                audio_params=audio_params,
            )
    except Exception as exc:
        sign_error = str(exc)
        print(f"  [{name}] SIGN FAILED: {exc}")
    sign_time = time.perf_counter() - t0

    if sign_error or result is None:
        return ClipResult(
            name=name, path=path, clip_type=clip_type, duration_s=duration_s,
            signed_path=None, signed_size_mb=0.0,
            sign_time_s=sign_time, sign_error=sign_error,
            short_clip_warning=short_warning,
        )

    # Save signed output
    signed_bytes = await infra.storage.get(result.signed_media_key)
    signed_path.write_bytes(signed_bytes)
    signed_size_mb = len(signed_bytes) / (1024 * 1024)
    print(
        f"  [{name}] signed → {signed_path.name}"
        f"  ({signed_size_mb:.1f} MB, {sign_time:.1f} s)"
    )

    # ── Verify under each degradation condition ────────────────────────────────
    svc = VerificationService()
    conditions = VIDEO_CONDITIONS if clip_type in ("video", "av") else AUDIO_CONDITIONS
    condition_results: list[ConditionResult] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for condition in conditions:
            print(f"  [{name}] verifying {condition}...")
            try:
                if clip_type in ("video", "av"):
                    degraded = apply_video_degradation(signed_path, condition, tmp_dir)
                else:
                    degraded = apply_audio_degradation(signed_path, condition, tmp_dir)

                t1 = time.perf_counter()

                if clip_type == "av":
                    vr = await svc.verify_av(
                        degraded, infra.media, infra.storage, infra.registry, infra.pepper,
                    )
                    verify_time = time.perf_counter() - t1
                    cr = ConditionResult(
                        condition=condition,
                        verdict=vr.verdict.value,
                        red_reason=vr.red_reason.value if vr.red_reason else None,
                        wid_match=vr.wid_match,
                        signature_valid=vr.signature_valid,
                        fingerprint_confidence=vr.fingerprint_confidence,
                        audio_verdict=vr.audio_verdict.value,
                        video_verdict=vr.video_verdict.value,
                        audio_n_segments=vr.audio_n_segments,
                        audio_n_decoded=vr.audio_n_decoded,
                        audio_n_erasures=vr.audio_n_erasures,
                        video_n_segments=vr.video_n_segments,
                        video_n_decoded=vr.video_n_decoded,
                        video_n_erasures=vr.video_n_erasures,
                        verify_time_s=verify_time,
                    )

                elif clip_type == "video":
                    vr = await svc.verify(
                        degraded, infra.media, infra.storage, infra.registry, infra.pepper,
                    )
                    verify_time = time.perf_counter() - t1
                    cr = ConditionResult(
                        condition=condition,
                        verdict=vr.verdict.value,
                        red_reason=vr.red_reason.value if vr.red_reason else None,
                        wid_match=vr.wid_match,
                        signature_valid=vr.signature_valid,
                        n_segments_total=vr.n_segments_total,
                        n_segments_decoded=vr.n_segments_decoded,
                        n_erasures=vr.n_erasures,
                        fingerprint_confidence=vr.fingerprint_confidence,
                        verify_time_s=verify_time,
                    )

                else:  # audio
                    vr = await svc.verify_audio(
                        degraded, infra.media, infra.storage, infra.registry, infra.pepper,
                    )
                    verify_time = time.perf_counter() - t1
                    cr = ConditionResult(
                        condition=condition,
                        verdict=vr.verdict.value,
                        red_reason=vr.red_reason.value if vr.red_reason else None,
                        wid_match=vr.wid_match,
                        signature_valid=vr.signature_valid,
                        n_segments_total=vr.n_segments_total,
                        n_segments_decoded=vr.n_segments_decoded,
                        n_erasures=vr.n_erasures,
                        fingerprint_confidence=vr.fingerprint_confidence,
                        verify_time_s=verify_time,
                    )

                sym = "✓" if cr.verdict == "VERIFIED" else "✗"
                reason = f"  ({cr.red_reason})" if cr.red_reason else ""
                print(f"  [{name}] {condition}: {cr.verdict} {sym}{reason}")
                condition_results.append(cr)

            except Exception as exc:
                print(f"  [{name}] {condition}: ERROR — {exc}")
                condition_results.append(ConditionResult(
                    condition=condition,
                    verdict="ERROR",
                    error=str(exc),
                ))

    return ClipResult(
        name=name, path=path, clip_type=clip_type, duration_s=duration_s,
        signed_path=signed_path, signed_size_mb=signed_size_mb,
        sign_time_s=sign_time, sign_error=None,
        conditions=condition_results, short_clip_warning=short_warning,
    )


# ── Report generation ──────────────────────────────────────────────────────────


def _vs(verdict: str) -> str:
    """Compact verdict string."""
    if verdict == "VERIFIED":
        return "VERIFIED ✓"
    if verdict == "RED":
        return "RED ✗    "
    return "ERROR ✗  "


def _b(flag: bool) -> str:
    return "✓" if flag else "✗"


def make_report(
    results: list[ClipResult],
    output_dir: Path,
    audio_params: AudioEmbeddingParams | None = None,
    video_params: VideoEmbeddingParams | None = None,
) -> str:
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines += [
        "# End-to-End Signing + Verification Report",
        f"Date: {now}",
        "",
    ]

    # Show watermark energy configuration
    a_snr = audio_params.target_snr_db if audio_params else _DEFAULT_AUDIO_PARAMS.target_snr_db
    v_step = video_params.qim_step_base if video_params else _DEFAULT_VIDEO_PARAMS.qim_step_base
    a_label = "CUSTOM" if audio_params else "production"
    v_label = "CUSTOM" if video_params else "production"
    lines += [
        "## Watermark Parameters",
        f"  Audio target_snr_db: {a_snr} ({a_label},"
        f" production = {_DEFAULT_AUDIO_PARAMS.target_snr_db})",
        f"  Video qim_step_base: {v_step} ({v_label},"
        f" production = {_DEFAULT_VIDEO_PARAMS.qim_step_base})",
        "",
    ]

    audio_results = [r for r in results if r.clip_type == "audio"]
    video_results = [r for r in results if r.clip_type in ("video", "av")]

    # ── Audio section ──────────────────────────────────────────────────────────
    if audio_results:
        lines.append("## Audio Clips")
        lines.append("")
        for r in audio_results:
            lines.append(f"### {r.name}  [WAV · {r.duration_s:.1f} s · AUDIO-ONLY]")
            if r.short_clip_warning:
                lines.append(r.short_clip_warning)
            if r.sign_error:
                lines.append(f"  SIGN FAILED: {r.sign_error}")
                lines.append("")
                continue
            lines.append(
                f"Signed:  {r.signed_path.name}"
                f"  ({r.signed_size_mb:.1f} MB, {r.sign_time_s:.1f} s)"
            )
            lines.append("")
            lines.append(
                "  Condition    │ Verdict      │ WID │ Sig │"
                " Decoded          │ FP conf │ Time"
            )
            lines.append(
                "  ─────────────┼──────────────┼─────┼─────┼"
                "──────────────────┼─────────┼──────"
            )
            for c in r.conditions:
                if c.error:
                    lines.append(
                        f"  {c.condition:<12} │ ERROR ✗       │  -  │  -  │"
                        "      -/-         │    -    │   -"
                    )
                    continue
                decoded = f"{c.n_segments_decoded}/{c.n_segments_total}"
                erasures = f"({c.n_erasures} erasures)"
                lines.append(
                    f"  {c.condition:<12} │ {_vs(c.verdict):<12} │"
                    f"  {_b(c.wid_match)}  │  {_b(c.signature_valid)}  │"
                    f"  {decoded:<6} {erasures:<14} │"
                    f"  {c.fingerprint_confidence:.3f}  │"
                    f"  {c.verify_time_s:.1f}s"
                )
            lines.append("")

    # ── Video section ──────────────────────────────────────────────────────────
    if video_results:
        lines.append("## Video Clips")
        lines.append("")
        for r in video_results:
            type_label = "AV" if r.clip_type == "av" else "VIDEO-ONLY"
            lines.append(f"### {r.name}  [MP4 · {r.duration_s:.1f} s · {type_label}]")
            if r.short_clip_warning:
                lines.append(r.short_clip_warning)
            if r.sign_error:
                lines.append(f"  SIGN FAILED: {r.sign_error}")
                lines.append("")
                continue
            lines.append(
                f"Signed:  {r.signed_path.name}"
                f"  ({r.signed_size_mb:.1f} MB, {r.sign_time_s:.1f} s)"
            )
            lines.append("")
            if r.clip_type == "av":
                lines.append(
                    "  Condition       │ Verdict      │ Audio WID          │"
                    " Video WID          │ FP conf │ Sig │ Time"
                )
                lines.append(
                    "  ────────────────┼──────────────┼────────────────────┼"
                    "────────────────────┼─────────┼─────┼──────"
                )
                for c in r.conditions:
                    if c.error:
                        lines.append(
                            f"  {c.condition:<15} │ ERROR ✗       │         -          │"
                            "         -          │    -    │  -  │   -"
                        )
                        continue
                    a_sym = _b(c.audio_verdict == "VERIFIED")
                    v_sym = _b(c.video_verdict == "VERIFIED")
                    a_col = f"{a_sym} {c.audio_n_decoded}/{c.audio_n_segments} ({c.audio_n_erasures} era)"
                    v_col = f"{v_sym} {c.video_n_decoded}/{c.video_n_segments} ({c.video_n_erasures} era)"
                    lines.append(
                        f"  {c.condition:<15} │ {_vs(c.verdict):<12} │"
                        f" {a_col:<20} │ {v_col:<20} │"
                        f"  {c.fingerprint_confidence:.3f}  │"
                        f"  {_b(c.signature_valid)}  │"
                        f"  {c.verify_time_s:.1f}s"
                    )
            else:
                lines.append(
                    "  Condition       │ Verdict      │ WID │ Sig │"
                    " Decoded          │ FP conf │ Time"
                )
                lines.append(
                    "  ────────────────┼──────────────┼─────┼─────┼"
                    "──────────────────┼─────────┼──────"
                )
                for c in r.conditions:
                    if c.error:
                        lines.append(
                            f"  {c.condition:<15} │ ERROR ✗       │  -  │  -  │"
                            "      -/-         │    -    │   -"
                        )
                        continue
                    decoded = f"{c.n_segments_decoded}/{c.n_segments_total}"
                    erasures = f"({c.n_erasures} erasures)"
                    lines.append(
                        f"  {c.condition:<15} │ {_vs(c.verdict):<12} │"
                        f"  {_b(c.wid_match)}  │  {_b(c.signature_valid)}  │"
                        f"  {decoded:<6} {erasures:<14} │"
                        f"  {c.fingerprint_confidence:.3f}  │"
                        f"  {c.verify_time_s:.1f}s"
                    )
            lines.append("")

    # ── Summary table ──────────────────────────────────────────────────────────
    lines += ["## Summary", ""]
    # Collect all conditions that appear across results
    audio_extra = AUDIO_CONDITIONS[1:]
    video_extra = VIDEO_CONDITIONS[1:]
    all_extra = list(dict.fromkeys(audio_extra + video_extra))

    hdr = f"  {'Clip':<22} │ {'Type':5} │ {'Dur':>6} │ {'Signed':6} │ {'clean':>8}"
    for cond in all_extra:
        hdr += f" │ {cond[:9]:>9}"
    lines.append(hdr)
    sep = (
        "  " + "─" * 22 + "┼" + "─" * 7 + "┼" + "─" * 8
        + "┼" + "─" * 8 + "┼" + "─" * 10
        + ("┼" + "─" * 11) * len(all_extra)
    )
    lines.append(sep)

    for r in results:
        cond_map = {c.condition: c for c in r.conditions}
        clean = cond_map.get("clean")
        clean_str = (
            "✓ VERIFIED" if (clean and clean.verdict == "VERIFIED")
            else ("✗ RED" if (clean and clean.verdict == "RED") else "✗ ERROR" if clean else "n/a")
        )
        row = (
            f"  {r.name:<22} │ {r.clip_type.upper():5} │"
            f" {r.duration_s:>5.0f}s │"
            f" {'✗ FAIL' if r.sign_error else '✓ OK':<6} │"
            f" {clean_str:>8}"
        )
        applicable = AUDIO_CONDITIONS[1:] if r.clip_type == "audio" else VIDEO_CONDITIONS[1:]
        for cond in all_extra:
            if cond in applicable:
                c = cond_map.get(cond)
                cell = (
                    "✓" if (c and c.verdict == "VERIFIED")
                    else ("✗" if c else "-")
                )
            else:
                cell = "n/a"
            row += f" │ {cell:>9}"
        lines.append(row)

    lines += [
        "",
        "## Output Files",
        f"  {output_dir}/",
    ]
    for r in results:
        if r.signed_path and r.signed_path.exists():
            lines.append(f"    {r.signed_path.name}")
    lines += [
        "    report.md",
        "",
        "## How to Inspect",
        "  Audio: open *_signed.wav in Audacity and compare against the original",
        "  Video: play *_signed.mp4 — check for visible blocking or banding",
        "  Tamper: flip 1 byte in the signed file and re-run — verdict must be RED",
        "  Tune:  edit signing_service.py target_snr_db or QIM_STEP_WID constants",
        "         and re-run to measure impact on robustness vs quality trade-off",
    ]
    return "\n".join(lines)


# ── Entry point ────────────────────────────────────────────────────────────────


async def _async_main(args: argparse.Namespace) -> None:
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Kernel Security — End-to-End Watermark Inspection")
    print(f"Output: {output_dir}")
    print()

    infra = bootstrap_infra()
    print(f"Keypair : {infra.certificate.public_key_pem.splitlines()[1][:40]}...")
    print(f"Pepper  : {infra.pepper.hex()[:16]}...")

    # ── Build custom params from CLI flags ────────────────────────────────────
    audio_params: AudioEmbeddingParams | None = None
    video_params: VideoEmbeddingParams | None = None

    if args.audio_snr is not None:
        audio_params = AudioEmbeddingParams(
            dwt_levels=_DEFAULT_AUDIO_PARAMS.dwt_levels,
            chips_per_bit=_DEFAULT_AUDIO_PARAMS.chips_per_bit,
            psychoacoustic=_DEFAULT_AUDIO_PARAMS.psychoacoustic,
            safety_margin_db=_DEFAULT_AUDIO_PARAMS.safety_margin_db,
            target_snr_db=args.audio_snr,
        )
        print(f"Audio   : custom target_snr_db = {args.audio_snr} dB"
              f"  (production = {_DEFAULT_AUDIO_PARAMS.target_snr_db})")
    else:
        print(f"Audio   : production defaults (target_snr_db = {_DEFAULT_AUDIO_PARAMS.target_snr_db})")

    if args.video_step is not None:
        video_params = VideoEmbeddingParams(
            jnd_adaptive=True,
            qim_step_base=args.video_step,
            qim_step_min=args.video_step,
            qim_step_max=args.video_step,
            qim_quantize_to=1.0,
        )
        print(f"Video   : custom qim_step = {args.video_step}"
              f"  (production = {_DEFAULT_VIDEO_PARAMS.qim_step_base})")
    else:
        print(f"Video   : production defaults (qim_step_base = {_DEFAULT_VIDEO_PARAMS.qim_step_base})")

    print()

    clips: list[tuple[str, Path]] = []

    if not args.video_only:
        for name, path in POLYGON_AUDIO_CLIPS.items():
            if path.exists():
                clips.append((name, path))
            else:
                print(f"  [SKIP] {name}: not found at {path}")

    if args.all or args.video_only:
        for name, path in POLYGON_VIDEO_CLIPS.items():
            if path.exists():
                clips.append((name, path))
            else:
                print(f"  [SKIP] {name}: not found at {path}")

    if not clips:
        print("No clips found. Check data/ directory and run setup_polygon_audio.py.")
        return

    results: list[ClipResult] = []
    for name, path in clips:
        print(f"\n{'─' * 60}")
        print(f"Processing: {name}  ({path.name})")
        r = await process_clip(name, path, infra, output_dir,
                               audio_params=audio_params, video_params=video_params)
        results.append(r)

    print(f"\n{'═' * 60}")
    report = make_report(results, output_dir, audio_params, video_params)
    print(report)

    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved → {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-End Watermark Inspection Tool",
    )
    parser.add_argument(
        "--polygon", action="store_true", required=True,
        help="Use polygon clips from data/",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process audio + video clips (default: audio only)",
    )
    parser.add_argument(
        "--video-only", action="store_true",
        help="Process video clips only",
    )
    parser.add_argument(
        "--audio-snr", type=float, default=None,
        help="Override audio target_snr_db (e.g. -22.0). Default: production value.",
    )
    parser.add_argument(
        "--video-step", type=float, default=None,
        help="Override video qim_step_base (e.g. 40.0). Default: production value.",
    )
    parser.add_argument(
        "--out-dir", default="scripts/output/listening_test/",
        help="Output directory (default: scripts/output/listening_test/)",
    )
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
