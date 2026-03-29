#!/usr/bin/env python3
"""
Calibration sweep — find the optimal (safety_margin_db, target_snr_db) pair.

Sweeps a grid of parameters, embedding pilot + WID into each clip, measuring
imperceptibility (seg-SNR, spectral distortion, max_abs_diff) and robustness
(pilot detection, WID Z-scores, RS decode after codec degradation).

Usage:
    uv run python scripts/calibration_sweep.py                 # librosa clips, coarse grid
    uv run python scripts/calibration_sweep.py --polygon       # polygon clips
    uv run python scripts/calibration_sweep.py --keep-all      # don't delete intermediate WAVs
    uv run python scripts/calibration_sweep.py --fine           # fine grid around defaults

Output:  scripts/output/calibration/  (CSV + Markdown report + Pareto WAVs)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kernel_backend.core.domain.watermark import BandConfig
from kernel_backend.engine.audio.pilot_tone import detect_pilot, embed_pilot
from kernel_backend.engine.audio.wid_beacon import (
    ERASURE_THRESHOLD_Z,
    embed_segment,
    extract_symbol_segment,
)
from kernel_backend.engine.codec.hopping import plan_audio_hopping
from kernel_backend.engine.codec.reed_solomon import ReedSolomonCodec

# ── Constants ────────────────────────────────────────────────────────────────

SR = 44100
HASH_48 = 0xABCDEF012345
PILOT_SEED = 0xDEADBEEFCAFEBABE
PEPPER = b"calibration-pepper-32bytes-pad!!"
CONTENT_ID = "calibration-content-id"
PUBKEY = "calibration-public-key"
WID = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10"
SEGMENT_S = 2.0

CODEC_CONDITIONS = ["clean", "aac_256k", "aac_192k", "aac_128k", "mp3_128k"]

POLYGON_DIR = Path(__file__).resolve().parent.parent / "data"
POLYGON_CLIPS = {
    "brahms_piano":  POLYGON_DIR / "audio" / "music"  / "brahms_piano_01.wav",
    "vibeace":       POLYGON_DIR / "audio" / "music"  / "vibeace_01.wav",
    "choice_hiphop": POLYGON_DIR / "audio" / "speech" / "choice_hiphop_01.wav",
    "libri_male":    POLYGON_DIR / "audio" / "speech" / "libri_male_01.wav",
    "libri_female":  POLYGON_DIR / "audio" / "speech" / "libri_female_01.wav",
}

# ── Parameter grids ──────────────────────────────────────────────────────────

# safety_margin_db is irrelevant (Bark model non-functional, Watson doesn't use it)
# Sweep only target_snr_db for WID
COARSE_SAFETY = [0.0]  # placeholder, not used by Watson path
COARSE_SNR = [-30.0, -28.0, -26.0, -24.0, -22.0, -20.0, -18.0, -16.0, -14.0, -12.0, -10.0]

FINE_SAFETY = [0.0]
FINE_SNR = [-24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0, -16.0, -15.0, -14.0]


# ── Metrics (from perceptual_validation.py) ──────────────────────────────────


def segmental_snr(original: np.ndarray, watermarked: np.ndarray,
                  frame_ms: float = 20.0) -> float:
    frame_len = int(SR * frame_ms / 1000.0)
    n_frames = len(original) // frame_len
    snrs: list[float] = []
    for i in range(n_frames):
        s = i * frame_len
        e = s + frame_len
        sig = original[s:e]
        noise = watermarked[s:e] - sig
        sig_power = float(np.mean(sig ** 2))
        noise_power = float(np.mean(noise ** 2))
        if sig_power < 1e-10 or noise_power < 1e-10:
            continue
        snrs.append(10.0 * np.log10(sig_power / noise_power))
    return float(np.mean(snrs)) if snrs else float("inf")


def spectral_distortion(original: np.ndarray, watermarked: np.ndarray,
                        n_fft: int = 2048) -> float:
    from scipy.signal import stft
    _, _, Zo = stft(original, fs=SR, nperseg=n_fft, noverlap=n_fft // 2)
    _, _, Zw = stft(watermarked, fs=SR, nperseg=n_fft, noverlap=n_fft // 2)
    Po = np.abs(Zo) ** 2 + 1e-10
    Pw = np.abs(Zw) ** 2 + 1e-10
    lsd = (10.0 * np.log10(Po) - 10.0 * np.log10(Pw)) ** 2
    return float(np.mean(lsd))


def max_abs_diff(original: np.ndarray, watermarked: np.ndarray) -> float:
    return float(np.max(np.abs(watermarked - original)))


# ── Embedding ────────────────────────────────────────────────────────────────


def _pn_seed(i: int) -> int:
    msg = f"wid|{CONTENT_ID}|{PUBKEY}|{i}".encode()
    return int.from_bytes(hmac.new(PEPPER, msg, hashlib.sha256).digest()[:8], "big")


PILOT_SNR_DB = -26.0  # Z-score detection (threshold 1.5) allows -26 dB.
                      # Old normalized-correlation detection needed -14 dB.
                      # Pilot not used in production verification — only scripts.


def embed_full(audio: np.ndarray, wid_target_snr_db: float,
               safety_margin_db: float = 0.0,
               pilot_snr_db: float = PILOT_SNR_DB) -> np.ndarray:
    """Embed pilot + WID with separate SNR controls.

    Uses Watson masking (perceptual_shaping + temporal_shaping) for both pilot
    and WID.  The Bark psychoacoustic model (use_psychoacoustic=True) is
    DISABLED because bark_amplitude_profile_for_dwt_level() returns values in
    STFT power units (~1e-6) while DWT coefficients need ~1e-3 — the floor
    always overrides 100% of coefficients, making the model non-functional.
    """
    result = embed_pilot(
        audio, SR, HASH_48, PILOT_SEED,
        target_snr_db=pilot_snr_db,
        perceptual_shaping=True,
        temporal_shaping=True,
        use_psychoacoustic=False,
    )

    seg_len = int(SR * SEGMENT_S)
    n_segments = len(audio) // seg_len
    if n_segments < 17:
        return result

    codec = ReedSolomonCodec(n_symbols=n_segments)
    rs_symbols = codec.encode(WID)
    band_configs = plan_audio_hopping(n_segments, CONTENT_ID, PUBKEY, PEPPER)
    seeds = [_pn_seed(i) for i in range(n_segments)]

    for i in range(n_segments):
        s = i * seg_len
        e = s + seg_len
        result[s:e] = embed_segment(
            result[s:e], rs_symbols[i], band_configs[i], seeds[i],
            chips_per_bit=32,
            target_snr_db=wid_target_snr_db,
            perceptual_shaping=True,
            temporal_shaping=True,
            use_psychoacoustic=False,
        )

    return result


# ── Robustness extraction ────────────────────────────────────────────────────


@dataclass
class RobustnessResult:
    pilot_detected: bool
    wid_mean_z: float
    wid_n_erasures: int
    wid_n_segments: int
    rs_decode_ok: bool
    rs_margin: int  # (N-K)/2 - n_erasures


def measure_robustness(audio: np.ndarray) -> RobustnessResult:
    """Measure pilot detection + WID recovery on (possibly degraded) audio."""
    pilot_detected = detect_pilot(audio, SR, PILOT_SEED) == HASH_48

    seg_len = int(SR * SEGMENT_S)
    n_segments = len(audio) // seg_len
    if n_segments < 17:
        return RobustnessResult(
            pilot_detected=pilot_detected,
            wid_mean_z=0.0, wid_n_erasures=n_segments,
            wid_n_segments=n_segments, rs_decode_ok=False, rs_margin=-1,
        )

    codec = ReedSolomonCodec(n_symbols=n_segments)
    band_configs = plan_audio_hopping(n_segments, CONTENT_ID, PUBKEY, PEPPER)
    seeds = [_pn_seed(i) for i in range(n_segments)]

    recovered: list[int | None] = []
    z_scores: list[float] = []
    for i in range(n_segments):
        s = i * seg_len
        e = s + seg_len
        sym, z = extract_symbol_segment(audio[s:e], band_configs[i], seeds[i],
                                        chips_per_bit=32)
        z_scores.append(z)
        recovered.append(sym if z >= ERASURE_THRESHOLD_Z else None)

    n_erasures = sum(1 for s in recovered if s is None)
    mean_z = float(np.mean(z_scores))
    parity = (n_segments - 16) // 2
    rs_margin = parity - n_erasures

    try:
        decoded = codec.decode(recovered)
        rs_ok = decoded == WID
    except Exception:
        rs_ok = False

    return RobustnessResult(
        pilot_detected=pilot_detected,
        wid_mean_z=mean_z,
        wid_n_erasures=n_erasures,
        wid_n_segments=n_segments,
        rs_decode_ok=rs_ok,
        rs_margin=rs_margin,
    )


# ── Codec degradation ────────────────────────────────────────────────────────


def apply_codec(src_path: Path, condition: str, tmp_dir: str) -> Path:
    """Apply codec degradation; returns path to degraded file."""
    if condition == "clean":
        return src_path

    codec_map = {
        "aac_256k": ("aac",        "256k", ".m4a"),
        "aac_192k": ("aac",        "192k", ".m4a"),
        "aac_128k": ("aac",        "128k", ".m4a"),
        "mp3_128k": ("libmp3lame", "128k", ".mp3"),
    }
    acodec, bitrate, ext = codec_map[condition]
    out = Path(tmp_dir) / f"degraded_{condition}{ext}"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src_path),
         "-acodec", acodec, "-b:a", bitrate,
         "-loglevel", "quiet", str(out)],
        check=True, capture_output=True,
    )
    return out


def read_audio(path: Path) -> np.ndarray:
    """Read audio file, convert to float32 mono at SR.

    For formats not supported by libsndfile (AAC/M4A/MP3), decodes via FFmpeg
    to a temporary WAV first.
    """
    suffix = path.suffix.lower()
    if suffix in (".m4a", ".aac", ".mp3", ".mp4", ".ogg"):
        # Decode to raw PCM via FFmpeg pipe — no temp file needed
        proc = subprocess.run(
            ["ffmpeg", "-i", str(path),
             "-f", "s16le", "-acodec", "pcm_s16le",
             "-ar", str(SR), "-ac", "1",
             "-loglevel", "quiet", "-"],
            capture_output=True, check=True,
        )
        audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return audio

    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != SR:
        from scipy.signal import resample
        n_target = int(len(audio) * SR / file_sr)
        audio = resample(audio, n_target).astype(np.float32)
    return audio


# ── Single configuration evaluation ─────────────────────────────────────────


@dataclass
class SweepRow:
    clip_name: str
    safety_margin_db: float
    target_snr_db: float
    codec: str
    seg_snr: float
    spectral_dist: float
    max_diff: float
    pilot_ok: bool
    wid_mean_z: float
    wid_erasures: int
    wid_segments: int
    rs_ok: bool
    rs_margin: int


def evaluate_config(
    clip_name: str,
    audio: np.ndarray,
    safety_margin_db: float,
    target_snr_db: float,
    codecs: list[str],
    tmp_dir: str,
) -> list[SweepRow]:
    """Embed with given params, measure quality + robustness under each codec."""
    watermarked = embed_full(audio, wid_target_snr_db=target_snr_db,
                             safety_margin_db=safety_margin_db)

    # Imperceptibility metrics (always measured on clean)
    snr = segmental_snr(audio, watermarked)
    sd = spectral_distortion(audio, watermarked)
    md = max_abs_diff(audio, watermarked)

    rows: list[SweepRow] = []

    # Write watermarked WAV to temp for codec degradation
    wav_path = Path(tmp_dir) / "watermarked.wav"
    sf.write(str(wav_path), watermarked, SR)

    for codec_cond in codecs:
        if codec_cond == "clean":
            test_audio = watermarked
        else:
            degraded_path = apply_codec(wav_path, codec_cond, tmp_dir)
            test_audio = read_audio(degraded_path)
            # Trim/pad to match original length for fair comparison
            if len(test_audio) > len(audio):
                test_audio = test_audio[:len(audio)]
            elif len(test_audio) < len(audio):
                test_audio = np.pad(test_audio, (0, len(audio) - len(test_audio)))
            # Clean up degraded file immediately
            if degraded_path != wav_path:
                degraded_path.unlink(missing_ok=True)

        rob = measure_robustness(test_audio)
        rows.append(SweepRow(
            clip_name=clip_name,
            safety_margin_db=safety_margin_db,
            target_snr_db=target_snr_db,
            codec=codec_cond,
            seg_snr=snr,
            spectral_dist=sd,
            max_diff=md,
            pilot_ok=rob.pilot_detected,
            wid_mean_z=rob.wid_mean_z,
            wid_erasures=rob.wid_n_erasures,
            wid_segments=rob.wid_n_segments,
            rs_ok=rob.rs_decode_ok,
            rs_margin=rob.rs_margin,
        ))

    # Clean up watermarked WAV
    wav_path.unlink(missing_ok=True)
    return rows


# ── Pareto frontier ──────────────────────────────────────────────────────────


def find_pareto_configs(rows: list[SweepRow], top_n: int = 5) -> list[tuple[float, float]]:
    """Find Pareto-optimal (safety_margin, target_snr) pairs.

    Optimising: maximise seg_snr (imperceptibility) AND maximise worst-case
    rs_margin (robustness).  A config dominates another if it is >= on both
    objectives and strictly > on at least one.
    """
    # Aggregate: per (safety, snr) → (mean seg_snr across clips, min rs_margin across clips+codecs)
    from collections import defaultdict
    agg: dict[tuple[float, float], dict] = defaultdict(lambda: {"seg_snrs": [], "rs_margins": []})

    for r in rows:
        key = (r.safety_margin_db, r.target_snr_db)
        agg[key]["seg_snrs"].append(r.seg_snr)
        agg[key]["rs_margins"].append(r.rs_margin)

    points: list[tuple[float, float, float, float]] = []  # (safety, snr, mean_seg_snr, min_rs_margin)
    for (safety, snr), v in agg.items():
        mean_seg = float(np.mean(v["seg_snrs"]))
        min_margin = min(v["rs_margins"])
        points.append((safety, snr, mean_seg, min_margin))

    # Pareto filter: keep points not dominated by any other
    pareto: list[tuple[float, float, float, float]] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q[2] >= p[2] and q[3] >= p[3] and (q[2] > p[2] or q[3] > p[3]):
                dominated = True
                break
        if not dominated:
            pareto.append(p)

    # Sort by seg_snr descending (imperceptibility first)
    pareto.sort(key=lambda x: -x[2])
    return [(p[0], p[1]) for p in pareto[:top_n]]


# ── Report generation ────────────────────────────────────────────────────────


def write_csv(rows: list[SweepRow], path: Path) -> None:
    fieldnames = [
        "clip_name", "safety_margin_db", "target_snr_db", "codec",
        "seg_snr", "spectral_dist", "max_diff",
        "pilot_ok", "wid_mean_z", "wid_erasures", "wid_segments",
        "rs_ok", "rs_margin",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "clip_name": r.clip_name,
                "safety_margin_db": r.safety_margin_db,
                "target_snr_db": r.target_snr_db,
                "codec": r.codec,
                "seg_snr": f"{r.seg_snr:.2f}",
                "spectral_dist": f"{r.spectral_dist:.4f}",
                "max_diff": f"{r.max_diff:.6f}",
                "pilot_ok": r.pilot_ok,
                "wid_mean_z": f"{r.wid_mean_z:.3f}",
                "wid_erasures": r.wid_erasures,
                "wid_segments": r.wid_segments,
                "rs_ok": r.rs_ok,
                "rs_margin": r.rs_margin,
            })


def write_report(rows: list[SweepRow], pareto: list[tuple[float, float]],
                 path: Path) -> None:
    from collections import defaultdict

    lines: list[str] = ["# Calibration Sweep Report\n"]

    # Summary table: one row per (safety, snr) config
    agg: dict[tuple[float, float], dict] = defaultdict(lambda: {
        "seg_snrs": [], "rs_margins": [], "rs_oks": [], "pilot_oks": [],
        "wid_zs": [],
    })
    for r in rows:
        key = (r.safety_margin_db, r.target_snr_db)
        agg[key]["seg_snrs"].append(r.seg_snr)
        agg[key]["rs_margins"].append(r.rs_margin)
        agg[key]["rs_oks"].append(r.rs_ok)
        agg[key]["pilot_oks"].append(r.pilot_ok)
        agg[key]["wid_zs"].append(r.wid_mean_z)

    lines.append("## Summary (all clips, all codecs)\n")
    lines.append("| safety_db | snr_db | seg_SNR | min_RS_margin | RS_pass% | pilot% | mean_Z | Pareto |")
    lines.append("|-----------|--------|---------|---------------|----------|--------|--------|--------|")

    sorted_keys = sorted(agg.keys())
    for key in sorted_keys:
        v = agg[key]
        mean_seg = float(np.mean(v["seg_snrs"]))
        min_margin = min(v["rs_margins"])
        rs_pct = 100.0 * sum(v["rs_oks"]) / len(v["rs_oks"])
        pilot_pct = 100.0 * sum(v["pilot_oks"]) / len(v["pilot_oks"])
        mean_z = float(np.mean(v["wid_zs"]))
        is_pareto = "**YES**" if key in pareto else ""
        lines.append(
            f"| {key[0]:>9.1f} | {key[1]:>6.1f} | {mean_seg:>7.1f} | "
            f"{min_margin:>13d} | {rs_pct:>7.0f}% | {pilot_pct:>5.0f}% | "
            f"{mean_z:>6.2f} | {is_pareto} |"
        )

    # Pareto section
    lines.append("\n## Pareto Frontier (imperceptibility vs robustness)\n")
    if pareto:
        lines.append("These configurations are not dominated by any other:\n")
        for i, (safety, snr) in enumerate(pareto, 1):
            v = agg[(safety, snr)]
            lines.append(
                f"{i}. safety_margin_db={safety:.1f}, target_snr_db={snr:.1f}  "
                f"(seg-SNR={np.mean(v['seg_snrs']):.1f} dB, "
                f"min RS margin={min(v['rs_margins'])}, "
                f"RS pass={100*sum(v['rs_oks'])/len(v['rs_oks']):.0f}%)"
            )
    else:
        lines.append("No Pareto-optimal configurations found.\n")

    # Recommendation
    lines.append("\n## Recommendation\n")
    # Find the Pareto point with the best imperceptibility that still passes RS
    best = None
    for safety, snr in pareto:
        v = agg[(safety, snr)]
        if all(v["rs_oks"]) and all(v["pilot_oks"]):
            if best is None or np.mean(v["seg_snrs"]) > np.mean(agg[best]["seg_snrs"]):
                best = (safety, snr)
    if best:
        v = agg[best]
        lines.append(
            f"Best fully-passing config: `safety_margin_db={best[0]:.1f}`, "
            f"`target_snr_db={best[1]:.1f}`\n"
        )
        lines.append(f"- Mean seg-SNR: {np.mean(v['seg_snrs']):.1f} dB")
        lines.append(f"- Min RS margin: {min(v['rs_margins'])}")
        lines.append(f"- Mean WID Z-score: {np.mean(v['wid_zs']):.2f}")
    else:
        # Fallback: best RS pass rate with highest seg-SNR
        candidates = [(k, v) for k, v in agg.items()
                       if sum(v["rs_oks"]) / len(v["rs_oks"]) > 0.8]
        if candidates:
            candidates.sort(key=lambda x: -np.mean(x[1]["seg_snrs"]))
            best = candidates[0][0]
            v = candidates[0][1]
            lines.append(
                f"Best >80% passing config: `safety_margin_db={best[0]:.1f}`, "
                f"`target_snr_db={best[1]:.1f}`\n"
            )
            lines.append(f"- Mean seg-SNR: {np.mean(v['seg_snrs']):.1f} dB")
            lines.append(f"- RS pass rate: {100*sum(v['rs_oks'])/len(v['rs_oks']):.0f}%")
        else:
            lines.append("No configuration achieves >80% RS decode. Consider reducing safety_margin_db.")

    path.write_text("\n".join(lines) + "\n")


# ── Audio loading ────────────────────────────────────────────────────────────


def load_librosa_clip(name: str) -> np.ndarray | None:
    try:
        import librosa
        audio, _ = librosa.load(librosa.ex(name), sr=SR, mono=True)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"  Could not load librosa '{name}': {e}")
        return None


def load_wav(path: Path) -> np.ndarray | None:
    try:
        return read_audio(path)
    except Exception as e:
        print(f"  Could not load '{path}': {e}")
        return None


def synthetic_noise(duration_s: float = 40.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(SR * duration_s)) * 0.3).astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibration sweep: find optimal (safety_margin_db, target_snr_db)")
    parser.add_argument("--polygon", action="store_true",
                        help="Use polygon clips from data/")
    parser.add_argument("--fine", action="store_true",
                        help="Fine grid (smaller steps around current defaults)")
    parser.add_argument("--keep-all", action="store_true",
                        help="Keep all intermediate WAV files (uses more disk)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of Pareto-optimal WAVs to keep (default: 5)")
    parser.add_argument("--codecs", nargs="*", default=None,
                        help="Override codec conditions (default: all)")
    args = parser.parse_args()

    safety_grid = FINE_SAFETY if args.fine else COARSE_SAFETY
    snr_grid = FINE_SNR if args.fine else COARSE_SNR
    codecs = args.codecs or CODEC_CONDITIONS

    # Load clips
    clips: dict[str, np.ndarray] = {}
    if args.polygon:
        print("Source: POLYGON clips")
        for name, wav_path in POLYGON_CLIPS.items():
            if not wav_path.exists():
                print(f"  Skipping {name} — not found: {wav_path}")
                continue
            audio = load_wav(wav_path)
            if audio is not None and len(audio) / SR >= 34.0:
                clips[name] = audio
            elif audio is not None:
                print(f"  Skipping {name} — too short ({len(audio)/SR:.1f}s < 34s)")
    else:
        print("Source: LIBROSA + synthetic")
        for name in ["libri1", "trumpet"]:
            audio = load_librosa_clip(name)
            if audio is not None and len(audio) / SR >= 34.0:
                clips[name] = audio
        clips["synthetic_noise"] = synthetic_noise(40.0)

    if not clips:
        print("ERROR: No clips loaded. Check data/ directory or install librosa.")
        sys.exit(1)

    n_configs = len(safety_grid) * len(snr_grid)
    n_evals = n_configs * len(clips) * len(codecs)

    # Disk estimate: each WAV ~= duration_s * SR * 2 bytes (int16)
    max_dur = max(len(a) for a in clips.values()) / SR
    wav_bytes = int(max_dur * SR * 2)
    print(f"\nGrid: {len(safety_grid)} safety x {len(snr_grid)} snr = {n_configs} configs")
    print(f"Clips: {len(clips)} ({', '.join(clips.keys())})")
    print(f"Codecs: {len(codecs)} ({', '.join(codecs)})")
    print(f"Total evaluations: {n_evals}")
    print(f"Max WAV size: {wav_bytes / 1024 / 1024:.1f} MB")
    print(f"Temp disk peak: ~{wav_bytes * 2 / 1024 / 1024:.1f} MB (watermarked + 1 degraded)\n")

    output_dir = Path(__file__).resolve().parent / "output" / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[SweepRow] = []
    total = n_configs * len(clips)
    done = 0

    for safety in safety_grid:
        for snr in snr_grid:
            for clip_name, audio in clips.items():
                done += 1
                tag = f"[{done}/{total}] safety={safety:.0f} snr={snr:.0f} clip={clip_name}"
                t0 = time.time()

                with tempfile.TemporaryDirectory() as tmp_dir:
                    rows = evaluate_config(
                        clip_name, audio, safety, snr, codecs, tmp_dir,
                    )
                    all_rows.extend(rows)

                elapsed = time.time() - t0
                # Show quick summary for this config
                clean_row = next((r for r in rows if r.codec == "clean"), rows[0])
                worst_row = min(rows, key=lambda r: r.rs_margin)
                print(
                    f"  {tag}  "
                    f"seg-SNR={clean_row.seg_snr:.1f}dB  "
                    f"RS_ok={sum(r.rs_ok for r in rows)}/{len(rows)}  "
                    f"min_margin={worst_row.rs_margin}  "
                    f"({elapsed:.1f}s)"
                )

    # Write results
    csv_path = output_dir / "sweep_results.csv"
    write_csv(all_rows, csv_path)
    print(f"\nCSV: {csv_path}")

    # Find Pareto frontier
    pareto = find_pareto_configs(all_rows, top_n=args.top_n)
    print(f"Pareto-optimal configs: {pareto}")

    # Save WAVs only for Pareto configs
    if pareto:
        wav_dir = output_dir / "pareto_wavs"
        wav_dir.mkdir(exist_ok=True)
        for safety, snr in pareto:
            for clip_name, audio in clips.items():
                watermarked = embed_full(audio, wid_target_snr_db=snr,
                                         safety_margin_db=safety)
                fname = f"{clip_name}_s{safety:.0f}_snr{snr:.0f}.wav"
                sf.write(str(wav_dir / fname), watermarked, SR)
        print(f"Pareto WAVs: {wav_dir}/")

    # Write report
    report_path = output_dir / "report.md"
    write_report(all_rows, pareto, report_path)
    print(f"Report: {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
