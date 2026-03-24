#!/usr/bin/env python3
"""
Perceptual validation script — Phase 10.B.

Compares watermarked audio quality across embedding modes:
  - Flat:          No perceptual shaping (legacy)
  - Spectral:      Watson's masking gain only (Phase 10.A)
  - Temporal:      Full temporal JND + silence gate (Phase 10.B)

Generates:
  1. Objective metrics (seg-SNR, spectral distortion, max abs diff)
  2. WAV files for manual A/B listening comparison
  3. Detection verification (pilot + WID roundtrip)

Usage:
    uv run python scripts/perceptual_validation.py                    # librosa clips, 3-way
    uv run python scripts/perceptual_validation.py --polygon          # polygon clips, 3-way
    uv run python scripts/perceptual_validation.py --polygon --temporal  # polygon, temporal only

Output goes to scripts/output/ (gitignored).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kernel_backend.core.domain.watermark import BandConfig
from kernel_backend.engine.audio.pilot_tone import detect_pilot, embed_pilot
from kernel_backend.engine.audio.wid_beacon import embed_segment, extract_symbol_segment
from kernel_backend.engine.codec.hopping import plan_audio_hopping
from kernel_backend.engine.codec.reed_solomon import ReedSolomonCodec

SR = 44100
HASH_48 = 0xABCDEF012345
PILOT_SEED = 0xDEADBEEFCAFEBABE
PEPPER = b"validation-pepper-32-bytes-pad!!"
CONTENT_ID = "validation-content-id"
PUBKEY = "validation-public-key"
WID = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10"
SEGMENT_S = 2.0
TARGET_SNR_DB = -6.0  # AV mode — worst case for audibility


def _pn_seed(i: int) -> int:
    msg = f"wid|{CONTENT_ID}|{PUBKEY}|{i}".encode()
    return int.from_bytes(hmac.new(PEPPER, msg, hashlib.sha256).digest()[:8], "big")


# ── Metrics ───────────────────────────────────────────────────────────────────


def segmental_snr(original: np.ndarray, watermarked: np.ndarray,
                  frame_ms: float = 20.0) -> float:
    """Mean segmental SNR in dB (20ms frames, skip silent frames)."""
    frame_len = int(SR * frame_ms / 1000.0)
    n_frames = len(original) // frame_len
    snrs = []
    for i in range(n_frames):
        s = i * frame_len
        e = s + frame_len
        sig = original[s:e]
        noise = watermarked[s:e] - sig
        sig_power = float(np.mean(sig ** 2))
        noise_power = float(np.mean(noise ** 2))
        if sig_power < 1e-10 or noise_power < 1e-10:
            continue  # skip silent frames
        snrs.append(10.0 * np.log10(sig_power / noise_power))
    return float(np.mean(snrs)) if snrs else float("inf")


def spectral_distortion(original: np.ndarray, watermarked: np.ndarray,
                        n_fft: int = 2048) -> float:
    """Mean-squared log-spectral distance in dB^2."""
    from scipy.signal import stft
    _, _, Zo = stft(original, fs=SR, nperseg=n_fft, noverlap=n_fft // 2)
    _, _, Zw = stft(watermarked, fs=SR, nperseg=n_fft, noverlap=n_fft // 2)
    Po = np.abs(Zo) ** 2 + 1e-10
    Pw = np.abs(Zw) ** 2 + 1e-10
    lsd = (10.0 * np.log10(Po) - 10.0 * np.log10(Pw)) ** 2
    return float(np.mean(lsd))


def max_abs_diff(original: np.ndarray, watermarked: np.ndarray) -> float:
    return float(np.max(np.abs(watermarked - original)))


# ── Embedding helpers ─────────────────────────────────────────────────────────


def embed_full(audio: np.ndarray, perceptual_shaping: bool,
               temporal_shaping: bool = False,
               target_snr_db: float = -6.0) -> np.ndarray:
    """Embed pilot + WID into audio, return watermarked audio."""
    # Step 1: Embed pilot
    result = embed_pilot(
        audio, SR, HASH_48, PILOT_SEED,
        target_snr_db=target_snr_db,
        perceptual_shaping=perceptual_shaping,
        temporal_shaping=temporal_shaping,
    )

    # Step 2: Embed WID segments
    seg_len = int(SR * SEGMENT_S)
    n_segments = len(audio) // seg_len
    if n_segments < 17:
        print(f"  Warning: only {n_segments} segments (need ≥17 for RS)")
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
            target_snr_db=target_snr_db,
            perceptual_shaping=perceptual_shaping,
            temporal_shaping=temporal_shaping,
        )

    return result


def verify_detection(watermarked: np.ndarray, label: str) -> bool:
    """Verify pilot and WID detection on watermarked audio."""
    ok = True

    # Pilot
    detected = detect_pilot(watermarked, SR, PILOT_SEED)
    if detected == HASH_48:
        print(f"  {label}: Pilot DETECTED (hash match)")
    else:
        print(f"  {label}: Pilot FAILED (got {detected})")
        ok = False

    # WID
    seg_len = int(SR * SEGMENT_S)
    n_segments = len(watermarked) // seg_len
    if n_segments < 17:
        print(f"  {label}: WID SKIPPED (too short)")
        return ok

    codec = ReedSolomonCodec(n_symbols=n_segments)
    band_configs = plan_audio_hopping(n_segments, CONTENT_ID, PUBKEY, PEPPER)
    seeds = [_pn_seed(i) for i in range(n_segments)]

    recovered: list[int | None] = []
    confidences: list[float] = []
    for i in range(n_segments):
        s = i * seg_len
        e = s + seg_len
        sym, conf = extract_symbol_segment(watermarked[s:e], band_configs[i], seeds[i])
        recovered.append(sym)
        confidences.append(conf)

    n_erasures = sum(1 for s in recovered if s is None)
    mean_conf = float(np.mean(confidences))
    try:
        decoded = codec.decode(recovered)
        wid_ok = decoded == WID
    except Exception:
        wid_ok = False

    status = "RECOVERED" if wid_ok else "FAILED"
    print(f"  {label}: WID {status} (erasures={n_erasures}/{n_segments}, "
          f"mean_conf={mean_conf:.3f})")
    if not wid_ok:
        ok = False

    return ok


# ── Main ──────────────────────────────────────────────────────────────────────


def load_audio(name: str) -> np.ndarray | None:
    """Try to load a librosa example audio clip."""
    try:
        import librosa
        audio, _ = librosa.load(librosa.ex(name), sr=SR, mono=True)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"  Could not load '{name}': {e}")
        return None


def process_clip(name: str, audio: np.ndarray, output_dir: Path,
                 target_snr_db: float = -6.0,
                 modes: list[str] | None = None) -> None:
    """Process a single audio clip across embedding modes, compute metrics, save WAVs.

    modes: subset of ["flat", "spectral", "temporal"]. Default: all three.
    """
    if modes is None:
        modes = ["flat", "spectral", "temporal"]

    print(f"\n{'='*70}")
    print(f"  Clip: {name}  ({len(audio)/SR:.1f}s, {len(audio)} samples)")
    print(f"  target_snr_db = {target_snr_db} dB   modes = {modes}")
    print(f"{'='*70}")

    # Embed each mode
    embedded: dict[str, np.ndarray] = {}
    for mode in modes:
        if mode == "flat":
            embedded[mode] = embed_full(audio, perceptual_shaping=False,
                                        target_snr_db=target_snr_db)
        elif mode == "spectral":
            embedded[mode] = embed_full(audio, perceptual_shaping=True,
                                        temporal_shaping=False,
                                        target_snr_db=target_snr_db)
        elif mode == "temporal":
            embedded[mode] = embed_full(audio, perceptual_shaping=True,
                                        temporal_shaping=True,
                                        target_snr_db=target_snr_db)

    # Metrics table
    metrics: dict[str, dict[str, float]] = {}
    for mode, wm in embedded.items():
        metrics[mode] = {
            "seg_snr": segmental_snr(audio, wm),
            "spectral_dist": spectral_distortion(audio, wm),
            "max_abs_diff": max_abs_diff(audio, wm),
        }

    header = f"  {'Metric':<30}"
    for mode in modes:
        header += f" {mode:>12}"
    if len(modes) >= 2:
        header += f" {'Δ(last-first)':>14}"
    print(f"\n{header}")
    print(f"  {'-'*(30 + 13*len(modes) + (15 if len(modes) >= 2 else 0))}")

    for label, key in [("Seg-SNR (dB) ↑", "seg_snr"),
                       ("Spectral Distortion (dB²) ↓", "spectral_dist"),
                       ("Max Abs Diff ↓", "max_abs_diff")]:
        row = f"  {label:<30}"
        fmt = ".2f" if key == "seg_snr" else ".4f" if key == "spectral_dist" else ".6f"
        for mode in modes:
            row += f" {metrics[mode][key]:>12{fmt}}"
        if len(modes) >= 2:
            delta = metrics[modes[-1]][key] - metrics[modes[0]][key]
            row += f" {delta:>+14{fmt}}"
        print(row)

    # Detection verification
    print()
    all_ok = True
    for mode, wm in embedded.items():
        ok = verify_detection(wm, mode.capitalize())
        if not ok:
            all_ok = False

    if not all_ok:
        print(f"\n  *** DETECTION FAILURE — check parameters ***")

    # Save WAV files
    prefix = output_dir / name
    sf.write(f"{prefix}_original.wav", audio, SR)
    for mode, wm in embedded.items():
        sf.write(f"{prefix}_{mode}_watermark.wav", wm, SR)
        diff = (wm - audio) * 10.0
        sf.write(f"{prefix}_difference_{mode}_x10.wav", diff, SR)

    print(f"\n  WAV files saved to {output_dir}/")


def load_wav(path: str | Path) -> np.ndarray | None:
    """Load a WAV file from disk, resample to SR, mono."""
    import soundfile as _sf

    try:
        audio, file_sr = _sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != SR:
            # Simple resample via scipy
            from scipy.signal import resample

            n_target = int(len(audio) * SR / file_sr)
            audio = resample(audio, n_target).astype(np.float32)
        return audio
    except Exception as e:
        print(f"  Could not load '{path}': {e}")
        return None


# Polygon WAV files — real-world clips from data/
POLYGON_DIR = Path(__file__).resolve().parent.parent / "data"
POLYGON_CLIPS = {
    "brahms_piano":     POLYGON_DIR / "audio" / "music" / "brahms_piano_01.wav",
    "vibeace":          POLYGON_DIR / "audio" / "music" / "vibeace_01.wav",
    "choice_hiphop":    POLYGON_DIR / "audio" / "speech" / "choice_hiphop_01.wav",
    "libri_male":       POLYGON_DIR / "audio" / "speech" / "libri_male_01.wav",
    "libri_female":     POLYGON_DIR / "audio" / "speech" / "libri_female_01.wav",
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Perceptual validation: flat vs spectral vs temporal watermark")
    parser.add_argument(
        "--polygon", action="store_true",
        help="Use real polygon WAV clips from data/ (brahms, vibeace, choice, libri)."
             " Without this flag, uses librosa built-in clips.",
    )
    parser.add_argument(
        "--temporal", action="store_true",
        help="Compare only temporal (Phase 10.B) mode against flat."
             " Without this flag, compares all three modes (flat, spectral, temporal).",
    )
    parser.add_argument(
        "--snr", type=float, default=TARGET_SNR_DB,
        help=f"target_snr_db for embedding (default: {TARGET_SNR_DB})",
    )
    args = parser.parse_args()

    snr = args.snr
    modes = ["flat", "temporal"] if args.temporal else ["flat", "spectral", "temporal"]
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)

    print(f"\n  Modes: {modes}")

    if args.polygon:
        print("  Source: POLYGON (real-world clips from data/)")
        for name, wav_path in POLYGON_CLIPS.items():
            if not wav_path.exists():
                print(f"  Skipping {name} — file not found: {wav_path}")
                continue
            audio = load_wav(wav_path)
            if audio is not None:
                process_clip(name, audio, output_dir, target_snr_db=snr, modes=modes)
    else:
        print("  Source: LIBROSA (built-in clips)")
        clips = ["libri1", "trumpet"]
        fallback_used = False
        for name in clips:
            audio = load_audio(name)
            if audio is None:
                if not fallback_used:
                    print(f"\n  Falling back to synthetic white noise (40s)")
                    rng = np.random.default_rng(42)
                    audio = rng.standard_normal(SR * 40).astype(np.float32) * 0.3
                    name = "synthetic_noise"
                    fallback_used = True
                else:
                    continue
            process_clip(name, audio, output_dir, target_snr_db=snr, modes=modes)

    print(f"\n{'='*70}")
    print(f"  Output WAV files in: {output_dir}/")
    print(f"  To clean up:  rm -rf {output_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
