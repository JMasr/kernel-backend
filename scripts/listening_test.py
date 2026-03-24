#!/usr/bin/env python3
"""
Listening test — Phase 10.C.

Generates watermarked WAV files at multiple SNR levels for A/B comparison.
The goal: find the target_snr_db where the watermark becomes inaudible,
then check which levels still allow reliable detection.

Usage:
    uv run python scripts/listening_test.py                # librosa clips
    uv run python scripts/listening_test.py --polygon      # real polygon clips

Output goes to scripts/output/listening_test/ (gitignored).
"""

from __future__ import annotations

import hashlib
import hmac
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
CONTENT_ID = "listening-test-content"
PUBKEY = "listening-test-pubkey"
WID = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10"
SEGMENT_S = 2.0

SNR_LEVELS = [-6.0, -14.0, -20.0, -26.0, -30.0, -35.0, -40.0]


def _pn_seed(i: int) -> int:
    msg = f"wid|{CONTENT_ID}|{PUBKEY}|{i}".encode()
    return int.from_bytes(hmac.new(PEPPER, msg, hashlib.sha256).digest()[:8], "big")


def embed_full(audio: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Embed pilot + WID into audio at the given SNR level."""
    result = embed_pilot(
        audio, SR, HASH_48, PILOT_SEED,
        target_snr_db=target_snr_db,
        perceptual_shaping=True,
        temporal_shaping=True,
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
            target_snr_db=target_snr_db,
            perceptual_shaping=True,
            temporal_shaping=True,
        )

    return result


def check_detection(watermarked: np.ndarray) -> dict:
    """Check pilot and WID detection, return results dict."""
    # Pilot
    detected = detect_pilot(watermarked, SR, PILOT_SEED)
    pilot_ok = detected == HASH_48

    # WID
    seg_len = int(SR * SEGMENT_S)
    n_segments = len(watermarked) // seg_len
    wid_ok = False
    n_erasures = 0
    mean_conf = 0.0

    if n_segments >= 17:
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

    return {
        "pilot_ok": pilot_ok,
        "wid_ok": wid_ok,
        "n_erasures": n_erasures,
        "n_segments": n_segments,
        "mean_conf": mean_conf,
    }


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
            continue
        snrs.append(10.0 * np.log10(sig_power / noise_power))
    return float(np.mean(snrs)) if snrs else float("inf")


def process_clip(name: str, audio: np.ndarray, output_dir: Path) -> None:
    """Process one clip at all SNR levels."""
    print(f"\n{'='*75}")
    print(f"  Clip: {name}  ({len(audio)/SR:.1f}s, {len(audio)} samples)")
    print(f"{'='*75}")

    # Save original
    sf.write(str(output_dir / f"{name}_original.wav"), audio, SR)

    # Results table
    results = []
    for snr in SNR_LEVELS:
        print(f"  Embedding at {snr:+.0f} dB ...", end=" ", flush=True)
        wm = embed_full(audio, target_snr_db=snr)
        det = check_detection(wm)
        seg_snr = segmental_snr(audio, wm)
        results.append((snr, det, seg_snr))

        # Save watermarked
        sf.write(str(output_dir / f"{name}_{snr:+.0f}dB_watermarked.wav"), wm, SR)

        # Save amplified difference (×20 for easy listening)
        diff = (wm - audio) * 20.0
        sf.write(str(output_dir / f"{name}_{snr:+.0f}dB_difference_x20.wav"), diff, SR)
        print("done")

    # Print results table
    print(f"\n  {'SNR (dB)':>10}  {'Pilot':>7}  {'WID':>7}  {'MeanConf':>10}  "
          f"{'Erasures':>10}  {'SegSNR':>10}")
    print(f"  {'-'*62}")
    for snr, det, seg_snr in results:
        pilot_str = "OK" if det["pilot_ok"] else "FAIL"
        wid_str = "OK" if det["wid_ok"] else "FAIL"
        era_str = f"{det['n_erasures']}/{det['n_segments']}"
        print(f"  {snr:>+10.1f}  {pilot_str:>7}  {wid_str:>7}  "
              f"{det['mean_conf']:>10.4f}  {era_str:>10}  {seg_snr:>+10.1f}")

    print(f"\n  WAV files saved to {output_dir}/")
    print(f"  Listen to {{name}}_{{snr}}dB_watermarked.wav vs {{name}}_original.wav")
    print(f"  The _difference_x20.wav files isolate the watermark signal (amplified 20×)")


def load_audio(name: str) -> np.ndarray | None:
    """Try to load a librosa example audio clip."""
    try:
        import librosa
        audio, _ = librosa.load(librosa.ex(name), sr=SR, mono=True)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"  Could not load '{name}': {e}")
        return None


def load_wav(path: str | Path) -> np.ndarray | None:
    """Load a WAV file from disk, resample to SR, mono."""
    try:
        audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != SR:
            from scipy.signal import resample
            n_target = int(len(audio) * SR / file_sr)
            audio = resample(audio, n_target).astype(np.float32)
        return audio
    except Exception as e:
        print(f"  Could not load '{path}': {e}")
        return None


POLYGON_DIR = Path(__file__).resolve().parent.parent / "data"
POLYGON_CLIPS = {
    "brahms_piano":   POLYGON_DIR / "audio" / "music" / "brahms_piano_01.wav",
    "vibeace":        POLYGON_DIR / "audio" / "music" / "vibeace_01.wav",
    "choice_hiphop":  POLYGON_DIR / "audio" / "speech" / "choice_hiphop_01.wav",
    "libri_male":     POLYGON_DIR / "audio" / "speech" / "libri_male_01.wav",
    "libri_female":   POLYGON_DIR / "audio" / "speech" / "libri_female_01.wav",
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Listening test: find the inaudibility threshold for watermark SNR"
    )
    parser.add_argument(
        "--polygon", action="store_true",
        help="Use real polygon WAV clips from data/",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "output" / "listening_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Phase 10.C — Listening Test")
    print(f"  SNR levels: {SNR_LEVELS}")
    print(f"  Perceptual shaping: ON (Phase 10.B)")

    if args.polygon:
        print(f"  Source: POLYGON clips from data/")
        for name, wav_path in POLYGON_CLIPS.items():
            if not wav_path.exists():
                print(f"  Skipping {name} — not found: {wav_path}")
                continue
            audio = load_wav(wav_path)
            if audio is not None:
                process_clip(name, audio, output_dir)
    else:
        print(f"  Source: LIBROSA built-in clips")
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
            process_clip(name, audio, output_dir)

    print(f"\n{'='*75}")
    print(f"  Output: {output_dir}/")
    print(f"  Instructions:")
    print(f"    1. Open the _original.wav and _watermarked.wav files in an audio player")
    print(f"    2. A/B compare at each SNR level — find where you stop hearing the watermark")
    print(f"    3. Check the detection table above — note which levels still detect")
    print(f"    4. Report the threshold SNR where watermark is inaudible")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
