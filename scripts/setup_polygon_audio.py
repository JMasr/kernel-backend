"""
Exports librosa example recordings to data/audio/ as WAV files.
Run once before using polygon audio tests:

    uv run python scripts/setup_polygon_audio.py

After running, copy the printed sha256 and duration_s values
into data/manifest.yaml.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

try:
    import librosa
except ImportError:
    sys.exit("librosa not installed. Run: uv add --dev librosa")

DATA_ROOT = Path(__file__).parent.parent / "data"
SR = 44100

# (librosa_name, relative_output_path, manifest_id)
EXPORTS: list[tuple[str, str, str]] = [
    ("libri1",  "audio/speech/libri_male_01.wav",    "libri_male_01"),
    ("libri2",  "audio/speech/libri_female_01.wav",  "libri_female_01"),
    ("choice",  "audio/speech/choice_hiphop_01.wav", "choice_hiphop_01"),
    ("brahms",  "audio/music/brahms_piano_01.wav",   "brahms_piano_01"),
    ("trumpet", "audio/music/trumpet_01.wav",        "trumpet_01"),
    ("vibeace", "audio/music/vibeace_01.wav",        "vibeace_01"),
]


def export_clip(
    librosa_name: str,
    dest: Path,
    manifest_id: str,
    sr: int = SR,
) -> None:
    if dest.exists():
        size_kb = dest.stat().st_size // 1024
        sha = hashlib.sha256(dest.read_bytes()).hexdigest()
        print(f"  skip  {dest.relative_to(DATA_ROOT)}  ({size_kb} KB, already exists)")
        print(f"        sha256: {sha}")
        print(f"        manifest id: {manifest_id}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  load  {librosa_name} from librosa...", end="", flush=True)
    audio, _ = librosa.load(librosa.ex(librosa_name), sr=sr, mono=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(str(dest), sr, pcm)
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    size_kb = dest.stat().st_size // 1024
    duration_s = round(len(audio) / sr, 2)
    print(f" done ({size_kb} KB, {duration_s}s)")
    print(f"        sha256: {sha}")
    print(f"        duration_s: {duration_s}")
    print(f"        manifest id: {manifest_id}")


def main() -> None:
    print(f"Exporting librosa recordings to {DATA_ROOT}/\n")
    for librosa_name, rel_path, manifest_id in EXPORTS:
        export_clip(librosa_name, DATA_ROOT / rel_path, manifest_id)
    print(
        "\nDone. Copy the sha256 and duration_s values "
        "printed above into data/manifest.yaml."
    )


if __name__ == "__main__":
    main()
