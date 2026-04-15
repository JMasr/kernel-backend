"""
Subprocess-side fixture helpers.

Imported by worker functions running inside ``multiprocessing`` children.
Kept out of conftest.py so they can be safely imported under spawn.
"""
from __future__ import annotations

import secrets
import subprocess
import tempfile
from pathlib import Path

# 22 MB checked-in fixture used by the AV signing benchmark.
SPEECH_AV_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "video" / "speech" / "speech.mp4"
)

# Single canonical pepper for sign benchmarks. Padded to 32 bytes.
BENCH_PEPPER = b"benchmark-pepper-padded-to-32!!!"


def make_synthetic_audio_120s(out_dir: Path | None = None) -> Path:
    """Generate a 120 s mono WAV of broadband noise via ffmpeg.

    Matches the synthetic_audio_120s fixture in tests/unit/test_pipeline_sign_verify.py.
    Returns the path to the generated file.
    """
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="bench_audio_"))
    out = out_dir / "audio_120s.wav"
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


def make_keypair_and_cert():
    """Generate a fresh RSA keypair + Certificate for signing."""
    from kernel_backend.core.domain.identity import Certificate
    from kernel_backend.core.services.crypto_service import generate_keypair

    private_pem, public_pem = generate_keypair()
    cert = Certificate(
        author_id="benchmark-author",
        name="Benchmark Author",
        institution="Bench Org",
        public_key_pem=public_pem,
        created_at="2026-01-01T00:00:00+00:00",
    )
    return private_pem, public_pem, cert


def random_pepper() -> bytes:
    """A fresh 32-byte pepper, distinct from BENCH_PEPPER."""
    return secrets.token_bytes(32)
