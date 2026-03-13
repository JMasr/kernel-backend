"""
Audio test signal golden dataset for Kernel Security.

Target domain: SPEECH and music (real-world production content).
Synthetic signals are retained as informational baselines only.

Golden Dataset tiers:
  Tier 1 — Clean speech: high-quality librosa recordings
  Tier 2 — Degraded speech: babble noise and pink noise injection
  Tier 3 — Transcoded speech: MP3-like compression simulation

Test gate (blocking for release):
  - Same speaker clean vs noisy: hamming <= 10 for >= 80% of segments
  - Different speakers: hamming >= 20 for >= 80% of segment pairs

Synthetic signals (white_noise, multitone_dwt):
  - Retained for DWT subband coverage verification
  - NOT blocking if speech tests pass
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

SR = 44100

# ── Tier 1: Clean speech (librosa built-in) ───────────────────────────────────

# Available librosa speech/vocal recordings
SPEECH_RECORDINGS = {
    "libri1":   "LibriSpeech: male speech, clear diction, studio quality",
    "libri2":   "LibriSpeech: female speech, clear diction, studio quality",
    "choice":   "Hip-hop vocal: mixed voice + music, common production case",
}

# Music recordings retained for non-speech content tests
MUSIC_RECORDINGS = {
    "brahms":  "Piano: percussive attack + sustain, non-stationary",
    "trumpet": "Brass: fundamental + harmonics, quasi-periodic",
    "vibeace": "Vibraphone + ambience: low sustained energy, edge case",
}

ALL_LIBROSA_RECORDINGS = {**SPEECH_RECORDINGS, **MUSIC_RECORDINGS}


def load_librosa(name: str, sr: int = SR) -> np.ndarray:
    """Load a librosa built-in recording. Raises ImportError if not installed."""
    import librosa
    audio, _ = librosa.load(librosa.ex(name), sr=sr, mono=True)
    return audio.astype(np.float32)


# ── Tier 2: Noise-degraded speech ────────────────────────────────────────────

def add_babble_noise(
    speech: np.ndarray,
    snr_db: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Babble noise simulation: sum of multiple pink noise sources
    at random phases, approximating restaurant/crowd background.
    Spectrally similar to speech — the hardest masker for speech identity.
    """
    rng = np.random.default_rng(seed)
    n = len(speech)
    # Babble = sum of 6 pink noise sources (different seeds)
    babble = np.zeros(n, dtype=np.float32)
    for i in range(6):
        white = rng.standard_normal(n).astype(np.float32)
        f = np.fft.rfftfreq(n)
        f[0] = 1.0
        pink = np.fft.irfft(np.fft.rfft(white) / np.sqrt(f), n=n).astype(np.float32)
        peak = np.max(np.abs(pink))
        if peak > 0:
            pink /= peak
        babble += pink * rng.uniform(0.5, 1.0)

    speech_rms = max(float(np.sqrt(np.mean(speech ** 2))), 1e-6)
    babble_rms = max(float(np.sqrt(np.mean(babble ** 2))), 1e-6)
    target_noise_rms = speech_rms * (10.0 ** (-snr_db / 20.0))
    babble = babble * (target_noise_rms / babble_rms)
    return np.clip(speech + babble, -1.0, 1.0)


def add_pink_noise(
    speech: np.ndarray,
    snr_db: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Pink noise injection — simulates car/engine/HVAC background.
    Less speech-like than babble, but dominant in mobile scenarios.
    """
    rng = np.random.default_rng(seed)
    n = len(speech)
    white = rng.standard_normal(n).astype(np.float32)
    f = np.fft.rfftfreq(n)
    f[0] = 1.0
    pink = np.fft.irfft(np.fft.rfft(white) / np.sqrt(f), n=n).astype(np.float32)
    pink_rms = max(float(np.sqrt(np.mean(pink ** 2))), 1e-6)
    speech_rms = max(float(np.sqrt(np.mean(speech ** 2))), 1e-6)
    target = speech_rms * (10.0 ** (-snr_db / 20.0))
    pink = pink * (target / pink_rms)
    return np.clip(speech + pink, -1.0, 1.0)


# ── Tier 3: Transcoded speech (codec simulation) ─────────────────────────────

def simulate_mp3_compression(
    speech: np.ndarray,
    sr: int = SR,
    bitrate_kbps: int = 32,
) -> np.ndarray:
    """
    Simulates low-bitrate MP3 compression via frequency band limiting
    and mild quantization noise.

    At 32kbps, MP3 cuts frequencies above ~11kHz and introduces
    pre-echo artifacts. We model this as:
      1. Low-pass filter at cutoff = min(11000, sr/2 * 0.95) Hz
      2. Add quantization noise at -40dB

    Does not require ffmpeg or an actual MP3 encoder.
    Deterministic. Suitable for unit tests.
    """
    from scipy import signal as scipy_signal
    # Low-pass filter: simulate codec bandwidth limiting
    cutoff_hz = min(11000.0, sr / 2.0 * 0.95)
    nyquist = sr / 2.0
    sos = scipy_signal.butter(
        8, cutoff_hz / nyquist, btype='low', output='sos'
    )
    filtered = scipy_signal.sosfilt(sos, speech).astype(np.float32)

    # Quantization noise at -40dB
    rng = np.random.default_rng(0)
    signal_rms = max(float(np.sqrt(np.mean(filtered ** 2))), 1e-6)
    quant_noise = (rng.standard_normal(len(filtered)).astype(np.float32)
                   * signal_rms * 10 ** (-40.0 / 20.0))
    return np.clip(filtered + quant_noise, -1.0, 1.0)


def simulate_voip_codec(
    speech: np.ndarray,
    sr: int = SR,
) -> np.ndarray:
    """
    Simulates G.711/G.729 VoIP codec:
      - Bandwidth limit to 300–3400 Hz (telephone band)
      - 8-bit mu-law quantization and de-quantization
      - Mild additive noise from packet jitter

    This is the most aggressive degradation tier.
    """
    from scipy import signal as scipy_signal

    # Bandpass: 300–3400 Hz telephone band
    nyquist = sr / 2.0
    sos = scipy_signal.butter(
        4,
        [300.0 / nyquist, 3400.0 / nyquist],
        btype='band',
        output='sos',
    )
    filtered = scipy_signal.sosfilt(sos, speech).astype(np.float32)

    # mu-law quantization (8-bit, 256 levels)
    mu = 255.0
    compressed = (np.sign(filtered)
                  * np.log1p(mu * np.abs(filtered))
                  / np.log1p(mu))
    quantized = np.round(compressed * 128) / 128.0
    decompressed = (np.sign(quantized)
                    * (np.exp(np.abs(quantized) * np.log1p(mu)) - 1.0)
                    / mu).astype(np.float32)

    # Jitter noise at -45dB
    rng = np.random.default_rng(1)
    rms = max(float(np.sqrt(np.mean(decompressed ** 2))), 1e-6)
    jitter = (rng.standard_normal(len(decompressed)).astype(np.float32)
              * rms * 10 ** (-45.0 / 20.0))
    return np.clip(decompressed + jitter, -1.0, 1.0)


# ── Synthetic baselines (informational only — not release-blocking) ───────────

def white_noise(
    duration_s: float = 5.0,
    sample_rate: int = SR,
    seed: int = 42,
) -> np.ndarray:
    """Flat spectrum. Used for DWT subband coverage verification only."""
    return (np.random.default_rng(seed)
            .standard_normal(int(duration_s * sample_rate))
            .astype(np.float32) * 0.3)

noise = white_noise  # alias


def pink_noise(
    duration_s: float = 5.0,
    sample_rate: int = SR,
    seed: int = 42,
) -> np.ndarray:
    """
    1/f spectrum. Energy decreases with frequency.
    Retained for DWT watermarking tests (pilot_tone, wid_beacon).
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    white = rng.standard_normal(n).astype(np.float32)
    f = np.fft.rfftfreq(n)
    f[0] = 1.0
    result = np.fft.irfft(
        np.fft.rfft(white) / np.sqrt(f), n=n
    ).astype(np.float32)
    peak = np.max(np.abs(result))
    return result / peak * 0.5 if peak > 0 else result


def multitone_dwt(
    duration_s: float = 5.0,
    sample_rate: int = SR,
    seed: int = 42,  # unused, kept for uniform signature
) -> np.ndarray:
    """Covers all db4 level-2 DWT subbands. Regression test for DWT params only."""
    n = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n, endpoint=False, dtype=np.float32)
    return (np.sin(2*np.pi*200*t) + np.sin(2*np.pi*3500*t) +
            np.sin(2*np.pi*7000*t) + np.sin(2*np.pi*15000*t)).astype(np.float32) * 0.25


def silence(
    duration_s: float = 5.0,
    sample_rate: int = SR,
    seed: int = 42,
) -> np.ndarray:
    """Edge case: RMS=0. System must not crash."""
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)
