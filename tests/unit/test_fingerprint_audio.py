"""
Audio fingerprint test suite.

Release gate (blocking):
  @pytest.mark.integration tests using speech_sample fixture.
  These MUST pass for Phase 3 to begin.

Informational (non-blocking):
  Tests using white_noise or multitone_dwt.
  Failures here are noted but do not block release.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.audio_signals import (
    white_noise, silence,
    add_babble_noise, add_pink_noise,
    simulate_mp3_compression, simulate_voip_codec,
)
from kernel_backend.engine.audio.fingerprint import extract_hashes, hamming_distance

SR = 44100
DURATION_S = 6.0
KEY_A = b"author-public-key-material-AAA"
KEY_B = b"author-public-key-material-BBB"
PEPPER = b"system-pepper-bytes-padded-32b!"


def _passing_rate(h_orig: list, h_noisy: list, threshold: int = 10) -> float:
    """Fraction of aligned segment pairs within Hamming threshold."""
    n = min(len(h_orig), len(h_noisy))
    if n == 0:
        return 0.0
    passing = sum(
        1 for i in range(n)
        if hamming_distance(h_orig[i].hash_hex, h_noisy[i].hash_hex) <= threshold
    )
    return passing / n


# ════════════════════════════════════════════════════════════════════════════════
# BLOCKING TESTS — Speech golden dataset
# All must pass before Phase 3 begins.
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
def test_speech_deterministic(speech_sample):
    """[BLOCKING] Same speech + same key → identical hashes."""
    name, audio = speech_sample
    h1 = extract_hashes(audio, SR, KEY_A, PEPPER)
    h2 = extract_hashes(audio, SR, KEY_A, PEPPER)
    for a, b in zip(h1, h2):
        assert a.hash_hex == b.hash_hex, f"[{name}] non-deterministic at {a.time_offset_ms}ms"


@pytest.mark.integration
def test_speech_keyed(speech_sample):
    """[BLOCKING] Same speech + different key → different hashes (keying works)."""
    name, audio = speech_sample
    h_a = extract_hashes(audio, SR, KEY_A, PEPPER)
    h_b = extract_hashes(audio, SR, KEY_B, PEPPER)
    assert any(a.hash_hex != b.hash_hex for a, b in zip(h_a, h_b)), \
        f"[{name}] key material has no effect"


@pytest.mark.integration
def test_speech_babble_noise_robustness(speech_sample):
    """
    [BLOCKING] Same speaker clean vs babble noise.
    Babble at 10dB SNR: >= 80% segments within Hamming 10.
    This is the hardest production scenario: crowded cafeteria.
    """
    name, audio = speech_sample
    noisy = add_babble_noise(audio, snr_db=10.0, seed=42)
    h_clean = extract_hashes(audio, SR, KEY_A, PEPPER)
    h_noisy = extract_hashes(noisy, SR, KEY_A, PEPPER)
    rate = _passing_rate(h_clean, h_noisy, threshold=10)
    assert rate >= 0.80, \
        f"[{name}] babble@10dB SNR: only {rate:.0%} segments within threshold"


@pytest.mark.integration
def test_speech_pink_noise_robustness(speech_sample):
    """
    [BLOCKING] Same speaker clean vs car/engine noise.
    Pink noise at 20dB SNR: >= 85% segments within Hamming 10.
    """
    name, audio = speech_sample
    noisy = add_pink_noise(audio, snr_db=20.0, seed=42)
    h_clean = extract_hashes(audio, SR, KEY_A, PEPPER)
    h_noisy = extract_hashes(noisy, SR, KEY_A, PEPPER)
    rate = _passing_rate(h_clean, h_noisy, threshold=10)
    assert rate >= 0.85, \
        f"[{name}] pink@20dB SNR: only {rate:.0%} segments within threshold"


@pytest.mark.integration
def test_speech_mp3_robustness(speech_sample):
    """
    [BLOCKING] Same speaker clean vs 32kbps MP3 simulation.
    >= 80% segments within Hamming 10.
    """
    name, audio = speech_sample
    compressed = simulate_mp3_compression(audio, SR, bitrate_kbps=32)
    h_clean = extract_hashes(audio, SR, KEY_A, PEPPER)
    h_comp  = extract_hashes(compressed, SR, KEY_A, PEPPER)
    rate = _passing_rate(h_clean, h_comp, threshold=10)
    assert rate >= 0.80, \
        f"[{name}] mp3@32kbps: only {rate:.0%} segments within threshold"


@pytest.mark.integration
def test_speech_voip_robustness(speech_sample):
    """
    [BLOCKING] Same speaker clean vs VoIP G.711 simulation.
    VoIP bandwidth-limits to 300-3400 Hz — harshest codec tier.
    >= 70% segments within Hamming 10 (lower bar due to extreme BW reduction).
    """
    name, audio = speech_sample
    voip = simulate_voip_codec(audio, SR)
    h_clean = extract_hashes(audio, SR, KEY_A, PEPPER)
    h_voip  = extract_hashes(voip, SR, KEY_A, PEPPER)
    rate = _passing_rate(h_clean, h_voip, threshold=10)
    assert rate >= 0.70, \
        f"[{name}] voip: only {rate:.0%} segments within threshold"


@pytest.mark.integration
def test_different_speakers_discriminated(golden_speech):
    """
    [BLOCKING] Different speakers → Hamming >= 20 for >= 80% of segment pairs.
    This is the anti-collision requirement: different content must not match.
    """
    names = list(golden_speech.keys())
    if len(names) < 2:
        pytest.skip("Need at least 2 speech recordings for discrimination test")

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            audio_a = golden_speech[name_a]
            audio_b = golden_speech[name_b]
            duration = min(len(audio_a), len(audio_b)) / SR
            a = audio_a[:int(duration * SR)]
            b = audio_b[:int(duration * SR)]

            h_a = extract_hashes(a, SR, KEY_A, PEPPER)
            h_b = extract_hashes(b, SR, KEY_A, PEPPER)
            n = min(len(h_a), len(h_b))
            dists = [hamming_distance(h_a[k].hash_hex, h_b[k].hash_hex)
                     for k in range(n)]
            high_dist = sum(1 for d in dists if d >= 20)
            rate = high_dist / n if n > 0 else 0.0
            assert rate >= 0.80, \
                f"[{name_a} vs {name_b}] only {rate:.0%} segments discriminated"


# ════════════════════════════════════════════════════════════════════════════════
# INFORMATIONAL TESTS — Synthetic baselines
# Non-blocking: failures noted but do not prevent Phase 3.
# ════════════════════════════════════════════════════════════════════════════════

def test_synthetic_discriminable_informational():
    """white_noise vs multitone_dwt spectral discrimination."""
    from tests.fixtures.audio_signals import multitone_dwt
    h_noise = extract_hashes(white_noise(6.0, SR), SR, KEY_A, PEPPER)
    h_tone  = extract_hashes(multitone_dwt(6.0, SR), SR, KEY_A, PEPPER)
    dists = [hamming_distance(a.hash_hex, b.hash_hex)
             for a, b in zip(h_noise, h_tone)]
    assert max(dists) > 10


# ════════════════════════════════════════════════════════════════════════════════
# STRUCTURAL TESTS — Always blocking regardless of signal type
# ════════════════════════════════════════════════════════════════════════════════

def test_deterministic_synthetic():
    """Determinism on white noise — validates no RNG state leaks."""
    sig = white_noise(6.0, SR, seed=42)
    h1 = extract_hashes(sig, SR, KEY_A, PEPPER)
    h2 = extract_hashes(sig, SR, KEY_A, PEPPER)
    for a, b in zip(h1, h2):
        assert a.hash_hex == b.hash_hex


def test_silence_does_not_crash():
    """RMS=0 must never raise. Output may be empty."""
    try:
        extract_hashes(silence(4.0, SR), SR, KEY_A, PEPPER)
    except Exception as e:
        pytest.fail(f"silence caused crash: {e}")


def test_overlap_produces_more_hashes():
    """50% overlap must produce more hashes than non-overlapping."""
    sig = white_noise(6.0, SR)
    h_overlap    = extract_hashes(sig, SR, KEY_A, PEPPER, overlap=0.5)
    h_no_overlap = extract_hashes(sig, SR, KEY_A, PEPPER, overlap=0.0)
    assert len(h_overlap) >= len(h_no_overlap) * 1.5


def test_keyed_synthetic():
    """Different key material must produce different hashes."""
    sig = white_noise(6.0, SR)
    h_a = extract_hashes(sig, SR, KEY_A, PEPPER)
    h_b = extract_hashes(sig, SR, KEY_B, PEPPER)
    assert any(a.hash_hex != b.hash_hex for a, b in zip(h_a, h_b))


def test_hamming_identical():
    """hamming_distance(h, h) == 0 for any hash."""
    hashes = extract_hashes(white_noise(DURATION_S, SR), SR, KEY_A, PEPPER)
    for fp in hashes:
        assert hamming_distance(fp.hash_hex, fp.hash_hex) == 0
