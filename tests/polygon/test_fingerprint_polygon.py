"""
Polygon audio fingerprint release gate.

Uses real WAV clips from data/audio/ (materialized by setup_polygon_audio.py).
All tests are marked @pytest.mark.polygon and are excluded from CI by default.

Run:
    pytest tests/ -m "polygon" -v
    pytest tests/ -m "polygon" -k "speech" -v   # blocking speech only
    pytest tests/ -m "polygon" -k "music" -v    # informational music only

Blocking categories (must pass for Phase 3):
    audio.speech — libri_male_01, libri_female_01, choice_hiphop_01

Informational categories (failures noted, do not block):
    audio.music — brahms_piano_01, trumpet_01, vibeace_01
"""
from __future__ import annotations

import pytest

from kernel_backend.engine.audio.fingerprint import extract_hashes, hamming_distance
from tests.fixtures.audio_signals import (
    add_babble_noise,
    add_pink_noise,
    simulate_mp3_compression,
    simulate_voip_codec,
)
from tests.fixtures.polygon.conftest import load_polygon_audio
from tests.fixtures.polygon.registry import DatasetRegistry

KEY_A = b"author-public-key-material-AAA"
KEY_B = b"author-public-key-material-BBB"
PEPPER = b"system-pepper-bytes-padded-32b!"


def _pass_rate(h_orig: list, h_noisy: list, threshold: int = 10) -> float:
    """Fraction of aligned segment pairs whose Hamming distance is within threshold."""
    n = min(len(h_orig), len(h_noisy))
    if n == 0:
        return 0.0
    return sum(
        1 for i in range(n)
        if hamming_distance(h_orig[i].hash_hex, h_noisy[i].hash_hex) <= threshold
    ) / n


# ════════════════════════════════════════════════════════════════════════════════
# SPEECH — Blocking release gate (audio.speech category)
# Mirrors test_fingerprint_audio.py but runs against real WAV files.
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.polygon
def test_polygon_speech_deterministic(speech_clip):
    """[POLYGON/BLOCKING] Real WAV: same clip + same key → identical hashes on every run."""
    samples, sr = load_polygon_audio(speech_clip)
    h1 = extract_hashes(samples, sr, KEY_A, PEPPER)
    h2 = extract_hashes(samples, sr, KEY_A, PEPPER)
    for a, b in zip(h1, h2):
        assert a.hash_hex == b.hash_hex, (
            f"[{speech_clip.id}] non-deterministic at {a.time_offset_ms} ms"
        )


@pytest.mark.polygon
def test_polygon_speech_keyed(speech_clip):
    """[POLYGON/BLOCKING] Real WAV: different key material → different hashes."""
    samples, sr = load_polygon_audio(speech_clip)
    h_a = extract_hashes(samples, sr, KEY_A, PEPPER)
    h_b = extract_hashes(samples, sr, KEY_B, PEPPER)
    assert any(a.hash_hex != b.hash_hex for a, b in zip(h_a, h_b)), (
        f"[{speech_clip.id}] key material has no effect on real clip"
    )


@pytest.mark.polygon
def test_polygon_speech_babble_robustness(speech_clip):
    """
    [POLYGON/BLOCKING] Real WAV vs babble at 10 dB SNR.
    Release gate: ≥ 80% of aligned segment pairs within Hamming 10.
    Hardest production scenario: crowded restaurant / cafeteria.
    """
    samples, sr = load_polygon_audio(speech_clip)
    noisy = add_babble_noise(samples, snr_db=10.0, seed=42)
    h_clean = extract_hashes(samples, sr, KEY_A, PEPPER)
    h_noisy = extract_hashes(noisy, sr, KEY_A, PEPPER)
    rate = _pass_rate(h_clean, h_noisy)
    assert rate >= 0.80, f"[{speech_clip.id}] babble@10dB: {rate:.0%} < 80%"


@pytest.mark.polygon
def test_polygon_speech_pink_robustness(speech_clip):
    """
    [POLYGON/BLOCKING] Real WAV vs pink noise at 20 dB SNR.
    Release gate: ≥ 85% of aligned segment pairs within Hamming 10.
    Models car engine / HVAC background noise.
    """
    samples, sr = load_polygon_audio(speech_clip)
    noisy = add_pink_noise(samples, snr_db=20.0, seed=42)
    h_clean = extract_hashes(samples, sr, KEY_A, PEPPER)
    h_noisy = extract_hashes(noisy, sr, KEY_A, PEPPER)
    rate = _pass_rate(h_clean, h_noisy)
    assert rate >= 0.85, f"[{speech_clip.id}] pink@20dB: {rate:.0%} < 85%"


@pytest.mark.polygon
def test_polygon_speech_mp3_robustness(speech_clip):
    """
    [POLYGON/BLOCKING] Real WAV vs simulated MP3 at 32 kbps.
    Release gate: ≥ 80% of aligned segment pairs within Hamming 10.
    Models mobile / streaming distribution.
    """
    samples, sr = load_polygon_audio(speech_clip)
    compressed = simulate_mp3_compression(samples, sr, bitrate_kbps=32)
    h_clean = extract_hashes(samples, sr, KEY_A, PEPPER)
    h_comp = extract_hashes(compressed, sr, KEY_A, PEPPER)
    rate = _pass_rate(h_clean, h_comp)
    assert rate >= 0.80, f"[{speech_clip.id}] mp3@32kbps: {rate:.0%} < 80%"


@pytest.mark.polygon
def test_polygon_speech_voip_robustness(speech_clip):
    """
    [POLYGON/BLOCKING] Real WAV vs VoIP G.711 simulation (300–3400 Hz).
    Release gate: ≥ 70% of aligned segment pairs within Hamming 10.
    Harshest codec tier — most aggressive bandwidth restriction.
    """
    samples, sr = load_polygon_audio(speech_clip)
    voip = simulate_voip_codec(samples, sr)
    h_clean = extract_hashes(samples, sr, KEY_A, PEPPER)
    h_voip = extract_hashes(voip, sr, KEY_A, PEPPER)
    rate = _pass_rate(h_clean, h_voip)
    assert rate >= 0.70, f"[{speech_clip.id}] voip: {rate:.0%} < 70%"


@pytest.mark.polygon
def test_polygon_speech_clips_discriminated(polygon: DatasetRegistry):
    """
    [POLYGON/BLOCKING] All speech clip pairs must be discriminated.
    ≥ 80% of aligned segment pairs at Hamming ≥ 20.
    Validates anti-collision across the real corpus (more diverse than librosa alone).
    """
    clips = polygon.get("audio.speech").available_audio()
    if len(clips) < 2:
        pytest.skip("Need at least 2 speech clips for discrimination test")

    for i in range(len(clips)):
        for j in range(i + 1, len(clips)):
            clip_a, clip_b = clips[i], clips[j]
            sa, sr = load_polygon_audio(clip_a)
            sb, _  = load_polygon_audio(clip_b)
            n_samples = min(len(sa), len(sb))
            sa, sb = sa[:n_samples], sb[:n_samples]

            h_a = extract_hashes(sa, sr, KEY_A, PEPPER)
            h_b = extract_hashes(sb, sr, KEY_A, PEPPER)
            n = min(len(h_a), len(h_b))
            if n == 0:
                continue
            high_dist_rate = sum(
                1 for k in range(n)
                if hamming_distance(h_a[k].hash_hex, h_b[k].hash_hex) >= 20
            ) / n
            assert high_dist_rate >= 0.80, (
                f"[{clip_a.id} vs {clip_b.id}] only {high_dist_rate:.0%} discriminated"
            )


# ════════════════════════════════════════════════════════════════════════════════
# MUSIC — Informational (audio.music category, non-blocking)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.polygon
def test_polygon_music_deterministic(music_clip):
    """[POLYGON/INFO] Real music WAV: same clip + same key → identical hashes."""
    samples, sr = load_polygon_audio(music_clip)
    h1 = extract_hashes(samples, sr, KEY_A, PEPPER)
    h2 = extract_hashes(samples, sr, KEY_A, PEPPER)
    for a, b in zip(h1, h2):
        assert a.hash_hex == b.hash_hex, (
            f"[{music_clip.id}] non-deterministic at {a.time_offset_ms} ms"
        )


@pytest.mark.polygon
def test_polygon_music_keyed(music_clip):
    """[POLYGON/INFO] Real music WAV: different key → different hashes."""
    samples, sr = load_polygon_audio(music_clip)
    h_a = extract_hashes(samples, sr, KEY_A, PEPPER)
    h_b = extract_hashes(samples, sr, KEY_B, PEPPER)
    assert any(a.hash_hex != b.hash_hex for a, b in zip(h_a, h_b)), (
        f"[{music_clip.id}] key material has no effect"
    )


# ════════════════════════════════════════════════════════════════════════════════
# CROSS-CATEGORY — Speech vs music must be discriminated (informational)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.polygon
def test_polygon_speech_vs_music_discriminated(polygon: DatasetRegistry):
    """
    [POLYGON/INFO] Speech clips vs music clips must be discriminated.
    ≥ 70% of aligned segment pairs at Hamming ≥ 20.
    The speech-band fingerprint (300–8000 Hz) is not optimized for music;
    some spectral overlap with the piano band is expected.
    Threshold matches the manifest music.discrimination spec.
    """
    speech_clips = polygon.get("audio.speech").available_audio()
    music_clips  = polygon.get("audio.music").available_audio()
    if not speech_clips or not music_clips:
        pytest.skip("Need at least one speech and one music clip")

    sr_ref = None
    for sc in speech_clips:
        for mc in music_clips:
            sa, sr_a = load_polygon_audio(sc)
            mb, sr_b = load_polygon_audio(mc)
            assert sr_a == sr_b, "sample rate mismatch between polygon clips"
            sr_ref = sr_a
            n_samples = min(len(sa), len(mb))
            sa, mb = sa[:n_samples], mb[:n_samples]

            h_s = extract_hashes(sa, sr_ref, KEY_A, PEPPER)
            h_m = extract_hashes(mb, sr_ref, KEY_A, PEPPER)
            n = min(len(h_s), len(h_m))
            if n == 0:
                continue
            high_dist_rate = sum(
                1 for k in range(n)
                if hamming_distance(h_s[k].hash_hex, h_m[k].hash_hex) >= 20
            ) / n
            assert high_dist_rate >= 0.70, (
                f"[{sc.id} vs {mc.id}] only {high_dist_rate:.0%} discriminated"
            )
