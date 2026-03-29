from __future__ import annotations

import pytest

from kernel_backend.engine.audio.pilot_tone import detect_pilot, embed_pilot

SR = 44100
HASH_48 = 0xABCDEF012345  # a valid 48-bit value
SEED = 0xDEADBEEFCAFEBABE
WRONG_SEED = 0x1234567890ABCDEF

def test_embed_detect_roundtrip_noise(audio_signal) -> None:
    """embed_pilot → detect_pilot roundtrip on 1s signal (pilot mechanism only).

    Perceptual shaping is disabled so this test is isolated to the pilot
    correlation mechanism.  Shaping-specific roundtrip tests live in
    test_perceptual_shaping.py.
    """
    _name, gen = audio_signal
    audio = gen(1.0, SR)
    embedded = embed_pilot(audio, SR, HASH_48, SEED, perceptual_shaping=False)
    detected = detect_pilot(embedded, SR, SEED)
    assert detected == HASH_48

def test_embed_detect_roundtrip_multitone(audio_signal) -> None:
    """embed_pilot → detect_pilot roundtrip on 1s multitone (covers all DWT subbands).

    Perceptual shaping disabled — tests the pilot correlation mechanism only.
    """
    _name, gen = audio_signal
    audio = gen(1.0, SR)
    embedded = embed_pilot(audio, SR, HASH_48, SEED, perceptual_shaping=False)
    detected = detect_pilot(embedded, SR, SEED)
    assert detected == HASH_48

def test_detect_returns_none_unmodified(audio_signal) -> None:
    """Unmodified audio with no pilot → None."""
    _name, gen = audio_signal
    audio = gen(1.0, SR)
    result = detect_pilot(audio, SR, SEED)
    assert result is None

def test_output_length_preserved(audio_signal) -> None:
    """embed_pilot output length == input length."""
    _name, gen = audio_signal
    for duration in [0.5, 1.0, 2.5, 5.0]:
        audio = gen(duration, SR)
        embedded = embed_pilot(audio, SR, HASH_48, SEED)
        assert len(embedded) == len(audio)

def test_wrong_seed_returns_none(audio_signal) -> None:
    """Different global_pn_seed during detection → None."""
    _name, gen = audio_signal
    audio = gen(1.0, SR)
    embedded = embed_pilot(audio, SR, HASH_48, SEED)
    # Detection with wrong seed should fail or give wrong hash
    detected = detect_pilot(embedded, SR, WRONG_SEED)
    # Either None or a different value (not the original hash)
    assert detected is None or detected != HASH_48
