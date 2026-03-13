from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st

from kernel_backend.engine.codec.hopping import plan_audio_hopping, plan_video_hopping


PEPPER = b"test-pepper-32-bytes-padding-here"
CONTENT_ID = "content-abc-123"
PUBKEY = "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VdAyEA\n-----END PUBLIC KEY-----\n"


def test_audio_hopping_deterministic() -> None:
    a = plan_audio_hopping(10, CONTENT_ID, PUBKEY, PEPPER)
    b = plan_audio_hopping(10, CONTENT_ID, PUBKEY, PEPPER)
    assert a == b


def test_audio_hopping_different_content_ids() -> None:
    a = plan_audio_hopping(10, "content-id-AAA", PUBKEY, PEPPER)
    b = plan_audio_hopping(10, "content-id-BBB", PUBKEY, PEPPER)
    assert any(ca != cb for ca, cb in zip(a, b))


def test_audio_hopping_length() -> None:
    for n in [1, 5, 32, 100]:
        result = plan_audio_hopping(n, CONTENT_ID, PUBKEY, PEPPER)
        assert len(result) == n


def test_audio_hopping_coeff_positions_empty() -> None:
    configs = plan_audio_hopping(20, CONTENT_ID, PUBKEY, PEPPER)
    for cfg in configs:
        assert cfg.coeff_positions == []


def test_audio_hopping_dwt_level_range() -> None:
    configs = plan_audio_hopping(50, CONTENT_ID, PUBKEY, PEPPER)
    for cfg in configs:
        assert cfg.dwt_level in (1, 2)


def test_audio_hopping_segment_indices() -> None:
    configs = plan_audio_hopping(10, CONTENT_ID, PUBKEY, PEPPER)
    for i, cfg in enumerate(configs):
        assert cfg.segment_index == i


def test_video_hopping_robust_subset() -> None:
    configs = plan_video_hopping(20, CONTENT_ID, PUBKEY, PEPPER)
    for cfg in configs:
        assert (0, 1) in cfg.coeff_positions
        assert (1, 0) in cfg.coeff_positions


def test_video_hopping_length() -> None:
    configs = plan_video_hopping(15, CONTENT_ID, PUBKEY, PEPPER)
    assert len(configs) == 15


def test_video_hopping_extra_from_pool() -> None:
    allowed = {(0, 1), (1, 0), (1, 1), (0, 2)}
    configs = plan_video_hopping(20, CONTENT_ID, PUBKEY, PEPPER)
    for cfg in configs:
        for pos in cfg.coeff_positions:
            assert pos in allowed


@given(st.integers(1, 50), st.text(min_size=1, max_size=36), st.text(min_size=1, max_size=100))
@settings(max_examples=30)
def test_video_hopping_deterministic(n: int, cid: str, pubkey: str) -> None:
    a = plan_video_hopping(n, cid, pubkey, PEPPER)
    b = plan_video_hopping(n, cid, pubkey, PEPPER)
    assert a == b
