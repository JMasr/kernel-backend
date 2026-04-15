"""Unit tests for plan_chunks."""
from __future__ import annotations

import pytest

from kernel_backend.core.services.chunk_planner import plan_chunks


def test_basic_60_segments_4_workers() -> None:
    manifest = plan_chunks(
        total_segments=60, segment_duration_s=5.0, n_workers=4, guard_segments=1
    )

    assert manifest.total_chunks == 4
    assert manifest.total_payload_segments == 60
    assert manifest.payload_segment_count() == 60

    c0, c1, c2, c3 = manifest.chunks

    assert (c0.payload_seg_start, c0.payload_seg_end) == (0, 15)
    assert (c0.guard_lead_segments, c0.guard_trail_segments) == (0, 1)
    assert c0.decode_start_s == 0.0
    assert c0.decode_end_s == 80.0
    assert c0.payload_start_s == 0.0
    assert c0.expected_payload_duration_s == 75.0

    assert (c1.payload_seg_start, c1.payload_seg_end) == (15, 30)
    assert (c1.guard_lead_segments, c1.guard_trail_segments) == (1, 1)
    assert c1.decode_start_s == 70.0
    assert c1.decode_end_s == 155.0
    assert c1.payload_start_s == 5.0

    assert (c2.payload_seg_start, c2.payload_seg_end) == (30, 45)
    assert (c2.guard_lead_segments, c2.guard_trail_segments) == (1, 1)
    assert c2.decode_start_s == 145.0
    assert c2.decode_end_s == 230.0
    assert c2.payload_start_s == 5.0

    assert (c3.payload_seg_start, c3.payload_seg_end) == (45, 60)
    assert (c3.guard_lead_segments, c3.guard_trail_segments) == (1, 0)
    assert c3.decode_start_s == 220.0
    assert c3.decode_end_s == 300.0
    assert c3.payload_start_s == 5.0


def test_odd_segments_distribution() -> None:
    manifest = plan_chunks(total_segments=61, n_workers=4, guard_segments=1)
    assert manifest.total_chunks == 4
    assert manifest.payload_segment_count() == 61
    payload_lens = [c.n_payload_segments for c in manifest.chunks]
    # Sprint rule: remainder goes to the earliest chunks.
    assert payload_lens == [16, 15, 15, 15]
    manifest.validate_coverage()


def test_short_video_fallback() -> None:
    manifest = plan_chunks(total_segments=3, n_workers=4, min_payload_segments=4)
    assert manifest.total_chunks == 1
    only = manifest.chunks[0]
    assert only.guard_lead_segments == 0
    assert only.guard_trail_segments == 0
    assert only.payload_seg_start == 0
    assert only.payload_seg_end == 3
    assert only.decode_start_s == 0.0
    assert only.decode_end_s == 15.0


def test_auto_reduce_workers() -> None:
    # 12 / 4 = 3 < min_payload=4 → reduce to 3 workers, 12 / 3 = 4 >= 4 → OK.
    manifest = plan_chunks(
        total_segments=12, n_workers=4, guard_segments=1, min_payload_segments=4
    )
    assert manifest.total_chunks == 3
    payload_lens = [c.n_payload_segments for c in manifest.chunks]
    assert payload_lens == [4, 4, 4]


def test_guard_first_last() -> None:
    manifest = plan_chunks(total_segments=80, n_workers=4, guard_segments=1)
    first = manifest.chunks[0]
    last = manifest.chunks[-1]
    assert first.guard_lead_segments == 0
    assert last.guard_trail_segments == 0
    for middle in manifest.chunks[1:-1]:
        assert middle.guard_lead_segments == 1
        assert middle.guard_trail_segments == 1


@pytest.mark.parametrize("total", list(range(17, 256)))
def test_coverage_no_gaps(total: int) -> None:
    manifest = plan_chunks(
        total_segments=total, n_workers=4, guard_segments=1, min_payload_segments=4
    )
    # validate_coverage is called inside plan_chunks; re-call defensively here.
    manifest.validate_coverage()
    assert manifest.payload_segment_count() == total


def test_decode_windows_include_guards() -> None:
    manifest = plan_chunks(
        total_segments=40, n_workers=4, guard_segments=1, segment_duration_s=5.0
    )
    for chunk in manifest.chunks:
        # decode window start: (seg_start - lead) * seg_s
        expected_start = max(
            0.0, (chunk.payload_seg_start - chunk.guard_lead_segments) * 5.0
        )
        expected_end = (chunk.payload_seg_end + chunk.guard_trail_segments) * 5.0
        assert chunk.decode_start_s == expected_start
        assert chunk.decode_end_s == expected_end
        # payload_start_s is purely a function of the leading guard.
        assert chunk.payload_start_s == chunk.guard_lead_segments * 5.0


def test_deterministic() -> None:
    a = plan_chunks(total_segments=61, n_workers=4, guard_segments=1)
    b = plan_chunks(total_segments=61, n_workers=4, guard_segments=1)
    assert a == b


def test_fallback_min_workers_when_everything_collides() -> None:
    # 5 segments, 4 workers, min_payload=4: 5/4=1 < 4 → reduce to 1.
    # 1 worker would give 5 >= 4 so single-chunk with guard_lead=guard_trail=0.
    manifest = plan_chunks(
        total_segments=5, n_workers=4, guard_segments=1, min_payload_segments=4
    )
    assert manifest.total_chunks == 1
    only = manifest.chunks[0]
    assert only.guard_lead_segments == 0
    assert only.guard_trail_segments == 0
    assert only.payload_seg_end == 5


def test_zero_segments_returns_empty_manifest() -> None:
    manifest = plan_chunks(total_segments=0, n_workers=4)
    assert manifest.total_chunks == 0
    assert manifest.chunks == ()
    manifest.validate_coverage()
