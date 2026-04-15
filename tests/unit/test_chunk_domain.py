"""Unit tests for core.domain.chunk — pure structural invariants."""
from __future__ import annotations

import pytest

from kernel_backend.core.domain.chunk import (
    ChunkManifest,
    ChunkResult,
    ChunkSpec,
    ChunkValidation,
)


def _spec(
    chunk_id: int,
    start: int,
    end: int,
    guard_lead: int = 0,
    guard_trail: int = 0,
    seg_s: float = 5.0,
) -> ChunkSpec:
    decode_start_s = max(0.0, (start - guard_lead) * seg_s)
    decode_end_s = (end + guard_trail) * seg_s
    payload_start_s = guard_lead * seg_s
    return ChunkSpec(
        chunk_id=chunk_id,
        payload_seg_start=start,
        payload_seg_end=end,
        guard_lead_segments=guard_lead,
        guard_trail_segments=guard_trail,
        decode_start_s=decode_start_s,
        decode_end_s=decode_end_s,
        payload_start_s=payload_start_s,
        expected_payload_duration_s=(end - start) * seg_s,
    )


def test_chunk_spec_properties() -> None:
    spec = _spec(0, 15, 30, guard_lead=1, guard_trail=1)
    assert spec.n_payload_segments == 15
    assert spec.total_segments == 17


def test_manifest_validate_coverage_ok() -> None:
    chunks = (
        _spec(0, 0, 15, guard_lead=0, guard_trail=1),
        _spec(1, 15, 30, guard_lead=1, guard_trail=1),
        _spec(2, 30, 45, guard_lead=1, guard_trail=1),
        _spec(3, 45, 60, guard_lead=1, guard_trail=0),
    )
    manifest = ChunkManifest(
        total_chunks=4,
        total_payload_segments=60,
        segment_duration_s=5.0,
        guard_segments=1,
        chunks=chunks,
    )
    manifest.validate_coverage()
    assert manifest.payload_segment_count() == 60


def test_manifest_validate_coverage_gap() -> None:
    chunks = (
        _spec(0, 0, 15),
        _spec(1, 20, 30),  # gap 15..20
    )
    manifest = ChunkManifest(
        total_chunks=2,
        total_payload_segments=30,
        segment_duration_s=5.0,
        guard_segments=1,
        chunks=chunks,
    )
    with pytest.raises(ValueError, match="gap"):
        manifest.validate_coverage()


def test_manifest_validate_coverage_overlap() -> None:
    chunks = (
        _spec(0, 0, 20),
        _spec(1, 15, 30),  # overlap
    )
    manifest = ChunkManifest(
        total_chunks=2,
        total_payload_segments=30,
        segment_duration_s=5.0,
        guard_segments=1,
        chunks=chunks,
    )
    with pytest.raises(ValueError, match="overlap"):
        manifest.validate_coverage()


def test_manifest_validate_coverage_first_not_zero() -> None:
    chunks = (_spec(0, 5, 20),)
    manifest = ChunkManifest(
        total_chunks=1,
        total_payload_segments=20,
        segment_duration_s=5.0,
        guard_segments=0,
        chunks=chunks,
    )
    with pytest.raises(ValueError, match="first chunk must start"):
        manifest.validate_coverage()


def test_manifest_validate_coverage_last_not_total() -> None:
    chunks = (
        _spec(0, 0, 15),
        _spec(1, 15, 30),
    )
    manifest = ChunkManifest(
        total_chunks=2,
        total_payload_segments=60,  # mismatch: last ends at 30, claims 60
        segment_duration_s=5.0,
        guard_segments=0,
        chunks=chunks,
    )
    with pytest.raises(ValueError, match="total_payload_segments"):
        manifest.validate_coverage()


def test_manifest_validate_coverage_non_contiguous_ids() -> None:
    chunks = (
        _spec(0, 0, 15),
        _spec(2, 15, 30),  # skipped chunk_id 1
    )
    manifest = ChunkManifest(
        total_chunks=2,
        total_payload_segments=30,
        segment_duration_s=5.0,
        guard_segments=0,
        chunks=chunks,
    )
    with pytest.raises(ValueError, match="chunk_id"):
        manifest.validate_coverage()


def test_chunk_validation_is_valid() -> None:
    good = ChunkValidation(
        all_present=True,
        all_durations_valid=True,
        total_duration_s=300.0,
        expected_duration_s=300.0,
    )
    assert good.is_valid is True

    missing = ChunkValidation(
        all_present=False,
        all_durations_valid=True,
        total_duration_s=225.0,
        expected_duration_s=300.0,
    )
    assert missing.is_valid is False

    with_errors = ChunkValidation(
        all_present=True,
        all_durations_valid=True,
        total_duration_s=300.0,
        expected_duration_s=300.0,
        errors=("chunk 2: boom",),
    )
    assert with_errors.is_valid is False

    dur_bad = ChunkValidation(
        all_present=True,
        all_durations_valid=False,
        total_duration_s=290.0,
        expected_duration_s=300.0,
    )
    assert dur_bad.is_valid is False


def test_chunk_result_defaults() -> None:
    ok = ChunkResult(
        chunk_id=0,
        output_path="/tmp/chunk_0.mp4",
        n_segments_processed=15,
        success=True,
    )
    assert ok.error is None
    fail = ChunkResult(
        chunk_id=1,
        output_path="",
        n_segments_processed=0,
        success=False,
        error="encoder failed",
    )
    assert fail.success is False
    assert fail.error == "encoder failed"
