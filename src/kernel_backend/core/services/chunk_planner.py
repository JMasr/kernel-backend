"""Chunk planning — pure function.

`plan_chunks` partitions a total-segment count into a `ChunkManifest`
describing contiguous payload ranges plus optional guard bands. The
output is deterministic for a given set of inputs and never touches I/O.
"""
from __future__ import annotations

from kernel_backend.core.domain.chunk import ChunkManifest, ChunkSpec


def plan_chunks(
    total_segments: int,
    segment_duration_s: float = 5.0,
    n_workers: int = 4,
    guard_segments: int = 1,
    min_payload_segments: int = 4,
) -> ChunkManifest:
    """Partition `total_segments` payload segments into chunks.

    Rules:
      1. Segments are distributed as evenly as possible; when
         `total_segments` is not divisible by the effective worker count,
         the earliest chunks absorb the +1 remainder.
      2. If `total_segments < min_payload_segments` a single no-guard
         chunk is returned (sequential fallback).
      3. Otherwise the effective worker count is reduced while any chunk
         would have fewer than `min_payload_segments` payload segments.
      4. The first chunk has `guard_lead_segments=0`; the last has
         `guard_trail_segments=0`; the middle chunks carry
         `guard_segments` on each side.
      5. `decode_start_s` and `decode_end_s` include the guards.
         `payload_start_s` is the offset, inside the encoded chunk, at
         which the payload starts (0.0 for the first chunk, otherwise
         `guard_lead_segments * segment_duration_s`).
      6. `ChunkManifest.validate_coverage` is invoked before returning so
         the caller always gets a structurally sound plan.
    """
    if total_segments < 0:
        raise ValueError(f"total_segments must be >= 0, got {total_segments}")
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if guard_segments < 0:
        raise ValueError(f"guard_segments must be >= 0, got {guard_segments}")
    if min_payload_segments < 1:
        raise ValueError(
            f"min_payload_segments must be >= 1, got {min_payload_segments}"
        )

    if total_segments == 0:
        return ChunkManifest(
            total_chunks=0,
            total_payload_segments=0,
            segment_duration_s=segment_duration_s,
            guard_segments=guard_segments,
            chunks=(),
        )

    # Rule 2 — single-chunk fallback for very short videos.
    if total_segments < min_payload_segments:
        single = ChunkSpec(
            chunk_id=0,
            payload_seg_start=0,
            payload_seg_end=total_segments,
            guard_lead_segments=0,
            guard_trail_segments=0,
            decode_start_s=0.0,
            decode_end_s=total_segments * segment_duration_s,
            payload_start_s=0.0,
            expected_payload_duration_s=total_segments * segment_duration_s,
        )
        manifest = ChunkManifest(
            total_chunks=1,
            total_payload_segments=total_segments,
            segment_duration_s=segment_duration_s,
            guard_segments=0,
            chunks=(single,),
        )
        manifest.validate_coverage()
        return manifest

    # Rule 3 — shrink effective workers until each chunk meets the floor.
    effective_workers = n_workers
    while (
        effective_workers > 1
        and total_segments // effective_workers < min_payload_segments
    ):
        effective_workers -= 1

    base = total_segments // effective_workers
    remainder = total_segments % effective_workers

    specs: list[ChunkSpec] = []
    cursor = 0
    for chunk_id in range(effective_workers):
        # Earlier chunks absorb the +1 remainder (sprint rule 1).
        payload_len = base + (1 if chunk_id < remainder else 0)
        seg_start = cursor
        seg_end = cursor + payload_len
        cursor = seg_end

        is_first = chunk_id == 0
        is_last = chunk_id == effective_workers - 1
        lead = 0 if is_first else guard_segments
        trail = 0 if is_last else guard_segments

        decode_start_s = max(0.0, (seg_start - lead) * segment_duration_s)
        decode_end_s = (seg_end + trail) * segment_duration_s
        payload_start_s = lead * segment_duration_s
        expected_payload_duration_s = payload_len * segment_duration_s

        specs.append(
            ChunkSpec(
                chunk_id=chunk_id,
                payload_seg_start=seg_start,
                payload_seg_end=seg_end,
                guard_lead_segments=lead,
                guard_trail_segments=trail,
                decode_start_s=decode_start_s,
                decode_end_s=decode_end_s,
                payload_start_s=payload_start_s,
                expected_payload_duration_s=expected_payload_duration_s,
            )
        )

    manifest = ChunkManifest(
        total_chunks=effective_workers,
        total_payload_segments=total_segments,
        segment_duration_s=segment_duration_s,
        guard_segments=guard_segments,
        chunks=tuple(specs),
    )
    manifest.validate_coverage()
    return manifest
