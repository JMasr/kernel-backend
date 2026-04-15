"""Chunked parallel video signing — domain types.

Pure dataclasses that describe how a video is split into parallelizable
chunks. No I/O, no side effects, no imports from infrastructure/, api/, or
engine/. `validate_coverage` is the only behavior: a structural check that
the chunks of a manifest cover the full payload contiguously.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkSpec:
    """Specification of a single chunk.

    A chunk covers a contiguous range of *payload* segments plus optional
    guard segments at each boundary. The guard segments are decoded and
    encoded but never receive a watermark — they exist to give the encoder
    ramp-up context so the trimmed payload is visually clean.
    """

    chunk_id: int
    payload_seg_start: int  # first payload segment_idx (inclusive)
    payload_seg_end: int    # exclusive
    guard_lead_segments: int
    guard_trail_segments: int
    decode_start_s: float   # where to seek in the source video
    decode_end_s: float     # where to stop decoding
    payload_start_s: float  # offset into the encoded chunk where payload begins
    expected_payload_duration_s: float

    @property
    def n_payload_segments(self) -> int:
        return self.payload_seg_end - self.payload_seg_start

    @property
    def total_segments(self) -> int:
        return (
            self.guard_lead_segments
            + self.n_payload_segments
            + self.guard_trail_segments
        )


@dataclass(frozen=True)
class ChunkManifest:
    """Full plan for splitting a video into chunks."""

    total_chunks: int
    total_payload_segments: int
    segment_duration_s: float
    guard_segments: int
    chunks: tuple[ChunkSpec, ...]

    def validate_coverage(self) -> None:
        """Assert that chunks tile [0, total_payload_segments) exactly once.

        Raises ValueError on the first detected inconsistency (gap, overlap,
        non-sequential chunk_id, or length mismatch).
        """
        if len(self.chunks) != self.total_chunks:
            raise ValueError(
                f"total_chunks={self.total_chunks} does not match "
                f"len(chunks)={len(self.chunks)}"
            )
        if not self.chunks:
            if self.total_payload_segments != 0:
                raise ValueError(
                    "empty chunks but total_payload_segments="
                    f"{self.total_payload_segments}"
                )
            return

        ordered = sorted(self.chunks, key=lambda c: c.chunk_id)
        for expected_id, chunk in enumerate(ordered):
            if chunk.chunk_id != expected_id:
                raise ValueError(
                    f"chunk_ids must be 0..N-1 contiguous; got chunk_id="
                    f"{chunk.chunk_id} at position {expected_id}"
                )

        first = ordered[0]
        if first.payload_seg_start != 0:
            raise ValueError(
                f"first chunk must start at payload_seg_start=0, got "
                f"{first.payload_seg_start}"
            )

        for prev, curr in zip(ordered, ordered[1:]):
            if curr.payload_seg_start < prev.payload_seg_end:
                raise ValueError(
                    f"overlap between chunk {prev.chunk_id} "
                    f"(ends at {prev.payload_seg_end}) and chunk "
                    f"{curr.chunk_id} (starts at {curr.payload_seg_start})"
                )
            if curr.payload_seg_start > prev.payload_seg_end:
                raise ValueError(
                    f"gap between chunk {prev.chunk_id} "
                    f"(ends at {prev.payload_seg_end}) and chunk "
                    f"{curr.chunk_id} (starts at {curr.payload_seg_start})"
                )

        last = ordered[-1]
        if last.payload_seg_end != self.total_payload_segments:
            raise ValueError(
                f"last chunk payload_seg_end={last.payload_seg_end} does not "
                f"match total_payload_segments={self.total_payload_segments}"
            )

    def payload_segment_count(self) -> int:
        return sum(c.n_payload_segments for c in self.chunks)


@dataclass(frozen=True)
class ChunkResult:
    """Outcome of processing a single chunk in a worker process."""

    chunk_id: int
    output_path: str
    n_segments_processed: int
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class ChunkValidation:
    """Result of validating a batch of ChunkResults before concat."""

    all_present: bool
    all_durations_valid: bool
    total_duration_s: float
    expected_duration_s: float
    errors: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return (
            self.all_present
            and self.all_durations_valid
            and len(self.errors) == 0
        )
