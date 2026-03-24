from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator, Iterator

import numpy as np

from kernel_backend.core.domain.media import MediaProfile


class MediaPort(ABC):
    """Port for media I/O. Concrete implementation: infrastructure/media/media_service.py."""

    @abstractmethod
    def probe(self, path: Path) -> MediaProfile:
        """ffprobe → MediaProfile. Raises ValueError if no streams found."""

    @abstractmethod
    def decode_audio_to_pcm(
        self,
        path: Path,
        target_sample_rate: int = 44100,
    ) -> tuple[np.ndarray, int]:
        """[DEPRECATED] Decode entire audio track → mono float32 PCM in [-1.0, 1.0].
        Do not use this for large files, use iter_audio_segments.
        Returns (samples_array, actual_sample_rate)."""

    @abstractmethod
    def iter_audio_segments(
        self,
        path: Path,
        segment_duration_s: float = 2.0,
        target_sample_rate: int = 44100,
    ) -> Generator[tuple[int, np.ndarray, int], None, None]:
        """
        Lazily yield (segment_idx, samples, sample_rate) for each audio segment.
        Reads from an FFmpeg subprocess pipe in chunks to prevent OOM on long files.
        """

    @abstractmethod
    def encode_audio_from_pcm(
        self,
        samples: np.ndarray,
        sample_rate: int,
        output_path: Path,
        codec: str = "aac",
        bitrate: str = "256k",
    ) -> None:
        """float32 PCM → encoded audio file."""

    @abstractmethod
    def encode_audio_stream(
        self,
        sample_rate: int,
        output_path: Path,
        codec: str = "aac",
        bitrate: str = "256k",
    ):
        """Returns a subprocess Popen to write s16le PCM bytes incrementally."""

    @abstractmethod
    def mux_video_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> None:
        """Combine video stream (copy) + new audio into output container."""

    @abstractmethod
    def read_video_frames(
        self,
        path: Path,
        start_frame: int = 0,
        n_frames: int | None = None,
    ) -> tuple[list[np.ndarray], float]:
        """Read BGR frames from video. Returns (frames, fps).
        If n_frames is None, reads all frames."""

    @abstractmethod
    def write_video_frames(
        self,
        frames: list[np.ndarray],
        fps: float,
        output_path: Path,
    ) -> None:
        """Write BGR frames to a video file (mp4/H.264)."""

    @abstractmethod
    def seek_frame(self, path: Path, time_s: float) -> np.ndarray:
        """
        Read exactly one frame at the given timestamp.

        Used by fingerprint extraction, which needs one representative frame
        per segment (at segment_start + frame_offset_s). Loading all segment
        frames for a single DCT hash is wasteful at any resolution.

        At 1080p: 1 frame = ~6 MB. Compared to 150 frames = ~900 MB.

        Returns:
            BGR frame as uint8 ndarray, shape (H, W, 3).
        Raises:
            ValueError if time_s is beyond the video duration or frame is unreadable.
        """

    @abstractmethod
    def open_video_encode_stream(
        self,
        width: int,
        height: int,
        fps: float,
        output_path: Path,
    ) -> Any:
        """
        Opens an FFmpeg subprocess to receive raw YUV frames via stdin and
        encode them as H.264 lossless (crf=0) into output_path.

        Returns the Popen object. Caller writes yuv.tobytes() per frame,
        then calls stdin.close() and wait().

        Analogous to encode_audio_stream — single-pass, no intermediate file.
        """

    @abstractmethod
    def iter_video_segments(
        self,
        path: Path,
        segment_duration_s: float = 5.0,
        frame_stride: int = 1,
    ) -> Iterator[tuple[int, list[np.ndarray], float]]:
        """
        Yields (segment_idx, frames, fps) for each full segment of a video file.
        Reads frames lazily — never holds more than one segment in memory at a time.

        frame_stride controls which frames are loaded within each segment:
        - frame_stride=1  → all frames (original behavior, backwards compatible)
        - frame_stride=3  → every 3rd frame (recommended for WID extraction)

        Memory impact at 1080p 30fps (5s segment = 150 frames):
        - frame_stride=1: 150 frames × 6 MB = ~900 MB
        - frame_stride=3:  50 frames × 6 MB = ~300 MB

        There is NO robustness loss with frame_stride=3.
        extract_segment() was designed for strided frame access —
        the majority vote operates on available frames regardless of count.

        frame_stride=1 is the default to preserve all existing callers unchanged.
        All new callers in Phase 5 must explicitly pass frame_stride=3.

        Yields:
            segment_idx : int              — 0-based segment index
            frames      : list[np.ndarray] — BGR frames belonging to this segment
            fps         : float            — frames per second of the source
        """
