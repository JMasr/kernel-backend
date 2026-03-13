from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

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
    def iter_video_segments(
        self,
        path: Path,
        segment_duration_s: float = 5.0,
        frame_offset_s: float = 0.5,
    ) -> Generator[tuple[int, list[np.ndarray], float], None, None]:
        """
        Yields (segment_idx, frames, fps) for each full segment of a video file.
        Reads frames lazily — never holds more than one segment in memory at a time.

        This is the required access pattern for long-form video verification.
        camping_01 is 1058 seconds (211 segments). Loading all frames at once
        is not acceptable for a production verification service.

        Yields:
            segment_idx : int              — 0-based segment index
            frames      : list[np.ndarray] — BGR frames belonging to this segment
            fps         : float            — frames per second of the source

        The caller is responsible for calling detect_pilot() and extract_segment()
        on the yielded frames within the loop body, not accumulating them.
        """
