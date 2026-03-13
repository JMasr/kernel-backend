from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MediaProfile:
    has_video: bool
    has_audio: bool
    width: int
    height: int
    fps: float
    duration_s: float
    sample_rate: int

    def __post_init__(self) -> None:
        if not self.has_video and not self.has_audio:
            raise ValueError(
                "MediaProfile must have at least one of has_video=True or has_audio=True"
            )

    @property
    def container_type(self) -> Literal["av", "audio_only", "video_only"]:
        if self.has_video and self.has_audio:
            return "av"
        if self.has_audio:
            return "audio_only"
        return "video_only"
