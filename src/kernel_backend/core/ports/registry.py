from __future__ import annotations

from abc import ABC, abstractmethod

from kernel_backend.core.domain.watermark import SegmentFingerprint, VideoEntry


class RegistryPort(ABC):
    @abstractmethod
    async def save_video(self, entry: VideoEntry) -> None: ...

    @abstractmethod
    async def get_by_content_id(self, content_id: str) -> VideoEntry | None: ...

    @abstractmethod
    async def get_valid_candidates(self) -> list[VideoEntry]: ...

    @abstractmethod
    async def save_segments(
        self,
        content_id: str,
        segments: list[SegmentFingerprint],
        is_original: bool,
    ) -> None: ...

    @abstractmethod
    async def match_fingerprints(
        self,
        hashes: list[str],
        max_hamming: int = 10,
    ) -> list[VideoEntry]: ...
