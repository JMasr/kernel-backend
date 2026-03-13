from abc import ABC, abstractmethod


class VerificationPort(ABC):
    @abstractmethod
    async def verify(self, key: str) -> dict:
        """Verify a media file at the given storage key."""
