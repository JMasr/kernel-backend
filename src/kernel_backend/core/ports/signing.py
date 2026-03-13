from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.domain.signing import SigningResult


class SigningPort(ABC):
    @abstractmethod
    async def sign(
        self,
        media_path: Path,
        certificate: Certificate,
        private_key_pem: str,
    ) -> SigningResult: ...
