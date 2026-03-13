from __future__ import annotations

from dataclasses import dataclass

from kernel_backend.core.domain.manifest import CryptographicManifest
from kernel_backend.core.domain.watermark import WatermarkID


@dataclass(frozen=True)
class SigningResult:
    content_id: str
    signed_media_key: str       # storage key where signed file was stored
    manifest: CryptographicManifest
    signature: bytes            # 64-byte Ed25519 signature
    wid: WatermarkID
    active_signals: list[str]
    rs_n: int
    pilot_hash_48: int
