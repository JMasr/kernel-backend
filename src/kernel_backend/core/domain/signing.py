from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

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


class _FingerprintDict(TypedDict):
    time_offset_ms: int
    hash_hex: str


class RawSigningPayload(TypedDict):
    """Serialisable intermediate produced by the CPU phase of signing.

    The subprocess (_sign_sync) returns this dict so the parent async loop can
    run the I/O phase (storage.put + registry.save_*) with real adapters.
    All bytes are encoded as base64/hex strings; UUIDs as plain strings.
    """
    # Identity
    content_id: str
    author_id: str
    author_public_key: str       # PEM string
    org_id: str | None           # UUID string or None
    content_hash_sha256: str

    # Cryptographic proof
    manifest_json: str           # from _manifest_to_json()
    manifest_signature: str      # base64(64-byte Ed25519 sig)
    wid_hex: str                 # hex(16-byte WID)

    # DSP config
    rs_n: int
    pilot_hash_48: int
    active_signals: list[str]

    # Storage
    storage_key: str             # "signed/{content_id}/output{ext}"
    signed_media_key: str        # same as storage_key
    signed_file_path: str        # temp path written by subprocess — must be cleaned up
    content_type: str            # "audio/aac" | "video/mp4"

    # Fingerprints (None for absent channels)
    audio_fingerprints: list[_FingerprintDict] | None
    video_fingerprints: list[_FingerprintDict] | None

    # Routing
    media_type: str              # "audio" | "video" | "av"
