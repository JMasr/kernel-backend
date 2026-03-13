from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SegmentFingerprint:
    time_offset_ms: int
    hash_hex: str   # 16-char hex string = 64-bit hash


@dataclass(frozen=True)
class VideoEntry:
    content_id: str
    author_id: str
    author_public_key: str
    active_signals: list[str]   # e.g. ["pilot_audio", "wid_audio", "fingerprint_audio"]
    rs_n: int                   # total RS symbols used at sign time
    pilot_hash_48: int          # 48-bit int for fast pilot index lookup
    manifest_signature: bytes   # 64-byte Ed25519 signature — stored for WID re-derivation
    manifest_json: str = ""     # canonical manifest JSON — stored for signature verification
    schema_version: int = 2
    status: str = "VALID"       # "VALID" | "REVOKED"


@dataclass(frozen=True)
class WatermarkID:
    data: bytes

    def __post_init__(self) -> None:
        if len(self.data) != 16:
            raise ValueError(f"WatermarkID.data must be exactly 16 bytes, got {len(self.data)}")


@dataclass(frozen=True)
class BandConfig:
    segment_index: int
    coeff_positions: list[tuple[int, int]]
    dwt_level: int


@dataclass(frozen=True)
class EmbeddingRecipe:
    content_id: str
    rs_n: int
    pilot_hash_48: bytes
    band_configs: list[BandConfig]
    prng_seeds: list[int]
    rs_k: int = 16

    def __post_init__(self) -> None:
        if not (16 < self.rs_n <= 255):
            raise ValueError(
                f"rs_n must be in the range (16, 255], got {self.rs_n}"
            )
