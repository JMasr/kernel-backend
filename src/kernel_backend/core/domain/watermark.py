from __future__ import annotations

from dataclasses import dataclass, asdict, field
from uuid import UUID


@dataclass(frozen=True)
class AudioEmbeddingParams:
    dwt_levels: tuple[int, ...]   # active DWT levels, e.g. (1, 2)
    chips_per_bit: int             # chips per bit in DSSS
    psychoacoustic: bool           # True → psychoacoustic masking S2
    safety_margin_db: float        # margin below masking threshold (dB)
    target_snr_db: float           # SNR fallback if psychoacoustic=False


@dataclass(frozen=True)
class VideoEmbeddingParams:
    jnd_adaptive: bool             # True → adaptive QIM S3
    qim_step_base: float           # base QIM step (mid-gray)
    qim_step_min: float            # minimum step (H.264 survival)
    qim_step_max: float            # maximum step (dark blocks)
    qim_quantize_to: float         # step quantization granularity


@dataclass(frozen=True)
class EmbeddingParams:
    audio: AudioEmbeddingParams
    video: VideoEmbeddingParams | None  # None for audio-only content


def embedding_params_to_dict(p: EmbeddingParams) -> dict:
    """Serialize EmbeddingParams to a flat dict suitable for JSONB storage."""
    return {
        "audio": asdict(p.audio),
        "video": asdict(p.video) if p.video else None,
    }


def embedding_params_from_dict(d: dict) -> EmbeddingParams:
    """Deserialize EmbeddingParams from a dict (read from JSONB)."""
    audio_data = d["audio"]
    # tuple[int, ...] is stored as list in JSON — convert back
    audio_data = dict(audio_data)
    audio_data["dwt_levels"] = tuple(audio_data["dwt_levels"])
    return EmbeddingParams(
        audio=AudioEmbeddingParams(**audio_data),
        video=VideoEmbeddingParams(**d["video"]) if d.get("video") else None,
    )


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
    embedding_params: EmbeddingParams  # DSP parameters used at sign time
    manifest_json: str = ""     # canonical manifest JSON — stored for signature verification
    schema_version: int = 2
    status: str = "VALID"       # "VALID" | "REVOKED"
    org_id: UUID | None = None  # organization owning this entry (Phase 6.A)
    signed_media_key: str = ""  # storage key for the signed media file (Phase 6.B-2)


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
    extra_dwt_levels: tuple[int, ...] = ()
    # extra_dwt_levels == () → single-band behaviour (v1)
    # extra_dwt_levels == (2,) with dwt_level=1 → embed in levels 1 and 2 (EGC)


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
