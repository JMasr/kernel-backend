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
    # Content-adaptive routing fields (backward-compatible defaults)
    target_subband: str = "detail"       # "detail" | "approximation"
    frame_length_ms: float = 0.0         # 0.0 = legacy 2s segments
    pn_sequence_length: int = 0          # 0 = derive from chips_per_bit * 8


@dataclass(frozen=True)
class VideoEmbeddingParams:
    jnd_adaptive: bool             # True → adaptive QIM S3
    qim_step_base: float           # base QIM step (mid-gray)
    qim_step_min: float            # minimum step (H.264 survival)
    qim_step_max: float            # maximum step (dark blocks)
    qim_quantize_to: float         # step quantization granularity
    min_block_variance: float = 0.0   # 0 = no filtering (backward compatible)
    block_oversample: int = 1         # 1 = no oversampling (backward compatible)


@dataclass(frozen=True)
class EmbeddingParams:
    audio: AudioEmbeddingParams | None  # None for video-only content
    video: VideoEmbeddingParams | None  # None for audio-only content


def embedding_params_to_dict(p: EmbeddingParams) -> dict:
    """Serialize EmbeddingParams to a flat dict suitable for JSONB storage."""
    return {
        "audio": asdict(p.audio) if p.audio else None,
        "video": asdict(p.video) if p.video else None,
    }


def embedding_params_from_dict(d: dict) -> EmbeddingParams:
    """Deserialize EmbeddingParams from a dict (read from JSONB)."""
    audio_obj = None
    if d.get("audio"):
        audio_data = dict(d["audio"])
        # tuple[int, ...] is stored as list in JSON — convert back
        audio_data["dwt_levels"] = tuple(audio_data["dwt_levels"])
        # Backward-compatible defaults for content-adaptive fields
        audio_data.setdefault("target_subband", "detail")
        audio_data.setdefault("frame_length_ms", 0.0)
        audio_data.setdefault("pn_sequence_length", 0)
        audio_obj = AudioEmbeddingParams(**audio_data)
    video_obj = None
    if d.get("video"):
        video_data = dict(d["video"])
        video_data.setdefault("min_block_variance", 0.0)
        video_data.setdefault("block_oversample", 1)
        video_obj = VideoEmbeddingParams(**video_data)
    return EmbeddingParams(
        audio=audio_obj,
        video=video_obj,
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
    active_signals: list[str]   # e.g. ["wid_audio", "fingerprint_audio"]
    rs_n: int                   # total RS symbols used at sign time
    manifest_signature: bytes   # 64-byte Ed25519 signature — stored for WID re-derivation
    embedding_params: EmbeddingParams  # DSP parameters used at sign time
    manifest_json: str = ""     # canonical manifest JSON — stored for signature verification
    schema_version: int = 2
    status: str = "VALID"       # "VALID" | "REVOKED"
    org_id: UUID | None = None  # organization owning this entry (Phase 6.A)
    signed_media_key: str = ""  # storage key for the signed media file (Phase 6.B-2)
    output_encoding_params: dict | None = None
    # Example: {'audio': {'codec': 'aac', 'bitrate': '256k', 'sample_rate': 44100},
    #           'video': {'codec': 'libx264', 'crf': 18, 'preset': 'ultrafast'}}
    routing_metadata: dict | None = None  # RoutingDecision as dict, or None for legacy


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
    target_subband: str = "detail"  # "detail" → coeffs[-2], "approximation" → coeffs[0]


@dataclass(frozen=True)
class EmbeddingRecipe:
    content_id: str
    rs_n: int
    band_configs: list[BandConfig]
    prng_seeds: list[int]
    rs_k: int = 16

    def __post_init__(self) -> None:
        if not (16 < self.rs_n <= 255):
            raise ValueError(
                f"rs_n must be in the range (16, 255], got {self.rs_n}"
            )
