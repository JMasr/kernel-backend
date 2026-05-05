"""
Centralized DSP configuration manifest.

Single source of truth for all calibrated watermarking constants.
Engine modules and signing_service derive their defaults from this manifest.
Tests import via tests/helpers/signing_defaults.py to stay in sync.

All dataclasses are frozen — values are immutable at runtime.
To change a calibrated constant, change it HERE and all consumers update.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioWIDConfig:
    """Audio WID beacon (DSSS + DWT) calibrated constants."""
    chips_per_bit: int = 32
    # These are fallback defaults used only when audio_params is passed explicitly
    # (bypassing the content-adaptive router). The router overrides these per content
    # type: speech → -12.0 dB (cD2 band has low energy; floor=5e-4 also applied),
    # music → -16.0 dB, classical/ambient → -18.0 dB. See algorithm_router.py.
    target_snr_db_audio_only: float = -20.0
    target_snr_db_av: float = -18.0
    dwt_levels: tuple[int, ...] = (2,)        # single-band cD2 (5.5-11 kHz)
    erasure_threshold_z: float = 1.0
    masking_alpha: float = 0.70
    masking_min_floor: float = 0.12
    masking_energy_floor: float = 0.15
    psychoacoustic: bool = False              # Bark model disabled (unit mismatch)
    safety_margin_db: float = 12.0


@dataclass(frozen=True)
class VideoWIDConfig:
    """Video WID watermark (QIM + 4x4 DCT) calibrated constants."""
    qim_step_wid: float = 48.0
    n_blocks_per_segment: int = 128
    agreement_threshold: float = 0.52
    block_size: int = 4
    jnd_adaptive: bool = False
    qim_step_base: float = 64.0
    qim_step_min: float = 44.0
    qim_step_max: float = 128.0
    qim_quantize_to: float = 4.0


@dataclass(frozen=True)
class AudioPilotConfig:
    """Audio pilot tone (DSSS + DWT approximation band) -- diagnostic only."""
    chips_per_bit: int = 64
    target_snr_db: float = -14.0
    masking_alpha: float = 0.65
    masking_min_floor: float = 0.05


@dataclass(frozen=True)
class VideoPilotConfig:
    """Video pilot tone (QIM on DC) -- diagnostic only."""
    qim_step: float = 28.0
    n_blocks_per_frame: int = 256
    agreement_threshold: float = 0.75


@dataclass(frozen=True)
class ReedSolomonConfig:
    """Reed-Solomon codec parameters."""
    k: int = 16                    # WID = 16 bytes, fixed
    max_n: int = 255               # max RS symbols
    min_segments: int = 17         # minimum segments for signing


@dataclass(frozen=True)
class AudioFingerprintConfig:
    """Audio fingerprint (speech-optimized log-mel -> DCT -> keyed projection)."""
    segment_duration_s: float = 2.0
    overlap: float = 0.5
    f_min: float = 300.0
    f_max: float = 8000.0
    fingerprint_bits: int = 64


@dataclass(frozen=True)
class VideoFingerprintConfig:
    """Video fingerprint (grayscale 32x32 -> 2D DCT -> keyed projection)."""
    segment_duration_s: float = 5.0
    frame_offset_s: float = 0.5
    fingerprint_bits: int = 64


@dataclass(frozen=True)
class DSPManifest:
    """Complete DSP configuration for the Kernel watermarking system."""
    audio_wid: AudioWIDConfig = AudioWIDConfig()
    video_wid: VideoWIDConfig = VideoWIDConfig()
    audio_pilot: AudioPilotConfig = AudioPilotConfig()
    video_pilot: VideoPilotConfig = VideoPilotConfig()
    reed_solomon: ReedSolomonConfig = ReedSolomonConfig()
    audio_fingerprint: AudioFingerprintConfig = AudioFingerprintConfig()
    video_fingerprint: VideoFingerprintConfig = VideoFingerprintConfig()


# Singleton -- the canonical production configuration
PRODUCTION_MANIFEST = DSPManifest()
