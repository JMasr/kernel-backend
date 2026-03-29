"""
Coherence tests for the centralized DSP manifest.

Verifies that engine module constants and signing service defaults
are derived from PRODUCTION_MANIFEST — catches drift if someone
hardcodes a value instead of referencing the manifest.
"""
from dataclasses import fields

from kernel_backend.core.domain.dsp_manifest import (
    DSPManifest,
    PRODUCTION_MANIFEST,
    AudioWIDConfig,
    VideoWIDConfig,
    AudioPilotConfig,
    VideoPilotConfig,
    ReedSolomonConfig,
    AudioFingerprintConfig,
    VideoFingerprintConfig,
)


# -- Engine module constant coherence ------------------------------------------

def test_audio_wid_erasure_threshold_matches():
    from kernel_backend.engine.audio.wid_beacon import ERASURE_THRESHOLD_Z
    assert ERASURE_THRESHOLD_Z == PRODUCTION_MANIFEST.audio_wid.erasure_threshold_z


def test_video_qim_step_matches():
    from kernel_backend.engine.video.wid_watermark import QIM_STEP_WID
    assert QIM_STEP_WID == PRODUCTION_MANIFEST.video_wid.qim_step_wid


def test_video_blocks_per_segment_matches():
    from kernel_backend.engine.video.wid_watermark import N_WID_BLOCKS_PER_SEGMENT
    assert N_WID_BLOCKS_PER_SEGMENT == PRODUCTION_MANIFEST.video_wid.n_blocks_per_segment


def test_video_agreement_threshold_matches():
    from kernel_backend.engine.video.wid_watermark import WID_AGREEMENT_THRESHOLD
    assert WID_AGREEMENT_THRESHOLD == PRODUCTION_MANIFEST.video_wid.agreement_threshold


def test_video_pilot_constants_match():
    from kernel_backend.engine.video.pilot_tone import (
        QIM_STEP_PILOT,
        N_PILOT_BLOCKS_PER_FRAME,
        PILOT_AGREEMENT_THRESHOLD,
    )
    assert QIM_STEP_PILOT == PRODUCTION_MANIFEST.video_pilot.qim_step
    assert N_PILOT_BLOCKS_PER_FRAME == PRODUCTION_MANIFEST.video_pilot.n_blocks_per_frame
    assert PILOT_AGREEMENT_THRESHOLD == PRODUCTION_MANIFEST.video_pilot.agreement_threshold


# -- Signing service defaults coherence ----------------------------------------

def test_signing_audio_params_match_manifest():
    from kernel_backend.core.services.signing_service import _DEFAULT_AUDIO_PARAMS
    m = PRODUCTION_MANIFEST.audio_wid
    assert _DEFAULT_AUDIO_PARAMS.dwt_levels == m.dwt_levels
    assert _DEFAULT_AUDIO_PARAMS.chips_per_bit == m.chips_per_bit
    assert _DEFAULT_AUDIO_PARAMS.psychoacoustic == m.psychoacoustic
    assert _DEFAULT_AUDIO_PARAMS.safety_margin_db == m.safety_margin_db
    assert _DEFAULT_AUDIO_PARAMS.target_snr_db == m.target_snr_db_audio_only


def test_signing_video_params_match_manifest():
    from kernel_backend.core.services.signing_service import _DEFAULT_VIDEO_PARAMS
    m = PRODUCTION_MANIFEST.video_wid
    assert _DEFAULT_VIDEO_PARAMS.jnd_adaptive == m.jnd_adaptive
    assert _DEFAULT_VIDEO_PARAMS.qim_step_base == m.qim_step_base
    assert _DEFAULT_VIDEO_PARAMS.qim_step_min == m.qim_step_min
    assert _DEFAULT_VIDEO_PARAMS.qim_step_max == m.qim_step_max
    assert _DEFAULT_VIDEO_PARAMS.qim_quantize_to == m.qim_quantize_to


# -- Manifest structural guarantees -------------------------------------------

def test_manifest_configs_are_frozen():
    """All config dataclasses must be frozen (immutable)."""
    for f in fields(DSPManifest):
        config = getattr(PRODUCTION_MANIFEST, f.name)
        assert config.__dataclass_params__.frozen, (
            f"{type(config).__name__} is not frozen"
        )


def test_manifest_singleton_identity():
    """PRODUCTION_MANIFEST is a module-level singleton — no accidental copies."""
    from kernel_backend.core.domain.dsp_manifest import PRODUCTION_MANIFEST as m2
    assert PRODUCTION_MANIFEST is m2
