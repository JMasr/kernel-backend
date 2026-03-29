"""
Canonical signing defaults and DSP manifest re-exported for test use.

Tests import from here instead of hardcoding their own constants.
Any change to the DSP manifest automatically propagates to all tests.
"""
from kernel_backend.core.domain.dsp_manifest import (
    DSPManifest,
    PRODUCTION_MANIFEST,
)
from kernel_backend.core.services.signing_service import (
    _DEFAULT_AUDIO_PARAMS as DEFAULT_AUDIO_PARAMS,
    _DEFAULT_VIDEO_PARAMS as DEFAULT_VIDEO_PARAMS,
    _DEFAULT_EMBEDDING_PARAMS as DEFAULT_EMBEDDING_PARAMS,
)

__all__ = [
    "DEFAULT_AUDIO_PARAMS",
    "DEFAULT_VIDEO_PARAMS",
    "DEFAULT_EMBEDDING_PARAMS",
    "DSPManifest",
    "PRODUCTION_MANIFEST",
]
