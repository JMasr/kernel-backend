from __future__ import annotations

import hashlib
import hmac

import numpy as np

from kernel_backend.core.domain.watermark import BandConfig


def plan_audio_hopping(
    n_segments: int,
    content_id: str,
    author_pubkey: str,
    pepper: bytes,
    force_levels: list[int] | None = None,
    target_subband: str = "detail",
) -> list[BandConfig]:
    """
    Per-segment DWT band config for audio WID embedding.

    If force_levels is None (legacy): dwt_level alternates between 1 and 2
    based on HMAC-derived seed; extra_dwt_levels == () for all segments.

    If force_levels is [1, 2] (multi-band): all segments use levels 1 and 2
    simultaneously — dwt_level=force_levels[0], extra_dwt_levels=tuple(force_levels[1:]).

    Seed per segment i (legacy path):
      HMAC-SHA256(pepper, f"audio_hop|{content_id}|{author_pubkey}|{i}".encode())
    Use first 8 bytes as int for np.random.default_rng(seed).

    coeff_positions is empty list for audio (not used — kept for interface symmetry).
    Returns exactly n_segments BandConfig objects.
    Deterministic: same inputs always produce same output.
    """
    configs: list[BandConfig] = []
    # Precompute the HMAC with the file-level prefix; per-segment work becomes
    # a single copy() + update() + digest() instead of a fresh hmac.new each
    # iteration (saves ~8 µs × n_segments per sign call).
    prefix = f"audio_hop|{content_id}|{author_pubkey}|".encode()
    base_hmac = hmac.new(pepper, prefix, hashlib.sha256)
    for i in range(n_segments):
        if force_levels is not None:
            primary = force_levels[0]
            extra = tuple(force_levels[1:])
            configs.append(BandConfig(
                segment_index=i,
                coeff_positions=[],
                dwt_level=primary,
                extra_dwt_levels=extra,
                target_subband=target_subband,
            ))
        else:
            h = base_hmac.copy()
            h.update(str(i).encode())
            digest = h.digest()
            seed = int.from_bytes(digest[:8], "big")
            rng = np.random.default_rng(seed)
            dwt_level = int(rng.integers(1, 3))  # 1 or 2
            configs.append(BandConfig(
                segment_index=i,
                coeff_positions=[],
                dwt_level=dwt_level,
                extra_dwt_levels=(),
                target_subband=target_subband,
            ))
    return configs


def plan_video_hopping(
    n_segments: int,
    content_id: str,
    author_pubkey: str,
    pepper: bytes,
) -> list[BandConfig]:
    """
    Per-segment 4x4 DCT coefficient selection for video WID embedding.
    Coefficient pool: {(0,1), (1,0), (1,1), (0,2)}
    Constraint: every BandConfig must include both (0,1) and (1,0)
    (robust subset — survives H.264 compression at QP up to ~36).
    Additional 1-2 positions chosen from {(1,1),(0,2)} via HMAC seed.
    Deterministic: same inputs always produce same output.
    """
    robust_required = [(0, 1), (1, 0)]
    optional_pool = [(1, 1), (0, 2)]
    configs: list[BandConfig] = []
    prefix = f"video_hop|{content_id}|{author_pubkey}|".encode()
    base_hmac = hmac.new(pepper, prefix, hashlib.sha256)
    for i in range(n_segments):
        h = base_hmac.copy()
        h.update(str(i).encode())
        digest = h.digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        n_extra = int(rng.integers(1, 3))  # 1 or 2 extra positions
        extra_indices = rng.choice(len(optional_pool), size=n_extra, replace=False)
        extra = [optional_pool[int(j)] for j in extra_indices]
        coeff_positions = robust_required + extra
        configs.append(BandConfig(
            segment_index=i,
            coeff_positions=coeff_positions,
            dwt_level=1,  # DCT-based; level field kept for interface symmetry
        ))
    return configs
