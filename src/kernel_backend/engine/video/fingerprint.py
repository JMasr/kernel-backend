"""
engine/video/fingerprint.py

Video perceptual fingerprint — one 64-bit hash per 5-second segment.

Algorithm per segment:
  1. Select representative frame at segment_start + 0.5s.
  2. Convert to grayscale, resize to 32×32.
  3. Zero-mean normalization — MANDATORY:
       resized = resized - resized.mean(axis=1, keepdims=True)
     Without this, all frames with similar overall brightness hash identically.
     Same DC dominance bug as audio fingerprint (Phase 2a, lesson L4).
  4. 2D DCT, take top-left 8×8 block (low frequencies) → flatten to 64 floats.
  5. L2 normalize the vector.
  6. Keyed projection: HMAC(pepper, key_material)[:8] → seed → 64×64 Gaussian matrix.
  7. bits = projected >= median(projected) → 64-bit hash.

key_material = author_public_key (same as audio fingerprint).
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from kernel_backend.core.domain.watermark import SegmentFingerprint

SEGMENT_DURATION_S = 5.0
FRAME_OFFSET_S = 0.5
FINGERPRINT_SIZE = 64  # bits

# Pepper-free intermediate: pairs time_offset_ms with a unit-normalized DCT block.
SegmentFeature = tuple[int, np.ndarray]


def extract_hashes(
    video_path: str,
    key_material: bytes,
    pepper: bytes,
    segment_duration_s: float = SEGMENT_DURATION_S,
    frame_offset_s: float = FRAME_OFFSET_S,
) -> list[SegmentFingerprint]:
    """File-based extraction — never buffers all frames in memory."""
    features = extract_features(
        video_path,
        segment_duration_s=segment_duration_s,
        frame_offset_s=frame_offset_s,
    )
    return project_features_to_fingerprints(features, key_material, pepper)


def extract_hashes_from_frames(
    frames: list[np.ndarray],
    key_material: bytes,
    pepper: bytes,
    fps: float = 25.0,
    segment_duration_s: float = SEGMENT_DURATION_S,
    frame_offset_s: float = FRAME_OFFSET_S,
) -> list[SegmentFingerprint]:
    """Frame-list based extraction — for unit tests without video files."""
    features = extract_features_from_frames(
        frames,
        fps=fps,
        segment_duration_s=segment_duration_s,
        frame_offset_s=frame_offset_s,
    )
    return project_features_to_fingerprints(features, key_material, pepper)


def extract_features(
    video_path: str,
    segment_duration_s: float = SEGMENT_DURATION_S,
    frame_offset_s: float = FRAME_OFFSET_S,
) -> list[SegmentFeature]:
    """Pepper-free feature extraction — decode + DCT only.

    Pair with `project_features_to_fingerprints` to obtain hashes. The public
    verification endpoint reuses one features list against every org pepper,
    so the expensive decode happens once per upload rather than once per pepper.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        return []

    duration_s = total_frames / fps

    results: list[SegmentFeature] = []
    t = 0.0
    while t + frame_offset_s < duration_s:
        target_time = t + frame_offset_s
        target_frame = int(target_time * fps)
        if target_frame >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
        if not ok:
            break

        results.append((int(t * 1000), _compute_features(frame)))
        t += segment_duration_s

    cap.release()
    return results


def extract_features_from_frames(
    frames: list[np.ndarray],
    fps: float = 25.0,
    segment_duration_s: float = SEGMENT_DURATION_S,
    frame_offset_s: float = FRAME_OFFSET_S,
) -> list[SegmentFeature]:
    """In-memory counterpart of `extract_features` for unit tests."""
    if not frames or fps <= 0:
        return []

    total_frames = len(frames)
    duration_s = total_frames / fps

    results: list[SegmentFeature] = []
    t = 0.0
    while t + frame_offset_s < duration_s:
        target_frame = int((t + frame_offset_s) * fps)
        if target_frame >= total_frames:
            break

        results.append((int(t * 1000), _compute_features(frames[target_frame])))
        t += segment_duration_s

    return results


def project_features_to_fingerprints(
    features: Iterable[SegmentFeature],
    key_material: bytes,
    pepper: bytes,
) -> list[SegmentFingerprint]:
    """Cheap keyed projection: one projection matrix reused across segments."""
    features_list = list(features)
    if not features_list:
        return []
    dimension = features_list[0][1].shape[0]
    proj = _projection_matrix(key_material, pepper, dimension=dimension)

    result: list[SegmentFingerprint] = []
    for time_offset_ms, feature in features_list:
        projected = proj @ feature
        med = float(np.median(projected))
        bits = (projected >= med).astype(np.uint8)
        hash_int = 0
        for b in bits:
            hash_int = (hash_int << 1) | int(b)
        result.append(SegmentFingerprint(
            time_offset_ms=time_offset_ms,
            hash_hex=f"{hash_int:016x}",
        ))
    return result


def project_features_batch(
    features: Iterable[SegmentFeature],
    key_materials: Sequence[bytes],
    peppers: Sequence[bytes],
) -> list[list[SegmentFingerprint]]:
    """Project the same pepper-free features against N (key_material, pepper) pairs.

    See `engine.audio.fingerprint.project_features_batch` — same contract for
    the video pipeline (64-dim DCT block).
    """
    if len(key_materials) != len(peppers):
        raise ValueError("key_materials and peppers must have the same length")

    features_list = list(features)
    n_peppers = len(peppers)
    if n_peppers == 0:
        return []
    if not features_list:
        return [[] for _ in range(n_peppers)]

    dimension = features_list[0][1].shape[0]
    time_offsets = [t for t, _ in features_list]
    feature_matrix = np.stack([f for _, f in features_list], axis=1)  # (dim, n_segments)

    projections = np.stack(
        [_projection_matrix(km, p, dimension=dimension)
         for km, p in zip(key_materials, peppers)],
        axis=0,
    )  # (n_peppers, dim, dim)

    projected = projections @ feature_matrix  # (n_peppers, dim, n_segments)

    medians = np.median(projected, axis=1, keepdims=True)
    bits = (projected >= medians).astype(np.uint64)

    powers = (np.uint64(1) << np.arange(dimension - 1, -1, -1, dtype=np.uint64))
    values = (bits * powers[None, :, None]).sum(axis=1, dtype=np.uint64)

    out: list[list[SegmentFingerprint]] = []
    for pi in range(n_peppers):
        per_pepper = [
            SegmentFingerprint(
                time_offset_ms=time_offsets[si],
                hash_hex=f"{int(values[pi, si]):016x}",
            )
            for si in range(len(time_offsets))
        ]
        out.append(per_pepper)
    return out


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def _compute_features(frame: np.ndarray) -> np.ndarray:
    """Pepper-free per-frame feature: 64-dim unit-normalized 2D DCT block."""
    # Convert to grayscale and resize to 32×32
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)

    # MANDATORY: per-row zero-mean normalization
    resized = resized - resized.mean(axis=1, keepdims=True)

    # 2D DCT, take top-left 8×8 (low frequencies)
    dct_full = cv2.dct(resized)
    dct_block = dct_full[:8, :8].flatten().astype(np.float32)

    # L2 normalize
    norm = np.linalg.norm(dct_block)
    if norm > 1e-10:
        dct_block = dct_block / norm
    return dct_block


def _projection_matrix(
    key_material: bytes,
    pepper: bytes,
    dimension: int = FINGERPRINT_SIZE,
) -> np.ndarray:
    seed_bytes = hmac.new(pepper, key_material, hashlib.sha256).digest()
    seed = int.from_bytes(seed_bytes[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((dimension, dimension)).astype(np.float32)
