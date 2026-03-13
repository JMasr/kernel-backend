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

import cv2
import numpy as np

from kernel_backend.core.domain.watermark import SegmentFingerprint

SEGMENT_DURATION_S = 5.0
FRAME_OFFSET_S = 0.5
FINGERPRINT_SIZE = 64  # bits


def extract_hashes(
    video_path: str,
    key_material: bytes,
    pepper: bytes,
    segment_duration_s: float = SEGMENT_DURATION_S,
    frame_offset_s: float = FRAME_OFFSET_S,
) -> list[SegmentFingerprint]:
    """File-based extraction — never buffers all frames in memory."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        return []

    duration_s = total_frames / fps
    proj = _projection_matrix(key_material, pepper)

    results = []
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

        hash_hex = _compute_hash(frame, proj)
        results.append(SegmentFingerprint(
            time_offset_ms=int(t * 1000),
            hash_hex=hash_hex,
        ))
        t += segment_duration_s

    cap.release()
    return results


def extract_hashes_from_frames(
    frames: list[np.ndarray],
    key_material: bytes,
    pepper: bytes,
    fps: float = 25.0,
    segment_duration_s: float = SEGMENT_DURATION_S,
    frame_offset_s: float = FRAME_OFFSET_S,
) -> list[SegmentFingerprint]:
    """Frame-list based extraction — for unit tests without video files."""
    if not frames or fps <= 0:
        return []

    total_frames = len(frames)
    duration_s = total_frames / fps
    proj = _projection_matrix(key_material, pepper)

    results = []
    t = 0.0
    while t + frame_offset_s < duration_s:
        target_frame = int((t + frame_offset_s) * fps)
        if target_frame >= total_frames:
            break

        hash_hex = _compute_hash(frames[target_frame], proj)
        results.append(SegmentFingerprint(
            time_offset_ms=int(t * 1000),
            hash_hex=hash_hex,
        ))
        t += segment_duration_s

    return results


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def _compute_hash(frame: np.ndarray, proj: np.ndarray) -> str:
    """
    Single-frame hash. Called once per segment.
    Applies the zero-mean normalization before DCT — non-negotiable.
    """
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
    dct_block = dct_full[:8, :8].flatten()

    # L2 normalize
    norm = np.linalg.norm(dct_block)
    if norm > 1e-10:
        dct_block = dct_block / norm

    # Keyed projection
    projected = proj @ dct_block.astype(np.float32)

    # Median threshold → 64-bit hash
    med = float(np.median(projected))
    bits = (projected >= med).astype(np.uint8)

    # Pack into 16-char hex string (64 bits)
    hash_int = 0
    for b in bits:
        hash_int = (hash_int << 1) | int(b)
    return f"{hash_int:016x}"


def _projection_matrix(
    key_material: bytes,
    pepper: bytes,
    dimension: int = FINGERPRINT_SIZE,
) -> np.ndarray:
    seed_bytes = hmac.new(pepper, key_material, hashlib.sha256).digest()
    seed = int.from_bytes(seed_bytes[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((dimension, dimension)).astype(np.float32)
