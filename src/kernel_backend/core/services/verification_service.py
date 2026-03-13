"""
Phase 4 — core/services/verification_service.py

Two-phase pipeline:
  Phase A — _identify_candidate(): fingerprint lookup → (content_id, pubkey, confidence)
  Phase B — _authenticate_wid():  segment iteration + RS decode + WID comparison + Ed25519

The two phases are intentionally separated. Fingerprints are candidate lookup,
not authentication. The watermark proof is the WID comparison.

ARCHITECTURAL INVARIANT:
  fingerprint_confidence MUST NEVER appear in any conditional that sets the verdict.
  This method is the ONLY producer of Verdict.VERIFIED.
"""
from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

import numpy as np

from kernel_backend.core.domain.manifest import CryptographicManifest
from kernel_backend.core.domain.verification import RedReason, VerificationResult, Verdict
from kernel_backend.core.ports.media import MediaPort
from kernel_backend.core.ports.registry import RegistryPort
from kernel_backend.core.ports.storage import StoragePort
from kernel_backend.core.services.crypto_service import derive_wid, verify_manifest
from kernel_backend.engine.audio.fingerprint import (
    extract_hashes as extract_audio_hashes,
    extract_hashes_from_stream as extract_audio_hashes_from_stream,
)
from kernel_backend.engine.codec.reed_solomon import ReedSolomonCodec, ReedSolomonError
from kernel_backend.engine.video.fingerprint import (
    SEGMENT_DURATION_S as VIDEO_SEGMENT_S,
    extract_hashes as extract_video_hashes,
)
from kernel_backend.engine.video.wid_watermark import WID_AGREEMENT_THRESHOLD, extract_segment

_AUDIO_FINGERPRINT_SEGMENT_S = 2.0  # matches signing_service.py
_MAX_HAMMING_CANDIDATE = 10         # max Hamming distance to consider a fingerprint match


def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


class VerificationService:
    """
    Stateless verification service. Pass infrastructure ports as arguments so
    core/ stays free of infra imports (hexagonal boundary).
    """

    async def verify(
        self,
        media_path: Path,
        media: MediaPort,
        storage: StoragePort,
        registry: RegistryPort,
        pepper: bytes,
    ) -> VerificationResult:
        """
        Two-phase verification pipeline.

        Phase A — Candidate identification (fingerprint lookup):
            Uses perceptual fingerprints to find the matching registry entry.
            Fast: O(registry_size × Hamming distance).
            Output: candidate (content_id, author_public_key) or None.

        Phase B — Cryptographic authentication (WID + Ed25519):
            Extracts watermark symbols from video segments.
            Decodes WID via Reed-Solomon.
            Compares extracted_WID with stored_WID.
            Verifies Ed25519 signature of the manifest.
            Slow: O(n_segments). Runs only if Phase A found a candidate.

        The two phases must not be merged.
        """
        candidate = await self._identify_candidate(media_path, media, registry, pepper)

        if candidate is None:
            return VerificationResult(
                verdict=Verdict.RED,
                red_reason=RedReason.CANDIDATE_NOT_FOUND,
            )

        content_id, author_public_key, confidence = candidate

        # Fetch the stored entry for Phase B
        entry = await registry.get_by_content_id(content_id)
        if entry is None:
            # Registry inconsistency — fingerprint matched but entry is gone
            return VerificationResult(
                verdict=Verdict.RED,
                content_id=content_id,
                red_reason=RedReason.CANDIDATE_NOT_FOUND,
                fingerprint_confidence=confidence,
            )

        # Re-derive stored WID from stored signature
        stored_wid = derive_wid(entry.manifest_signature, content_id)

        # Reconstruct manifest from stored JSON (needed for Ed25519 verify)
        stored_manifest = _manifest_from_json(entry.manifest_json) if entry.manifest_json else None

        return await self._authenticate_wid(
            media_path=media_path,
            media=media,
            content_id=content_id,
            author_id=entry.author_id,
            author_public_key=author_public_key,
            stored_wid=stored_wid.data,
            stored_manifest=stored_manifest,
            stored_signature=entry.manifest_signature,
            rs_n=entry.rs_n,
            pepper=pepper,
            fingerprint_confidence=confidence,
        )

    async def _identify_candidate(
        self,
        media_path: Path,
        media: MediaPort,
        registry: RegistryPort,
        pepper: bytes,
    ) -> tuple[str, str, float] | None:
        """
        Phase A: fingerprint extraction + registry lookup.
        Returns (content_id, author_public_key, hamming_confidence) | None.
        """
        profile = media.probe(media_path)

        # Extract fingerprints appropriate for the media type
        query_hashes: list[str] = []

        if profile.has_video:
            video_fps = extract_video_hashes(
                str(media_path),
                key_material=pepper,
                pepper=pepper,
            )
            query_hashes = [fp.hash_hex for fp in video_fps]
        elif profile.has_audio:
            target_sample_rate = 44100
            chunk_stream = (chunk for _, chunk, _ in media.iter_audio_segments(
                media_path, segment_duration_s=2.0, target_sample_rate=target_sample_rate
            ))
            audio_fps = extract_audio_hashes_from_stream(
                chunk_stream, target_sample_rate,
                key_material=pepper,
                pepper=pepper,
            )
            query_hashes = [fp.hash_hex for fp in audio_fps]

        if not query_hashes:
            return None

        candidates = await registry.match_fingerprints(
            query_hashes,
            max_hamming=_MAX_HAMMING_CANDIDATE,
        )

        if not candidates:
            return None

        # Select best candidate: the one returned first by match_fingerprints.
        # match_fingerprints already filters by max_hamming; with a single registry
        # entry this is trivially correct. Multi-entry ranking is a Phase 5 concern.
        best_entry = candidates[0]
        # Confidence = fraction of query segments within max_hamming of ANY stored hash
        # (already guaranteed by match_fingerprints contract — use 1.0 as proxy)
        confidence = 1.0

        return best_entry.content_id, best_entry.author_public_key, confidence

    async def _authenticate_wid(
        self,
        media_path: Path,
        media: MediaPort,
        content_id: str,
        author_id: str,
        author_public_key: str,
        stored_wid: bytes,
        stored_manifest: CryptographicManifest | None,
        stored_signature: bytes,
        rs_n: int,
        pepper: bytes,
        fingerprint_confidence: float,
    ) -> VerificationResult:
        """
        Phase B: segment iteration + RS decode + WID comparison + Ed25519 verify.
        This method is the sole producer of Verdict.VERIFIED.

        Step ordering is intentional:
          1. WID extracted
          2. WID compared (before signature check)
          3. Ed25519 verified
        WID_MISMATCH must take priority over SIGNATURE_INVALID — they carry
        different forensic meaning.
        """
        symbols: list[int | None] = []
        erasure_positions: list[int] = []
        n_segments_total = 0

        for seg_idx, frames, _fps in media.iter_video_segments(
            media_path,
            segment_duration_s=VIDEO_SEGMENT_S,
            frame_offset_s=0.5,
        ):
            if seg_idx >= rs_n:
                break  # only process segments used at sign time

            n_segments_total += 1
            result = extract_segment(
                frames,
                content_id,
                author_public_key,
                seg_idx,
                pepper,
            )

            if result.agreement < WID_AGREEMENT_THRESHOLD:
                erasure_positions.append(seg_idx)
                symbols.append(None)
            else:
                # extracted_bits is a bytes object with one byte (the RS symbol)
                symbol_byte = result.extracted_bits[0] if result.extracted_bits else 0
                symbols.append(symbol_byte)

        n_erasures = len(erasure_positions)
        n_segments_decoded = n_segments_total - n_erasures

        # RS decode
        codec = ReedSolomonCodec(rs_n)
        try:
            decoded_wid = codec.decode(symbols)
        except ReedSolomonError:
            # Too many erasures — cannot recover WID (quality issue, NOT tampering)
            return VerificationResult(
                verdict=Verdict.RED,
                content_id=content_id,
                author_id=author_id,
                author_public_key=author_public_key,
                red_reason=RedReason.WID_UNDECODABLE,
                n_segments_total=n_segments_total,
                n_segments_decoded=n_segments_decoded,
                n_erasures=n_erasures,
                fingerprint_confidence=fingerprint_confidence,
            )

        # Step 6: WID comparison — checked BEFORE signature (intentional ordering)
        if decoded_wid != stored_wid:
            return VerificationResult(
                verdict=Verdict.RED,
                content_id=content_id,
                author_id=author_id,
                author_public_key=author_public_key,
                red_reason=RedReason.WID_MISMATCH,
                wid_match=False,
                n_segments_total=n_segments_total,
                n_segments_decoded=n_segments_decoded,
                n_erasures=n_erasures,
                fingerprint_confidence=fingerprint_confidence,
            )

        # Step 7: Ed25519 signature verification
        sig_valid = (
            stored_manifest is not None
            and verify_manifest(stored_manifest, stored_signature, author_public_key)
        )
        if not sig_valid:
            return VerificationResult(
                verdict=Verdict.RED,
                content_id=content_id,
                author_id=author_id,
                author_public_key=author_public_key,
                red_reason=RedReason.SIGNATURE_INVALID,
                wid_match=True,   # WID matched — but signature proof is broken
                signature_valid=False,
                n_segments_total=n_segments_total,
                n_segments_decoded=n_segments_decoded,
                n_erasures=n_erasures,
                fingerprint_confidence=fingerprint_confidence,
            )

        # VERIFIED — the only path to Verdict.VERIFIED
        return VerificationResult(
            verdict=Verdict.VERIFIED,
            content_id=content_id,
            author_id=author_id,
            author_public_key=author_public_key,
            wid_match=True,
            signature_valid=True,
            n_segments_total=n_segments_total,
            n_segments_decoded=n_segments_decoded,
            n_erasures=n_erasures,
            fingerprint_confidence=fingerprint_confidence,
        )


def _manifest_from_json(manifest_json: str) -> CryptographicManifest:
    """Reconstruct a CryptographicManifest from the JSON stored at sign time."""
    import json as _json
    data = _json.loads(manifest_json)
    return CryptographicManifest(
        content_id=data["content_id"],
        content_hash_sha256=data["content_hash_sha256"],
        fingerprints_audio=data.get("fingerprints_audio", []),
        fingerprints_video=data.get("fingerprints_video", []),
        author_id=data["author_id"],
        author_public_key=data["author_public_key"],
        created_at=data["created_at"],
        schema_version=data.get("schema_version", 2),
    )
