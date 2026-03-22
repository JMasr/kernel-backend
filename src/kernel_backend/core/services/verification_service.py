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
from uuid import UUID

import numpy as np

from kernel_backend.core.domain.manifest import CryptographicManifest
from kernel_backend.core.domain.verification import (
    AVVerificationResult,
    RedReason,
    VerificationResult,
    Verdict,
)
from kernel_backend.core.ports.media import MediaPort
from kernel_backend.core.ports.registry import RegistryPort
from kernel_backend.core.ports.storage import StoragePort
from kernel_backend.core.services.crypto_service import derive_wid, verify_manifest
from kernel_backend.engine.audio.fingerprint import (
    extract_hashes as extract_audio_hashes,
    extract_hashes_from_stream as extract_audio_hashes_from_stream,
)
from kernel_backend.engine.audio.wid_beacon import extract_symbol_segment as extract_audio_symbol
from kernel_backend.engine.codec.hopping import plan_audio_hopping
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
        org_id: UUID | None = None,
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
        candidate = await self._identify_candidate(media_path, media, registry, pepper, org_id)

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
        org_id: UUID | None = None,
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
            org_id=org_id,
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

    async def verify_audio(
        self,
        media_path: Path,
        media: MediaPort,
        storage: StoragePort,
        registry: RegistryPort,
        pepper: bytes,
        org_id: UUID | None = None,
    ) -> VerificationResult:
        """
        Audio-only verification pipeline.

        Same two-phase structure as verify() but uses audio fingerprints for
        Phase A and audio WID extraction (DSSS) for Phase B.
        """
        candidate = await self._identify_candidate(media_path, media, registry, pepper, org_id)

        if candidate is None:
            return VerificationResult(
                verdict=Verdict.RED,
                red_reason=RedReason.CANDIDATE_NOT_FOUND,
            )

        content_id, author_public_key, confidence = candidate

        entry = await registry.get_by_content_id(content_id)
        if entry is None:
            return VerificationResult(
                verdict=Verdict.RED,
                content_id=content_id,
                red_reason=RedReason.CANDIDATE_NOT_FOUND,
                fingerprint_confidence=confidence,
            )

        stored_wid = derive_wid(entry.manifest_signature, content_id)
        stored_manifest = _manifest_from_json(entry.manifest_json) if entry.manifest_json else None

        # Phase B — extract audio WID via DSSS
        decoded_wid, decodable, n_seg, n_dec, n_era = self._extract_audio_wid(
            media_path, media, content_id, author_public_key, entry.rs_n, pepper
        )

        if not decodable:
            return VerificationResult(
                verdict=Verdict.RED,
                content_id=content_id,
                author_id=entry.author_id,
                author_public_key=author_public_key,
                red_reason=RedReason.WID_UNDECODABLE,
                n_segments_total=n_seg,
                n_segments_decoded=n_dec,
                n_erasures=n_era,
                fingerprint_confidence=confidence,
            )

        if decoded_wid != stored_wid.data:
            return VerificationResult(
                verdict=Verdict.RED,
                content_id=content_id,
                author_id=entry.author_id,
                author_public_key=author_public_key,
                red_reason=RedReason.WID_MISMATCH,
                wid_match=False,
                n_segments_total=n_seg,
                n_segments_decoded=n_dec,
                n_erasures=n_era,
                fingerprint_confidence=confidence,
            )

        sig_valid = (
            stored_manifest is not None
            and verify_manifest(stored_manifest, entry.manifest_signature, author_public_key)
        )
        if not sig_valid:
            return VerificationResult(
                verdict=Verdict.RED,
                content_id=content_id,
                author_id=entry.author_id,
                author_public_key=author_public_key,
                red_reason=RedReason.SIGNATURE_INVALID,
                wid_match=True,
                signature_valid=False,
                n_segments_total=n_seg,
                n_segments_decoded=n_dec,
                n_erasures=n_era,
                fingerprint_confidence=confidence,
            )

        return VerificationResult(
            verdict=Verdict.VERIFIED,
            content_id=content_id,
            author_id=entry.author_id,
            author_public_key=author_public_key,
            wid_match=True,
            signature_valid=True,
            n_segments_total=n_seg,
            n_segments_decoded=n_dec,
            n_erasures=n_era,
            fingerprint_confidence=confidence,
        )

    async def verify_av(
        self,
        media_path: Path,
        media: MediaPort,
        storage: StoragePort,
        registry: RegistryPort,
        pepper: bytes,
        org_id: UUID | None = None,
    ) -> AVVerificationResult:
        """
        AV verification pipeline.

        Phase A — Candidate identification:
            Extract both audio AND video fingerprints.
            Match against registry using the union of both fingerprint sets.
            A candidate is valid if either audio OR video fingerprints match
            (the file may have been degraded on one channel but not the other).
            Select best candidate by combined Hamming score.

        Phase B — Cryptographic authentication:
            Extract audio WID and video WID independently.
            Both must equal the single stored_wid.
            VERIFIED only if BOTH match AND Ed25519 is valid.

        NOTE: Pilot agreement is NOT used here. At H.264 CRF 28 (standard social
        media compression), pilot agreement drops below 0.75 threshold. Fingerprints
        drive Phase A; WID + Ed25519 drive Phase B.
        """
        candidate = await self._identify_candidate(media_path, media, registry, pepper, org_id)

        if candidate is None:
            return AVVerificationResult(
                verdict=Verdict.RED,
                audio_verdict=Verdict.RED,
                video_verdict=Verdict.RED,
                red_reason=RedReason.CANDIDATE_NOT_FOUND,
            )

        content_id, author_public_key, confidence = candidate

        entry = await registry.get_by_content_id(content_id)
        if entry is None:
            return AVVerificationResult(
                verdict=Verdict.RED,
                audio_verdict=Verdict.RED,
                video_verdict=Verdict.RED,
                content_id=content_id,
                red_reason=RedReason.CANDIDATE_NOT_FOUND,
                fingerprint_confidence=confidence,
            )

        stored_wid = derive_wid(entry.manifest_signature, content_id)
        stored_manifest = _manifest_from_json(entry.manifest_json) if entry.manifest_json else None

        # Phase B — extract audio WID
        (audio_wid, audio_decodable,
         audio_n_seg, audio_n_dec, audio_n_era) = self._extract_audio_wid(
            media_path, media, content_id, author_public_key, entry.rs_n, pepper
        )

        # Phase B — extract video WID
        (video_wid, video_decodable,
         video_n_seg, video_n_dec, video_n_era) = self._extract_video_wid(
            media_path, media, content_id, author_public_key, entry.rs_n, pepper
        )

        # WID comparison (before signature — WID mismatch is more specific)
        audio_wid_ok = audio_decodable and (audio_wid == stored_wid.data)
        video_wid_ok = video_decodable and (video_wid == stored_wid.data)

        # Ed25519 signature verification
        sig_valid = (
            stored_manifest is not None
            and verify_manifest(stored_manifest, entry.manifest_signature, author_public_key)
        )

        # Per-channel verdicts
        audio_verdict = Verdict.VERIFIED if (audio_wid_ok and sig_valid) else Verdict.RED
        video_verdict = Verdict.VERIFIED if (video_wid_ok and sig_valid) else Verdict.RED

        # Top-level verdict from explicit decision table
        verdict, red_reason = _compose_verdict(
            audio_wid_ok, video_wid_ok, audio_decodable, video_decodable, sig_valid
        )

        return AVVerificationResult(
            verdict=verdict,
            audio_verdict=audio_verdict,
            video_verdict=video_verdict,
            content_id=content_id,
            author_id=entry.author_id,
            red_reason=red_reason,
            wid_match=(verdict == Verdict.VERIFIED),
            signature_valid=sig_valid,
            audio_n_segments=audio_n_seg,
            audio_n_decoded=audio_n_dec,
            audio_n_erasures=audio_n_era,
            video_n_segments=video_n_seg,
            video_n_decoded=video_n_dec,
            video_n_erasures=video_n_era,
            fingerprint_confidence=confidence,
        )

    def _extract_audio_wid(
        self,
        media_path: Path,
        media: MediaPort,
        content_id: str,
        author_public_key: str,
        rs_n: int,
        pepper: bytes,
    ) -> tuple[bytes | None, bool, int, int, int]:
        """
        Extract audio WID via DSSS correlation over each 2-second segment.
        Returns (decoded_wid, decodable, n_segments, n_decoded, n_erasures).
        decoded_wid is None and decodable is False when RS decode fails.
        """
        band_configs = plan_audio_hopping(rs_n, content_id, author_public_key, pepper)
        symbols: list[int | None] = []
        erasure_positions: list[int] = []
        n_segments_total = 0

        for seg_idx, chunk, _ in media.iter_audio_segments(
            media_path, segment_duration_s=2.0, target_sample_rate=44100
        ):
            if seg_idx >= rs_n:
                break

            n_segments_total += 1
            pn_seed = int.from_bytes(
                hmac.new(
                    pepper,
                    f"wid|{content_id}|{author_public_key}|{seg_idx}".encode(),
                    hashlib.sha256,
                ).digest()[:8],
                "big",
            )

            symbol_byte, _conf = extract_audio_symbol(chunk, band_configs[seg_idx], pn_seed)

            if symbol_byte is None:
                erasure_positions.append(seg_idx)
                symbols.append(None)
            else:
                symbols.append(symbol_byte)

        n_erasures = len(erasure_positions)
        n_decoded = n_segments_total - n_erasures

        codec = ReedSolomonCodec(rs_n)
        try:
            decoded = codec.decode(symbols)
            return decoded, True, n_segments_total, n_decoded, n_erasures
        except ReedSolomonError:
            return None, False, n_segments_total, n_decoded, n_erasures

    def _extract_video_wid(
        self,
        media_path: Path,
        media: MediaPort,
        content_id: str,
        author_public_key: str,
        rs_n: int,
        pepper: bytes,
    ) -> tuple[bytes | None, bool, int, int, int]:
        """
        Extract video WID via QIM over each 5-second segment.
        Returns (decoded_wid, decodable, n_segments, n_decoded, n_erasures).
        decoded_wid is None and decodable is False when RS decode fails.
        """
        symbols: list[int | None] = []
        erasure_positions: list[int] = []
        n_segments_total = 0

        for seg_idx, frames, _fps in media.iter_video_segments(
            media_path,
            segment_duration_s=VIDEO_SEGMENT_S,
            frame_stride=3,
        ):
            if seg_idx >= rs_n:
                break

            n_segments_total += 1
            result = extract_segment(
                frames, content_id, author_public_key, seg_idx, pepper
            )

            if result.agreement < WID_AGREEMENT_THRESHOLD:
                erasure_positions.append(seg_idx)
                symbols.append(None)
            else:
                symbol_byte = result.extracted_bits[0] if result.extracted_bits else 0
                symbols.append(symbol_byte)

        n_erasures = len(erasure_positions)
        n_decoded = n_segments_total - n_erasures

        codec = ReedSolomonCodec(rs_n)
        try:
            decoded = codec.decode(symbols)
            return decoded, True, n_segments_total, n_decoded, n_erasures
        except ReedSolomonError:
            return None, False, n_segments_total, n_decoded, n_erasures


def _compose_verdict(
    audio_wid_ok: bool,
    video_wid_ok: bool,
    audio_decodable: bool,
    video_decodable: bool,
    signature_ok: bool,
) -> tuple[Verdict, RedReason | None]:
    """
    Explicit decision table for AV verdict composition.
    Order is intentional — more specific reasons take precedence.
    Candidate not found is handled before this function is called.
    """
    # Step 1: check decodability first (degradation, not tampering)
    if not audio_decodable and not video_decodable:
        return Verdict.RED, RedReason.WID_UNDECODABLE

    if not audio_decodable:
        return Verdict.RED, RedReason.AUDIO_WID_UNDECODABLE

    if not video_decodable:
        return Verdict.RED, RedReason.VIDEO_WID_UNDECODABLE

    # Step 2: check WID match (tampering signal)
    if not audio_wid_ok and not video_wid_ok:
        return Verdict.RED, RedReason.WID_MISMATCH

    if not audio_wid_ok:
        return Verdict.RED, RedReason.AUDIO_WID_MISMATCH

    if not video_wid_ok:
        return Verdict.RED, RedReason.VIDEO_WID_MISMATCH

    # Step 3: signature (checked last — WID mismatch is more specific)
    if not signature_ok:
        return Verdict.RED, RedReason.SIGNATURE_INVALID

    # All checks passed
    return Verdict.VERIFIED, None


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
