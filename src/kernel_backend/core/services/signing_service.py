from __future__ import annotations

import hashlib
import hmac
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.domain.manifest import CryptographicManifest
from kernel_backend.core.domain.signing import SigningResult
from kernel_backend.core.domain.watermark import VideoEntry
from kernel_backend.core.ports.media import MediaPort
from kernel_backend.core.ports.registry import RegistryPort
from kernel_backend.core.ports.storage import StoragePort
from kernel_backend.core.services.crypto_service import derive_wid, sign_manifest
from kernel_backend.engine.audio.fingerprint import (
    extract_hashes as extract_audio_hashes,
    extract_hashes_from_stream as extract_audio_hashes_from_stream,
)
from kernel_backend.engine.audio.pilot_tone import embed_pilot as embed_audio_pilot
from kernel_backend.engine.audio.wid_beacon import embed_segment as embed_audio_segment
from kernel_backend.engine.codec.hopping import plan_audio_hopping, plan_video_hopping
from kernel_backend.engine.codec.reed_solomon import ReedSolomonCodec
from kernel_backend.engine.video.fingerprint import extract_hashes as extract_video_hashes
from kernel_backend.engine.video.pilot_tone import embed_pilot as embed_video_pilot
from kernel_backend.engine.video.pilot_tone import pilot_hash_48 as compute_pilot_hash_48
from kernel_backend.engine.video.fingerprint import SEGMENT_DURATION_S as VIDEO_SEGMENT_S
from kernel_backend.engine.video.wid_watermark import embed_segment as embed_video_segment


def _manifest_to_json(manifest: "CryptographicManifest") -> str:
    """Serialize manifest to a JSON string for storage — used for signature verification."""
    return json.dumps({
        "author_id": manifest.author_id,
        "author_public_key": manifest.author_public_key,
        "content_hash_sha256": manifest.content_hash_sha256,
        "content_id": manifest.content_id,
        "created_at": manifest.created_at,
        "fingerprints_audio": manifest.fingerprints_audio,
        "fingerprints_video": manifest.fingerprints_video,
        "schema_version": manifest.schema_version,
    })


async def sign_audio(
    media_path: Path,
    certificate: Certificate,
    private_key_pem: str,
    storage: StoragePort,
    registry: RegistryPort,
    pepper: bytes,
    media: MediaPort,
) -> SigningResult:
    """
    Full audio signing pipeline. Orchestrates DSP, cryptography, storage, and
    registry operations. Raises ValueError on unsupported container or too-short
    audio.
    """
    # 1. Probe
    profile = media.probe(media_path)
    if profile.container_type == "video_only":
        raise ValueError("Container has no audio track — cannot sign audio-only pipeline")

    # 2–3. IDs and content hash
    content_id = str(uuid4())
    content_hash = hashlib.sha256(media_path.read_bytes()).hexdigest()

    # 4. Target sample rate for audio orchestration
    target_sample_rate = 44100

    # 5. Fingerprint (drives segment count) - Pass 1 Streaming
    chunk_stream = (chunk for _, chunk, _ in media.iter_audio_segments(
        media_path, segment_duration_s=2.0, target_sample_rate=target_sample_rate
    ))
    fingerprints = extract_audio_hashes_from_stream(
        chunk_stream,
        target_sample_rate,
        key_material=pepper,
        pepper=pepper,
    )

    # 6. RS parameters
    n_segments = len(fingerprints)
    rs_n = min(n_segments, 255)
    if rs_n < 17:
        raise ValueError(
            f"Audio too short: only {rs_n} segments (need ≥ 17 for RS with K=16)"
        )

    # 7. Manifest
    manifest = CryptographicManifest(
        content_id=content_id,
        content_hash_sha256=content_hash,
        fingerprints_audio=[fp.hash_hex for fp in fingerprints],
        fingerprints_video=[],
        author_id=certificate.author_id,
        author_public_key=certificate.public_key_pem,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    # 8. Sign
    signature = sign_manifest(manifest, private_key_pem)

    # 9. Derive WID
    wid = derive_wid(signature, content_id)

    # 10. Pilot seed
    pilot_hash_48 = int.from_bytes(
        hashlib.sha256(content_id.encode()).digest()[:6], "big"
    )
    global_pn_seed = int.from_bytes(
        hmac.new(pepper, b"global_pilot_seed", hashlib.sha256).digest()[:8], "big"
    )

    # 11–12. Hopping plan + RS symbols
    band_configs = plan_audio_hopping(rs_n, content_id, certificate.public_key_pem, pepper)
    rs_symbols = ReedSolomonCodec(rs_n).encode(wid.data)

    # 13-15. Pass 2 Streaming: Encode and Embed
    with tempfile.NamedTemporaryFile(suffix=media_path.suffix, delete=False) as tmp:
        signed_path = Path(tmp.name)

    encoder_proc = media.encode_audio_stream(
        sample_rate=target_sample_rate,
        output_path=signed_path,
        codec="aac",
        bitrate="256k"
    )

    try:
        for seg_idx, chunk, _ in media.iter_audio_segments(
            media_path, segment_duration_s=2.0, target_sample_rate=target_sample_rate
        ):
            # 13. Embed pilot on the chunk
            chunk = embed_audio_pilot(chunk, target_sample_rate, pilot_hash_48, global_pn_seed)

            # 14. Embed WID beacon (if within rs_n range)
            if seg_idx < rs_n:
                pn_seed = int.from_bytes(
                    hmac.new(
                        pepper,
                        f"wid|{content_id}|{certificate.public_key_pem}|{seg_idx}".encode(),
                        hashlib.sha256,
                    ).digest()[:8],
                    "big",
                )
                chunk = embed_audio_segment(
                    chunk, rs_symbols[seg_idx], band_configs[seg_idx], pn_seed
                )

            # 15. Write to encoder pipe
            if encoder_proc.stdin:
                pcm_bytes = (chunk * 32768.0).astype(np.int16).tobytes()
                encoder_proc.stdin.write(pcm_bytes)
    finally:
        if encoder_proc.stdin:
            encoder_proc.stdin.close()
        encoder_proc.wait()

    try:
        # 16. Store

        storage_key = f"signed/{content_id}/output{media_path.suffix}"
        await storage.put(storage_key, signed_path.read_bytes(), "audio/aac")
    finally:
        signed_path.unlink(missing_ok=True)

    # 17. Persist to registry
    active_signals = ["pilot_audio", "wid_audio", "fingerprint_audio"]
    await registry.save_video(VideoEntry(
        content_id=content_id,
        author_id=certificate.author_id,
        author_public_key=certificate.public_key_pem,
        active_signals=active_signals,
        rs_n=rs_n,
        pilot_hash_48=pilot_hash_48,
        manifest_signature=signature,
        manifest_json=_manifest_to_json(manifest),
    ))
    await registry.save_segments(content_id, fingerprints, is_original=True)

    # 18. Return
    return SigningResult(
        content_id=content_id,
        signed_media_key=storage_key,
        manifest=manifest,
        signature=signature,
        wid=wid,
        active_signals=active_signals,
        rs_n=rs_n,
        pilot_hash_48=pilot_hash_48,
    )


async def sign_video(
    media_path: Path,
    certificate: Certificate,
    private_key_pem: str,
    storage: StoragePort,
    registry: RegistryPort,
    pepper: bytes,
    media: MediaPort,
) -> SigningResult:
    """
    Video-only signing pipeline.
    Raises ValueError on audio-only containers or too-short video.
    """
    # 1. Probe — reject audio-only containers
    profile = media.probe(media_path)
    if not profile.has_video:
        raise ValueError("Container has no video track — cannot sign video-only pipeline")

    # 2–3. IDs and content hash
    content_id = str(uuid4())
    content_hash = hashlib.sha256(media_path.read_bytes()).hexdigest()

    # 4. Pilot hash
    pilot_hash = compute_pilot_hash_48(content_id)

    # 5. Video fingerprint (drives segment count)
    video_fingerprints = extract_video_hashes(
        str(media_path),
        key_material=pepper,
        pepper=pepper,
    )

    # 6. RS parameters
    n_segments = len(video_fingerprints)
    rs_n = min(n_segments, 255)
    if rs_n < 17:
        raise ValueError(
            f"Video too short: only {rs_n} segments (need ≥ 17 for RS with K=16)"
        )

    # 7. Manifest
    active_signals = ["video_pilot", "video_wid", "video_fingerprint"]
    manifest = CryptographicManifest(
        content_id=content_id,
        content_hash_sha256=content_hash,
        fingerprints_audio=[],
        fingerprints_video=[fp.hash_hex for fp in video_fingerprints],
        author_id=certificate.author_id,
        author_public_key=certificate.public_key_pem,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    # 8. Sign
    signature = sign_manifest(manifest, private_key_pem)

    # 9. Derive WID
    wid = derive_wid(signature, content_id)

    # 10. RS symbols
    rs_symbols = ReedSolomonCodec(rs_n).encode(wid.data)

    # 11. Read all frames
    all_frames, fps = media.read_video_frames(media_path)
    frames_per_segment = int(VIDEO_SEGMENT_S * fps)

    # 12. Per-segment embedding loop
    for seg_idx in range(rs_n):
        start_frame = seg_idx * frames_per_segment
        end_frame = start_frame + frames_per_segment
        if end_frame > len(all_frames):
            break

        segment_frames = all_frames[start_frame:end_frame]

        # Embed pilot on each frame
        for j, frame in enumerate(segment_frames):
            segment_frames[j] = embed_video_pilot(frame, content_id, pepper)

        # Embed WID
        symbol_bits = np.array(
            [(rs_symbols[seg_idx] >> (7 - k)) & 1 for k in range(8)],
            dtype=np.uint8,
        )
        segment_frames = embed_video_segment(
            segment_frames,
            symbol_bits,
            content_id,
            certificate.public_key_pem,
            seg_idx,
            pepper,
        )

        all_frames[start_frame:end_frame] = segment_frames

    # 13. Write embedded video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        signed_path = Path(tmp.name)
    try:
        media.write_video_frames(all_frames, fps, signed_path)

        # 14. Store
        storage_key = f"signed/{content_id}/output.mp4"
        await storage.put(storage_key, signed_path.read_bytes(), "video/mp4")
    finally:
        signed_path.unlink(missing_ok=True)

    # 15. Persist
    await registry.save_video(VideoEntry(
        content_id=content_id,
        author_id=certificate.author_id,
        author_public_key=certificate.public_key_pem,
        active_signals=active_signals,
        rs_n=rs_n,
        pilot_hash_48=pilot_hash,
        manifest_signature=signature,
        manifest_json=_manifest_to_json(manifest),
    ))
    await registry.save_segments(content_id, video_fingerprints, is_original=True)

    return SigningResult(
        content_id=content_id,
        signed_media_key=storage_key,
        manifest=manifest,
        signature=signature,
        wid=wid,
        active_signals=active_signals,
        rs_n=rs_n,
        pilot_hash_48=pilot_hash,
    )


async def sign_av(
    media_path: Path,
    certificate: Certificate,
    private_key_pem: str,
    storage: StoragePort,
    registry: RegistryPort,
    pepper: bytes,
    media: MediaPort,
) -> SigningResult:
    """
    Audio+Video signing pipeline with a SINGLE shared WID.

    A single CryptographicManifest covers both signals. The WID is derived
    once from the single Ed25519 signature and embedded in BOTH the audio DWT
    band AND the video DCT coefficients.

    An adversary who replaces either track cannot produce a valid WID for that
    channel without the original Ed25519 private key — the WID is derived from
    the signature over the original manifest which committed to BOTH channels.
    """
    # 1. Probe — require both tracks
    profile = media.probe(media_path)
    if not profile.has_video or not profile.has_audio:
        raise ValueError(
            "sign_av requires both audio and video tracks. "
            f"has_video={profile.has_video}, has_audio={profile.has_audio}"
        )

    # 2–3. IDs + content hash
    content_id = str(uuid4())
    content_hash = hashlib.sha256(media_path.read_bytes()).hexdigest()

    # 4. Pilot hash (video)
    pilot_hash = compute_pilot_hash_48(content_id)

    # 5. Audio fingerprints (streaming — avoids loading all audio into memory)
    target_sample_rate = 44100
    chunk_stream = (chunk for _, chunk, _ in media.iter_audio_segments(
        media_path, segment_duration_s=2.0, target_sample_rate=target_sample_rate
    ))
    audio_fingerprints = extract_audio_hashes_from_stream(
        chunk_stream, target_sample_rate, key_material=pepper, pepper=pepper
    )

    # 6. Video fingerprints
    video_fingerprints = extract_video_hashes(
        str(media_path), key_material=pepper, pepper=pepper
    )

    # 7. RS parameters — use the SMALLER rs_n so both channels stay in sync
    rs_n_audio = min(len(audio_fingerprints), 255)
    rs_n_video = min(len(video_fingerprints), 255)
    if rs_n_audio < 17:
        raise ValueError(
            f"Audio too short: only {rs_n_audio} segments (need ≥ 17 for RS with K=16)"
        )
    if rs_n_video < 17:
        raise ValueError(
            f"Video too short: only {rs_n_video} segments (need ≥ 17 for RS with K=16)"
        )
    rs_n = min(rs_n_audio, rs_n_video)

    # 8. Single manifest covering BOTH channels
    active_signals = [
        "audio_pilot", "audio_wid", "audio_fingerprint",
        "video_pilot", "video_wid", "video_fingerprint",
    ]
    manifest = CryptographicManifest(
        content_id=content_id,
        content_hash_sha256=content_hash,
        fingerprints_audio=[fp.hash_hex for fp in audio_fingerprints],
        fingerprints_video=[fp.hash_hex for fp in video_fingerprints],
        author_id=certificate.author_id,
        author_public_key=certificate.public_key_pem,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    # 9. Single Ed25519 signature
    signature = sign_manifest(manifest, private_key_pem)

    # 10. Single shared WID — embedded in BOTH audio and video
    wid = derive_wid(signature, content_id)

    # 11. Shared RS codeword
    rs_symbols = ReedSolomonCodec(rs_n).encode(wid.data)

    # 12. Audio pilot seed
    global_pn_seed = int.from_bytes(
        hmac.new(pepper, b"global_pilot_seed", hashlib.sha256).digest()[:8], "big"
    )
    band_configs = plan_audio_hopping(rs_n, content_id, certificate.public_key_pem, pepper)

    # 13. Embed audio — streaming pass (avoids loading all audio into memory)
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        audio_signed_path = Path(tmp.name)

    encoder_proc = media.encode_audio_stream(
        sample_rate=target_sample_rate,
        output_path=audio_signed_path,
        codec="aac",
        bitrate="192k",
    )
    try:
        for seg_idx, chunk, _ in media.iter_audio_segments(
            media_path, segment_duration_s=2.0, target_sample_rate=target_sample_rate
        ):
            chunk = embed_audio_pilot(chunk, target_sample_rate, pilot_hash, global_pn_seed)
            if seg_idx < rs_n:
                pn_seed = int.from_bytes(
                    hmac.new(
                        pepper,
                        f"wid|{content_id}|{certificate.public_key_pem}|{seg_idx}".encode(),
                        hashlib.sha256,
                    ).digest()[:8],
                    "big",
                )
                chunk = embed_audio_segment(
                    chunk, rs_symbols[seg_idx], band_configs[seg_idx], pn_seed,
                    target_snr_db=-6.0,  # -14 dB is destroyed by AAC 192k; -6 dB survives
                )
            if encoder_proc.stdin:
                encoder_proc.stdin.write((chunk * 32768.0).astype(np.int16).tobytes())
    finally:
        if encoder_proc.stdin:
            encoder_proc.stdin.close()
        encoder_proc.wait()

    # 14. Embed video — load all frames, embed with frame_stride=3 for efficiency
    _VIDEO_EMBED_STRIDE = 3
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_signed_path = Path(tmp.name)

    try:
        all_frames, fps = media.read_video_frames(media_path)
        frames_per_segment = int(VIDEO_SEGMENT_S * fps)

        for seg_idx in range(rs_n):
            start = seg_idx * frames_per_segment
            end = min(start + frames_per_segment, len(all_frames))
            if start >= len(all_frames):
                break

            symbol_bits = np.array(
                [(rs_symbols[seg_idx] >> (7 - k)) & 1 for k in range(8)],
                dtype=np.uint8,
            )

            # Embed pilot + WID only in strided frames (memory efficient; majority vote
            # at extraction time operates on available frames regardless of count)
            strided_indices = list(range(start, end, _VIDEO_EMBED_STRIDE))
            strided_frames = [all_frames[i] for i in strided_indices]

            for j, frame in enumerate(strided_frames):
                strided_frames[j] = embed_video_pilot(frame, content_id, pepper)

            strided_frames = embed_video_segment(
                strided_frames,
                symbol_bits,
                content_id,
                certificate.public_key_pem,
                seg_idx,
                pepper,
            )

            for j, i in enumerate(strided_indices):
                all_frames[i] = strided_frames[j]

        media.write_video_frames(all_frames, fps, video_signed_path)

        # 15. Mux signed audio + signed video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = Path(tmp.name)

        try:
            media.mux_video_audio(video_signed_path, audio_signed_path, output_path)

            # 16. Store
            storage_key = f"signed/{content_id}/output.mp4"
            await storage.put(storage_key, output_path.read_bytes(), "video/mp4")
        finally:
            output_path.unlink(missing_ok=True)

    finally:
        video_signed_path.unlink(missing_ok=True)
        audio_signed_path.unlink(missing_ok=True)

    # 17. Persist — single VideoEntry with both active signals
    await registry.save_video(VideoEntry(
        content_id=content_id,
        author_id=certificate.author_id,
        author_public_key=certificate.public_key_pem,
        active_signals=active_signals,
        rs_n=rs_n,
        pilot_hash_48=pilot_hash,
        manifest_signature=signature,
        manifest_json=_manifest_to_json(manifest),
    ))
    # Store both audio and video fingerprints under the same content_id so
    # Phase A candidate lookup succeeds when either channel is queried.
    await registry.save_segments(content_id, audio_fingerprints, is_original=True)
    await registry.save_segments(content_id, video_fingerprints, is_original=True)

    return SigningResult(
        content_id=content_id,
        signed_media_key=storage_key,
        manifest=manifest,
        signature=signature,
        wid=wid,
        active_signals=active_signals,
        rs_n=rs_n,
        pilot_hash_48=pilot_hash,
    )
