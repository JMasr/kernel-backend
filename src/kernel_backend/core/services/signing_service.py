from __future__ import annotations

import base64
import hashlib
import hmac
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np

from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.domain.manifest import CryptographicManifest
from kernel_backend.core.domain.signing import RawSigningPayload, SigningResult
from kernel_backend.core.domain.watermark import (
    AudioEmbeddingParams,
    EmbeddingParams,
    VideoEmbeddingParams,
    embedding_params_to_dict,
    SegmentFingerprint,
    VideoEntry,
    WatermarkID,
)
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
from kernel_backend.engine.video.wid_watermark import (
    embed_segment as embed_video_segment,
    embed_video_frame,
    frame_to_yuv420,
)


_DEFAULT_AUDIO_PARAMS = AudioEmbeddingParams(
    dwt_levels=(1, 2),        # multi-band S4: embed in both levels 1 and 2 simultaneously
    chips_per_bit=32,         # calibrated WID constant (CLAUDE.md: 32 chips/bit for WID)
    psychoacoustic=True,       # activated Sprint 2: MPEG-1 Bark-domain amplitude profile
    safety_margin_db=3.0,
    target_snr_db=-14.0,      # fallback when psychoacoustic=False
)

_DEFAULT_VIDEO_PARAMS = VideoEmbeddingParams(
    jnd_adaptive=True,         # activated Sprint 3: Chou-Li JND adaptive QIM step
    qim_step_base=64.0,
    qim_step_min=44.0,
    qim_step_max=128.0,
    qim_quantize_to=4.0,
)

_DEFAULT_EMBEDDING_PARAMS = EmbeddingParams(
    audio=_DEFAULT_AUDIO_PARAMS,
    video=_DEFAULT_VIDEO_PARAMS,
)


def _make_signed_name(original_filename: str, fallback_ext: str) -> str:
    """Build a storage-safe signed filename from the original upload name."""
    if original_filename:
        p = Path(original_filename)
        stem = p.stem
        ext = p.suffix or fallback_ext
    else:
        stem = "output"
        ext = fallback_ext
    return f"{stem}_signed{ext}"


def _manifest_to_json(manifest: CryptographicManifest) -> str:
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


def _payload_to_signing_result(payload: RawSigningPayload) -> SigningResult:
    """Reconstruct a SigningResult from a RawSigningPayload (for public sign_* callers)."""
    manifest_data = json.loads(payload["manifest_json"])
    manifest = CryptographicManifest(
        content_id=manifest_data["content_id"],
        content_hash_sha256=manifest_data["content_hash_sha256"],
        fingerprints_audio=manifest_data["fingerprints_audio"],
        fingerprints_video=manifest_data["fingerprints_video"],
        author_id=manifest_data["author_id"],
        author_public_key=manifest_data["author_public_key"],
        created_at=manifest_data["created_at"],
    )
    return SigningResult(
        content_id=payload["content_id"],
        signed_media_key=payload["signed_media_key"],
        manifest=manifest,
        signature=base64.b64decode(payload["manifest_signature"]),
        wid=WatermarkID(data=bytes.fromhex(payload["wid_hex"])),
        active_signals=payload["active_signals"],
        rs_n=payload["rs_n"],
        pilot_hash_48=payload["pilot_hash_48"],
    )


async def _persist_payload(
    payload: RawSigningPayload,
    storage: StoragePort,
    registry: RegistryPort,
) -> None:
    """I/O phase: upload signed file to storage and persist metadata to registry.

    Reads and deletes the temp file at payload['signed_file_path'], then calls
    storage.put() and registry.save_video() / save_segments() with real adapters.
    """
    signed_path = Path(payload["signed_file_path"])
    try:
        signed_bytes = signed_path.read_bytes()
        await storage.put(payload["storage_key"], signed_bytes, payload["content_type"])
    finally:
        signed_path.unlink(missing_ok=True)

    signature = base64.b64decode(payload["manifest_signature"])
    org_id = UUID(payload["org_id"]) if payload["org_id"] is not None else None

    raw_ep = payload.get("embedding_params")
    embedding_params = (
        EmbeddingParams(
            audio=AudioEmbeddingParams(**dict(raw_ep["audio"], dwt_levels=tuple(raw_ep["audio"]["dwt_levels"]))),
            video=VideoEmbeddingParams(**raw_ep["video"]) if raw_ep.get("video") else None,
        )
        if raw_ep is not None
        else _DEFAULT_EMBEDDING_PARAMS
    )

    await registry.save_video(VideoEntry(
        content_id=payload["content_id"],
        author_id=payload["author_id"],
        author_public_key=payload["author_public_key"],
        active_signals=payload["active_signals"],
        rs_n=payload["rs_n"],
        pilot_hash_48=payload["pilot_hash_48"],
        manifest_signature=signature,
        embedding_params=embedding_params,
        manifest_json=payload["manifest_json"],
        org_id=org_id,
        signed_media_key=payload["signed_media_key"],
    ))

    if payload.get("audio_fingerprints"):
        fp_list = [
            SegmentFingerprint(f["time_offset_ms"], f["hash_hex"])
            for f in payload["audio_fingerprints"]  # type: ignore[union-attr]
        ]
        await registry.save_segments(payload["content_id"], fp_list, is_original=True)

    if payload.get("video_fingerprints"):
        fp_list = [
            SegmentFingerprint(f["time_offset_ms"], f["hash_hex"])
            for f in payload["video_fingerprints"]  # type: ignore[union-attr]
        ]
        await registry.save_segments(payload["content_id"], fp_list, is_original=True)


# ── CPU phases ────────────────────────────────────────────────────────────────
# These sync helpers perform all DSP work and write the signed file to a temp
# path. They return a RawSigningPayload dict — no async, no storage, no DB.
# The parent async loop calls _persist_payload() with the real adapters after
# the subprocess (or inline call) returns.

def _sign_audio_cpu(
    media_path: Path,
    certificate: Certificate,
    private_key_pem: str,
    pepper: bytes,
    media: MediaPort,
    org_id: UUID | None = None,
    original_filename: str = "",
) -> RawSigningPayload:
    """CPU phase of audio signing. Returns a serialisable RawSigningPayload."""
    # 1. Probe
    profile = media.probe(media_path)
    if profile.container_type == "video_only":
        raise ValueError("Container has no audio track — cannot sign audio-only pipeline")

    # 2–3. IDs and content hash
    content_id = str(uuid4())
    content_hash = hashlib.sha256(media_path.read_bytes()).hexdigest()

    # 4. Target sample rate for audio orchestration
    target_sample_rate = 44100

    # 5. Fingerprint (drives segment count) — Pass 1 Streaming
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
        duration_s = n_segments * 2  # 2 seconds per segment
        raise ValueError(
            f"Audio is too short to sign. Your file is approximately {duration_s} seconds, "
            f"but the minimum required duration is 34 seconds."
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
    band_configs = plan_audio_hopping(
        rs_n, content_id, certificate.public_key_pem, pepper,
        force_levels=list(_DEFAULT_AUDIO_PARAMS.dwt_levels),
    )
    rs_symbols = ReedSolomonCodec(rs_n).encode(wid.data)

    # 13–15. Pass 2 Streaming: Encode and Embed
    with tempfile.NamedTemporaryFile(suffix=media_path.suffix, delete=False) as tmp:
        signed_path = Path(tmp.name)

    encoder_proc = media.encode_audio_stream(
        sample_rate=target_sample_rate,
        output_path=signed_path,
        codec="aac",
        bitrate="256k",
    )
    try:
        for seg_idx, chunk, _ in media.iter_audio_segments(
            media_path, segment_duration_s=2.0, target_sample_rate=target_sample_rate
        ):
            chunk = embed_audio_pilot(
                chunk, target_sample_rate, pilot_hash_48, global_pn_seed,
                use_psychoacoustic=_DEFAULT_AUDIO_PARAMS.psychoacoustic,
            )
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
                    chips_per_bit=_DEFAULT_AUDIO_PARAMS.chips_per_bit,
                    use_psychoacoustic=_DEFAULT_AUDIO_PARAMS.psychoacoustic,
                    safety_margin_db=_DEFAULT_AUDIO_PARAMS.safety_margin_db,
                )
            if encoder_proc.stdin:
                encoder_proc.stdin.write((chunk * 32768.0).astype(np.int16).tobytes())
    finally:
        if encoder_proc.stdin:
            encoder_proc.stdin.close()
        encoder_proc.wait()

    signed_name = _make_signed_name(original_filename, media_path.suffix)
    storage_key = f"signed/{content_id}/{signed_name}"
    active_signals = ["pilot_audio", "wid_audio", "fingerprint_audio"]

    return RawSigningPayload(
        content_id=content_id,
        author_id=certificate.author_id,
        author_public_key=certificate.public_key_pem,
        org_id=str(org_id) if org_id is not None else None,
        content_hash_sha256=content_hash,
        manifest_json=_manifest_to_json(manifest),
        manifest_signature=base64.b64encode(signature).decode("ascii"),
        wid_hex=wid.data.hex(),
        rs_n=rs_n,
        pilot_hash_48=pilot_hash_48,
        active_signals=active_signals,
        storage_key=storage_key,
        signed_media_key=storage_key,
        signed_file_path=str(signed_path),
        content_type="audio/aac",
        audio_fingerprints=[
            {"time_offset_ms": fp.time_offset_ms, "hash_hex": fp.hash_hex}
            for fp in fingerprints
        ],
        video_fingerprints=None,
        media_type="audio",
        embedding_params=embedding_params_to_dict(
            EmbeddingParams(audio=_DEFAULT_AUDIO_PARAMS, video=None)
        ),
    )


def _sign_video_cpu(
    media_path: Path,
    certificate: Certificate,
    private_key_pem: str,
    pepper: bytes,
    media: MediaPort,
    org_id: UUID | None = None,
    original_filename: str = "",
) -> RawSigningPayload:
    """CPU phase of video signing. Returns a serialisable RawSigningPayload."""
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
        duration_s = n_segments * 5  # 5 seconds per segment
        raise ValueError(
            f"Video is too short to sign. Your file is approximately {duration_s} seconds, "
            f"but the minimum required duration is 85 seconds."
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

    # 11. Get dimensions without loading frames
    profile = media.probe(media_path)
    fps = profile.fps
    width, height = profile.width, profile.height

    # 12. Stream-encode: read one segment at a time, embed, pipe to FFmpeg
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        signed_path = Path(tmp.name)

    encoder_proc = media.open_video_encode_stream(width, height, fps, signed_path)

    try:
        for seg_idx, seg_frames, _ in media.iter_video_segments(
            media_path,
            segment_duration_s=VIDEO_SEGMENT_S,
            frame_stride=1,   # signing: all frames, no striding
        ):
            if seg_idx >= rs_n:
                # write remaining frames unmodified so video length is preserved
                for frame in seg_frames:
                    encoder_proc.stdin.write(frame_to_yuv420(frame))
                continue

            symbol_bits = np.array(
                [(rs_symbols[seg_idx] >> (7 - k)) & 1 for k in range(8)],
                dtype=np.uint8,
            )

            for frame in seg_frames:
                frame = embed_video_pilot(frame, content_id, pepper)
                frame = embed_video_frame(
                    frame, symbol_bits, content_id,
                    certificate.public_key_pem, seg_idx, pepper,
                    use_jnd_adaptive=_DEFAULT_VIDEO_PARAMS.jnd_adaptive,
                    jnd_params=_DEFAULT_VIDEO_PARAMS,
                )
                encoder_proc.stdin.write(frame_to_yuv420(frame))

    finally:
        if encoder_proc.stdin:
            encoder_proc.stdin.close()
        encoder_proc.wait()
        if encoder_proc.returncode != 0:
            signed_path.unlink(missing_ok=True)
            raise ValueError(
                f"FFmpeg video encode failed (returncode={encoder_proc.returncode})"
            )

    signed_name = _make_signed_name(original_filename, ".mp4")
    storage_key = f"signed/{content_id}/{signed_name}"

    return RawSigningPayload(
        content_id=content_id,
        author_id=certificate.author_id,
        author_public_key=certificate.public_key_pem,
        org_id=str(org_id) if org_id is not None else None,
        content_hash_sha256=content_hash,
        manifest_json=_manifest_to_json(manifest),
        manifest_signature=base64.b64encode(signature).decode("ascii"),
        wid_hex=wid.data.hex(),
        rs_n=rs_n,
        pilot_hash_48=pilot_hash,
        active_signals=active_signals,
        storage_key=storage_key,
        signed_media_key=storage_key,
        signed_file_path=str(signed_path),
        content_type="video/mp4",
        audio_fingerprints=None,
        video_fingerprints=[
            {"time_offset_ms": fp.time_offset_ms, "hash_hex": fp.hash_hex}
            for fp in video_fingerprints
        ],
        media_type="video",
        embedding_params=embedding_params_to_dict(_DEFAULT_EMBEDDING_PARAMS),
    )


def _sign_av_cpu(
    media_path: Path,
    certificate: Certificate,
    private_key_pem: str,
    pepper: bytes,
    media: MediaPort,
    org_id: UUID | None = None,
    original_filename: str = "",
) -> RawSigningPayload:
    """CPU phase of AV signing (single shared WID). Returns a serialisable RawSigningPayload."""
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
        duration_s = rs_n_audio * 2  # 2 seconds per segment
        raise ValueError(
            f"Audio track is too short to sign. Your file's audio is approximately {duration_s} seconds, "
            f"but the minimum required duration is 34 seconds."
        )
    if rs_n_video < 17:
        duration_s = rs_n_video * 5  # 5 seconds per segment
        raise ValueError(
            f"Video track is too short to sign. Your file's video is approximately {duration_s} seconds, "
            f"but the minimum required duration is 85 seconds."
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
    band_configs = plan_audio_hopping(
        rs_n, content_id, certificate.public_key_pem, pepper,
        force_levels=list(_DEFAULT_AUDIO_PARAMS.dwt_levels),
    )

    # 13. Embed audio — streaming pass
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
            chunk = embed_audio_pilot(
                chunk, target_sample_rate, pilot_hash, global_pn_seed,
                use_psychoacoustic=_DEFAULT_AUDIO_PARAMS.psychoacoustic,
            )
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
                    chips_per_bit=_DEFAULT_AUDIO_PARAMS.chips_per_bit,
                    target_snr_db=-6.0,  # -14 dB is destroyed by AAC 192k; -6 dB survives
                    use_psychoacoustic=_DEFAULT_AUDIO_PARAMS.psychoacoustic,
                    safety_margin_db=_DEFAULT_AUDIO_PARAMS.safety_margin_db,
                )
            if encoder_proc.stdin:
                encoder_proc.stdin.write((chunk * 32768.0).astype(np.int16).tobytes())
    finally:
        if encoder_proc.stdin:
            encoder_proc.stdin.close()
        encoder_proc.wait()

    # 14. Embed video — streaming encode, one segment at a time
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_signed_path = Path(tmp.name)

    av_profile = media.probe(media_path)
    av_fps = av_profile.fps
    av_width, av_height = av_profile.width, av_profile.height

    video_encoder = media.open_video_encode_stream(av_width, av_height, av_fps, video_signed_path)
    output_path: Path | None = None
    try:
        for seg_idx, seg_frames, _ in media.iter_video_segments(
            media_path,
            segment_duration_s=VIDEO_SEGMENT_S,
            frame_stride=1,   # signing: all frames, no striding
        ):
            if seg_idx >= rs_n:
                for frame in seg_frames:
                    video_encoder.stdin.write(frame_to_yuv420(frame))
                continue

            symbol_bits = np.array(
                [(rs_symbols[seg_idx] >> (7 - k)) & 1 for k in range(8)],
                dtype=np.uint8,
            )

            for frame in seg_frames:
                frame = embed_video_pilot(frame, content_id, pepper)
                frame = embed_video_frame(
                    frame, symbol_bits, content_id,
                    certificate.public_key_pem, seg_idx, pepper,
                    use_jnd_adaptive=_DEFAULT_VIDEO_PARAMS.jnd_adaptive,
                    jnd_params=_DEFAULT_VIDEO_PARAMS,
                )
                video_encoder.stdin.write(frame_to_yuv420(frame))

    finally:
        if video_encoder.stdin:
            video_encoder.stdin.close()
        video_encoder.wait()
        if video_encoder.returncode != 0:
            video_signed_path.unlink(missing_ok=True)
            raise ValueError(
                f"FFmpeg video encode failed in sign_av (returncode={video_encoder.returncode})"
            )

    output_path = None
    try:
        # 15. Mux signed audio + signed video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = Path(tmp.name)
        media.mux_video_audio(video_signed_path, audio_signed_path, output_path)

    except Exception:
        # Clean up the final output file on failure (intermediate files cleaned in finally)
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        raise
    finally:
        video_signed_path.unlink(missing_ok=True)
        audio_signed_path.unlink(missing_ok=True)

    assert output_path is not None  # always true when try block completed without exception
    signed_name = _make_signed_name(original_filename, ".mp4")
    storage_key = f"signed/{content_id}/{signed_name}"

    return RawSigningPayload(
        content_id=content_id,
        author_id=certificate.author_id,
        author_public_key=certificate.public_key_pem,
        org_id=str(org_id) if org_id is not None else None,
        content_hash_sha256=content_hash,
        manifest_json=_manifest_to_json(manifest),
        manifest_signature=base64.b64encode(signature).decode("ascii"),
        wid_hex=wid.data.hex(),
        rs_n=rs_n,
        pilot_hash_48=pilot_hash,
        active_signals=active_signals,
        storage_key=storage_key,
        signed_media_key=storage_key,
        signed_file_path=str(output_path),
        content_type="video/mp4",
        audio_fingerprints=[
            {"time_offset_ms": fp.time_offset_ms, "hash_hex": fp.hash_hex}
            for fp in audio_fingerprints
        ],
        video_fingerprints=[
            {"time_offset_ms": fp.time_offset_ms, "hash_hex": fp.hash_hex}
            for fp in video_fingerprints
        ],
        media_type="av",
        embedding_params=embedding_params_to_dict(_DEFAULT_EMBEDDING_PARAMS),
    )


# ── Public async functions ────────────────────────────────────────────────────
# Unchanged signatures — existing callers (tests, API routers) are unaffected.
# Each calls the corresponding CPU helper then _persist_payload.

async def sign_audio(
    media_path: Path,
    certificate: Certificate,
    private_key_pem: str,
    storage: StoragePort,
    registry: RegistryPort,
    pepper: bytes,
    media: MediaPort,
    org_id: UUID | None = None,
    original_filename: str = "",
) -> SigningResult:
    """
    Full audio signing pipeline. Orchestrates DSP, cryptography, storage, and
    registry operations. Raises ValueError on unsupported container or too-short
    audio.
    """
    payload = _sign_audio_cpu(media_path, certificate, private_key_pem, pepper, media, org_id, original_filename)
    await _persist_payload(payload, storage, registry)
    return _payload_to_signing_result(payload)


async def sign_video(
    media_path: Path,
    certificate: Certificate,
    private_key_pem: str,
    storage: StoragePort,
    registry: RegistryPort,
    pepper: bytes,
    media: MediaPort,
    org_id: UUID | None = None,
    original_filename: str = "",
) -> SigningResult:
    """
    Video-only signing pipeline.
    Raises ValueError on audio-only containers or too-short video.
    """
    payload = _sign_video_cpu(media_path, certificate, private_key_pem, pepper, media, org_id, original_filename)
    await _persist_payload(payload, storage, registry)
    return _payload_to_signing_result(payload)


async def sign_av(
    media_path: Path,
    certificate: Certificate,
    private_key_pem: str,
    storage: StoragePort,
    registry: RegistryPort,
    pepper: bytes,
    media: MediaPort,
    org_id: UUID | None = None,
    original_filename: str = "",
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
    payload = _sign_av_cpu(media_path, certificate, private_key_pem, pepper, media, org_id, original_filename)
    await _persist_payload(payload, storage, registry)
    return _payload_to_signing_result(payload)
