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

from kernel_backend.core.domain.dsp_manifest import PRODUCTION_MANIFEST as _M
from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.domain.manifest import CryptographicManifest
from kernel_backend.core.domain.signing import RawSigningPayload, SigningResult
from kernel_backend.core.domain.watermark import (
    AudioEmbeddingParams,
    EmbeddingParams,
    VideoEmbeddingParams,
    embedding_params_from_dict,
    embedding_params_to_dict,
    SegmentFingerprint,
    VideoEntry,
    WatermarkID,
)
from kernel_backend.core.ports.media import MediaPort
from kernel_backend.core.ports.registry import RegistryPort
from kernel_backend.core.ports.storage import StoragePort
from kernel_backend.core.services.crypto_service import derive_wid, sign_manifest
from kernel_backend.core.domain.content_profile import routing_decision_to_dict
from kernel_backend.core.domain.video_content_profile import (
    video_routing_decision_to_dict as _video_routing_to_dict,
)
from kernel_backend.engine.audio.algorithm_router import (
    route as route_audio,
    routing_decision_to_audio_params,
)
from kernel_backend.engine.audio.content_profiler import (
    SUBSAMPLE_DURATION_S,
    SUBSAMPLE_N_SEGMENTS,
    SUBSAMPLE_THRESHOLD_S,
    profile_audio,
)
from kernel_backend.engine.video.content_profiler import profile_video as _profile_video
from kernel_backend.engine.video.algorithm_router import (
    route as _route_video,
    video_routing_decision_to_video_params as _video_routing_to_params,
)
from kernel_backend.engine.audio.fingerprint import (
    extract_hashes as extract_audio_hashes,
    extract_hashes_from_stream as extract_audio_hashes_from_stream,
)
from kernel_backend.engine.audio.wid_beacon import embed_segment as embed_audio_segment
from kernel_backend.engine.codec.hopping import plan_audio_hopping, plan_video_hopping
from kernel_backend.engine.codec.reed_solomon import ReedSolomonCodec
from kernel_backend.engine.video.fingerprint import extract_hashes as extract_video_hashes
from kernel_backend.engine.video.fingerprint import SEGMENT_DURATION_S as VIDEO_SEGMENT_S
from kernel_backend.engine.video.wid_watermark import (
    embed_segment as embed_video_segment,
    embed_video_frame,
    frame_to_yuv420,
)


# ── Output encoding quality helpers ──────────────────────────────────────────

_LOSSLESS_MARKER_BPS: int = 1_411_200   # ≥ this → source is lossless
_AUDIO_FLOOR_BPS: int = 256_000
_AUDIO_CEILING_BPS: int = 384_000


def _compute_output_audio_bitrate(
    source_bps: int,
    cap_lossless: bool = False,
) -> str | None:
    """
    Adaptive AAC bitrate for signed audio output.

    Returns None when source is lossless AND cap_lossless=False — caller uses
    pcm_s16le / WAV instead of AAC (audio-only path).

    Returns '384k' when source is lossless AND cap_lossless=True — AV path
    avoids PCM-in-MP4 for universal container compatibility.

    Floor: 256k. Ceiling: 384k.
    """
    if source_bps >= _LOSSLESS_MARKER_BPS:
        return "384k" if cap_lossless else None
    target = max(source_bps, _AUDIO_FLOOR_BPS)
    target = min(target, _AUDIO_CEILING_BPS)
    if target >= 350_000:
        return "384k"
    if target >= 280_000:
        return "320k"
    return "256k"


def _compute_output_video_crf(duration_s: float) -> int:
    """
    Duration-adaptive CRF to bound output file size.

    Policy (calibration: 0/24 watermark errors at CRF 18/20/23/28):
      ≤ 15 min → CRF 18  (~<1 GB at 1080p, maximum quality)
      ≤ 30 min → CRF 20  (~<2 GB at 1080p, high quality)
      ≤ 60 min → CRF 23  (bounded size, calibrated safe)
    """
    if duration_s <= 900:    # 15 min
        return 18
    if duration_s <= 1800:   # 30 min
        return 20
    return 23


# ── End output encoding helpers ───────────────────────────────────────────────

_DEFAULT_AUDIO_PARAMS = AudioEmbeddingParams(
    dwt_levels=_M.audio_wid.dwt_levels,
    chips_per_bit=_M.audio_wid.chips_per_bit,
    psychoacoustic=_M.audio_wid.psychoacoustic,
    safety_margin_db=_M.audio_wid.safety_margin_db,
    target_snr_db=_M.audio_wid.target_snr_db_audio_only,
)

_DEFAULT_VIDEO_PARAMS = VideoEmbeddingParams(
    jnd_adaptive=_M.video_wid.jnd_adaptive,
    qim_step_base=_M.video_wid.qim_step_base,
    qim_step_min=_M.video_wid.qim_step_min,
    qim_step_max=_M.video_wid.qim_step_max,
    qim_quantize_to=_M.video_wid.qim_quantize_to,
)

_DEFAULT_AV_AUDIO_PARAMS = AudioEmbeddingParams(
    dwt_levels=_M.audio_wid.dwt_levels,
    chips_per_bit=_M.audio_wid.chips_per_bit,
    psychoacoustic=_M.audio_wid.psychoacoustic,
    safety_margin_db=_M.audio_wid.safety_margin_db,
    target_snr_db=_M.audio_wid.target_snr_db_av,
)

_DEFAULT_EMBEDDING_PARAMS = EmbeddingParams(
    audio=_DEFAULT_AUDIO_PARAMS,
    video=_DEFAULT_VIDEO_PARAMS,
)


def _make_signed_name(
    original_filename: str,
    fallback_ext: str,
    force_ext: str | None = None,
) -> str:
    """Build a storage-safe signed filename from the original upload name.

    force_ext overrides the source file extension — used when the output format
    differs from the input (e.g. FLAC input → WAV output for lossless audio).
    """
    if original_filename:
        p = Path(original_filename)
        stem = p.stem
        ext = force_ext or p.suffix or fallback_ext
    else:
        stem = "output"
        ext = force_ext or fallback_ext
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
        embedding_params_from_dict(raw_ep)
        if raw_ep is not None
        else _DEFAULT_EMBEDDING_PARAMS
    )

    await registry.save_video(VideoEntry(
        content_id=payload["content_id"],
        author_id=payload["author_id"],
        author_public_key=payload["author_public_key"],
        active_signals=payload["active_signals"],
        rs_n=payload["rs_n"],
        manifest_signature=signature,
        embedding_params=embedding_params,
        manifest_json=payload["manifest_json"],
        org_id=org_id,
        signed_media_key=payload["signed_media_key"],
        output_encoding_params=payload.get("output_encoding_params"),
        routing_metadata=payload.get("routing_metadata"),
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


# ── Content profiling helper ────────────────────────────────────────────────

def _load_profiling_segments(media_path: Path, duration_s: float) -> list[np.ndarray]:
    """Load audio segments for content profiling.

    For files <= 10 min: load full audio at 22050 Hz.
    For files > 10 min: load 3x30s segments (start, middle, end).
    """
    import librosa

    profiling_sr = 22050
    if duration_s <= SUBSAMPLE_THRESHOLD_S:
        samples, _ = librosa.load(media_path, sr=profiling_sr, mono=True)
        return [samples]

    offsets = [
        0.0,
        max(0.0, duration_s / 2 - SUBSAMPLE_DURATION_S / 2),
        max(0.0, duration_s - SUBSAMPLE_DURATION_S),
    ]
    segments = []
    for offset in offsets:
        seg, _ = librosa.load(
            media_path, sr=profiling_sr, mono=True,
            offset=offset, duration=SUBSAMPLE_DURATION_S,
        )
        segments.append(seg)
    return segments


def _sample_video_frames(
    media_path: Path,
    media: MediaPort,
    n_frames: int = 5,
) -> list[np.ndarray]:
    """Sample representative BGR frames for video content profiling.

    Grabs frames at positions [10%, 25%, 50%, 75%, 90%] of the video.
    Uses iter_video_segments with large stride to minimize I/O.
    """
    import cv2

    cap = cv2.VideoCapture(str(media_path))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 300  # fallback

        positions = [0.10, 0.25, 0.50, 0.75, 0.90]
        frame_indices = [int(p * total_frames) for p in positions[:n_frames]]

        frames: list[np.ndarray] = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        if not frames:
            # Fallback: read first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        if not frames:
            raise ValueError("Could not read any frames from video for profiling")

        return frames
    finally:
        cap.release()


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
    audio_params: AudioEmbeddingParams | None = None,
) -> RawSigningPayload:
    """CPU phase of audio signing. Returns a serialisable RawSigningPayload."""
    # 1. Probe
    profile = media.probe(media_path)
    if profile.container_type == "video_only":
        raise ValueError("Container has no audio track — cannot sign audio-only pipeline")

    # 1b. Content profiling + adaptive routing (unless caller supplied explicit params)
    routing_meta: dict | None = None
    if audio_params is not None:
        ap = audio_params
    else:
        profiling_segments = _load_profiling_segments(media_path, profile.duration_s)
        content_profile = profile_audio(np.concatenate(profiling_segments), 22050)
        routing = route_audio(content_profile)
        ap = routing_decision_to_audio_params(routing)
        routing_meta = routing_decision_to_dict(routing)

    # 2–3. IDs and content hash
    content_id = str(uuid4())
    content_hash = hashlib.sha256(media_path.read_bytes()).hexdigest()

    # 4. Target sample rate for audio orchestration
    target_sample_rate = 44100

    # 5. Fingerprint (drives segment count) — Pass 1 Streaming
    # Count non-overlapping 2s chunks alongside fingerprint extraction.
    # extract_hashes_from_stream uses overlap=0.5, so len(fingerprints) > n_wid_segments.
    # rs_n must match the number of non-overlapping WID segments, not fingerprint count.
    wid_segment_counter: list[int] = [0]

    def _counting_stream():
        for _, chunk, _ in media.iter_audio_segments(
            media_path, segment_duration_s=2.0, target_sample_rate=target_sample_rate
        ):
            wid_segment_counter[0] += 1
            yield chunk

    fingerprints = extract_audio_hashes_from_stream(
        _counting_stream(),
        target_sample_rate,
        key_material=pepper,
        pepper=pepper,
    )
    n_wid_segments = wid_segment_counter[0]

    # 6. RS parameters — based on non-overlapping WID segment count
    rs_n = min(n_wid_segments, 255)
    if rs_n < 17:
        duration_s = n_wid_segments * 2  # 2 seconds per segment
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

    # 10–11. Hopping plan + RS symbols
    band_configs = plan_audio_hopping(
        rs_n, content_id, certificate.public_key_pem, pepper,
        force_levels=list(ap.dwt_levels),
        target_subband=ap.target_subband,
    )
    rs_symbols = ReedSolomonCodec(rs_n).encode(wid.data)

    # 13–15. Pass 2 Streaming: Encode and Embed
    # Determine output format before creating temp file so the suffix matches.
    # For lossless sources (WAV/FLAC/ALAC): write watermarked PCM back as WAV —
    # no lossy re-encoding stage; DWT-DSSS works on raw samples internally.
    # For lossy sources: AAC with adaptive bitrate ≥ source, floor 256k, ceiling 384k.
    #
    # Note on .m4a vs .wav: .m4a creates an edit list (elst) that compensates for
    # the ~1024-sample AAC encoder priming delay. For WAV output the delay is zero.
    output_bitrate = _compute_output_audio_bitrate(profile.audio_bitrate_bps)
    is_lossless_out = output_bitrate is None
    out_suffix       = ".wav" if is_lossless_out else ".m4a"
    out_codec        = "pcm_s16le" if is_lossless_out else "aac"
    out_content_type = "audio/wav" if is_lossless_out else "audio/aac"

    with tempfile.NamedTemporaryFile(suffix=out_suffix, delete=False) as tmp:
        signed_path = Path(tmp.name)

    encoder_proc = media.encode_audio_stream(
        sample_rate=target_sample_rate,
        output_path=signed_path,
        codec=out_codec,
        bitrate=output_bitrate,   # None → no -b:a flag for PCM
    )
    try:
        for seg_idx, chunk, _ in media.iter_audio_segments(
            media_path, segment_duration_s=2.0, target_sample_rate=target_sample_rate
        ):
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
                    chips_per_bit=ap.chips_per_bit,
                    target_snr_db=ap.target_snr_db,
                    use_psychoacoustic=ap.psychoacoustic,
                )
            if encoder_proc.stdin:
                encoder_proc.stdin.write(
                    (np.clip(chunk, -1.0, 1.0) * 32768.0).astype(np.int16).tobytes()
                )
    finally:
        if encoder_proc.stdin:
            encoder_proc.stdin.close()
        encoder_proc.wait()

    signed_name = _make_signed_name(original_filename, media_path.suffix, force_ext=out_suffix)
    storage_key = f"signed/{content_id}/{signed_name}"
    active_signals = ["wid_audio", "fingerprint_audio"]

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
        active_signals=active_signals,
        storage_key=storage_key,
        signed_media_key=storage_key,
        signed_file_path=str(signed_path),
        content_type=out_content_type,
        audio_fingerprints=[
            {"time_offset_ms": fp.time_offset_ms, "hash_hex": fp.hash_hex}
            for fp in fingerprints
        ],
        video_fingerprints=None,
        media_type="audio",
        embedding_params=embedding_params_to_dict(
            EmbeddingParams(audio=ap, video=None)
        ),
        output_encoding_params={
            "audio": {
                "codec": out_codec,
                "bitrate": output_bitrate,   # None if lossless (WAV output)
                "sample_rate": target_sample_rate,
            }
        },
        routing_metadata=routing_meta,
    )


def _sign_video_cpu(
    media_path: Path,
    certificate: Certificate,
    private_key_pem: str,
    pepper: bytes,
    media: MediaPort,
    org_id: UUID | None = None,
    original_filename: str = "",
    video_params: VideoEmbeddingParams | None = None,
    output_crf: int | None = None,
) -> RawSigningPayload:
    """CPU phase of video signing. Returns a serialisable RawSigningPayload."""
    # 1. Probe — reject audio-only containers
    profile = media.probe(media_path)
    if not profile.has_video:
        raise ValueError("Container has no video track — cannot sign video-only pipeline")

    # 1b. Video content profiling + adaptive routing
    video_routing_meta: dict | None = None
    if video_params is not None:
        vp = video_params
    else:
        sample_frames = _sample_video_frames(media_path, media)
        video_profile = _profile_video(sample_frames)
        video_routing = _route_video(video_profile)
        vp = _video_routing_to_params(video_routing)
        video_routing_meta = _video_routing_to_dict(video_routing)

    # 2–3. IDs and content hash
    content_id = str(uuid4())
    content_hash = hashlib.sha256(media_path.read_bytes()).hexdigest()

    # 4. Video fingerprint (drives segment count)
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
    active_signals = ["video_wid", "video_fingerprint"]
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

    crf_to_use = output_crf if output_crf is not None else _compute_output_video_crf(profile.duration_s)
    encoder_proc = media.open_video_encode_stream(width, height, fps, signed_path, crf=crf_to_use)

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
                frame = embed_video_frame(
                    frame, symbol_bits, content_id,
                    certificate.public_key_pem, seg_idx, pepper,
                    use_jnd_adaptive=vp.jnd_adaptive,
                    jnd_params=vp,
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
        embedding_params=embedding_params_to_dict(
            EmbeddingParams(audio=None, video=vp)
        ),
        output_encoding_params={
            "video": {"codec": "libx264", "crf": crf_to_use, "preset": "ultrafast"},
        },
        routing_metadata=(
            {"version": 2, "video": video_routing_meta}
            if video_routing_meta is not None else None
        ),
    )


def _sign_av_cpu(
    media_path: Path,
    certificate: Certificate,
    private_key_pem: str,
    pepper: bytes,
    media: MediaPort,
    org_id: UUID | None = None,
    original_filename: str = "",
    audio_params: AudioEmbeddingParams | None = None,
    video_params: VideoEmbeddingParams | None = None,
    output_crf: int | None = None,
) -> RawSigningPayload:
    """CPU phase of AV signing (single shared WID). Returns a serialisable RawSigningPayload."""
    # 1. Probe — require both tracks
    profile = media.probe(media_path)
    if not profile.has_video or not profile.has_audio:
        raise ValueError(
            "sign_av requires both audio and video tracks. "
            f"has_video={profile.has_video}, has_audio={profile.has_audio}"
        )

    # 1b. Content profiling + adaptive routing for audio track
    audio_routing_meta: dict | None = None
    if audio_params is not None:
        ap = audio_params
    else:
        profiling_segments = _load_profiling_segments(media_path, profile.duration_s)
        content_profile = profile_audio(np.concatenate(profiling_segments), 22050)
        routing = route_audio(content_profile)
        ap = routing_decision_to_audio_params(routing)
        audio_routing_meta = routing_decision_to_dict(routing)

        # Silent audio track → delegate to video-only pipeline (audio WID
        # cannot survive on a silent carrier, so embedding it only produces
        # false-RED verdicts on verification).
        if content_profile.content_type == "silence":
            import logging
            logging.getLogger(__name__).warning(
                "Audio track classified as silence (conf=%.2f) — "
                "falling back to video-only signing pipeline.",
                content_profile.confidence,
            )
            return _sign_video_cpu(
                media_path, certificate, private_key_pem, pepper, media,
                org_id=org_id, original_filename=original_filename,
                video_params=video_params, output_crf=output_crf,
            )

    # 1c. Video content profiling + adaptive routing
    video_routing_meta: dict | None = None
    if video_params is not None:
        vp = video_params
    else:
        sample_frames = _sample_video_frames(media_path, media)
        video_profile = _profile_video(sample_frames)
        video_routing = _route_video(video_profile)
        vp = _video_routing_to_params(video_routing)
        video_routing_meta = _video_routing_to_dict(video_routing)

    # Merge audio + video routing metadata
    routing_meta: dict | None = None
    if audio_routing_meta is not None or video_routing_meta is not None:
        routing_meta = {
            "version": 2,
            "audio": audio_routing_meta,
            "video": video_routing_meta,
        }

    # 2–3. IDs + content hash
    content_id = str(uuid4())
    content_hash = hashlib.sha256(media_path.read_bytes()).hexdigest()

    # 4. Audio fingerprints (streaming — avoids loading all audio into memory)
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
        "audio_wid", "audio_fingerprint",
        "video_wid", "video_fingerprint",
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

    # 12. Audio hopping plan
    band_configs = plan_audio_hopping(
        rs_n, content_id, certificate.public_key_pem, pepper,
        force_levels=list(ap.dwt_levels),
        target_subband=ap.target_subband,
    )

    # 13. Embed audio — streaming pass
    # Adaptive bitrate: ≥ source, floor 256k (raised from 192k), ceiling 384k.
    # For lossless source audio, cap at 384k (PCM-in-MP4 has compatibility issues).
    output_audio_bitrate = _compute_output_audio_bitrate(
        profile.audio_bitrate_bps, cap_lossless=True
    )

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        audio_signed_path = Path(tmp.name)

    encoder_proc = media.encode_audio_stream(
        sample_rate=target_sample_rate,
        output_path=audio_signed_path,
        codec="aac",
        bitrate=output_audio_bitrate,
    )
    try:
        for seg_idx, chunk, _ in media.iter_audio_segments(
            media_path, segment_duration_s=2.0, target_sample_rate=target_sample_rate
        ):
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
                    chips_per_bit=ap.chips_per_bit,
                    target_snr_db=ap.target_snr_db,
                    use_psychoacoustic=ap.psychoacoustic,
                )
            if encoder_proc.stdin:
                encoder_proc.stdin.write(
                    (np.clip(chunk, -1.0, 1.0) * 32768.0).astype(np.int16).tobytes()
                )
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

    av_crf_to_use = output_crf if output_crf is not None else _compute_output_video_crf(profile.duration_s)
    video_encoder = media.open_video_encode_stream(av_width, av_height, av_fps, video_signed_path, crf=av_crf_to_use)
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
                frame = embed_video_frame(
                    frame, symbol_bits, content_id,
                    certificate.public_key_pem, seg_idx, pepper,
                    use_jnd_adaptive=vp.jnd_adaptive,
                    jnd_params=vp,
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
        embedding_params=embedding_params_to_dict(
            EmbeddingParams(audio=ap, video=vp)
        ),
        output_encoding_params={
            "audio": {
                "codec": "aac",
                "bitrate": output_audio_bitrate,
                "sample_rate": target_sample_rate,
            },
            "video": {"codec": "libx264", "crf": av_crf_to_use, "preset": "ultrafast"},
        },
        routing_metadata=routing_meta,
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
    audio_params: AudioEmbeddingParams | None = None,
) -> SigningResult:
    """
    Full audio signing pipeline. Orchestrates DSP, cryptography, storage, and
    registry operations. Raises ValueError on unsupported container or too-short
    audio.
    """
    payload = _sign_audio_cpu(media_path, certificate, private_key_pem, pepper, media, org_id, original_filename, audio_params=audio_params)
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
    video_params: VideoEmbeddingParams | None = None,
    output_crf: int | None = None,
) -> SigningResult:
    """
    Video-only signing pipeline.
    Raises ValueError on audio-only containers or too-short video.
    """
    payload = _sign_video_cpu(media_path, certificate, private_key_pem, pepper, media, org_id, original_filename, video_params=video_params, output_crf=output_crf)
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
    audio_params: AudioEmbeddingParams | None = None,
    video_params: VideoEmbeddingParams | None = None,
    output_crf: int | None = None,
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
    payload = _sign_av_cpu(media_path, certificate, private_key_pem, pepper, media, org_id, original_filename, audio_params=audio_params, video_params=video_params, output_crf=output_crf)
    await _persist_payload(payload, storage, registry)
    return _payload_to_signing_result(payload)
