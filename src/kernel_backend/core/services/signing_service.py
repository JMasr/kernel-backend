from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import hmac
import json
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np

from kernel_backend.config import get_settings
from kernel_backend.core.domain.chunk import ChunkResult
from kernel_backend.core.domain.dsp_manifest import PRODUCTION_MANIFEST as _M
from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.domain.manifest import CryptographicManifest
from kernel_backend.core.domain.media import MediaProfile
from kernel_backend.core.domain.signing import RawSigningPayload, SigningResult
from kernel_backend.core.domain.watermark import (
    AudioEmbeddingParams,
    EmbeddingParams,
    SegmentMap,
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
from kernel_backend.core.services.chunk_assembler import (
    cleanup_chunks,
    concatenate_chunks,
    validate_chunks,
)
from kernel_backend.core.services.chunk_planner import plan_chunks
from kernel_backend.core.services.chunk_worker import process_video_chunk
from kernel_backend.core.services.crypto_service import (
    derive_wid,
    sign_manifest,
    streaming_file_hash,
)
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
from kernel_backend.engine.audio.segment_scorer import (
    score_segment,
    scores_from_raw_metrics,
    select_best,
)
from kernel_backend.engine.audio.wid_beacon import embed_segment as embed_audio_segment
from kernel_backend.engine.codec.hopping import plan_audio_hopping, plan_video_hopping
from kernel_backend.engine.codec.reed_solomon import ReedSolomonCodec
from kernel_backend.engine.video.fingerprint import extract_hashes as extract_video_hashes
from kernel_backend.engine.video.fingerprint import SEGMENT_DURATION_S as VIDEO_SEGMENT_S
from kernel_backend.engine.video.wid_watermark import (
    embed_segment as embed_video_segment,
    embed_video_frame_yuvj420_planes,
    frame_to_yuvj420_planes,
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


class _PCMChunkScratch:
    """
    On-disk spill buffer for float32 PCM segments.

    Pass 1 of the signing pipeline decodes the input audio once with ffmpeg
    and streams every 2s chunk through here instead of a Python list. Pass 2
    reads the chunks back sequentially, embeds the selected segments, and
    pipes the result to the output encoder.

    Peak RSS stays at ~one chunk (~353 KB) regardless of audio duration;
    disk usage is ~10 MB per minute of audio, transient, cleaned on exit.
    """

    def __init__(self, suffix: str = ".pcm") -> None:
        self._file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        self._path = Path(self._file.name)
        self._lengths: list[int] = []
        self._finalized = False

    def append(self, chunk: np.ndarray) -> None:
        """Write one chunk. Must be called before ``finalize()``."""
        arr = np.ascontiguousarray(chunk, dtype=np.float32)
        self._lengths.append(int(arr.size))
        self._file.write(arr.tobytes())

    def finalize(self) -> None:
        """Close the writer so the file can be re-opened for reading."""
        if not self._finalized:
            self._file.close()
            self._finalized = True

    def __len__(self) -> int:
        return len(self._lengths)

    def __iter__(self) -> "Iterator[np.ndarray]":
        """Yield chunks in the order they were appended."""
        if not self._finalized:
            self.finalize()
        with self._path.open("rb") as fh:
            for n in self._lengths:
                raw = fh.read(n * 4)
                yield np.frombuffer(raw, dtype=np.float32)

    def __enter__(self) -> "_PCMChunkScratch":
        return self

    def __exit__(self, *exc_info: object) -> None:
        try:
            self.finalize()
        finally:
            self._path.unlink(missing_ok=True)


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
    profile: MediaProfile | None = None,
) -> list[np.ndarray]:
    """Sample representative BGR frames for video content profiling.

    Grabs frames at positions [10%, 25%, 50%, 75%, 90%] of the video.
    Uses seek_frame to minimize I/O. Accepts a preloaded ``profile`` to
    skip the extra ffprobe spawn when the caller already has one.
    """
    if profile is None:
        profile = media.probe(media_path)
    duration = profile.duration_s
    if duration <= 0:
        duration = 10.0  # fallback

    positions = [0.10, 0.25, 0.50, 0.75, 0.90]
    times = [p * duration for p in positions[:n_frames]]

    frames: list[np.ndarray] = []
    for t in times:
        try:
            frame = media.seek_frame(media_path, t)
            frames.append(frame)
        except ValueError:
            continue

    if not frames:
        try:
            frame = media.seek_frame(media_path, 0.0)
            frames.append(frame)
        except ValueError:
            pass

    if not frames:
        raise ValueError("Could not read any frames from video for profiling")

    return frames


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
    profile: MediaProfile | None = None,
) -> RawSigningPayload:
    """CPU phase of audio signing. Returns a serialisable RawSigningPayload."""
    # 1. Probe (reuse caller-supplied profile to avoid redundant ffprobe spawn)
    if profile is None:
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
    content_hash = streaming_file_hash(media_path)

    # 4. Target sample rate for audio orchestration
    target_sample_rate = 44100

    # 5. Fingerprint (drives segment count) — Pass 1 Streaming
    # Count non-overlapping 2s chunks alongside fingerprint extraction.
    # extract_hashes_from_stream uses overlap=0.5, so len(fingerprints) > n_wid_segments.
    # rs_n must match the number of non-overlapping WID segments, not fingerprint count.
    # Also score each segment for content-adaptive placement, and cache the decoded
    # chunks so the embed pass reuses them instead of re-running ffmpeg on the file.
    segment_scores: list[tuple[int, float, float, float, float]] = []

    with _PCMChunkScratch() as scratch:
        def _counting_stream():
            for seg_idx, chunk, _ in media.iter_audio_segments(
                media_path, segment_duration_s=2.0, target_sample_rate=target_sample_rate
            ):
                scratch.append(chunk)
                sc = score_segment(chunk, target_sample_rate,
                                   dwt_level=ap.dwt_levels[0],
                                   target_subband=ap.target_subband)
                segment_scores.append(
                    (seg_idx, sc.rms_db, sc.band_energy_db,
                     sc.spectral_flatness, sc.transient_density)
                )
                yield chunk

        fingerprints = extract_audio_hashes_from_stream(
            _counting_stream(),
            target_sample_rate,
            key_material=pepper,
            pepper=pepper,
        )
        scratch.finalize()
        n_wid_segments = len(scratch)

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

        # 10a. Content-adaptive segment selection
        # Build scored SegmentScores from the raw metrics collected during Pass 1.
        scored_segments = scores_from_raw_metrics(segment_scores)
        selected_indices = select_best(scored_segments, n_needed=rs_n)
        segment_map = SegmentMap(
            selected_indices=tuple(selected_indices),
            total_segments=n_wid_segments,
        )

        # Attach segment map to audio params for serialization
        ap = AudioEmbeddingParams(
            dwt_levels=ap.dwt_levels,
            chips_per_bit=ap.chips_per_bit,
            psychoacoustic=ap.psychoacoustic,
            safety_margin_db=ap.safety_margin_db,
            target_snr_db=ap.target_snr_db,
            target_subband=ap.target_subband,
            frame_length_ms=ap.frame_length_ms,
            pn_sequence_length=ap.pn_sequence_length,
            segment_map=segment_map,
        )

        # 10b–11. Hopping plan + RS symbols
        # band_configs and rs_symbols are indexed by RS symbol position (0..rs_n-1),
        # NOT by absolute segment index. The segment_map provides the indirection.
        band_configs = plan_audio_hopping(
            rs_n, content_id, certificate.public_key_pem, pepper,
            force_levels=list(ap.dwt_levels),
            target_subband=ap.target_subband,
        )
        rs_symbols = ReedSolomonCodec(rs_n).encode(wid.data)

        # Build segment→RS mapping for the embedding loop
        selected_set = set(segment_map.selected_indices)
        _seg_to_rs = {seg_idx: rs_idx
                      for rs_idx, seg_idx in enumerate(segment_map.selected_indices)}

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
        # Prebuild the per-file HMAC prefix once; per-segment seeding is then a
        # cheap copy() + update() instead of reconstructing the HMAC from scratch.
        _wid_prefix = f"wid|{content_id}|{certificate.public_key_pem}|".encode()
        _wid_base_hmac = hmac.new(pepper, _wid_prefix, hashlib.sha256)
        try:
            # Pass 2: stream chunks back from the on-disk scratch — no second
            # ffmpeg subprocess, and peak RSS is bounded to ~one chunk.
            for seg_idx, chunk in enumerate(scratch):
                if seg_idx in selected_set:
                    rs_idx = _seg_to_rs[seg_idx]
                    h = _wid_base_hmac.copy()
                    h.update(str(rs_idx).encode())
                    pn_seed = int.from_bytes(h.digest()[:8], "big")
                    chunk = embed_audio_segment(
                        chunk, rs_symbols[rs_idx], band_configs[rs_idx], pn_seed,
                        chips_per_bit=ap.chips_per_bit,
                        target_snr_db=ap.target_snr_db,
                        use_psychoacoustic=ap.psychoacoustic,
                        safety_margin_db=ap.safety_margin_db,
                    )
                if encoder_proc.stdin:
                    encoder_proc.stdin.write(
                        (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                    )
        finally:
            if encoder_proc.stdin:
                encoder_proc.stdin.close()
            encoder_proc.wait()

    signed_name = _make_signed_name(original_filename, media_path.suffix, force_ext=out_suffix)
    storage_key = f"signed/{content_id}/{signed_name}"
    active_signals = ["audio_wid", "audio_fingerprint"]

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


def _encode_signed_video_sequential(
    media_path: Path,
    media: MediaPort,
    rs_n: int,
    rs_symbols: list[int],
    content_id: str,
    author_public_key: str,
    pepper: bytes,
    vp: VideoEmbeddingParams,
    width: int,
    height: int,
    fps: float,
    crf: int,
) -> Path:
    """Single-process encode: decode the whole file once, embed per segment,
    pipe to one ffmpeg. Raises ValueError on encoder failure."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        signed_path = Path(tmp.name)

    encoder_proc = media.open_video_encode_stream(
        width, height, fps, signed_path, crf=crf
    )
    try:
        for seg_idx, seg_frames, _ in media.iter_video_segments(
            media_path,
            segment_duration_s=VIDEO_SEGMENT_S,
            frame_stride=1,
        ):
            if seg_idx >= rs_n:
                for frame in seg_frames:
                    y, u, v = frame_to_yuvj420_planes(frame)
                    encoder_proc.stdin.write(y)
                    encoder_proc.stdin.write(u)
                    encoder_proc.stdin.write(v)
                continue

            symbol_bits = np.array(
                [(rs_symbols[seg_idx] >> (7 - k)) & 1 for k in range(8)],
                dtype=np.uint8,
            )

            for frame in seg_frames:
                y, u, v = embed_video_frame_yuvj420_planes(
                    frame, symbol_bits, content_id,
                    author_public_key, seg_idx, pepper,
                    use_jnd_adaptive=vp.jnd_adaptive,
                    jnd_params=vp,
                )
                encoder_proc.stdin.write(y)
                encoder_proc.stdin.write(u)
                encoder_proc.stdin.write(v)
    finally:
        if encoder_proc.stdin:
            encoder_proc.stdin.close()
        encoder_proc.wait()
        if encoder_proc.returncode != 0:
            signed_path.unlink(missing_ok=True)
            raise ValueError(
                f"FFmpeg video encode failed (returncode={encoder_proc.returncode})"
            )
    return signed_path


def _encode_signed_video_chunked(
    media_path: Path,
    media: MediaPort,
    rs_n: int,
    rs_symbols: list[int],
    content_id: str,
    author_public_key: str,
    pepper: bytes,
    vp: VideoEmbeddingParams,
    width: int,
    height: int,
    fps: float,
    crf: int,
    duration_s: float,
) -> Path | None:
    """Parallel chunked encode. Returns the signed Path, or None when the
    planner collapses to a single chunk and the caller should fall back to
    :func:`_encode_signed_video_sequential`. Raises ValueError on worker /
    validation / concat failure.
    """
    settings = get_settings()
    manifest = plan_chunks(
        total_segments=rs_n,
        segment_duration_s=VIDEO_SEGMENT_S,
        n_workers=settings.CHUNK_WORKERS,
        guard_segments=settings.CHUNK_GUARD_SEGMENTS,
        min_payload_segments=settings.CHUNK_MIN_PAYLOAD_SEGMENTS,
        total_duration_s=duration_s,
    )
    if manifest.total_chunks == 1:
        return None

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        signed_path = Path(tmp.name)

    work_dir = Path(tempfile.mkdtemp(prefix="kernel_chunk_"))
    vp_dict = asdict(vp)
    rs_symbols_list = list(rs_symbols)
    source_str = str(media_path)

    results_dicts: list[dict] = []
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=manifest.total_chunks
        ) as pool:
            futures = [
                pool.submit(
                    process_video_chunk,
                    source_str,
                    asdict(spec),
                    rs_symbols_list,
                    content_id,
                    author_public_key,
                    pepper,
                    vp_dict,
                    width,
                    height,
                    fps,
                    crf,
                    str(work_dir),
                    VIDEO_SEGMENT_S,
                )
                for spec in manifest.chunks
            ]
            for fut in concurrent.futures.as_completed(futures):
                results_dicts.append(fut.result())

        results = [ChunkResult(**r) for r in results_dicts]
        validation = validate_chunks(manifest, results, media.probe)
        if not validation.is_valid:
            raise ValueError(
                "chunked signing failed validation: "
                + "; ".join(validation.errors)
            )
        concatenate_chunks(results, signed_path)
        return signed_path
    except Exception:
        signed_path.unlink(missing_ok=True)
        raise
    finally:
        cleanup_chunks([ChunkResult(**r) for r in results_dicts])
        shutil.rmtree(work_dir, ignore_errors=True)


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
    profile: MediaProfile | None = None,
) -> RawSigningPayload:
    """CPU phase of video signing. Returns a serialisable RawSigningPayload."""
    # 1. Probe — reject audio-only containers (reuse caller-supplied profile)
    if profile is None:
        profile = media.probe(media_path)
    if not profile.has_video:
        raise ValueError("Container has no video track — cannot sign video-only pipeline")

    # 1b. Video content profiling + adaptive routing
    video_routing_meta: dict | None = None
    if video_params is not None:
        vp = video_params
    else:
        sample_frames = _sample_video_frames(media_path, media, profile=profile)
        video_profile = _profile_video(sample_frames)
        video_routing = _route_video(video_profile)
        vp = _video_routing_to_params(video_routing)
        video_routing_meta = _video_routing_to_dict(video_routing)

    # 2–3. IDs and content hash
    content_id = str(uuid4())
    content_hash = streaming_file_hash(media_path)

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

    # 11. Get dimensions without loading frames — reuse the profile we already have
    fps = profile.fps
    width, height = profile.width, profile.height
    crf_to_use = output_crf if output_crf is not None else _compute_output_video_crf(profile.duration_s)

    # 12. Encode: try the chunked parallel pipeline first; the planner returns
    # None when the video is too short or the worker count collapses to 1 —
    # in that case we fall through to the sequential single-process encode.
    signed_path = _encode_signed_video_chunked(
        media_path=media_path,
        media=media,
        rs_n=rs_n,
        rs_symbols=list(rs_symbols),
        content_id=content_id,
        author_public_key=certificate.public_key_pem,
        pepper=pepper,
        vp=vp,
        width=width,
        height=height,
        fps=fps,
        crf=crf_to_use,
        duration_s=profile.duration_s,
    )
    if signed_path is None:
        signed_path = _encode_signed_video_sequential(
            media_path=media_path,
            media=media,
            rs_n=rs_n,
            rs_symbols=list(rs_symbols),
            content_id=content_id,
            author_public_key=certificate.public_key_pem,
            pepper=pepper,
            vp=vp,
            width=width,
            height=height,
            fps=fps,
            crf=crf_to_use,
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
    profile: MediaProfile | None = None,
) -> RawSigningPayload:
    """CPU phase of AV signing (single shared WID). Returns a serialisable RawSigningPayload."""
    # 1. Probe — require both tracks (reuse caller-supplied profile)
    if profile is None:
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
                profile=profile,
            )

    # 1c. Video content profiling + adaptive routing
    video_routing_meta: dict | None = None
    if video_params is not None:
        vp = video_params
    else:
        sample_frames = _sample_video_frames(media_path, media, profile=profile)
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
    content_hash = streaming_file_hash(media_path)

    # 4. Audio fingerprints (streaming) + segment scoring + chunk cache.
    #    Caching the chunks lets the embed pass reuse them instead of starting
    #    a second ffmpeg decode of the same file.
    target_sample_rate = 44100
    av_segment_scores: list[tuple[int, float, float, float, float]] = []

    with _PCMChunkScratch() as av_scratch:
        def _av_scoring_stream():
            for seg_idx, chunk, _ in media.iter_audio_segments(
                media_path, segment_duration_s=2.0, target_sample_rate=target_sample_rate
            ):
                av_scratch.append(chunk)
                sc = score_segment(chunk, target_sample_rate,
                                   dwt_level=ap.dwt_levels[0],
                                   target_subband=ap.target_subband)
                av_segment_scores.append(
                    (seg_idx, sc.rms_db, sc.band_energy_db,
                     sc.spectral_flatness, sc.transient_density)
                )
                yield chunk

        audio_fingerprints = extract_audio_hashes_from_stream(
            _av_scoring_stream(), target_sample_rate, key_material=pepper, pepper=pepper
        )
        av_scratch.finalize()

        # 6. Video fingerprints
        video_fingerprints = extract_video_hashes(
            str(media_path), key_material=pepper, pepper=pepper
        )

        # 7. RS parameters — use the SMALLER rs_n so both channels stay in sync
        n_av_wid_segments = len(av_scratch)
        rs_n_audio = min(n_av_wid_segments, 255)
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

        # 11a. Content-adaptive audio segment selection
        av_scored = scores_from_raw_metrics(av_segment_scores)
        av_selected = select_best(av_scored, n_needed=rs_n)
        av_segment_map = SegmentMap(
            selected_indices=tuple(av_selected),
            total_segments=n_av_wid_segments,
        )
        ap = AudioEmbeddingParams(
            dwt_levels=ap.dwt_levels,
            chips_per_bit=ap.chips_per_bit,
            psychoacoustic=ap.psychoacoustic,
            safety_margin_db=ap.safety_margin_db,
            target_snr_db=ap.target_snr_db,
            target_subband=ap.target_subband,
            frame_length_ms=ap.frame_length_ms,
            pn_sequence_length=ap.pn_sequence_length,
            segment_map=av_segment_map,
        )

        # 12. Audio hopping plan
        band_configs = plan_audio_hopping(
            rs_n, content_id, certificate.public_key_pem, pepper,
            force_levels=list(ap.dwt_levels),
            target_subband=ap.target_subband,
        )

        # Build segment→RS mapping
        av_selected_set = set(av_segment_map.selected_indices)
        _av_seg_to_rs = {seg_idx: rs_idx
                         for rs_idx, seg_idx in enumerate(av_segment_map.selected_indices)}

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
        _av_wid_prefix = f"wid|{content_id}|{certificate.public_key_pem}|".encode()
        _av_wid_base_hmac = hmac.new(pepper, _av_wid_prefix, hashlib.sha256)
        try:
            # Pass 2: stream chunks back from the on-disk scratch — no second
            # ffmpeg subprocess, and peak RSS is bounded to ~one chunk.
            for seg_idx, chunk in enumerate(av_scratch):
                if seg_idx in av_selected_set:
                    rs_idx = _av_seg_to_rs[seg_idx]
                    h = _av_wid_base_hmac.copy()
                    h.update(str(rs_idx).encode())
                    pn_seed = int.from_bytes(h.digest()[:8], "big")
                    chunk = embed_audio_segment(
                        chunk, rs_symbols[rs_idx], band_configs[rs_idx], pn_seed,
                        chips_per_bit=ap.chips_per_bit,
                        target_snr_db=ap.target_snr_db,
                        use_psychoacoustic=ap.psychoacoustic,
                        safety_margin_db=ap.safety_margin_db,
                    )
                if encoder_proc.stdin:
                    encoder_proc.stdin.write(
                        (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                    )
        finally:
            if encoder_proc.stdin:
                encoder_proc.stdin.close()
            encoder_proc.wait()

    # 14. Embed video — chunked parallel encode when the clip is long enough,
    # otherwise fall back to the sequential single-process path. Both helpers
    # return a Path to the encoded signed video.
    av_fps = profile.fps
    av_width, av_height = profile.width, profile.height
    av_crf_to_use = output_crf if output_crf is not None else _compute_output_video_crf(profile.duration_s)

    video_signed_path = _encode_signed_video_chunked(
        media_path=media_path,
        media=media,
        rs_n=rs_n,
        rs_symbols=list(rs_symbols),
        content_id=content_id,
        author_public_key=certificate.public_key_pem,
        pepper=pepper,
        vp=vp,
        width=av_width,
        height=av_height,
        fps=av_fps,
        crf=av_crf_to_use,
        duration_s=profile.duration_s,
    )
    if video_signed_path is None:
        video_signed_path = _encode_signed_video_sequential(
            media_path=media_path,
            media=media,
            rs_n=rs_n,
            rs_symbols=list(rs_symbols),
            content_id=content_id,
            author_public_key=certificate.public_key_pem,
            pepper=pepper,
            vp=vp,
            width=av_width,
            height=av_height,
            fps=av_fps,
            crf=av_crf_to_use,
        )

    output_path: Path | None = None
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
