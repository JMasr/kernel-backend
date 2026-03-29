# CLAUDE.md — engine/video/

> **Phase 3 — COMPLETE**

## Responsibility

Video-domain watermark embedding and extraction across two active layers:
- Layer 1 (`wid_watermark.py`): 1 Reed-Solomon symbol per segment via QIM on 4×4 DCT AC coefficients
- Layer 2 (`fingerprint.py`): grayscale 32×32 → zero-mean → 2D DCT → keyed projection → 64-bit hash
- Layer 0 (`pilot_tone.py`): 48-bit content hash via QIM on DC — **removed from signing pipeline**, diagnostic only

Frames are passed as `np.ndarray` BGR (OpenCV convention). No file I/O.

## Boundary rules

MUST NOT import: `fastapi`, `sqlalchemy`, `arq`, `boto3`, `ffmpeg`, `pywt`
MAY import: `numpy`, `scipy`, `cv2`, `engine/codec/`

## 4×4 DCT — critical constraint

H.264 uses a 4×4 integer transform (Baseline and Main Profile).
All watermarking operates on **4×4 luma blocks**, never 8×8.

```
Coefficient pool:     {(0,1), (1,0), (1,1), (0,2)}
Robust subset:        {(0,1), (1,0)}          ← always include ≥2 per segment
High-freq subset:     {(1,1), (0,2)}          ← used when hopping allows
```

Each 4×4 block is extracted from the Y channel of YCrCb. DCT is applied with
`cv2.dct()` on a float32 4×4 patch.

## Color space handling — critical

Convert BGR→YCrCb as **uint8**, extract Y channel to float32 for DCT work,
modify DCT coefficients, clip back to uint8, then convert YCrCb→BGR.
**Never** convert to float32 before color space conversion — OpenCV uses
different formulas for uint8 vs float32 YCrCb, causing ~0.54 agreement
instead of ~1.0.

## Normalized coordinates

Block positions are generated in [0,1] normalized space via `rng.random((n_blocks, 2))`,
then mapped to pixel coordinates:
```
row = int(ny * n_rows)
col = int(nx * n_cols)
y0 = row * block_size
x0 = col * block_size
```
This makes block selection resolution-independent — same content_id works at any resolution.

## QIM step sizes (calibrated)

| Constant | Value | Target | Survives |
|---|---|---|---|
| `QIM_STEP_PILOT` | 28.0 | DC coefficient `(0,0)` | H.264 CRF ≤ 28 |
| `QIM_STEP_WID` | **48.0** | AC coefficients `{(0,1),(1,0),(1,1),(0,2)}` | H.264 CRF ≤ 28 |

CRF 35 pilot is xfail (informational), but passed unexpectedly (xpassed) on noise-enriched synthetic video — the noise source adds energy to AC coefficients that benefits pilot survival too.
QIM_STEP_PILOT was calibrated iteratively: 12→20→24→28.
QIM_STEP_WID was calibrated iteratively: 8.0→64.0→48.0. 8.0 caused quantization grid capture at H.264 QP≈28. 64.0 caused visible 4×4 block artifacts; 48.0 reduces visibility while maintaining CRF 28 robustness via majority voting (12,800 votes/bit).

## write_video_frames pixel format — critical

`MediaService.write_video_frames()` must use `yuvj420p` (JPEG/full-range YUV420).
- `yuv420p` (BT.601 limited-range) uses Y∈[16,235] — does NOT match `cv2.COLOR_BGR2YCrCb` (full-range Y∈[0,255]). This formula mismatch causes systematic QIM errors.
- `yuvj444p` (full-range, no chroma subsampling) gives correct Y but 3× pipe bandwidth → BrokenPipeError on sequential tests.
- `yuvj420p` (JPEG/full-range YUV420): correct full-range Y, chroma 2× subsampled, 1.5× bandwidth. Confirmed: 0/24 QIM errors at CRF 0/18/23/28.

## Module contracts

### pilot_tone.py

> **Not used in signing pipeline.** Kept for calibration scripts and diagnostic tests only.

```python
embed_pilot(frame: np.ndarray, content_id: str, pepper: bytes) -> np.ndarray
detect_pilot(frame: np.ndarray, content_id: str, pepper: bytes) -> PilotDetection
pilot_hash_48(content_id: str) -> int  # 48-bit hash
```
- N_PILOT_BLOCKS_PER_FRAME = 256
- PILOT_AGREEMENT_THRESHOLD = 0.75

### wid_watermark.py

```python
embed_segment(frames, symbol_bits, content_id, author_public_key, segment_idx, pepper) -> list[np.ndarray]
extract_segment(frames, content_id, author_public_key, segment_idx, pepper) -> SegmentWIDResult
```
- N_WID_BLOCKS_PER_SEGMENT = 128
- WID_AGREEMENT_THRESHOLD = 0.52

### fingerprint.py

```python
extract_hashes(video_path: str, key_material: bytes, pepper: bytes) -> list[SegmentFingerprint]
extract_hashes_from_frames(frames, key_material, pepper, fps) -> list[SegmentFingerprint]
hamming_distance(hash_a: str, hash_b: str) -> int
```
- SEGMENT_DURATION_S = 5.0
- FRAME_OFFSET_S = 0.5
- FINGERPRINT_SIZE = 64 bits

## Validation

```bash
python -m pytest tests/unit/test_video_pilot.py -v      # 5/5
python -m pytest tests/unit/test_video_wid.py -v         # 5/5
python -m pytest tests/unit/test_fingerprint_video.py -v # 5/5
python -m pytest tests/ -m "polygon" -k "video" -v       # polygon validation
```

## Diagnostic results (Phase 3 release gate)

WID agreement after H.264 recompression (speech_01 clip):
- Clean: 1.000, CRF 18: 0.764, CRF 23: 0.810, CRF 28: 0.857, CRF 35: 0.877

DC dominance check — all pairwise Hamming distances > 10:
- speech vs outside: 32, speech vs dark: 28, speech vs show: 32
- outside vs dark: 16, outside vs show: 32, dark vs show: 32
