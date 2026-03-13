# Polygon Test Data

This directory contains the real-world media clips used by the test polygon.
Media files are NOT committed to git. Only `manifest.yaml` is versioned.

## Setup

### Audio (automatic)

Run once to export librosa example recordings as WAV files:

```bash
uv run python scripts/setup_polygon_audio.py
```

This creates all files listed under `audio/` in `manifest.yaml`.
Copy the printed sha256 values into `manifest.yaml`.

### Video (manual)

Place video clips in the following directories:

- `data/video/speech/`        — monologue, conference, single speaker
- `data/video/outside/`       — outdoor interviews, ambient noise
- `data/video/without_audio/` — video-only, no audio stream
- `data/video/others/`        — mixed or uncategorized

Name convention: `{descriptive_name}_{index}.mp4`
After adding a clip, add its entry to `manifest.yaml` and fill sha256.

## Running polygon tests

```bash
# Only polygon tests
pytest tests/ -m "polygon" -v

# Skip polygon tests (default CI)
pytest tests/ -m "not polygon" -v

# Blocking categories only
pytest tests/ -m "polygon" -k "speech" -v
```
