"""
DatasetRegistry — single entry point for all polygon tests.

Reads data/manifest.yaml and exposes typed ClipSets by category.
Never downloads, never generates, never raises on missing files.
Missing clips are silently excluded from available_* lists — tests
that use those clips are skipped by the fixture parametrization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DATA_ROOT = Path(__file__).parents[3] / "data"
MANIFEST_PATH = DATA_ROOT / "manifest.yaml"


@dataclass(frozen=True)
class AudioClip:
    id: str
    path: Path
    duration_s: float
    sample_rate: int
    category: str
    conditions: list[str]
    speaker: str | None = None
    language: str | None = None
    source: str | None = None

    def exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0


@dataclass(frozen=True)
class VideoClip:
    id: str
    path: Path
    duration_s: float
    fps: float
    resolution: tuple[int, int]
    has_audio: bool
    category: str
    conditions: list[str]

    def exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0


@dataclass
class CategorySet:
    name: str
    blocking: bool
    audio_clips: list[AudioClip] = field(default_factory=list)
    video_clips: list[VideoClip] = field(default_factory=list)

    def available_audio(self) -> list[AudioClip]:
        return [c for c in self.audio_clips if c.exists()]

    def available_video(self) -> list[VideoClip]:
        return [c for c in self.video_clips if c.exists()]

    def is_empty(self) -> bool:
        return (
            len(self.available_audio()) == 0
            and len(self.available_video()) == 0
        )


class DatasetRegistry:
    """
    Loads manifest.yaml and resolves all clips to absolute paths.
    Provides typed access by category key ('audio.speech', 'video.outside', etc.)
    and threshold lookup by (category, degradation) pair.
    """

    def __init__(self, manifest_path: Path = MANIFEST_PATH) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                "Run scripts/setup_polygon_audio.py to initialize."
            )
        self._manifest: dict = yaml.safe_load(manifest_path.read_text())
        self._categories: dict[str, CategorySet] = {}
        self._build()

    def _build(self) -> None:
        for cat_name, cat_data in self._manifest.get("audio", {}).items():
            key = f"audio.{cat_name}"
            cs = CategorySet(
                name=key,
                blocking=cat_data.get("blocking", False),
            )
            for raw in cat_data.get("clips", []):
                cs.audio_clips.append(AudioClip(
                    id=raw["id"],
                    path=DATA_ROOT / raw["path"],
                    duration_s=float(raw.get("duration_s", 0.0)),
                    sample_rate=int(raw.get("sample_rate", 44100)),
                    category=cat_name,
                    conditions=raw.get("conditions", []),
                    speaker=raw.get("speaker"),
                    language=raw.get("language"),
                    source=raw.get("source"),
                ))
            self._categories[key] = cs

        for cat_name, cat_data in self._manifest.get("video", {}).items():
            key = f"video.{cat_name}"
            cs = CategorySet(
                name=key,
                blocking=cat_data.get("blocking", False),
            )
            for raw in cat_data.get("clips", []):
                resolution = raw.get("resolution", [0, 0])
                cs.video_clips.append(VideoClip(
                    id=raw["id"],
                    path=DATA_ROOT / raw["path"],
                    duration_s=float(raw.get("duration_s", 0.0)),
                    fps=float(raw.get("fps", 30.0)),
                    resolution=(int(resolution[0]), int(resolution[1])),
                    has_audio=bool(raw.get("has_audio", True)),
                    category=cat_name,
                    conditions=raw.get("conditions", []),
                ))
            self._categories[key] = cs

    def get(self, category: str) -> CategorySet:
        """
        category: 'audio.speech' | 'audio.music' |
                  'video.speech' | 'video.outside' |
                  'video.without_audio' | 'video.others'
        Returns an empty CategorySet if category is unknown.
        """
        return self._categories.get(
            category,
            CategorySet(name=category, blocking=False),
        )

    def get_threshold(self, category: str, degradation: str) -> dict:
        """
        Returns threshold dict for (category, degradation) pair.
        Falls back to conservative defaults if not specified in manifest.

        category:    'audio.speech' | 'video.outside' etc.
        degradation: 'babble_10db' | 'pink_20db' | 'h264_crf23' etc.
        """
        domain, cat_name = category.split(".", 1)
        found = (
            self._manifest
            .get("thresholds", {})
            .get(domain, {})
            .get(cat_name, {})
            .get(degradation)
        )
        if found is not None:
            return found
        # Conservative defaults
        if domain == "audio":
            return {"pass_rate": 0.80, "max_hamming": 10}
        return {"pass_rate": 0.75, "min_agreement": 0.52}

    def blocking_categories(self) -> list[str]:
        return [k for k, v in self._categories.items() if v.blocking]

    def all_categories(self) -> list[str]:
        return list(self._categories.keys())
