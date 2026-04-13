"""Input format validation and video normalization.

Ensures incoming uploads are in formats the pipeline can process reliably.
Video files not already in H.264/MP4 are transcoded to an intermediate
H.264 MP4 before the signing/verification pipeline touches them.
"""
from __future__ import annotations

from pathlib import Path

ACCEPTED_EXTENSIONS: frozenset[str] = frozenset({
    # Video
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp", ".flv", ".wmv", ".mpg", ".mpeg",
    # Audio
    ".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".au",
})


def validate_media_extension(filename: str) -> None:
    """Raise ValueError if the file extension is not in the accepted set."""
    ext = Path(filename).suffix.lower()
    if ext not in ACCEPTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Accepted formats: {', '.join(sorted(ACCEPTED_EXTENSIONS))}"
        )
