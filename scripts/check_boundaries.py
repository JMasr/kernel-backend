"""
Boundary lint: ensures src/kernel_backend/core/ and src/kernel_backend/engine/
have zero infra imports.

Runs standalone:  python scripts/check_boundaries.py
Via pytest:       tests/boundary/test_import_boundaries.py imports this module.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Paths are relative to the repo root, inside src/kernel_backend/
RULES: dict[str, set[str]] = {
    "src/kernel_backend/core": {
        "fastapi", "starlette", "sqlalchemy", "alembic",
        "arq", "boto3", "botocore", "ffmpeg", "cv2", "pywt",
        "soundfile", "aiofiles",
    },
    "src/kernel_backend/engine": {
        "fastapi", "starlette", "sqlalchemy", "alembic",
        "arq", "boto3", "botocore", "ffmpeg",
    },
}

def check(root: Path = Path(".")) -> list[str]:
    """Return violation strings. Empty list = clean."""
    violations: list[str] = []
    for layer, forbidden in RULES.items():
        layer_path = root / layer
        if not layer_path.exists():
            continue
        for py_file in sorted(layer_path.rglob("*.py")):
            source = py_file.read_text(encoding="utf-8")
            for lib in forbidden:
                if f"import {lib}" in source or f"from {lib}" in source:
                    rel = py_file.relative_to(root)
                    violations.append(f"BOUNDARY VIOLATION: {rel} imports '{lib}'")
    return violations

if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    violations = check(root)
    if violations:
        for v in violations:
            print(v, file=sys.stderr)
        sys.exit(1)
    print("✔ Boundary check passed.")
    sys.exit(0)
