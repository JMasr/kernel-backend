"""Generate synthetic test media using ffmpeg.

Reads data/fixtures/manifest.yaml for ffmpeg commands and writes outputs to tests/fixtures/.
Skips gracefully if ffmpeg is not available.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
MANIFEST = ROOT / "data" / "fixtures" / "manifest.yaml"
OUTPUT_DIR = ROOT / "tests" / "fixtures"


def main() -> None:
    if not shutil.which("ffmpeg"):
        print(
            "WARNING: ffmpeg not found in PATH. Skipping fixture generation.",
            file=sys.stderr,
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with MANIFEST.open() as f:
        manifest = yaml.safe_load(f)

    for item in manifest.get("synthetic", []):
        name = item["name"]
        cmd_str: str = item["ffmpeg_cmd"].strip()
        # Split on whitespace, handling multi-line YAML scalars
        cmd = cmd_str.split()
        print(f"Generating {name} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR generating {name}:\n{result.stderr}", file=sys.stderr)
        else:
            print(f"  done: {name}")


if __name__ == "__main__":
    main()
