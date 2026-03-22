"""Export the OpenAPI schema to a static JSON file.

Usage:
    uv run python scripts/export_schema.py

The generated openapi.json is version-controlled and serves as the formal
contract between backend and frontend.  Frontend runs:
    npm run generate-types
to produce TypeScript types from this file.
"""

import json
import sys
from pathlib import Path

# Ensure the backend root is on sys.path so `main` can be imported
backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from main import create_app

app = create_app()
schema = app.openapi()

out = backend_root / "openapi.json"
out.write_text(json.dumps(schema, indent=2) + "\n")
print(f"Wrote {out}")
