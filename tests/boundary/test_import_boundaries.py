"""Pytest wrapper for the boundary lint — runs in CI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.check_boundaries import check

ROOT = Path(__file__).parent.parent.parent


def test_core_boundary():
    violations = [v for v in check(ROOT) if "core" in v]
    assert not violations, "\n".join(violations)


def test_engine_boundary():
    violations = [v for v in check(ROOT) if "engine" in v]
    assert not violations, "\n".join(violations)
