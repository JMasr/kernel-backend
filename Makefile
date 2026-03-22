# ──────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_NAME = kernel-backend
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = ./.venv/bin/python

# ── Environment ───────────────────────────────────────────────────────────────
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"


## Install Production Python dependencies
.PHONY: requirements
requirements:
	uv sync


## Install Development Python dependencies
.PHONY: requirements-dev
requirements-dev: requirements
	uv sync --extra dev

# ── Dev server ───────────────────────────────────────────────────────────────
.PHONY: up
up:
	uv run fastapi dev main.py

# ── ARQ worker ───────────────────────────────────────────────────────────────
.PHONY: worker
worker:
	$(PYTHON_INTERPRETER) -m arq src.kernel_backend.infrastructure.queue.worker.WorkerSettings

# ── Tests ────────────────────────────────────────────────────────────────────

## CI default — all tiers except polygon (~30 s)
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests/ -m "not polygon" -v

## Fast unit tests only — no librosa, no polygon (~10 s)
.PHONY: unit-fast
unit-fast:
	$(PYTHON_INTERPRETER) -m pytest tests/unit/ -m "not integration" -v

## All unit tests including speech integration (~80 s)
.PHONY: unit
unit:
	$(PYTHON_INTERPRETER) -m pytest tests/unit/ -v

## Phase 3 release gate — speech fingerprint blocking tests
.PHONY: release-gate
release-gate:
	$(PYTHON_INTERPRETER) -m pytest tests/unit/test_fingerprint_audio.py -v

## Real-world polygon tests — requires setup-polygon first (~80 s)
.PHONY: polygon
polygon:
	$(PYTHON_INTERPRETER) -m pytest tests/ -m "polygon" -v

## Export librosa clips to data/audio/ (run once before make polygon)
.PHONY: setup-polygon
setup-polygon:
	uv run $(PYTHON_INTERPRETER) scripts/setup_polygon_audio.py

## Phase 4 integration stubs (currently empty)
.PHONY: integration
integration:
	$(PYTHON_INTERPRETER) -m pytest tests/integration/ -v

## Boundary lint — import invariant enforcement
.PHONY: boundary
boundary:
	$(PYTHON_INTERPRETER) -m pytest tests/boundary/ -v
	$(PYTHON_INTERPRETER) scripts/check_boundaries.py .

# ── Code quality ─────────────────────────────────────────────────────────────
.PHONY: lint
lint:
	ruff check .

.PHONY: format
format:
	ruff format .
	ruff check --fix .

.PHONY: typecheck
typecheck:
	mypy src/kernel_backend/ --ignore-missing-imports

# ── Database ─────────────────────────────────────────────────────────────────
.PHONY: migrate
migrate:
	alembic upgrade head

.PHONY: migrate-create
migrate-create:
	@read -p "Migration name: " name; alembic revision --autogenerate -m "$$name"

.PHONY: reset-db
reset-db:
	$(PYTHON_INTERPRETER) scripts/reset_db.py

# ── Fixtures ─────────────────────────────────────────────────────────────────
.PHONY: gen-fixtures
gen-fixtures:
	$(PYTHON_INTERPRETER) scripts/gen_fixtures.py

# ── Cleanup ──────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
