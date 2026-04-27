.PHONY: help install install-dev fmt lint typecheck test test-unit test-integration \
        seed-data train-occupancy train-hvac train-anomaly up down logs clean

PYTHON := python
PIP    := pip
PYTEST := pytest

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Smart Building AI Dashboard — Dev Commands"
	@echo "  ──────────────────────────────────────────"
	@echo "  make install            Install production dependencies"
	@echo "  make install-dev        Install all dev dependencies"
	@echo "  make fmt                Format code (black + isort)"
	@echo "  make lint               Run flake8"
	@echo "  make typecheck          Run mypy"
	@echo "  make check              fmt + lint + typecheck"
	@echo "  make test               Run all tests with coverage"
	@echo "  make test-unit          Unit tests only"
	@echo "  make test-integration   Integration tests (needs Docker)"
	@echo "  make seed-data          Generate and load synthetic sensor data"
	@echo "  make train-occupancy    Train the occupancy transformer"
	@echo "  make train-hvac         Train the HVAC load forecaster"
	@echo "  make train-anomaly      Train anomaly detection models"
	@echo "  make up                 Start all Docker services"
	@echo "  make down               Stop all Docker services"
	@echo "  make logs               Tail logs from all services"
	@echo "  make clean              Remove caches and build artifacts"
	@echo ""

# ── Install ───────────────────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt
	pre-commit install

# ── Code Quality ──────────────────────────────────────────────────────────────
fmt:
	black src/ tests/ --line-length 100
	isort src/ tests/

lint:
	flake8 src/ tests/ --max-line-length 100 --extend-ignore E203,W503

typecheck:
	mypy src/ --ignore-missing-imports

check: fmt lint typecheck

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ --cov=src --cov-report=term-missing --cov-report=html -v

test-unit:
	$(PYTEST) tests/unit/ -v

test-integration:
	$(PYTEST) tests/integration/ -v --timeout=60

# ── Data ──────────────────────────────────────────────────────────────────────
seed-data:
	$(PYTHON) data/synthetic/generate.py --zones 20 --days 90
	bash scripts/seed_data.sh

# ── Training ──────────────────────────────────────────────────────────────────
train-occupancy:
	$(PYTHON) src/models/occupancy/train.py --config configs/model_config.yaml

train-hvac:
	$(PYTHON) src/models/hvac_forecast/train.py --config configs/model_config.yaml

train-anomaly:
	$(PYTHON) src/models/anomaly/ensemble.py --config configs/model_config.yaml

train-all: train-occupancy train-hvac train-anomaly

# ── Docker ────────────────────────────────────────────────────────────────────
up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

rebuild:
	docker-compose up -d --build

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	find . -name ".coverage" -delete
