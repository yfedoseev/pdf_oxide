# Makefile for PDF Library Python bindings
#
# Common development tasks for building and testing the Python package

.PHONY: dev install test build clean help lint-py fmt-py fmt-py-check check-py \
        benchmark benchmark-fetch benchmark-run benchmark-compare

# ─── Benchmark harness (#320) ───────────────────────────────────────────
# Defaults override on the command line, e.g.
#   make benchmark-run ENGINE=pdftotext CORPUS=/path/to/pdfs OUTPUT=head.json
ENGINE ?= pdf_oxide
CORPUS ?= tools/benchmark-harness/fixtures/kreuzberg/pdfs
GROUND_TRUTH ?= tools/benchmark-harness/fixtures/kreuzberg/gt
OUTPUT ?= target/benchmark.json
BASE ?= base.json
HEAD ?= head.json

benchmark: benchmark-run

benchmark-fetch:
	tools/benchmark-harness/scripts/fetch-fixtures.sh

benchmark-run:
	cargo run --release -p benchmark-harness -- run \
		--engine $(ENGINE) \
		--corpus $(CORPUS) \
		--ground-truth $(GROUND_TRUTH) \
		--output $(OUTPUT)

benchmark-compare:
	cargo run --release -p benchmark-harness -- diff $(BASE) $(HEAD)

# Development install (editable mode)
# Builds the Rust extension and installs the Python package in development mode
dev:
	maturin develop

# Install in release mode
# Builds the optimized Rust extension and installs the Python package
install:
	maturin develop --release

# Run Python tests
# Executes pytest on the tests/ directory
test:
	pytest tests/

# Run Python tests with verbose output
test-verbose:
	pytest tests/ -v

# Run Python tests with coverage
test-coverage:
	pytest tests/ --cov=pdf_library --cov-report=html

# Build wheel package
# Creates a distributable Python wheel in target/wheels/
build:
	maturin build --release

# Build wheel for all Python versions
build-all:
	maturin build --release --interpreter python3.8 python3.9 python3.10 python3.11 python3.12

# Clean build artifacts
# Removes all build artifacts and compiled extensions
clean:
	cargo clean
	rm -rf target/
	rm -rf python/pdf_library/*.so
	rm -rf python/pdf_library/*.pyd
	rm -rf python/pdf_library/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov/
	rm -rf .coverage

# Run Rust tests with Python feature enabled
test-rust:
	cargo test --features python

# Run Clippy linter on Rust code
lint:
	cargo clippy --features python -- -D warnings

# Run Ruff linter on Python code
lint-py:
	ruff check .

# Auto-fix Python linting issues
lint-py-fix:
	ruff check --fix .

# Format Rust code
fmt:
	cargo fmt

# Format Python code with Ruff
fmt-py:
	ruff format .

# Check formatting without modifying files
fmt-check:
	cargo fmt -- --check

# Check Python formatting without modifying files
fmt-py-check:
	ruff format --check .

# Run all Rust checks (format, lint, test)
check: fmt-check lint test-rust

# Run all Python checks (format, lint)
check-py: fmt-py-check lint-py

# Run all checks for both Rust and Python
check-all: check check-py

# Display help
help:
	@echo "PDF Library Python Bindings - Makefile Commands"
	@echo ""
	@echo "Development:"
	@echo "  make dev              - Install in development mode (fast rebuilds)"
	@echo "  make install          - Install in release mode (optimized)"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run Python tests"
	@echo "  make test-verbose     - Run Python tests with verbose output"
	@echo "  make test-coverage    - Run Python tests with coverage report"
	@echo "  make test-rust        - Run Rust tests with Python feature"
	@echo ""
	@echo "Building:"
	@echo "  make build            - Build release wheel"
	@echo "  make build-all        - Build wheels for all Python versions"
	@echo ""
	@echo "Code Quality (Rust):"
	@echo "  make lint             - Run Clippy linter on Rust code"
	@echo "  make fmt              - Format Rust code"
	@echo "  make fmt-check        - Check Rust formatting without modifying"
	@echo "  make check            - Run all Rust checks (format, lint, test)"
	@echo ""
	@echo "Code Quality (Python):"
	@echo "  make lint-py          - Run Ruff linter on Python code"
	@echo "  make lint-py-fix      - Auto-fix Python linting issues"
	@echo "  make fmt-py           - Format Python code with Ruff"
	@echo "  make fmt-py-check     - Check Python formatting without modifying"
	@echo "  make check-py         - Run all Python checks (format, lint)"
	@echo ""
	@echo "Code Quality (All):"
	@echo "  make check-all        - Run all checks for both Rust and Python"
	@echo ""
	@echo "Benchmark harness (#320):"
	@echo "  make benchmark-fetch   - Clone + link Kreuzberg fixture corpus"
	@echo "  make benchmark-run     - Run TF1+SF1 scoring on current branch"
	@echo "                           (ENGINE=pdf_oxide|pdftotext, OUTPUT=report.json)"
	@echo "  make benchmark-compare - Diff two JSON reports with the regression gate"
	@echo "                           (BASE=base.json HEAD=head.json)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            - Remove all build artifacts"
	@echo ""
	@echo "Help:"
	@echo "  make help             - Display this help message"
