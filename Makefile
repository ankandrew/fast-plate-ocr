# Directories
SRC_PATHS := fast_lp_ocr/ \
            test/ \
            train.py \
            valid.py \
            onnx_converter.py \
            demo_recog.py

# Tasks
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  format           : Format code using Ruff format"
	@echo "  check_format     : Check code formatting with Ruff format"
	@echo "  lint             : Run linters (Ruff, Pylint and Mypy)"
	@echo "  test             : Run tests using pytest"
	@echo "  clean            : Clean up caches and build artifacts"
	@echo "  run_local_checks : Run format, lint, and test"
	@echo "  help             : Show this help message"

.PHONY: format
format:
	@echo "==> Sorting imports..."
	@# Currently, the Ruff formatter does not sort imports, see https://docs.astral.sh/ruff/formatter/#sorting-imports
	@poetry run ruff check --select I --fix $(SRC_PATHS)
	@echo "=====> Formatting code..."
	@poetry run ruff format $(SRC_PATHS)

.PHONY: check_format
check_format:
	@echo "=====> Checking format..."
	@poetry run ruff format --check --diff $(SRC_PATHS)
	@echo "=====> Checking imports are sorted..."
	@poetry run ruff check --select I --exit-non-zero-on-fix $(SRC_PATHS)

.PHONY: lint
lint:
	@echo "=====> Running Ruff..."
	@poetry run ruff check $(SRC_PATHS)
	@echo "=====> Running Pylint..."
	@poetry run pylint $(SRC_PATHS)
	@echo "=====> Running Mypy..."
	@poetry run mypy $(SRC_PATHS)

.PHONY: test
test:
	@echo "=====> Running tests..."
	@poetry run pytest

.PHONY: clean
clean:
	@echo "=====> Cleaning caches..."
	@poetry run ruff clean
	@rm -rf .cache .pytest_cache .mypy_cache build dist *.egg-info

run_local_checks: format lint test
