# Directories
SRC_DIRS := fast_lp_ocr/ test/

# Tasks
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  format           : Format code using Ruff format"
	@echo "  check_format     : Check code formatting with Ruff format"
	@echo "  lint             : Run Ruff linter and Mypy for static code analysis"
	@echo "  test             : Run tests using pytest"
	@echo "  clean            : Clean up caches and build artifacts"
	@echo "  run_local_checks : Run format, lint, and test"
	@echo "  help             : Show this help message"

.PHONY: format
format:
	@echo "==> Sorting imports..."
	@# Currently, the Ruff formatter does not sort imports, see https://docs.astral.sh/ruff/formatter/#sorting-imports
	@poetry run ruff check --select I --fix $(SRC_DIRS)
	@echo "=====> Formatting code..."
	@poetry run ruff format $(SRC_DIRS)

.PHONY: check_format
check_format:
	@echo "=====> Checking format..."
	@poetry run ruff format --check --diff $(SRC_DIRS)
	@echo "=====> Checking imports are sorted..."
	@poetry run ruff check --select I --exit-non-zero-on-fix $(SRC_DIRS)


.PHONY: lint
lint:
	@echo "=====> Running Ruff linter..."
	@poetry run ruff check $(SRC_DIRS)
	@echo "=====> Running Mypy..."
	@poetry run mypy $(SRC_DIRS)

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
