# Directories
SRC_PATHS := fast_plate_ocr/ test/
YAML_PATHS := .github/ models/ config/ mkdocs.yml

# Tasks
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  help             : Show this help message"
	@echo "  install          : Install project with all dev/test/docs/train dependencies"
	@echo "  format           : Format code using Ruff format"
	@echo "  check_format     : Check code formatting with Ruff format"
	@echo "  ruff             : Run Ruff linter"
	@echo "  pylint           : Run Pylint linter"
	@echo "  mypy             : Run MyPy static type checker"
	@echo "  lint             : Run linters (Ruff, Pylint and Mypy)"
	@echo "  test             : Run tests using pytest"
	@echo "  checks           : Check format, lint, and test"
	@echo "  clean            : Clean up caches and build artifacts"

install:
	@echo "==> Installing project with dev/test/docs/train dependencies..."
	@uv sync --locked --all-groups --extra train --extra onnx

.PHONY: format
format:
	@echo "==> Sorting imports..."
	@# Currently, the Ruff formatter does not sort imports, see https://docs.astral.sh/ruff/formatter/#sorting-imports
	@uv run ruff check --select I --fix $(SRC_PATHS)
	@echo "=====> Formatting code..."
	@uv run ruff format $(SRC_PATHS)

.PHONY: check_format
check_format:
	@echo "=====> Checking format..."
	@uv run ruff format --check --diff $(SRC_PATHS)
	@echo "=====> Checking imports are sorted..."
	@uv run ruff check --select I --exit-non-zero-on-fix $(SRC_PATHS)

.PHONY: ruff
ruff:
	@echo "=====> Running Ruff..."
	@uv run ruff check $(SRC_PATHS)

.PHONY: yamllint
yamllint:
	@echo "=====> Running yamllint..."
	@uv run yamllint $(YAML_PATHS)

.PHONY: pylint
pylint:
	@echo "=====> Running Pylint..."
	@uv run pylint $(SRC_PATHS)

.PHONY: mypy
mypy:
	@echo "=====> Running Mypy..."
	@uv run mypy $(SRC_PATHS)

.PHONY: lint
lint: ruff yamllint pylint mypy

.PHONY: test
test:
	@echo "=====> Running tests..."
	@uv run pytest test/

.PHONY: clean
clean:
	@echo "=====> Cleaning caches..."
	@uv run ruff clean
	@rm -rf .cache .pytest_cache .mypy_cache build dist *.egg-info

checks: format lint test
