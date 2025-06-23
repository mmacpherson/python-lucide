# Makefile for python-lucide development

# Variables
UV_CMD := uv
PRE_COMMIT_CMD := $(UV_CMD) run pre-commit
PYTEST_CMD := $(UV_CMD) run pytest
LUCIDE_DB_CMD := $(UV_CMD) run lucide-db

# Get default Lucide tag from the package's config.py
PYTHON_CMD_FOR_TAG := $(UV_CMD) run python -c "from lucide.config import DEFAULT_LUCIDE_TAG; print(DEFAULT_LUCIDE_TAG)"
DEFAULT_LUCIDE_TAG := $(shell $(PYTHON_CMD_FOR_TAG))
# Allow overriding the tag via make argument, e.g., make db TAG=0.500.0
TAG ?= $(DEFAULT_LUCIDE_TAG)
DB_OUTPUT_PATH := src/lucide/data/lucide-icons.db
VENV_DIR := .venv

# Phony targets prevent conflicts with files of the same name.
.PHONY: help default env db test install-hooks run-hooks-all-files check-version clean nuke

# Default target
default: help

help:
	@echo "Makefile for python-lucide development"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  help                   Show this help message."
	@echo "  env                    Set up the development environment (creates $(VENV_DIR) and installs dependencies)."
	@echo "  db                     (Re)builds the Lucide icon database into $(DB_OUTPUT_PATH)."
	@echo "                         Uses TAG=$(TAG). Default TAG is read from src/lucide/config.py (currently $(DEFAULT_LUCIDE_TAG))."
	@echo "                         Example: make db TAG=0.520.0"
	@echo "  test                   Run tests using pytest."
	@echo "  install-hooks          Install pre-commit hooks."
	@echo "  update-hooks           Update pre-commit hooks to latest version."
	@echo "  run-hooks-all-files    Run all pre-commit hooks on all files."
	@echo "  check-version          Check if Lucide version/artifacts need updating."
	@echo "  clean                  Remove build artifacts, __pycache__, .pytest_cache, .ruff_cache, coverage data, etc."
	@echo "  nuke                   A more thorough clean: runs 'clean', 'uv cache clean', and removes $(VENV_DIR)."
	@echo ""

env: $(VENV_DIR)/pyvenv.cfg
	@echo "Installing/updating development dependencies into $(VENV_DIR)..."
	$(UV_CMD) pip install -e ".[dev]"
	@echo "Development environment ready in $(VENV_DIR)/."

$(VENV_DIR)/pyvenv.cfg:
	@echo "Creating virtual environment in $(VENV_DIR)/ using $(UV_CMD)..."
	$(UV_CMD) venv $(VENV_DIR)

db:
	@echo "Building Lucide icon database with tag $(TAG) into $(DB_OUTPUT_PATH)..."
	@mkdir -p src/lucide/data # Ensure data directory exists
	$(LUCIDE_DB_CMD) -o $(DB_OUTPUT_PATH) -t $(TAG) -v
	@echo "Database build complete: $(DB_OUTPUT_PATH)"

test:
	@echo "Running tests..."
	$(PYTEST_CMD)

install-hooks:
	@echo "Installing pre-commit hooks..."
	$(PRE_COMMIT_CMD) install

update-hooks:
	@echo "Updating pre-commit hooks..."
	$(PRE_COMMIT_CMD) autoupdate

run-hooks-all-files:
	@echo "Running all pre-commit hooks on all files..."
	$(PRE_COMMIT_CMD) run --all-files

check-version:
	@echo "Checking Lucide version and artifact status..."
	$(UV_CMD) run python -c "from lucide.dev_utils import print_version_status; exit(print_version_status())"

clean:
	@echo "Cleaning up project..."
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf build/ dist/ .eggs/ *.egg-info/ site/
	rm -rf .pytest_cache/ .ruff_cache/ htmlcov/ .coverage .coverage.* coverage.xml
	rm -f lucide-icons.db # Remove db if built in root by mistake
	@echo "Clean complete."

nuke: clean
	@echo "Nuking project (includes 'clean', 'uv cache clean', and removing $(VENV_DIR))..."
	$(UV_CMD) cache clean
	rm -rf $(VENV_DIR)/
	@echo "Nuke complete."
