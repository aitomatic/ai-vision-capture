.PHONY: setup format lint test build publish all

##@ Setup

setup: ## install dependencies using uv
	@uv sync --all-extras

##@ Formatters

format-autoflake: ## remove unused imports and variables
	@echo "🔍 Removing unused imports and variables using autoflake..."
	@uv run autoflake --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --in-place --recursive aicapture tests

format-black: ## run black (code formatter)
	@echo "🔍 Formatting code using black..."
	@uv run black -S .

format-isort: ## run isort (import formatter)
	@echo "🔍 Formatting imports using isort..."
	@uv run isort .

format-autopep8: ## fix additional style issues
	@echo "🔍 Fixing additional style issues using autopep8..."
	@uv run autopep8 --in-place --recursive --aggressive --aggressive aicapture tests

format: format-autoflake format-black format-isort format-autopep8 ## run all formatters
	@echo "✨ Code formatting complete!"

##@ Linters

lint-black: ## run black in check mode
	@echo "🔍 Checking code format using black..."
	@uv run black -S . --check

lint-flake8: ## run flake8
	@echo "🔍 Checking code using flake8..."
	@uv run flake8 aicapture tests

lint-mypy: ## run mypy (static-type checker)
	@echo "🔍 Type checking using mypy..."
	@uv run mypy aicapture --show-error-codes --pretty

lint-isort: ## run isort in check mode
	@echo "🔍 Checking import format using isort..."
	@uv run isort . --check-only --diff

lint: lint-black lint-isort lint-flake8 lint-mypy ## run all linters (matches CI order)
	@echo "✨ Code linting complete!"

##@ Tests

test: ## run tests with coverage
	@echo "🧪 Running tests..."
	@uv run pytest -v --cov=aicapture --cov-report=term-missing

##@ Build & Publish

build: ## build package for distribution
	@echo "📦 Building package..."
	@uv build

publish: build ## publish package to PyPI
	@echo "📤 Publishing to PyPI..."
	@uv publish

##@ All

all: format lint test ## run format, lint and test
	@echo "✨ All checks completed!"
