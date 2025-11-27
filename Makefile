.PHONY: setup format lint test build publish all

##@ Setup

setup: ## install dependencies using uv
	@uv sync --all-extras

##@ Formatters

format: ## format code using ruff
	@echo "🚀 Formatting code using Ruff..."
	@uv run ruff format .
	@echo "✨ Code formatting complete!"

##@ Linters

lint: ## run ruff check and mypy (matches CI)
	@echo "🚀 Starting Ruff check..."
	@uv run ruff check . --exclude ".venv/"
	@echo "✅ Ruff check completed"
	@echo ""
	@echo "🚀 Starting MyPy type checking..."
	@echo "MyPy version:"
	@uv run python -m mypy --version
	@echo "Running MyPy on aicapture directory..."
	@uv run python -m mypy aicapture --show-error-codes --pretty
	@echo "✅ MyPy type checking completed"

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
