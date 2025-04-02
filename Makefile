.PHONY: setup format lint test all

##@ Setup

setup:
	@poetry env use python3.10
	@poetry install --with dev

##@ Formatters

format-black: ## run black (code formatter)
	@echo "🔍 Formatting code using black..."
	@poetry run black -S . 

format-isort: ## run isort (import formatter)
	@echo "🔍 Formatting imports using isort..."
	@poetry run isort .

format: format-black format-isort ## run all formatters
	@echo "✨ Code formatting complete!"

##@ Linters

lint-black: ## run black in check mode
	@echo "🔍 Checking code format using black..."
	@poetry run black -S . --check

lint-flake8: ## run flake8
	@echo "🔍 Checking code using flake8..."
	@poetry run flake8 aicapture tests

lint-mypy: ## run mypy (static-type checker)
	@echo "🔍 Type checking using mypy..."
	@poetry run mypy aicapture tests --show-error-codes --pretty

lint-isort: ## run isort in check mode
	@echo "🔍 Checking import format using isort..."
	@poetry run isort . --check-only --diff

lint: lint-black lint-flake8 lint-mypy lint-isort ## run all linters
	@echo "✨ Code linting complete!"

##@ Tests

test: ## run tests with coverage
	@echo "🧪 Running tests..."
	@poetry run pytest -v --cov=aicapture --cov-report=term-missing

##@ All

all: format lint test ## run format, lint and test
	@echo "✨ All checks completed!"
