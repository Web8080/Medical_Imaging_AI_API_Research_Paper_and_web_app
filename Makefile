# Medical Imaging AI API Makefile

.PHONY: help install test lint format clean docker-build docker-up docker-down

help: ## Show this help message
	@echo "Medical Imaging AI API - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -e .

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint: ## Run linting
	flake8 src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/

docker-build: ## Build Docker image
	docker build -t medical-imaging-api .

docker-up: ## Start Docker services
	docker-compose up -d

docker-down: ## Stop Docker services
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

docker-shell: ## Open shell in API container
	docker-compose exec api bash

run: ## Run the API locally
	python -m src.main

run-dev: ## Run the API in development mode
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

db-migrate: ## Run database migrations
	alembic upgrade head

db-revision: ## Create new database revision
	alembic revision --autogenerate -m "$(MSG)"

db-reset: ## Reset database
	docker-compose down -v
	docker-compose up -d postgres redis
	sleep 10
	make db-migrate

setup: install-dev ## Setup development environment
	pre-commit install
	make db-migrate

docs: ## Generate API documentation
	sphinx-build -b html docs/ docs/_build/html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8001
