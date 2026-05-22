.PHONY: help install setup test lint format clean run dashboard docs

help:
	@echo "ESG Analyzer - Available Commands"
	@echo "================================"
	@echo "make install       - Install dependencies"
	@echo "make setup         - Setup development environment"
	@echo "make test          - Run tests"
	@echo "make lint          - Run linters"
	@echo "make format        - Format code"
	@echo "make clean         - Clean temporary files"
	@echo "make run           - Run analysis pipeline"
	@echo "make dashboard     - Start Streamlit dashboard"
	@echo "make docs          - Generate documentation"

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

setup: install
	@echo "Setup complete!"
	@echo "Next steps:"
	@echo "1. Copy .env.example to .env"
	@echo "2. Add your Groq API key to .env"
	@echo "3. Add PDF reports to data/reports/"
	@echo "4. Run 'make run' to start analysis"

test:
	pytest tests/ -v --cov=. --cov-report=html

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	pylint pipeline/ taxonomy/ analysis/ models/

format:
	black . --line-length=100
	isort . --profile black

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage

run:
	python main.py

dashboard:
	streamlit run dashboard/dashboard.py

docs:
	@echo "Generating documentation..."
	@echo "Documentation is in README.md"

.DEFAULT_GOAL := help
