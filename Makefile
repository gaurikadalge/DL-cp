.PHONY: help install dev test lint format smoke-test streamlit eval clean frontend-install frontend-dev frontend-build frontend-lint

help:
	@echo "Available commands:"
	@echo "  make install         Install core and development Python dependencies"
	@echo "  make dev             Start local FastAPI development server"
	@echo "  make test            Run pytest test suite"
	@echo "  make lint            Check code formatting and linting with ruff"
	@echo "  make format          Auto-format code with ruff"
	@echo "  make smoke-test      Run smoke test against running API"
	@echo "  make streamlit       Launch Streamlit research portal"
	@echo "  make eval            Run empirical retrieval benchmark"
	@echo "  make frontend-install Install Next.js frontend dependencies"
	@echo "  make frontend-dev    Start Next.js frontend development server"
	@echo "  make frontend-build  Compile Next.js production build"
	@echo "  make frontend-lint   Run ESLint on Next.js frontend"
	@echo "  make clean           Clean build, cache, and bytecode files"

install:
	pip install -r requirements-dev.txt

dev:
	uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

test:
	pytest tests/

lint:
	ruff check .

format:
	ruff format .

smoke-test:
	python scripts/smoke_test.py

streamlit:
	streamlit run app/frontend.py --server.port 8501

eval:
	PYTHONPATH=. python evaluation/evaluate_retrieval.py

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

frontend-lint:
	cd frontend && npm run lint

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
