# Changelog

All notable changes to the Multimodal RAG Research Assistant project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-15

### Added - Phase 0: Repository Foundation
- Initialized clean repository structure connected to `gaurikadalge/DL-cp`.
- Created standardized development tooling configurations: `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, and `Makefile`.
- Established comprehensive `.gitignore` protecting secrets, virtualenvs, uploaded files, vector indexes, and SQLite databases while preserving `.gitkeep` placeholders.
- Added `.gitattributes` for LF line normalization and binary file safeguarding.
- Created `.env.example` template with safe defaults, upload bounds, and documented placeholders.
- Added licensing notice (`LICENSE`) explaining provisional unlicensed status pending owner selection.
- Created optional development `docker-compose.yml` for PostgreSQL service.
- Implemented minimal FastAPI entrypoint (`app/main.py`) with `GET /` and `GET /health` endpoints.
- Implemented centralized configuration management (`app/config.py`) using `pydantic-settings` with port validation and secret redaction (`safe_dict`).
- Implemented comprehensive Pydantic schemas (`app/schemas.py`) for API health, unified error responses, document processing statuses, stable chunk metadata, hybrid retrieval scoring, and citations.
- Implemented global error handling preventing system path or stack trace leakage to API clients.
- Created lightweight, dependency-free CLI smoke test script (`scripts/smoke_test.py`).
- Added unit tests for configuration and health endpoints with 100% pass rate (`tests/unit/test_config.py`, `tests/unit/test_health.py`).
- Authored full documentation suite under `docs/`:
  - `PROJECT_OVERVIEW.md`
  - `ARCHITECTURE.md` (with system topology & sequence diagrams)
  - `API_CONTRACTS.md`
  - `DEVELOPMENT_SETUP.md`
  - `TEAM_WORKFLOW.md` (with 4-person role matrix and branch strategy)
  - `TESTING_AND_EVALUATION.md`
  - `SECURITY_AND_PRIVACY.md`
  - `ROADMAP.md` (with Phase 0–8 acceptance gates)
- Created tracking documents: `IMPLEMENTATION_STATUS.md` and `CONTRIBUTING.md`.
