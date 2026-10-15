# Contributing to Multimodal RAG Research Assistant

Thank you for contributing to the **Multimodal RAG Research Assistant** academic project! To maintain high code quality and smooth collaboration across all four team members, please adhere to these guidelines.

---

## 1. Code of Conduct & Academic Integrity
- Write transparent, reproducible, and verifiable code.
- Cite any academic methods, open-source libraries, or external code snippets utilized.
- Never commit private research data, proprietary PDF papers without license, or proprietary API keys.

---

## 2. Getting Started
1. Review [docs/DEVELOPMENT_SETUP.md](docs/DEVELOPMENT_SETUP.md) for local environment configuration.
2. Ensure you have Python 3.11+ installed.
3. Set up a clean virtual environment and install development dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-dev.txt
   ```
4. Copy `.env.example` to `.env` and verify baseline health:
   ```bash
   make test
   ```

---

## 3. Branching & Commit Conventions

### Branch Naming Conventions
Always create feature branches off the latest `main`:
- `feature/backend`: Backend routes, database models, lifecycle logic (Person 2)
- `feature/ingestion`: Document parsing, PyMuPDF, OCR, chunking (Person 3)
- `feature/rag-engine`: FAISS, BM25, RRF, reranking, LLM synthesis (Person 1)
- `feature/frontend-evaluation`: Streamlit UI, evaluation harness, e2e tests (Person 4)
- `fix/<issue-name>`: Bug fixes
- `docs/<topic>`: Documentation updates

### Commit Messages
Follow the Conventional Commits specification:
```
<type>(<scope>): <short description>

[optional body]
```
Examples:
- `feat(ingestion): add PyMuPDF table extraction fallback`
- `fix(config): enforce port boundary between 1024 and 65535`
- `test(health): add integration test for dependency status audit`
- `docs(api): document query response schema timing fields`

---

## 4. Coding Standards & Tooling

### Python Standards
- Python 3.11+ syntax and type annotations.
- Explicit type hints on all public function signatures and Pydantic schemas.
- Clean docstrings (Google or PEP 257 format) explaining purpose, arguments, and return types.

### Linting and Formatting
Before opening a pull request, run Ruff to ensure consistency:
```bash
ruff check .
ruff format .
# Or simply:
make lint
make format
```

---

## 5. Testing Requirements
- Every new feature must be accompanied by relevant unit tests in `tests/unit/` or integration tests in `tests/integration/`.
- Ensure all tests pass prior to submitting code:
  ```bash
  pytest tests/
  ```
- Do not download large weights or rely on paid APIs inside automated tests. Use fixtures or mock objects.

---

## 6. Pull Request Process
1. Rebase your feature branch on the latest `main`:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
2. Verify all tests and linter checks pass locally.
3. Open a Pull Request targeting `main`.
4. Fill in the PR description detailing:
   - Summary of changes
   - Associated module / role
   - Verification commands and test outcomes
5. Request review from the relevant module partner. At least one approval is required before merging.
