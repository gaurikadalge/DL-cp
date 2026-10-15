# Development Environment Setup Guide

This guide walks through configuring a local development environment for the **Multimodal RAG Research Assistant** on macOS, Linux, and Windows.

---

## 1. Prerequisites
- **Python**: Version `3.11` is the recommended baseline (compatible with Python 3.11–3.13).
- **Git**: Version `2.30+`.
- **System OCR Library (Optional for Phase 1)**:
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install -y tesseract-ocr`
  - Windows: Download installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
- **Make (Optional)**: Standard build automation tool.

---

## 2. Step-by-Step Installation

### Step 2.1: Clone the Repository
```bash
git clone https://github.com/gaurikadalge/DL-cp.git
cd DL-cp
```

### Step 2.2: Create and Activate Virtual Environment

#### macOS / Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

### 3.1 Minimal Phase 0 / Core Development Setup (Recommended)
To keep the development environment light and fast:
```bash
pip install --upgrade pip
pip install -r requirements-dev.txt
```
or via pyproject:
```bash
pip install -e ".[dev]"
```

### 3.2 Full Pipeline Dependencies (For Later Phases)
When developing Phase 1 (Ingestion) or Phase 2 (Hybrid Retrieval):
```bash
pip install -e ".[all]"
```

---

## 4. Environment Configuration

Copy the template configuration file:
```bash
cp .env.example .env
```

Edit `.env` to configure your local parameters:
```ini
APP_NAME="Multimodal RAG Research Assistant"
APP_ENV=development
DEBUG=true
API_HOST=127.0.0.1
API_PORT=8000
DATABASE_URL=sqlite:///./data/research_assistant.db
MAX_UPLOAD_SIZE_MB=25
```

> **Security Rule**: Never commit `.env` or plain text API keys to Git. The `.gitignore` file automatically excludes `.env`.

---

## 5. Running the Application

### 5.1 Using Make (macOS & Linux)
```bash
make dev
```

### 5.2 Direct Uvicorn Command
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Once running, navigate your browser to:
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Alternative ReDoc UI: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- Service Health Audit: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 6. Running Tests & Smoke Verification

### Run Pytest Suite
```bash
pytest -v
# Or using Make:
make test
```

### Run Smoke Test against Running API
In a separate terminal while the API is active:
```bash
python scripts/smoke_test.py
# Or using Make:
make smoke-test
```

### Code Formatting and Linting
```bash
ruff check .
ruff format .
# Or using Make:
make lint
make format
```
