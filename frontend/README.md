# Multimodal RAG Research Assistant — Next.js Web Frontend

A modern, responsive, and type-safe web frontend for the **Multimodal RAG Research Assistant**, built with **Next.js 16 (App Router)**, **React 19**, **TypeScript 5**, and **Tailwind CSS v4**.

---

## 🌟 Key Features

1. **Executive Dashboard (`/`)**:
   - Real-time pipeline health indicator polling the FastAPI backend (`GET /health`).
   - High-level KPIs: Total indexed documents, FAISS vector chunks, retrieval benchmarks, and latency metrics.
   - Quick research query launchpad with prompt suggestions.
   - Interactive document library snapshot with direct navigation.
   - Pipeline topology visualizer displaying Ingestion → Hybrid Retrieval → Cross-Encoder → Grounded LLM.

2. **Research Assistant (`/assistant`)**:
   - Interactive conversational chat with streaming-ready design.
   - Grounded responses with precise inline citation tags `[1]`, `[2]`.
   - Clickable citation provenance chips revealing document title, page number, and exact text snippet.
   - Slide-over **Evidence Drawer** exposing FAISS dense similarity, BM25 term scores, RRF fusion rank, and Cross-Encoder reranker scores.
   - Configurable $k$-retrieval slider ($k \in [1, 20]$).
   - Low-evidence guardrail warnings when documents lack sufficient factual support.

3. **Document Library (`/documents`)**:
   - Drag-and-drop PDF uploader with real-time status feedback (`POST /documents/upload`).
   - Live document catalog with client-side search filtering.
   - Document metadata inspection (document ID, page count, chunk count, file size, timestamps).
   - Document deletion (`DELETE /documents/{id}`) with synchronous state synchronization.

4. **Retrieval Evaluation Dashboard (`/evaluation`)**:
   - Interactive visualization of the 35-query benchmark using **Recharts**.
   - Side-by-side metric comparison across 4 retrieval pipelines:
     - **Dense FAISS** (Cosine similarity on `all-MiniLM-L6-v2`)
     - **Lexical BM25** (BM25Okapi sparse index)
     - **Hybrid RRF** (Reciprocal Rank Fusion, $k=60$)
     - **Hybrid + Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`)
   - Visual charts for **Hit@5**, **Recall@5**, **MRR**, **NDCG@5**, and **Latency**.
   - Category breakdowns: Academic/Research, Table & Structural, OCR & Scanned, Cross-Page, and Unanswerable queries.
   - CSV export for academic submission and defense presentations.

5. **System Settings & Health Monitor (`/settings`)**:
   - Live round-trip ping latency measurement against the backend.
   - Subsystem diagnostic grid monitoring all 6 pipeline components:
     - SQLite Database
     - FAISS Vector Index
     - BM25 Keyword Index
     - LLM Provider
     - Tesseract OCR Engine
     - Cross-Encoder Reranker
   - Direct links to Swagger (`/docs`) and Redoc (`/redoc`).
   - Environment setup guidelines.

---

## 🛠️ Technology Stack

- **Framework**: [Next.js 16.4.2](https://nextjs.org/) (App Router)
- **UI Library**: [React 19](https://react.dev/)
- **Language**: [TypeScript 5](https://www.typescriptlang.org/) (Strict typing against `app/schemas.py`)
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/)
- **Charts**: [Recharts 2.15](https://recharts.org/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **Theme**: System/Dark/Light Mode with CSS variables

---

## 🚀 Getting Started

### 1. Prerequisites

- **Node.js**: v18.18.0 or later
- **Python**: v3.10+ (for running the FastAPI backend)

### 2. Configure Environment

Ensure `frontend/.env.local` contains the backend URL:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

### 3. Install Dependencies

```bash
cd frontend
npm install
```

### 4. Run the Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### 5. Production Build & Linting

```bash
# Type check and production build
npm run build

# Run ESLint
npm run lint

# Start production server
npm run start
```

---

## 🔌 API Contract Alignment

The frontend strictly implements the FastAPI schema defined in `app/schemas.py`:

| Endpoint | Method | TypeScript Type | Description |
|---|---|---|---|
| `/health` | `GET` | `HealthCheckResponse` | Subsystem diagnostics and service status |
| `/` | `GET` | `AppInfo` | API root metadata and documentation links |
| `/documents` | `GET` | `DocumentMetadata[]` | Paginated document library |
| `/documents/upload` | `POST` | `DocumentUploadResponse` | Multipart PDF upload and asynchronous ingestion |
| `/documents/{id}` | `DELETE` | `{ status, message }` | Document deletion and index purge |
| `/query` | `POST` | `QueryResponse` | Grounded RAG query answering with citations & timing |
| `/evaluation/benchmark` | `GET` | `BenchmarkResultsData` | Labeled benchmark evaluation dataset |

---

## 📁 Project Structure

```
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx            # Global layout with AppShell wrapper
│   │   ├── page.tsx              # Executive Dashboard
│   │   ├── globals.css           # Tailwind v4 theme variables
│   │   ├── assistant/
│   │   │   └── page.tsx          # Conversational Research Assistant
│   │   ├── documents/
│   │   │   └── page.tsx          # Document Library & Upload
│   │   ├── evaluation/
│   │   │   └── page.tsx          # Benchmark Visualizations & Metrics
│   │   └── settings/
│   │       └── page.tsx          # System Diagnostics & Config
│   ├── components/
│   │   ├── AppShell.tsx          # Sidebar + Header + Polling layout
│   │   ├── CitationChip.tsx      # Provenance badge with click handler
│   │   ├── EvidenceCard.tsx      # Retrieved chunk & scoring inspector
│   │   ├── FileUploader.tsx      # Drag-and-drop PDF uploader
│   │   ├── Header.tsx            # Navigation header with theme toggle
│   │   ├── MetricCard.tsx        # KPI statistics card
│   │   ├── Sidebar.tsx           # Collapsible navigation drawer
│   │   └── StatusBadge.tsx       # Color-coded status indicator
│   ├── data/
│   │   └── benchmarkData.ts      # 35-query benchmark fallback dataset
│   ├── lib/
│   │   ├── api.ts                # Centralized typed HTTP client
│   │   └── utils.ts              # Styling and formatting helpers
│   └── types/
│       └── api.ts                # Strict TypeScript interfaces matching backend
├── .env.example
├── .env.local
├── package.json
├── tsconfig.json
└── README.md
```
