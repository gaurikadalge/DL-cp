# Project Implementation Roadmap & Milestones

This document specifies the phased technical progression for the Multimodal RAG Research Assistant. Every phase defines a concrete deliverable and an explicit acceptance gate that must be satisfied before progressing.

---

## Roadmap Overview

```mermaid
gantt
    title Engineering Implementation Roadmap
    dateFormat  YYYY-MM-DD
    section Foundation
    Phase 0: Repository Foundation           :done,    p0, 2026-10-15, 1d
    section Core Pipeline
    Phase 1: Vertical Slice (PDF to Evidence):active,  p1, after p0, 3d
    Phase 2: Ingestion & OCR Robustness      :         p2, after p1, 4d
    Phase 3: Persistent Meta & Index Storage :         p3, after p2, 3d
    section Advanced RAG
    Phase 4: Hybrid Retrieval & Reranker     :         p4, after p3, 4d
    Phase 5: Grounded Answer & Citations     :         p5, after p4, 4d
    section Delivery
    Phase 6: Frontend Portal (Streamlit)     :         p6, after p5, 3d
    Phase 7: Quantitative Evaluation         :         p7, after p6, 3d
    Phase 8: Packaging & Final Demo          :         p8, after p7, 2d
```

---

## Detailed Phases & Acceptance Gates

### Phase 0: Repository Inspection & Foundation
- **Deliverables**:
  - Foundational repository structure with clean `.gitignore`, `.gitattributes`, and `.env.example`.
  - Minimal FastAPI entry point (`app/main.py`) with `GET /` and `GET /health`.
  - Configuration engine (`app/config.py`) with validation and secret redaction.
  - Complete documentation suite (`docs/`) and formal API contracts (`docs/API_CONTRACTS.md`).
  - Unit tests for configuration and health endpoints; smoke-test script.
- **Acceptance Gate**:
  - `pytest tests/unit/` passes 100%.
  - `scripts/smoke_test.py` validates live server with zero errors.
  - No secrets, large files, or private PDFs committed.
- **Status**: **COMPLETE**

---

### Phase 1: PDF-to-Retrieved-Evidence Vertical Slice
- **Deliverables**:
  - End-to-end barebones pipeline connecting PDF input $\to$ basic text extraction $\to$ embedding generation $\to$ retrieval.
  - Validates that the chosen embedding model and vector index operate cohesively.
- **Acceptance Gate**:
  - Script ingests a single sample academic paper, queries it with a domain question, and outputs top-3 text snippets with correct source page numbers.
- **Status**: Planned (Next Phase)

---

### Phase 2: Reliable Document Ingestion & OCR Fallback
- **Deliverables**:
  - PyMuPDF structured extractor capturing multi-column text, titles, sections, and tables.
  - Tesseract OCR integration with automated fallback trigger for image-heavy pages.
  - Deterministic chunking engine emitting stable chunk IDs (`{doc_id}#chunk_{idx}`) and page offsets.
- **Acceptance Gate**:
  - Tested across 5 diverse academic PDFs (scanned paper, double-column arXiv paper, paper with math/tables); zero chunking exceptions; accurate page mapping.
- **Status**: Planned

---

### Phase 3: Persistent Metadata & Index Lifecycle
- **Deliverables**:
  - Database persistence for document records and processing states (`uploaded`, `processing`, `indexed`, `failed`, `deleting`).
  - Safe index serialization and deserialization from disk (`./indexes/`).
  - Ingestion queue or async background processing for non-blocking uploads.
- **Acceptance Gate**:
  - Server restart preserves indexed documents and allows immediate querying without re-indexing.
- **Status**: Planned

---

### Phase 4: Hybrid Retrieval & Optional Reranking
- **Deliverables**:
  - Dense FAISS index integration + Sparse BM25 keyword index integration.
  - Reciprocal Rank Fusion (RRF) algorithm combining dense and lexical ranks without score blending.
  - Optional Cross-Encoder reranker module.
- **Acceptance Gate**:
  - Retrieval benchmark confirms RRF achieves higher Hit@5 on academic acronyms/terms than FAISS or BM25 individually.
- **Status**: Planned

---

### Phase 5: Grounded Answer Generation & Citation Validation
- **Deliverables**:
  - LLM provider abstraction layer (`openai`, `anthropic`, `ollama`, or mock).
  - Strict grounding prompt injecting retrieved evidence with citation index tags (`[1]`, `[2]`).
  - Post-generation citation parser validating that every cited tag maps to retrieved evidence.
  - Insufficient-evidence refusal mechanism (`insufficient_evidence: true`).
- **Acceptance Gate**:
  - Model refuses to answer ungrounded questions; 100% of emitted citations link to valid source chunk snippets.
- **Status**: Planned

---

### Phase 6: Frontend Integration
- **Deliverables**:
  - Interactive research portal (Streamlit prototype / React UI).
  - PDF document uploader with real-time processing status.
  - Research chat interface with interactive citation badges and snippet inspector.
- **Acceptance Gate**:
  - User can upload a paper via UI, ask a question, and click a citation badge to view source evidence and page location.
- **Status**: Planned

---

### Phase 7: Evaluation & Robustness Testing
- **Deliverables**:
  - Benchmark QA dataset of academic papers (`evaluation/datasets/`).
  - Automated evaluation harness computing Hit@K, MRR, context relevance, faithfulness, and answer relevance.
  - Latency and throughput benchmarking report.
- **Acceptance Gate**:
  - Automated evaluation report generated; faithfulness metric $> 90\%$ on test benchmark.
- **Status**: Planned

---

### Phase 8: Packaging & Final Demonstration
- **Deliverables**:
  - Complete demo script and reproducible sample papers.
  - Polished technical report and demonstration video/slides.
  - Finalized production-ready repository with complete documentation.
- **Acceptance Gate**:
  - Clean clone runs out of the box following `docs/DEVELOPMENT_SETUP.md` with zero undocumented manual steps.
- **Status**: Planned
