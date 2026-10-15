# System Architecture & Technical Design

## 1. Architectural Principles
1. **Modularity**: Strict separation between Document Ingestion, Retrieval Indexing, LLM Synthesis, and API Presentation.
2. **Determinism & Provenance**: Chunks possess immutable deterministic IDs; citations trace back to precise source files, page offsets, and verbatim excerpts.
3. **No Metric Conflation**: Raw dense vector cosine scores and BM25 scores are kept distinct and never mathematically added directly; Reciprocal Rank Fusion (RRF) operates strictly over ordinal rank positions.
4. **Resilient Degradation**: Graceful handling of image-heavy or degraded documents via OCR fallback; transparent reporting when evidence is insufficient.
5. **Security by Default**: Zero hardcoded secrets; internal server paths and tracebacks are hidden from external API clients.

---

## 2. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph ClientLayer ["Client Layer"]
        UI["Web Interface (Streamlit / React)"]
        CLI["CLI & Smoke Scripts"]
    end

    subgraph APILayer ["FastAPI Service Layer"]
        HealthEndpoint["GET /health"]
        UploadEndpoint["POST /documents/upload"]
        DocStatusEndpoint["GET /documents/{id}/status"]
        QueryEndpoint["POST /query"]
    end

    subgraph IngestionPipeline ["Document Ingestion Engine (Phase 1)"]
        Validator["File Validator (Size/Magic Type)"]
        PDFParser["PyMuPDF Layout Extractor"]
        OCRFallback["Tesseract OCR Fallback"]
        Chunker["Semantic Chunking Engine"]
    end

    subgraph RetrievalEngine ["Hybrid Retrieval Engine (Phase 2)"]
        DenseModel["SentenceTransformer Embedding"]
        FAISSIndex["FAISS Dense Vector Index"]
        BM25Index["BM25 Lexical Keyword Index"]
        RRF["Reciprocal Rank Fusion (RRF)"]
        Reranker["Optional Cross-Encoder Reranker"]
    end

    subgraph SynthesisEngine ["Grounded Answer Engine (Phase 3)"]
        ContextBuilder["Evidence & Prompt Assembler"]
        LLMProvider["Configurable LLM Provider Interface"]
        CitationEngine["Citation Anchoring & Validation"]
    end

    subgraph PersistenceLayer ["Storage Layer"]
        DiskUploads["Document File Storage (./data/uploads)"]
        MetaDB["Metadata Store (SQLite / PostgreSQL)"]
        IndexFiles["Index Storage (./indexes)"]
    end

    UI --> APILayer
    CLI --> APILayer
    APILayer --> IngestionPipeline
    APILayer --> RetrievalEngine
    APILayer --> SynthesisEngine

    UploadEndpoint --> Validator --> PDFParser
    PDFParser -. Scanned Pages .-> OCRFallback
    PDFParser --> Chunker
    Chunker --> DenseModel --> FAISSIndex
    Chunker --> BM25Index
    Chunker --> DiskUploads
    Chunker --> MetaDB

    QueryEndpoint --> RetrievalEngine
    RetrievalEngine --> FAISSIndex
    RetrievalEngine --> BM25Index
    FAISSIndex --> RRF
    BM25Index --> RRF
    RRF --> Reranker
    Reranker --> SynthesisEngine
    SynthesisEngine --> ContextBuilder --> LLMProvider --> CitationEngine
```

---

## 3. Subsystem Breakdown

### 3.1 Ingestion & Processing Subsystem (`ingestion/`)
- **Input Validation**: Verifies file extension (`.pdf`), validates magic bytes (header `%PDF-`), and enforces upload bounds (`MAX_UPLOAD_SIZE_MB`).
- **Extraction**: PyMuPDF extracts text blocks, tables, and bounding box positions per page.
- **OCR Fallback**: Pages with low text density or embedded images invoke Tesseract OCR.
- **Chunking**: Generates chunks with configured token limits (e.g., 300–500 tokens) with sliding overlap (50 tokens). Each chunk preserves `page_number`, `page_end`, and `section`.

### 3.2 Hybrid Retrieval Subsystem (`rag_engine/`)
- **Dense Representation**: `sentence-transformers/all-MiniLM-L6-v2` produces 384-dimensional dense vectors stored in a FAISS FlatIP or HNSW index.
- **Sparse Representation**: `rank-bm25` (Okapi BM25) indexes tokenized words, ensuring domain-specific names and equation symbols are recallable.
- **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{k + r_m(d)}$$
  where $k = 60$ (smoothing constant) and $r_m(d)$ is the 1-indexed rank in retrieval list $m$.
- **Reranker (Optional)**: Cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores top $N$ candidates.

### 3.3 Grounded Generation & Citation Subsystem (`app/services/`)
- **Prompt Architecture**: System instructions force the model to answer exclusively from supplied numbered evidence chunks.
- **Citation Anchoring**: In-text citations (e.g., `[1]`, `[2]`) map directly to chunk metadata and page numbers.
- **Abstention Mode**: If maximum chunk relevance falls below a threshold, the model emits `insufficient_evidence: true` with a refusal explanation.

---

## 4. Ingestion & Retrieval Sequence

```mermaid
sequenceDiagram
    autonumber
    actor User as Researcher
    participant API as FastAPI Backend
    participant Ingest as Ingestion Service
    participant Storage as File & Meta Store
    participant Hybrid as Retrieval Engine
    participant LLM as LLM Provider

    User->>API: POST /documents/upload (research_paper.pdf)
    API->>Ingest: Validate & parse PDF
    Ingest->>Storage: Save PDF to ./data/uploads
    Ingest->>Ingest: Extract text & generate semantic chunks
    Ingest->>Hybrid: Index dense vectors (FAISS) & keywords (BM25)
    Ingest->>Storage: Record document status = 'indexed'
    API-->>User: 201 Created (document_id, status='indexed')

    User->>API: POST /query {"query": "What is the training objective?"}
    API->>Hybrid: Hybrid search (FAISS Top-10, BM25 Top-10)
    Hybrid->>Hybrid: Compute RRF & select top-5 evidence chunks
    API->>LLM: Prompt with evidence chunks & strict citation instructions
    LLM-->>API: Synthesized answer with citations [1], [2]
    API->>API: Validate citations against retrieved chunk IDs
    API-->>User: 200 OK (answer, citations, evidence, timing)
```
