# Shared API Contracts & Interface Specifications

> **Notice**: This document defines the formal OpenAPI contracts agreed upon by the engineering team for Phase 1 through Phase 5. In **Phase 0**, only `GET /` and `GET /health` are active and verified. The document processing and query endpoints defined below represent agreed interfaces to enable parallel development.

---

## 1. System Endpoints (Implemented in Phase 0)

### 1.1 Service Information
- **Endpoint**: `GET /`
- **Description**: Returns basic application metadata and links to OpenAPI documentation.
- **Response (200 OK)**:
```json
{
  "app_name": "Multimodal RAG Research Assistant",
  "version": "0.1.0",
  "environment": "development",
  "docs_url": "/docs",
  "health_url": "/health"
}
```

### 1.2 Health Check & Dependency Audit
- **Endpoint**: `GET /health`
- **Description**: Verifies service responsiveness and explicitly audits external subsystem readiness.
- **Response (200 OK)**:
```json
{
  "status": "ok",
  "app_name": "Multimodal RAG Research Assistant",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-10-15T17:00:00Z",
  "dependencies": {
    "database": "configured",
    "vector_index": "not_loaded_scaffold",
    "keyword_retrieval": "not_loaded_scaffold",
    "llm_provider": "not_configured",
    "ocr_engine": "disabled",
    "reranker": "disabled"
  }
}
```

---

## 2. Document Ingestion Lifecycle Contracts (Planned: Phase 1 & 2)

### Allowed Processing Status Values
1. `uploaded`: PDF received, stored on disk, queued for parsing.
2. `processing`: PyMuPDF/OCR parsing and chunk generation in progress.
3. `indexed`: Embeddings computed, FAISS vector index and BM25 index updated.
4. `failed`: Document parsing or indexing failed (details in `error_message`).
5. `deleting`: Index removal and disk purge in progress.

### 2.1 Upload Document
- **Endpoint**: `POST /documents/upload`
- **Content-Type**: `multipart/form-data`
- **Request Parameters**:
  - `file`: Binary PDF file (Max `MAX_UPLOAD_SIZE_MB`, default 25MB)
- **Response (202 Accepted / 201 Created)**:
```json
{
  "document_id": "doc_a7c8e92f",
  "filename": "deep_residual_learning.pdf",
  "status": "uploaded",
  "upload_timestamp": "2026-10-15T17:05:00Z",
  "file_size_bytes": 1048576,
  "message": "File uploaded successfully and scheduled for processing."
}
```

### 2.2 List Ingested Documents
- **Endpoint**: `GET /documents`
- **Query Parameters**: `skip` (int, default 0), `limit` (int, default 50)
- **Response (200 OK)**:
```json
[
  {
    "document_id": "doc_a7c8e92f",
    "filename": "deep_residual_learning.pdf",
    "status": "indexed",
    "page_count": 12,
    "chunk_count": 48,
    "created_at": "2026-10-15T17:05:00Z",
    "updated_at": "2026-10-15T17:05:12Z",
    "error_message": null
  }
]
```

### 2.3 Get Document Details & Status
- **Endpoint**: `GET /documents/{document_id}`
- **Endpoint**: `GET /documents/{document_id}/status`
- **Response (200 OK)**:
```json
{
  "document_id": "doc_a7c8e92f",
  "filename": "deep_residual_learning.pdf",
  "status": "indexed",
  "page_count": 12,
  "chunk_count": 48,
  "created_at": "2026-10-15T17:05:00Z",
  "updated_at": "2026-10-15T17:05:12Z",
  "error_message": null
}
```

### 2.4 Delete Document
- **Endpoint**: `DELETE /documents/{document_id}`
- **Response (200 OK)**:
```json
{
  "document_id": "doc_a7c8e92f",
  "status": "deleting",
  "message": "Document and associated vectors scheduled for deletion."
}
```

---

## 3. Stable Chunk Metadata Specification

All ingested text chunks conform to this invariant metadata contract:

```json
{
  "chunk_id": "doc_a7c8e92f#chunk_0012",
  "document_id": "doc_a7c8e92f",
  "text": "To tackle the degradation problem, we introduce a deep residual learning framework...",
  "page_number": 2,
  "page_end": 2,
  "section": "3. Deep Residual Learning",
  "content_type": "text",
  "source_filename": "deep_residual_learning.pdf",
  "extraction_method": "pymupdf",
  "metadata": {
    "token_count": 142,
    "has_table": false,
    "has_math": true
  }
}
```

---

## 4. Query & Hybrid Retrieval Contracts (Planned: Phase 3)

### 4.1 Research Query
- **Endpoint**: `POST /query`
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
  "query": "How do residual connections address the degradation problem during training?",
  "document_ids": ["doc_a7c8e92f"],
  "top_k": 5,
  "stream": false
}
```

### 4.2 Query Response Schema
> **Important Note on Score Semantics**: Dense similarity scores (inner product / cosine $\in [-1, 1]$) and BM25 scores (unbounded $> 0$) are non-calibrated distinct metrics. They are never compared or added directly. The hybrid retrieval engine ranks candidates via **Reciprocal Rank Fusion (RRF)**, producing `fusion_score` and `fusion_rank`.

- **Response (200 OK)**:
```json
{
  "query": "How do residual connections address the degradation problem during training?",
  "answer": "Residual connections address the degradation problem by reformulating layers to fit residual mappings F(x) = H(x) - x rather than directly fitting the desired underlying mapping H(x) [1]. This enables identity shortcuts where gradients flow directly through the skip connections, eliminating vanishing gradient hurdles in deep networks [2].",
  "citations": [
    {
      "citation_id": "[1]",
      "chunk_id": "doc_a7c8e92f#chunk_0012",
      "document_id": "doc_a7c8e92f",
      "source_filename": "deep_residual_learning.pdf",
      "page_number": 2,
      "section": "3. Deep Residual Learning",
      "snippet": "Instead of hoping each few stacked layers directly fit a desired underlying mapping, we explicitly let these layers fit a residual mapping."
    },
    {
      "citation_id": "[2]",
      "chunk_id": "doc_a7c8e92f#chunk_0015",
      "document_id": "doc_a7c8e92f",
      "source_filename": "deep_residual_learning.pdf",
      "page_number": 3,
      "section": "3.1. Formulation",
      "snippet": "The formulation of F(x) + x by shortcut connections neither introduces extra parameter nor computation complexity."
    }
  ],
  "evidence": [
    {
      "chunk": {
        "chunk_id": "doc_a7c8e92f#chunk_0012",
        "document_id": "doc_a7c8e92f",
        "text": "Instead of hoping each few stacked layers directly fit a desired underlying mapping...",
        "page_number": 2,
        "page_end": 2,
        "section": "3. Deep Residual Learning",
        "content_type": "text",
        "source_filename": "deep_residual_learning.pdf",
        "extraction_method": "pymupdf",
        "metadata": {}
      },
      "scoring": {
        "dense_score": 0.842,
        "bm25_score": 14.28,
        "fusion_score": 0.0328,
        "fusion_rank": 1,
        "rerank_score": 0.924
      }
    }
  ],
  "insufficient_evidence": false,
  "timing": {
    "retrieval_ms": 14.2,
    "reranking_ms": 8.5,
    "generation_ms": 612.0,
    "total_ms": 634.7
  }
}
```

### 4.3 Fallback / Insufficient Evidence Response
When evidence relevance is below minimum certainty threshold:
```json
{
  "query": "What quantum computing hardware was used in this study?",
  "answer": "The provided research documents do not contain evidence regarding quantum computing hardware.",
  "citations": [],
  "evidence": [],
  "insufficient_evidence": true,
  "timing": {
    "retrieval_ms": 11.0,
    "reranking_ms": null,
    "generation_ms": 120.0,
    "total_ms": 131.0
  }
}
```

---

## 5. Unified Error Response Contract

All errors return the standard `ErrorDetail` JSON payload:

```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Request payload validation failed",
  "details": [
    {
      "loc": ["body", "query"],
      "msg": "String should have at least 3 characters",
      "type": "string_too_short"
    }
  ],
  "timestamp": "2026-10-15T17:10:00Z"
}
```
