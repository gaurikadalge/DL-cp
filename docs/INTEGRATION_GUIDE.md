# RAG Module Integration Guide for Teammates

This guide provides explicit interface specifications, code snippets, and handoff contracts for integrating the **RAG Engine** with the work of Person 2 (Backend), Person 3 (Ingestion), and Person 4 (Frontend/Evaluation).

---

## 1. Integration with Person 3 (Multimodal Processing Lead)

### Handoff Contract: Ingestion $\to$ RAG Engine
Person 3 extracts academic PDF content and produces structured, stable chunks.

```python
from app.schemas import StableChunkMetadata
from rag_engine import IndexManager

# 1. Person 3 generates chunks conforming to StableChunkMetadata:
chunks = [
    StableChunkMetadata(
        chunk_id=f"{document_id}#chunk_{idx:04d}",
        document_id=document_id,
        text="Extracted section text...",
        page_number=page_idx,
        page_end=page_idx,
        section="3. Methodology",
        content_type="text",
        source_filename="vaswani2017.pdf",
        extraction_method="pymupdf",
        metadata={"has_table": False, "token_count": 120},
    ),
    # ...
]

# 2. Feed chunks directly to IndexManager for synchronized indexing:
index_manager.add_chunks(chunks)

# 3. Save updated indexes to disk:
index_manager.save("./indexes")
```

---

## 2. Integration with Person 2 (Backend & Integration Lead)

### Route Integration: FastAPI $\to$ RAG Pipeline
Person 2 mounts the RAG pipeline into FastAPI dependency injection or endpoint handlers.

```python
from fastapi import APIRouter, Depends, HTTPException
from app.config import get_settings
from app.schemas import QueryRequest, QueryResponse
from rag_engine import RAGPipeline, create_pipeline

router = APIRouter(prefix="/query", tags=["Retrieval & QA"])

# Initialize pipeline as a singleton or lifespan dependency:
settings = get_settings()
pipeline: RAGPipeline = create_pipeline(settings)


@router.post("", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    try:
        response: QueryResponse = pipeline.answer_question(
            query=request.query,
            document_ids=request.document_ids,
            top_k=request.top_k,
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal query processing error")
```

### Document Deletion Contract (`DELETE /documents/{id}`)
When a user deletes a document via the API, Person 2 invokes:
```python
# Purge document chunks from both FAISS and BM25:
removed_count = index_manager.delete_document(document_id)
# Persist updated state:
index_manager.save("./indexes")
```

---

## 3. Integration with Person 4 (Frontend & Evaluation Lead)

### Consumption Contract: Response Parsing for UI
Person 4 renders the research chat, citation badges, and evidence inspector from the `QueryResponse` payload:

```json
{
  "query_id": "query_1a2b3c4d",
  "query": "How do residual connections prevent degradation?",
  "answer": "Residual connections address degradation by fitting residual mappings F(x) = H(x) - x [1]...",
  "citations": [
    {
      "citation_id": "[1]",
      "chunk_id": "resnet#chunk_002",
      "document_id": "doc_resnet",
      "source_filename": "deep_residual_learning.pdf",
      "page_number": 2,
      "section": "3. Deep Residual Learning",
      "snippet": "Instead of hoping each few stacked layers directly fit a desired underlying mapping..."
    }
  ],
  "evidence": [
    {
      "chunk": { "chunk_id": "resnet#chunk_002", "page_number": 2, "text": "..." },
      "scoring": { "dense_score": 0.84, "bm25_score": 14.2, "fusion_score": 0.032, "fusion_rank": 1 }
    }
  ],
  "insufficient_evidence": false,
  "timing": {
    "retrieval_ms": 12.4,
    "reranking_ms": null,
    "generation_ms": 540.2,
    "total_ms": 552.6
  }
}
```

### UI Interaction Recommendations
1. **Interactive Citations**: When the user clicks `[1]`, display the source PDF name, jump to `page_number: 2`, and highlight the `snippet` text in the document viewer.
2. **Evidence Sidebar**: Expand `evidence` to display chunk-level dense vs BM25 vs RRF scores.
3. **Insufficient Evidence Banner**: When `insufficient_evidence: true`, display a warning badge explaining that the retrieved documents lacked adequate evidence.

---

## 4. Verification & Integration Test Commands

```bash
# 1. Run full unit & integration tests
pytest tests/ -v

# 2. Run retrieval evaluation benchmark
PYTHONPATH=. python3 evaluation/evaluate_retrieval.py

# 3. Test end-to-end RAG pipeline
pytest tests/integration/test_rag_pipeline.py -v

# 4. Test index lifecycle (add, save, load, delete)
pytest tests/integration/test_index_lifecycle.py -v
```
