# Deep Learning & RAG Engine Technical Architecture

This document provides a comprehensive technical overview of the **Deep Learning & RAG Engine** implemented by Person 1 for the Multimodal RAG Research Assistant.

---

## 1. High-Level Architecture & Pipeline Flow

The RAG engine executes an end-to-end, multi-stage retrieval, ranking, and grounded generation pipeline:

```mermaid
flowchart TD
    subgraph QueryInput ["1. User Inquiry"]
        Q["User Query Text"]
        F["Optional Document Filters"]
    end

    subgraph RetrievalStage ["2. Hybrid Retrieval"]
        E["EmbeddingService (SentenceTransformers)"]
        D["DenseRetriever (FAISS IndexFlatIP)"]
        B["BM25Retriever (rank-bm25)"]
        RRF["Reciprocal Rank Fusion (RRF)"]
        
        Q --> E --> D
        Q --> B
        D -->|Top-K Dense Ranks| RRF
        B -->|Top-K BM25 Ranks| RRF
    end

    subgraph RankingStage ["3. Optional Reranking"]
        Reranker["RerankerService (CrossEncoder)"]
        RRF -->|Fused Candidates| Reranker
    end

    subgraph SynthesisStage ["4. Context & Generation"]
        CB["ContextBuilder (Token-Budgeted & Sanitized)"]
        LLM["AnswerGenerator (LLMProvider Interface)"]
        CV["CitationValidator (Source Provenance)"]
        
        Reranker --> CB
        CB -->|Context String [1],[2]| LLM
        LLM -->|Raw Answer| CV
        CB -->|Citation Map| CV
    end

    subgraph ResponseOutput ["5. Structured Response"]
        Resp["QueryResponse Schema"]
        CV --> Resp
    end
```

---

## 2. Component Design & Implementation

### 2.1 Embedding Service (`rag_engine/embedding_model.py`)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) by default; configurable to `BAAI/bge-base-en-v1.5` (768-dim) or domain checkpoints.
- **Normalization**: Mandatory L2 normalization ($||\mathbf{v}||_2 = 1.0$), ensuring that dot-product / inner product calculations directly yield cosine similarity.
- **Lazy Loading**: Model weights are loaded strictly upon first embedding invocation, preventing heavy startup penalties or unneeded memory consumption during testing.
- **Query Prefixing**: Supports asymmetric query instructions (e.g. `"Represent this sentence for searching relevant passages: "`).
- **Validation**: Strict checks rejecting empty strings, whitespace-only texts, non-string types, and non-finite outputs (NaN/Inf).

### 2.2 Dense Retrieval (`rag_engine/dense_retriever.py`)
- **Index Type**: `faiss.IndexFlatIP` (exact inner-product search on L2-normalized vectors).
- **Index Integrity**: Maintains a 1-to-1 sequential mapping between FAISS row IDs and `StableChunkMetadata`.
- **Filtering**: Supports document-scoped searches (`document_ids: List[str]`) by oversampling candidates and filtering post-search.
- **Persistence**: Serializes index to `dense_index.faiss` and metadata mapping to `dense_metadata.json`.
- **Deletion**: Removes a document's chunks and rebuilds the FAISS index to ensure zero ghost vectors remain.

### 2.3 BM25 Keyword Retrieval (`rag_engine/bm25_retriever.py`)
- **Algorithm**: Okapi BM25 via `rank_bm25.BM25Okapi`.
- **Tokenization**: Deterministic alphanumeric tokenization preserving hyphenated terms (`ResNet-50`), acronyms (`AdamW`), and numerical values (`O(n^2 * d)`).
- **Persistence**: Serializes corpus text chunks to `bm25_metadata.json` for deterministic rebuilds upon service restart.

### 2.4 Reciprocal Rank Fusion (`rag_engine/hybrid_retriever.py`)
- **Mathematical Formulation**:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{k + r_m(d)}$$
  where $k = 60$ (configurable smoothing constant) and $r_m(d)$ is the 1-indexed ordinal rank of chunk $d$ in system $m$.
- **Why RRF?**: Dense vector cosine scores ($\in [-1, 1]$) and BM25 scores ($\ge 0$, unbounded) inhabit uncalibrated scales and must never be added directly. RRF operates purely over rank positions.
- **Deduplication**: Chunks retrieved by both systems receive fused credit and appear once in the output list.
- **Tie-Breaking**: Deterministic alphabetical tie-breaking on `chunk_id`.

### 2.5 Optional Cross-Encoder Reranker (`rag_engine/reranker.py`)
- **Model**: `SentenceTransformers CrossEncoder` (`BAAI/bge-reranker-base` or `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **Scoring**: Computes joint cross-attention score on $(q, passage)$ pairs.
- **Fallback Policy**: If disabled or if inference fails (e.g., GPU OOM), gracefully falls back to the RRF candidate ordering without crashing.

### 2.6 Context Construction (`rag_engine/context_builder.py`)
- **Budgeting**: Enforces strict token budget limits (`context_budget_tokens: 2048` by default).
- **Injection Defense**: Delimits evidence blocks with clear markers (`--- BEGIN EVIDENCE [1] ---`) and defangs markdown code block escapes (`'''`) and control tokens (`<|im_start|>`).
- **Citation Keying**: Assigns sequential bracketed tags (`[1]`, `[2]`, ...) to selected chunks and builds a `citation_map` for validation.

### 2.7 Grounded Answer Generation (`rag_engine/answer_generator.py`)
- **Provider Abstraction**: Abstract `LLMProvider` interface with `MockLLMProvider` (offline/test) and `OpenAILLMProvider` (API-backed).
- **System Instructions**: Forces the model to synthesize answers solely from numbered evidence blocks and acknowledge uncertainty with `INSUFFICIENT_EVIDENCE` if evidence is lacking.
- **Prompt Isolation**: Documents are treated as untrusted data; the system prompt instructs the model to ignore commands within evidence text.

### 2.8 Citation Validation (`rag_engine/citations.py`)
- **Resolution**: Parses all `[N]` tags in the answer and resolves them against the authoritative `citation_map`.
- **Zero Hallucination Tolerance**: Rejects hallucinated tags (e.g. `[99]`). Never trusts LLM-emitted filenames or page numbers; pulls all citation metadata from the verified `StableChunkMetadata`.
- **Verbatim Excerpt**: Extracts a representative clean snippet directly from source chunk text.

### 2.9 Index Manager (`rag_engine/index_manager.py`)
- Provides synchronized `add_chunks()`, `delete_document()`, `save()`, `load()`, and `rebuild_indexes()`.
