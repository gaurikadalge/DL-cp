# Testing Strategy & Evaluation Framework

## 1. Multi-Tier Testing Pyramid

The project maintains a three-tier testing strategy to ensure reliability from low-level algorithms up to complete user workflows:

```
          / \
         /   \      End-to-End Tests (tests/e2e/)
        /     \     - Full upload-to-answer user journeys
       /-------\
      /         \   Integration Tests (tests/integration/)
     /           \  - FastAPI route handlers + DB + Ingestion mocks
    /-------------\
   /               \ Unit Tests (tests/unit/)
  /                 \ - Configuration validation, schema serialization,
 /                   \  RRF score calculation, text chunking algorithms
-----------------------
```

### 1.1 Unit Tests (`tests/unit/`)
- Fast, isolated tests running in milliseconds without external network or heavy model downloads.
- Examples implemented in Phase 0:
  - `test_config.py`: Default settings, port validation, secret redaction in `safe_dict()`, environment variable overrides.
  - `test_health.py`: HTTP status codes, root info schema, `/health` dependency reporting, 404 structured error responses.

### 1.2 Integration Tests (`tests/integration/`)
- Test collaboration between components:
  - Document upload route saving file to disk and triggering background processing.
  - State machine transitions (`uploaded` $\to$ `processing` $\to$ `indexed`).
  - Index save/load roundtrips to `./indexes/`.

### 1.3 End-to-End Tests (`tests/e2e/`)
- Test complete user research scenarios:
  - Upload sample academic paper $\to$ Poll status until `indexed` $\to$ Submit query $\to$ Verify answer has valid citations $\to$ Delete paper.

---

## 2. Retrieval Evaluation Methodology (Phase 5)

Academic retrieval cannot rely purely on manual checks. We evaluate retrieval components quantitatively:

### 2.1 Metrics
1. **Hit Rate @ K (`Hit@K`)**:
   $$\text{Hit@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{relevant chunk in top } K)$$
2. **Mean Reciprocal Rank (`MRR`)**:
   $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_1(q)}$$
3. **Normalized Discounted Cumulative Gain (`NDCG@K`)**:
   Measures ranking quality with graded relevance.

### 2.2 Ablation Matrix
We conduct empirical ablations comparing:
- Pure Dense Retrieval (Sentence Transformers + FAISS)
- Pure Sparse Retrieval (BM25)
- Hybrid RRF (Dense + BM25)
- Hybrid RRF + Cross-Encoder Reranker

---

## 3. Grounded Answer Evaluation (The RAG Triad)

Answer synthesis is validated against three core criteria:

1. **Context Relevance**: Are retrieved chunks relevant to the user's research question, or is there irrelevant clutter?
2. **Groundedness (Faithfulness)**: Can every factual assertion in the answer be directly inferred from the cited chunks? Hallucinated claims count as a critical failure.
3. **Answer Relevance**: Does the generated text directly answer the research prompt?
4. **Citation Precision**: Do citation identifiers (`[1]`, `[2]`) accurately map to the source chunk where the evidence originates?

---

## 4. Test Execution Commands

```bash
# Run all unit tests
pytest tests/unit -v

# Run with test coverage
pytest --cov=app --cov-report=term-missing tests/

# Run smoke verification against live server
python scripts/smoke_test.py --url http://127.0.0.1:8000
```
