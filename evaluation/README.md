# Evaluation & Quality Benchmarking Framework

This directory houses the evaluation datasets, scoring harnesses, and benchmarking scripts for the Multimodal RAG Research Assistant.

## Planned Evaluation Dimensions (Phases 6 & 7)

Our evaluation framework is designed around two rigorous axes:

### 1. Retrieval Performance
To ensure evidence extraction from complex academic documents is robust before generation:
- **Hit Rate @ K (`Hit@K`)**: Proportion of test queries where relevant source chunks appear in the top $K$ ($K \in \{1, 3, 5, 10\}$).
- **Mean Reciprocal Rank (`MRR`)**: Evaluates the position of the first relevant chunk returned by hybrid search.
- **NDCG (`NDCG@K`)**: Assesses ranking quality when multi-level relevance labels are provided.
- **Dense vs. Keyword vs. Hybrid Comparison**: Quantitative comparison validating that Reciprocal Rank Fusion (RRF) outperforms either FAISS or BM25 alone on academic terminology and acronyms.

### 2. Answer Quality & Grounding (The RAG Triad)
- **Context Relevance**: Did the retrieval engine select only chunks necessary to answer the research question?
- **Groundedness / Faithfulness**: Is every factual claim in the generated answer strictly supported by the cited chunks (zero hallucination tolerance)?
- **Answer Relevance**: Does the response directly and comprehensively address the user's research query?
- **Citation Precision & Traceability**: Verification that page numbers and chunk text excerpts cited accurately map to the source PDF.

---

## Benchmark Datasets Specification
Academic QA datasets will be placed in `evaluation/datasets/` following this JSON format:
```json
[
  {
    "id": "qa-001",
    "document_id": "vaswani2017attention",
    "source_pdf": "attention_is_all_you_need.pdf",
    "question": "What is the computational complexity per layer for Self-Attention compared to Recurrent layers?",
    "ground_truth_answer": "Self-attention complexity per layer is O(n^2 * d), whereas recurrent layers have complexity O(n * d^2)...",
    "relevant_pages": [6],
    "expected_citations": ["Table 1"]
  }
]
```

## Running Evaluations (Future Command)
```bash
python scripts/run_evaluation.py --dataset evaluation/datasets/academic_bench_v1.json
```
