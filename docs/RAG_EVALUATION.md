# Retrieval Evaluation & Quality Benchmarking

This guide outlines the evaluation methodology, dataset structure, metrics, and empirical baseline comparisons for the **Multimodal RAG Research Assistant**.

---

## 1. Information Retrieval Metrics

To evaluate evidence retrieval quality independently of LLM synthesis, we use four standard IR metrics:

### 1.1 Hit Rate @ K (`Hit@K`)
Indicates whether at least one relevant document chunk appears in the top $K$ retrieved candidates:
$$\text{Hit@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left( \text{Top-K}(q) \cap G_q \neq \emptyset \right)$$
where $G_q$ is the set of ground-truth relevant chunk IDs for query $q$.

### 1.2 Recall @ K (`Recall@K`)
Proportion of all known relevant chunks successfully retrieved in the top $K$:
$$\text{Recall@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\text{Top-K}(q) \cap G_q|}{|G_q|}$$

### 1.3 Mean Reciprocal Rank (`MRR`)
Evaluates where the first relevant chunk appears in the ranked candidate list:
$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_1(q)}$$
where $\text{rank}_1(q)$ is the 1-indexed position of the highest-ranked relevant chunk (or $0$ if not found).

### 1.4 Normalized Discounted Cumulative Gain (`NDCG@K`)
Measures graded relevance ranking quality, giving higher credit when highly relevant chunks appear near the top of the list:
$$\text{DCG@K} = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

---

## 2. Benchmark Dataset Format

The benchmark dataset (`evaluation/benchmark_dataset.py`) defines academic text chunks and query-relevance pairs:

```python
{
    "query_id": "q1",
    "query": "How do residual networks solve the degradation problem?",
    "relevant_chunks": {"resnet#chunk_001", "resnet#chunk_002", "resnet#chunk_003"},
    "graded_relevance": {
        "resnet#chunk_002": 2.0,
        "resnet#chunk_001": 1.0,
        "resnet#chunk_003": 1.0,
    },
    "expected_documents": {"doc_resnet"},
    "notes": "Targeted conceptual query matching ResNet formulation.",
}
```

---

## 3. Empirical Baseline Comparison Results

Evaluated on the synthetic academic benchmark ($N=5$ queries, $K=5$):

```
======================================================================
RETRIEVAL BENCHMARK EVALUATION (K = 5, N = 5 Queries)
======================================================================
Retrieval Configuration   | Hit@K    | Recall@K   | MRR      | NDCG@K  
----------------------------------------------------------------------
BM25 Only                 | 1.0000   | 1.0000     | 1.0000   | 1.0000  
Dense Only                | 1.0000   | 1.0000     | 1.0000   | 0.9903  
Hybrid RRF                | 1.0000   | 1.0000     | 1.0000   | 0.9903  
======================================================================
```

### Analysis of Results
- **BM25 Only**: Strong on direct terminology matches (e.g. `AdamW`, `residual mapping`, `O(n^2 * d)`).
- **Dense Only**: Robust on conceptual paraphrasing; slight variance in secondary chunk order results in NDCG@5 of 0.9903.
- **Hybrid RRF**: Combines the precision of BM25 on exact acronyms with semantic coverage from dense retrieval without requiring manual score weighting.

---

## 4. Running Evaluation Experiments

To run the evaluation script locally:
```bash
PYTHONPATH=. python3 evaluation/evaluate_retrieval.py
```

To run the unit tests verifying metric implementations:
```bash
pytest tests/unit/test_retrieval_metrics.py -v
```

---

## 5. Evaluation Limitations
1. **Synthetic Fixture Scale**: The initial benchmark dataset contains synthetic fixture papers and queries for unit test repeatability. Full-scale academic benchmarks (e.g., SciFact, BioASQ, or arXiv QA) will be added in Phase 7.
2. **LLM Evaluation Caution**: Answer evaluation in this phase relies on manual inspection and unit assertions. LLM-as-a-judge evaluators must be used with care as they do not provide objective ground truth.
