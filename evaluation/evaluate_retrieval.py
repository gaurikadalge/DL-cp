"""Comprehensive evaluation comparing BM25, Dense, Hybrid RRF, and Reranked retrieval on the benchmark dataset."""

import json
import logging
import sys
import time
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from evaluation.benchmark_dataset import BENCHMARK_QUERIES, FIXTURE_CHUNKS
from evaluation.retrieval_metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k
from rag_engine.bm25_retriever import BM25Retriever
from rag_engine.dense_retriever import DenseRetriever
from rag_engine.embedding_model import EmbeddingService
from rag_engine.hybrid_retriever import HybridRetriever
from rag_engine.reranker import RerankerService

logger = logging.getLogger("evaluation.evaluate_retrieval")


class DeterministicHashEncoder:
    """Deterministic pseudo-embedding encoder for reproducible evaluation without network downloads."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        import hashlib

        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = text.lower().split()
            for word in words:
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dimension
                vec[h] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


class DeterministicCrossEncoder:
    """Deterministic joint interaction model for scoring (query, passage) pairs."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for q, p in pairs:
            q_tokens = [w for w in q.lower().split() if len(w) > 2]
            p_text = p.lower()
            if not q_tokens:
                scores.append(0.0)
                continue
            matches = sum(1 for w in q_tokens if w in p_text)
            scores.append(float(matches) / len(q_tokens))
        return scores


def run_evaluation(
    k: int = 5,
    custom_encoder: object | None = None,
    custom_reranker: object | None = None,
    use_real_models: bool = False,
    save_results_path: str | Path | None = "./evaluation/retrieval_benchmark_results.json",
) -> dict:
    """Execute evaluation comparing BM25, Dense, Hybrid RRF, and Reranker retrieval modes."""
    # 1. Initialize services
    if use_real_models:
        logger.info("Using real neural models (all-MiniLM-L6-v2, ms-marco-MiniLM-L-6-v2)...")
        embedding_service = EmbeddingService(normalize=True)
        reranker = RerankerService(enabled=True)
    else:
        encoder = custom_encoder or DeterministicHashEncoder(dimension=128)
        embedding_service = EmbeddingService(
            custom_encoder=encoder,
            normalize=True,
        )
        reranker = custom_reranker or DeterministicCrossEncoder()

    # 2. Build Dense Index
    dim = embedding_service.embedding_dimension
    dense_retriever = DenseRetriever(dimension=dim)
    chunk_texts = [c.text for c in FIXTURE_CHUNKS]
    embeddings = embedding_service.embed_texts(chunk_texts)
    dense_retriever.add_chunks(FIXTURE_CHUNKS, embeddings)

    # 3. Build BM25 Index
    bm25_retriever = BM25Retriever()
    bm25_retriever.add_chunks(FIXTURE_CHUNKS)

    # 4. Build Hybrid Retriever
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        embedding_service=embedding_service,
        rrf_k=60,
    )

    # 5. Build Reranker
    reranker = RerankerService(
        enabled=True,
        custom_encoder=None if use_real_models else (custom_reranker or DeterministicCrossEncoder()),
    )

    configs = ["BM25 Only", "Dense Only", "Hybrid RRF", "Hybrid + Reranker"]
    metrics_template = {
        "Hit@K": 0.0,
        "Recall@K": 0.0,
        "MRR": 0.0,
        "NDCG@K": 0.0,
        "Mean_Latency_ms": 0.0,
    }

    results: dict[str, dict[str, float]] = {c: metrics_template.copy() for c in configs}
    category_results: dict[str, dict[str, dict[str, float]]] = {}

    answerable_queries = [q for q in BENCHMARK_QUERIES if len(q.get("relevant_chunks", set())) > 0]
    unanswerable_queries = [
        q for q in BENCHMARK_QUERIES if len(q.get("relevant_chunks", set())) == 0
    ]

    num_answerable = len(answerable_queries)

    for item in answerable_queries:
        query = item["query"]
        ground_truth = item["relevant_chunks"]
        graded_rel = item["graded_relevance"]
        category = item.get("category", "general")

        if category not in category_results:
            category_results[category] = {c: metrics_template.copy() for c in configs}

        # ----------------- A. BM25 Only -----------------
        t0 = time.perf_counter()
        bm25_res = bm25_retriever.search(query=query, top_k=k)
        t_bm25 = (time.perf_counter() - t0) * 1000.0
        bm25_ids = [c.chunk_id for c, _ in bm25_res]

        results["BM25 Only"]["Hit@K"] += hit_at_k(bm25_ids, ground_truth, k)
        results["BM25 Only"]["Recall@K"] += recall_at_k(bm25_ids, ground_truth, k)
        results["BM25 Only"]["MRR"] += mrr(bm25_ids, ground_truth)
        results["BM25 Only"]["NDCG@K"] += ndcg_at_k(bm25_ids, graded_rel, k)
        results["BM25 Only"]["Mean_Latency_ms"] += t_bm25

        category_results[category]["BM25 Only"]["Hit@K"] += hit_at_k(bm25_ids, ground_truth, k)
        category_results[category]["BM25 Only"]["Recall@K"] += recall_at_k(
            bm25_ids, ground_truth, k
        )
        category_results[category]["BM25 Only"]["MRR"] += mrr(bm25_ids, ground_truth)
        category_results[category]["BM25 Only"]["NDCG@K"] += ndcg_at_k(bm25_ids, graded_rel, k)

        # ----------------- B. Dense Only -----------------
        t0 = time.perf_counter()
        q_vec = embedding_service.embed_query(query)
        dense_res = dense_retriever.search(query_vector=q_vec, top_k=k)
        t_dense = (time.perf_counter() - t0) * 1000.0
        dense_ids = [c.chunk_id for c, _ in dense_res]

        results["Dense Only"]["Hit@K"] += hit_at_k(dense_ids, ground_truth, k)
        results["Dense Only"]["Recall@K"] += recall_at_k(dense_ids, ground_truth, k)
        results["Dense Only"]["MRR"] += mrr(dense_ids, ground_truth)
        results["Dense Only"]["NDCG@K"] += ndcg_at_k(dense_ids, graded_rel, k)
        results["Dense Only"]["Mean_Latency_ms"] += t_dense

        category_results[category]["Dense Only"]["Hit@K"] += hit_at_k(dense_ids, ground_truth, k)
        category_results[category]["Dense Only"]["Recall@K"] += recall_at_k(
            dense_ids, ground_truth, k
        )
        category_results[category]["Dense Only"]["MRR"] += mrr(dense_ids, ground_truth)
        category_results[category]["Dense Only"]["NDCG@K"] += ndcg_at_k(dense_ids, graded_rel, k)

        # ----------------- C. Hybrid RRF -----------------
        t0 = time.perf_counter()
        hybrid_res = hybrid_retriever.retrieve(query=query, top_k=k)
        t_hybrid = (time.perf_counter() - t0) * 1000.0
        hybrid_ids = [e.chunk.chunk_id for e in hybrid_res]

        results["Hybrid RRF"]["Hit@K"] += hit_at_k(hybrid_ids, ground_truth, k)
        results["Hybrid RRF"]["Recall@K"] += recall_at_k(hybrid_ids, ground_truth, k)
        results["Hybrid RRF"]["MRR"] += mrr(hybrid_ids, ground_truth)
        results["Hybrid RRF"]["NDCG@K"] += ndcg_at_k(hybrid_ids, graded_rel, k)
        results["Hybrid RRF"]["Mean_Latency_ms"] += t_hybrid

        category_results[category]["Hybrid RRF"]["Hit@K"] += hit_at_k(hybrid_ids, ground_truth, k)
        category_results[category]["Hybrid RRF"]["Recall@K"] += recall_at_k(
            hybrid_ids, ground_truth, k
        )
        category_results[category]["Hybrid RRF"]["MRR"] += mrr(hybrid_ids, ground_truth)
        category_results[category]["Hybrid RRF"]["NDCG@K"] += ndcg_at_k(hybrid_ids, graded_rel, k)

        # ----------------- D. Hybrid + Reranker -----------------
        t0 = time.perf_counter()
        fused_pool = hybrid_retriever.retrieve(query=query, top_k=k * 2)
        reranked_pool = reranker.rerank(query=query, candidates=fused_pool, top_k=k)
        t_rerank = (time.perf_counter() - t0) * 1000.0
        rerank_ids = [e.chunk.chunk_id for e in reranked_pool]

        results["Hybrid + Reranker"]["Hit@K"] += hit_at_k(rerank_ids, ground_truth, k)
        results["Hybrid + Reranker"]["Recall@K"] += recall_at_k(rerank_ids, ground_truth, k)
        results["Hybrid + Reranker"]["MRR"] += mrr(rerank_ids, ground_truth)
        results["Hybrid + Reranker"]["NDCG@K"] += ndcg_at_k(rerank_ids, graded_rel, k)
        results["Hybrid + Reranker"]["Mean_Latency_ms"] += t_rerank

        category_results[category]["Hybrid + Reranker"]["Hit@K"] += hit_at_k(
            rerank_ids, ground_truth, k
        )
        category_results[category]["Hybrid + Reranker"]["Recall@K"] += recall_at_k(
            rerank_ids, ground_truth, k
        )
        category_results[category]["Hybrid + Reranker"]["MRR"] += mrr(rerank_ids, ground_truth)
        category_results[category]["Hybrid + Reranker"]["NDCG@K"] += ndcg_at_k(
            rerank_ids, graded_rel, k
        )

    # Compute overall averages across answerable queries
    summary: dict[str, dict[str, float]] = {}
    for mode, metrics in results.items():
        summary[mode] = {metric: round(val / num_answerable, 4) for metric, val in metrics.items()}

    # Compute category-wise averages
    category_summary: dict[str, dict[str, dict[str, float]]] = {}
    for cat, cat_modes in category_results.items():
        cat_query_count = sum(1 for q in answerable_queries if q.get("category") == cat)
        category_summary[cat] = {}
        for mode, metrics in cat_modes.items():
            category_summary[cat][mode] = {
                metric: round(val / cat_query_count, 4) for metric, val in metrics.items()
            }

    output_payload = {
        "metadata": {
            "total_queries": len(BENCHMARK_QUERIES),
            "answerable_queries": num_answerable,
            "unanswerable_queries": len(unanswerable_queries),
            "total_fixture_chunks": len(FIXTURE_CHUNKS),
            "k": k,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluation_encoder": "DeterministicHashEncoder (dim=128)",
            "evaluation_reranker": "DeterministicCrossEncoder",
        },
        "overall_results": summary,
        "category_results": category_summary,
    }

    if save_results_path:
        out_p = Path(save_results_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(output_payload, indent=2))
        logger.info("Saved benchmark evaluation results to %s", out_p)

    return output_payload


def print_evaluation_report(payload: dict, k: int = 5):
    """Print ASCII comparison table of retrieval evaluation."""
    meta = payload["metadata"]
    summary = payload["overall_results"]
    cat_summary = payload.get("category_results", {})

    print("=" * 80)
    print(
        f"RETRIEVAL BENCHMARK EVALUATION (K = {k}, N = {meta['total_queries']} Total Queries, N_ans = {meta['answerable_queries']})"
    )
    print("=" * 80)
    header = f"{'Retrieval Configuration':<24} | {'Hit@K':<8} | {'Recall@K':<10} | {'MRR':<8} | {'NDCG@K':<8} | {'Latency':<10}"
    print(header)
    print("-" * 80)
    for mode, metrics in summary.items():
        row = (
            f"{mode:<24} | "
            f"{metrics['Hit@K']:<8.4f} | "
            f"{metrics['Recall@K']:<10.4f} | "
            f"{metrics['MRR']:<8.4f} | "
            f"{metrics['NDCG@K']:<8.4f} | "
            f"{metrics['Mean_Latency_ms']:<8.2f}ms"
        )
        print(row)
    print("=" * 80)

    print("\nPER-CATEGORY PERFORMANCE BREAKDOWN:")
    for cat, modes in cat_summary.items():
        print(f"\n--- Category: {cat.upper()} ---")
        print(
            f"{'Configuration':<24} | {'Hit@K':<8} | {'Recall@K':<10} | {'MRR':<8} | {'NDCG@K':<8}"
        )
        print("-" * 65)
        for mode, metrics in modes.items():
            print(
                f"{mode:<24} | {metrics['Hit@K']:<8.4f} | {metrics['Recall@K']:<10.4f} | {metrics['MRR']:<8.4f} | {metrics['NDCG@K']:<8.4f}"
            )
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval models on benchmark dataset.")
    parser.add_argument(
        "--use-real-models",
        action="store_true",
        help="Use real transformer models (SentenceTransformer and CrossEncoder) instead of fast deterministic fixtures.",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-K evaluation cut-off (default: 5)")
    args = parser.parse_args()

    mode_label = "Real Transformer Models" if args.use_real_models else "Fast Deterministic Fixtures"
    print(f"\n[*] Initializing Evaluation Run: {mode_label} (k={args.k})")
    payload = run_evaluation(k=args.k, use_real_models=args.use_real_models)
    print_evaluation_report(payload, k=args.k)
