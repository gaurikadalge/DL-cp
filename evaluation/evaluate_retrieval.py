"""Evaluation script comparing BM25, Dense, and Hybrid RRF retrieval on the benchmark dataset."""

import logging
from typing import Dict, List, Optional
import numpy as np

from evaluation.benchmark_dataset import BENCHMARK_QUERIES, FIXTURE_CHUNKS
from evaluation.retrieval_metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k
from rag_engine.bm25_retriever import BM25Retriever
from rag_engine.dense_retriever import DenseRetriever
from rag_engine.embedding_model import EmbeddingService
from rag_engine.hybrid_retriever import HybridRetriever

logger = logging.getLogger("evaluation.evaluate_retrieval")


class DeterministicHashEncoder:
    """Deterministic pseudo-embedding encoder for reproducible test evaluation without downloading weights."""

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        vectors = []
        for text in texts:
            # Generate deterministic feature vector based on word hashes
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = text.lower().split()
            for word in words:
                h = abs(hash(word)) % self.dimension
                vec[h] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


def run_evaluation(
    k: int = 5,
    custom_encoder: Optional[object] = None,
) -> Dict[str, Dict[str, float]]:
    """Execute evaluation comparing BM25, Dense, and Hybrid retrieval modes."""
    # 1. Initialize embedding service
    encoder = custom_encoder or DeterministicHashEncoder(dimension=64)
    embedding_service = EmbeddingService(
        custom_encoder=encoder,
        normalize=True,
    )

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

    results: Dict[str, Dict[str, float]] = {
        "BM25 Only": {"Hit@K": 0.0, "Recall@K": 0.0, "MRR": 0.0, "NDCG@K": 0.0},
        "Dense Only": {"Hit@K": 0.0, "Recall@K": 0.0, "MRR": 0.0, "NDCG@K": 0.0},
        "Hybrid RRF": {"Hit@K": 0.0, "Recall@K": 0.0, "MRR": 0.0, "NDCG@K": 0.0},
    }

    num_queries = len(BENCHMARK_QUERIES)

    for item in BENCHMARK_QUERIES:
        query = item["query"]
        ground_truth = item["relevant_chunks"]
        graded_rel = item["graded_relevance"]

        # Run BM25
        bm25_res = bm25_retriever.search(query=query, top_k=k)
        bm25_ids = [c.chunk_id for c, _ in bm25_res]
        results["BM25 Only"]["Hit@K"] += hit_at_k(bm25_ids, ground_truth, k)
        results["BM25 Only"]["Recall@K"] += recall_at_k(bm25_ids, ground_truth, k)
        results["BM25 Only"]["MRR"] += mrr(bm25_ids, ground_truth)
        results["BM25 Only"]["NDCG@K"] += ndcg_at_k(bm25_ids, graded_rel, k)

        # Run Dense
        q_vec = embedding_service.embed_query(query)
        dense_res = dense_retriever.search(query_vector=q_vec, top_k=k)
        dense_ids = [c.chunk_id for c, _ in dense_res]
        results["Dense Only"]["Hit@K"] += hit_at_k(dense_ids, ground_truth, k)
        results["Dense Only"]["Recall@K"] += recall_at_k(dense_ids, ground_truth, k)
        results["Dense Only"]["MRR"] += mrr(dense_ids, ground_truth)
        results["Dense Only"]["NDCG@K"] += ndcg_at_k(dense_ids, graded_rel, k)

        # Run Hybrid RRF
        hybrid_res = hybrid_retriever.retrieve(query=query, top_k=k)
        hybrid_ids = [e.chunk.chunk_id for e in hybrid_res]
        results["Hybrid RRF"]["Hit@K"] += hit_at_k(hybrid_ids, ground_truth, k)
        results["Hybrid RRF"]["Recall@K"] += recall_at_k(hybrid_ids, ground_truth, k)
        results["Hybrid RRF"]["MRR"] += mrr(hybrid_ids, ground_truth)
        results["Hybrid RRF"]["NDCG@K"] += ndcg_at_k(hybrid_ids, graded_rel, k)

    # Average metrics over queries
    summary: Dict[str, Dict[str, float]] = {}
    for mode, metrics in results.items():
        summary[mode] = {
            metric: round(val / num_queries, 4) for metric, val in metrics.items()
        }

    return summary


def print_evaluation_report(summary: Dict[str, Dict[str, float]], k: int = 5):
    """Print ASCII comparison table of retrieval evaluation."""
    print("=" * 70)
    print(f"RETRIEVAL BENCHMARK EVALUATION (K = {k}, N = {len(BENCHMARK_QUERIES)} Queries)")
    print("=" * 70)
    header = f"{'Retrieval Configuration':<25} | {'Hit@K':<8} | {'Recall@K':<10} | {'MRR':<8} | {'NDCG@K':<8}"
    print(header)
    print("-" * 70)
    for mode, metrics in summary.items():
        row = (
            f"{mode:<25} | "
            f"{metrics['Hit@K']:<8.4f} | "
            f"{metrics['Recall@K']:<10.4f} | "
            f"{metrics['MRR']:<8.4f} | "
            f"{metrics['NDCG@K']:<8.4f}"
        )
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    summary = run_evaluation(k=5)
    print_evaluation_report(summary, k=5)
