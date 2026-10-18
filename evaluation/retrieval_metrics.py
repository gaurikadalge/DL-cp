"""Information retrieval evaluation metrics: Hit@K, Recall@K, MRR, and NDCG@K."""

import math


def hit_at_k(retrieved_ids: list[str], ground_truth_ids: set[str], k: int = 5) -> float:
    """Return 1.0 if at least one relevant chunk ID appears in top-k, else 0.0."""
    if not ground_truth_ids or k <= 0:
        return 0.0
    top_k_ids = set(retrieved_ids[:k])
    return 1.0 if bool(top_k_ids.intersection(ground_truth_ids)) else 0.0


def recall_at_k(retrieved_ids: list[str], ground_truth_ids: set[str], k: int = 5) -> float:
    """Calculate Recall@K: proportion of relevant chunks retrieved in top-k."""
    if not ground_truth_ids or k <= 0:
        return 0.0
    top_k_ids = set(retrieved_ids[:k])
    hits = len(top_k_ids.intersection(ground_truth_ids))
    return float(hits) / float(len(ground_truth_ids))


def mrr(retrieved_ids: list[str], ground_truth_ids: set[str]) -> float:
    """Calculate Mean Reciprocal Rank: 1 / rank of the first relevant chunk."""
    if not ground_truth_ids:
        return 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in ground_truth_ids:
            return 1.0 / float(rank)
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    ground_truth_relevance: dict[str, float],
    k: int = 5,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K) with multi-level relevance.

    DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    NDCG@K = DCG@K / IDCG@K
    """
    if not ground_truth_relevance or k <= 0:
        return 0.0

    # 1. Compute DCG@K for retrieved ranking
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k], start=1):
        rel = ground_truth_relevance.get(cid, 0.0)
        dcg += (math.pow(2.0, rel) - 1.0) / math.log2(i + 1.0)

    # 2. Compute Ideal DCG@K (sorted by true relevance descending)
    ideal_rels = sorted(ground_truth_relevance.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels[:k], start=1):
        idcg += (math.pow(2.0, rel) - 1.0) / math.log2(i + 1.0)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg
