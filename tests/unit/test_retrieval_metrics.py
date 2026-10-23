"""Unit tests for information retrieval evaluation metrics."""

import pytest

from evaluation.retrieval_metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k


def test_hit_at_k():
    ground_truth = {"c1", "c2"}
    assert hit_at_k(["c3", "c1", "c4"], ground_truth, k=2) == 1.0
    assert hit_at_k(["c3", "c4", "c1"], ground_truth, k=2) == 0.0
    assert hit_at_k([], ground_truth, k=5) == 0.0
    assert hit_at_k(["c1"], set(), k=5) == 0.0


def test_recall_at_k():
    ground_truth = {"c1", "c2", "c3"}
    # Retrieved 2 of 3 in top 3
    assert (
        pytest.approx(recall_at_k(["c1", "c4", "c2", "c5"], ground_truth, k=3), 1e-4) == 2.0 / 3.0
    )
    # Retrieved 0 of 3
    assert recall_at_k(["c4", "c5"], ground_truth, k=2) == 0.0


def test_mrr():
    ground_truth = {"target"}
    # First item: rank 1 -> 1/1 = 1.0
    assert mrr(["target", "c2", "c3"], ground_truth) == 1.0
    # Second item: rank 2 -> 1/2 = 0.5
    assert mrr(["c1", "target", "c3"], ground_truth) == 0.5
    # Third item: rank 3 -> 1/3 = ~0.3333
    assert pytest.approx(mrr(["c1", "c2", "target"], ground_truth), 1e-4) == 1.0 / 3.0
    # Not found -> 0.0
    assert mrr(["c1", "c2"], ground_truth) == 0.0


def test_ndcg_at_k():
    graded = {"c1": 2.0, "c2": 1.0}
    # Ideal order: c1, c2 -> NDCG = 1.0
    assert pytest.approx(ndcg_at_k(["c1", "c2"], graded, k=2), 1e-4) == 1.0
    # Sub-optimal order: c2, c1 -> NDCG < 1.0
    sub_ndcg = ndcg_at_k(["c2", "c1"], graded, k=2)
    assert 0.0 < sub_ndcg < 1.0
    # Completely irrelevant -> 0.0
    assert ndcg_at_k(["c99"], graded, k=2) == 0.0
