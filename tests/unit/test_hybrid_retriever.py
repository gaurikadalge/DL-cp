"""Unit tests for HybridRetriever and Reciprocal Rank Fusion."""

import pytest

from app.schemas import StableChunkMetadata
from rag_engine.hybrid_retriever import HybridRetriever


def make_chunk(cid: str) -> StableChunkMetadata:
    return StableChunkMetadata(
        chunk_id=cid,
        document_id="doc_" + cid.split("#")[0],
        text=f"Text content for {cid}",
        page_number=1,
        source_filename="test.pdf",
    )


def test_compute_rrf_scoring_and_ranks():
    """Verify RRF mathematical formulation: sum 1 / (k + rank)."""
    c1 = make_chunk("c1")
    c2 = make_chunk("c2")
    c3 = make_chunk("c3")

    # Dense ranks: c1 (rank 1), c2 (rank 2)
    dense_results = [(c1, 0.95), (c2, 0.85)]
    # BM25 ranks: c2 (rank 1), c3 (rank 2)
    bm25_results = [(c2, 12.0), (c3, 8.0)]

    k = 60
    evidence = HybridRetriever.compute_rrf(dense_results, bm25_results, k=k)

    # c2 appears in both: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = ~0.03252
    # c1 appears only in dense: 1/(60+1) = 1/61 = ~0.01639
    # c3 appears only in bm25: 1/(60+2) = 1/62 = ~0.01613
    assert len(evidence) == 3
    assert evidence[0].chunk.chunk_id == "c2"
    assert evidence[0].scoring.fusion_rank == 1
    assert pytest.approx(evidence[0].scoring.fusion_score, 1e-4) == (1 / 62 + 1 / 61)

    assert evidence[1].chunk.chunk_id == "c1"
    assert evidence[1].scoring.fusion_rank == 2

    assert evidence[2].chunk.chunk_id == "c3"
    assert evidence[2].scoring.fusion_rank == 3


def test_rrf_deduplication():
    """Verify chunks present in both dense and BM25 receive a single deduplicated entry."""
    c1 = make_chunk("c1")
    dense_results = [(c1, 0.9)]
    bm25_results = [(c1, 5.0)]

    evidence = HybridRetriever.compute_rrf(dense_results, bm25_results, k=60)
    assert len(evidence) == 1
    assert evidence[0].chunk.chunk_id == "c1"
    assert evidence[0].scoring.dense_score == 0.9
    assert evidence[0].scoring.bm25_score == 5.0


def test_rrf_empty_inputs():
    """Verify empty dense and BM25 results yield empty list."""
    assert HybridRetriever.compute_rrf([], [], k=60) == []
