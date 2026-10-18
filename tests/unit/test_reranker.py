"""Unit tests for RerankerService."""

from app.schemas import RetrievedEvidence, RetrievalScoring, StableChunkMetadata
from rag_engine.reranker import RerankerService


class MockCrossEncoder:
    """Mock cross-encoder returning deterministic scores."""

    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        return self.scores[: len(pairs)]


def make_candidate(cid: str, text: str) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk=StableChunkMetadata(
            chunk_id=cid,
            document_id="doc1",
            text=text,
            page_number=1,
            source_filename="test.pdf",
        ),
        scoring=RetrievalScoring(fusion_score=0.03, fusion_rank=1),
    )


def test_reranker_disabled_mode():
    """Verify disabled reranker leaves candidates in their original order."""
    service = RerankerService(enabled=False)
    c1 = make_candidate("c1", "first passage")
    c2 = make_candidate("c2", "second passage")

    res = service.rerank(query="test", candidates=[c1, c2])
    assert res == [c1, c2]
    assert res[0].scoring.rerank_score is None


def test_reranker_rescoring():
    """Verify enabled reranker updates rerank_score and re-sorts descending."""
    # Scores: c1 gets 0.1, c2 gets 0.9 -> c2 should become rank 1
    mock_encoder = MockCrossEncoder(scores=[0.1, 0.9])
    service = RerankerService(enabled=True, custom_encoder=mock_encoder)

    c1 = make_candidate("c1", "first passage")
    c2 = make_candidate("c2", "second passage")

    res = service.rerank(query="test", candidates=[c1, c2])
    assert len(res) == 2
    assert res[0].chunk.chunk_id == "c2"
    assert res[0].scoring.rerank_score == 0.9
    assert res[1].chunk.chunk_id == "c1"
    assert res[1].scoring.rerank_score == 0.1


def test_reranker_failure_fallback():
    """Verify encoder exception falls back gracefully to RRF candidates."""
    class FailingEncoder:
        def predict(self, pairs):
            raise RuntimeError("CUDA out of memory simulation")

    service = RerankerService(enabled=True, custom_encoder=FailingEncoder())
    c1 = make_candidate("c1", "first passage")
    c2 = make_candidate("c2", "second passage")

    # Should not raise exception; should preserve input list
    res = service.rerank(query="test", candidates=[c1, c2])
    assert res == [c1, c2]
