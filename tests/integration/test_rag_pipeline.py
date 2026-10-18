"""Integration tests for the full end-to-end RAGPipeline."""

from app.config import Settings
from app.schemas import StableChunkMetadata
from rag_engine.answer_generator import MockLLMProvider
from rag_engine.embedding_model import EmbeddingService
from rag_engine.rag_pipeline import create_pipeline


class DeterministicEncoder:
    """Mock encoder creating deterministic word-hashed vectors."""
    def __init__(self, dimension: int = 32):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        import numpy as np
        vecs = []
        for t in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            for word in t.lower().split():
                idx = abs(hash(word)) % self.dimension
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            vecs.append(vec / norm if norm > 0 else vec)
        return np.array(vecs, dtype=np.float32)


def test_end_to_end_pipeline():
    """Verify end-to-end query processing, citation validation, and timing metadata."""
    settings = Settings()
    encoder = DeterministicEncoder()
    mock_llm = MockLLMProvider(
        fixed_answer="Deep residual learning addresses network degradation by fitting residual mappings [1]."
    )

    pipeline = create_pipeline(
        settings=settings,
        custom_encoder=encoder,
        custom_llm=mock_llm,
    )

    # Populate index
    chunks = [
        StableChunkMetadata(
            chunk_id="res_01",
            document_id="doc_resnet",
            text="Deep residual learning addresses degradation by fitting residual mappings F(x) = H(x) - x.",
            page_number=2,
            section="3. Residual Learning",
            source_filename="resnet.pdf",
        ),
        StableChunkMetadata(
            chunk_id="trans_01",
            document_id="doc_trans",
            text="Transformer multi-head attention mechanism without recurrence.",
            page_number=3,
            section="3. Attention",
            source_filename="transformer.pdf",
        ),
    ]

    # Index chunks
    texts = [c.text for c in chunks]
    embeddings = pipeline.hybrid_retriever.embedding_service.embed_texts(texts)
    pipeline.hybrid_retriever.dense_retriever.add_chunks(chunks, embeddings)
    pipeline.hybrid_retriever.bm25_retriever.add_chunks(chunks)

    # Query pipeline
    response = pipeline.answer_question(
        query="How does residual learning address degradation?",
        top_k=2,
    )

    assert response.query == "How does residual learning address degradation?"
    assert response.insufficient_evidence is False
    assert len(response.evidence) > 0
    assert len(response.citations) == 1
    assert response.citations[0].citation_id == "[1]"
    assert response.citations[0].document_id == "doc_resnet"
    assert response.citations[0].page_number == 2
    assert response.citations[0].source_filename == "resnet.pdf"

    # Verify timing
    assert response.timing.retrieval_ms >= 0.0
    assert response.timing.generation_ms >= 0.0
    assert response.timing.total_ms >= response.timing.retrieval_ms


def test_empty_index_insufficient_evidence():
    """Verify empty index produces explicit insufficient evidence response."""
    settings = Settings()
    encoder = DeterministicEncoder()
    pipeline = create_pipeline(settings=settings, custom_encoder=encoder)

    response = pipeline.answer_question("What is self-attention?")
    assert response.insufficient_evidence is True
    assert "insufficient evidence" in response.answer.lower()
    assert len(response.citations) == 0
    assert len(response.evidence) == 0
