"""Unit tests for EmbeddingService."""

import numpy as np
import pytest
from rag_engine.embedding_model import EmbeddingService


class DummyEncoder:
    """Mock encoder returning deterministic vectors without network downloads."""

    def __init__(self, dimension: int = 16):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, **kwargs):
        # Return simple deterministic vector based on text length
        vectors = []
        for t in texts:
            val = float(len(t))
            vec = np.full(self.dimension, val, dtype=np.float32)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


class NonFiniteEncoder:
    """Encoder returning NaNs to test validation."""

    def get_sentence_embedding_dimension(self) -> int:
        return 16

    def encode(self, texts, **kwargs):
        return np.full((len(texts), 16), np.nan, dtype=np.float32)


def test_embedding_dimensions_and_shapes():
    """Verify vector shapes and normalization."""
    dummy = DummyEncoder(dimension=16)
    service = EmbeddingService(custom_encoder=dummy, normalize=True)

    assert service.embedding_dimension == 16
    texts = ["First academic text block.", "Second passage for analysis."]
    embeddings = service.embed_texts(texts)

    assert embeddings.shape == (2, 16)
    # Verify L2 normalization: norm of each row vector should be ~1.0
    for vec in embeddings:
        norm = np.linalg.norm(vec)
        assert pytest.approx(norm, 1e-5) == 1.0


def test_embed_query_shape():
    """Verify query embedding produces 1D normalized vector."""
    dummy = DummyEncoder(dimension=16)
    service = EmbeddingService(custom_encoder=dummy, normalize=True)

    vec = service.embed_query("What is residual learning?")
    assert vec.ndim == 1
    assert vec.shape == (16,)
    assert pytest.approx(np.linalg.norm(vec), 1e-5) == 1.0


def test_empty_and_invalid_inputs():
    """Verify empty texts or lists raise appropriate ValueErrors."""
    dummy = DummyEncoder(dimension=16)
    service = EmbeddingService(custom_encoder=dummy)

    with pytest.raises(ValueError, match="cannot be empty"):
        service.embed_texts([])

    with pytest.raises(ValueError, match="empty or whitespace"):
        service.embed_texts(["   "])

    with pytest.raises(ValueError, match="cannot be empty or whitespace"):
        service.embed_query("   ")

    with pytest.raises(TypeError):
        service.embed_query(12345)  # type: ignore


def test_non_finite_values_raise_error():
    """Verify non-finite embeddings (NaN/Inf) are caught."""
    bad_encoder = NonFiniteEncoder()
    service = EmbeddingService(custom_encoder=bad_encoder)

    with pytest.raises(ValueError, match="non-finite values"):
        service.embed_texts(["Test text"])


def test_query_instruction_prefix():
    """Verify query prefix is applied to the encoded query text."""
    recorded_texts = []

    class InspectEncoder:
        def encode(self, texts, **kwargs):
            recorded_texts.extend(texts)
            return np.ones((len(texts), 8), dtype=np.float32)

    service = EmbeddingService(
        custom_encoder=InspectEncoder(),
        query_instruction="Represent this sentence:",
    )
    service.embed_query("research question")
    assert recorded_texts[0] == "Represent this sentence: research question"
