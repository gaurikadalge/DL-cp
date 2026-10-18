"""Unit tests for FAISS DenseRetriever."""

import numpy as np
import pytest
from app.schemas import StableChunkMetadata
from rag_engine.dense_retriever import DenseRetriever


def sample_chunks():
    return [
        StableChunkMetadata(
            chunk_id="doc1#0",
            document_id="doc1",
            text="Residual learning for deep networks.",
            page_number=1,
            source_filename="doc1.pdf",
        ),
        StableChunkMetadata(
            chunk_id="doc1#1",
            document_id="doc1",
            text="Identity shortcut mapping formulation.",
            page_number=2,
            source_filename="doc1.pdf",
        ),
        StableChunkMetadata(
            chunk_id="doc2#0",
            document_id="doc2",
            text="Transformer multi-head attention mechanism.",
            page_number=1,
            source_filename="doc2.pdf",
        ),
    ]


def test_dense_retrieval_and_mapping():
    """Verify FAISS exact indexing, top-k retrieval, and score accuracy."""
    dim = 4
    retriever = DenseRetriever(dimension=dim)
    chunks = sample_chunks()

    # Create 3 orthogonal unit vectors
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)

    retriever.add_chunks(chunks, embeddings)
    assert retriever.total_chunks == 3

    # Query matching vector 1 exactly
    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = retriever.search(query_vec, top_k=2)

    assert len(results) == 2
    top_chunk, top_score = results[0]
    assert top_chunk.chunk_id == "doc1#0"
    assert pytest.approx(top_score, 1e-4) == 1.0


def test_empty_index_behavior():
    """Verify empty dense index returns empty list without raising exception."""
    retriever = DenseRetriever(dimension=4)
    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = retriever.search(query_vec, top_k=5)
    assert results == []


def test_dimension_mismatch_raises():
    """Verify search raises error on dimension mismatch."""
    retriever = DenseRetriever(dimension=4)
    bad_query = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # dim 3 instead of 4
    with pytest.raises(ValueError, match="does not match index"):
        retriever.search(bad_query)


def test_document_filtering():
    """Verify document_ids filtering restricts results to allowed docs."""
    retriever = DenseRetriever(dimension=2)
    chunks = sample_chunks()[:2]
    # doc1#0, doc1#1
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    retriever.add_chunks(chunks, embeddings)

    # Filter to doc2 (which does not exist in index)
    results = retriever.search(np.array([1.0, 0.0]), top_k=5, document_ids=["doc2"])
    assert results == []

    # Filter to doc1 (both sample chunks belong to doc1)
    results = retriever.search(np.array([1.0, 0.0]), top_k=5, document_ids=["doc1"])
    assert len(results) == 2
    assert results[0][0].document_id == "doc1"


def test_persistence_and_reload(tmp_path):
    """Verify saving FAISS index to disk and reloading restores vectors and metadata."""
    retriever = DenseRetriever(dimension=2)
    chunks = sample_chunks()[:2]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    retriever.add_chunks(chunks, embeddings)

    retriever.save(tmp_path)

    loaded = DenseRetriever.load(tmp_path)
    assert loaded.total_chunks == 2
    assert loaded.dimension == 2

    res = loaded.search(np.array([1.0, 0.0]), top_k=1)
    assert res[0][0].chunk_id == "doc1#0"
