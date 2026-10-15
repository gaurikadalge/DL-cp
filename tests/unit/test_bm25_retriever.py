"""Unit tests for BM25Retriever."""

import pytest
from app.schemas import StableChunkMetadata
from rag_engine.bm25_retriever import BM25Retriever, tokenize


def sample_chunks():
    return [
        StableChunkMetadata(
            chunk_id="chunk_1",
            document_id="doc1",
            text="Residual connections prevent degradation in deep neural networks.",
            page_number=1,
            source_filename="doc1.pdf",
        ),
        StableChunkMetadata(
            chunk_id="chunk_2",
            document_id="doc2",
            text="Attention mechanisms allow sequence transduction without recurrence.",
            page_number=1,
            source_filename="doc2.pdf",
        ),
        StableChunkMetadata(
            chunk_id="chunk_3",
            document_id="doc3",
            text="Decoupled weight decay regularization in AdamW optimizer.",
            page_number=1,
            source_filename="doc3.pdf",
        ),
    ]


def test_tokenization():
    """Verify deterministic tokenization handles lowercase, punctuation, and terms."""
    tokens = tokenize("ResNet-50 / AdamW: 100% accuracy!")
    assert "resnet-50" in tokens
    assert "adamw" in tokens
    assert "100" in tokens
    assert "accuracy" in tokens
    assert tokenize("") == []


def test_bm25_keyword_retrieval():
    """Verify keyword retrieval returns relevant document based on terms."""
    retriever = BM25Retriever()
    retriever.add_chunks(sample_chunks())

    results = retriever.search(query="AdamW optimizer weight decay", top_k=2)
    assert len(results) > 0
    top_chunk, top_score = results[0]
    assert top_chunk.chunk_id == "chunk_3"
    assert top_score > 0.0


def test_empty_query_and_empty_corpus():
    """Verify empty queries and empty corpus return empty list safely."""
    empty_retriever = BM25Retriever()
    assert empty_retriever.search("test query") == []

    populated = BM25Retriever()
    populated.add_chunks(sample_chunks())
    assert populated.search("") == []
    assert populated.search("    ") == []


def test_document_deletion():
    """Verify delete_document removes matching chunks and rebuilds corpus."""
    retriever = BM25Retriever()
    retriever.add_chunks(sample_chunks())
    assert retriever.total_chunks == 3

    deleted_count = retriever.delete_document("doc2")
    assert deleted_count == 1
    assert retriever.total_chunks == 2

    # Query for doc2 content should now return no results
    results = retriever.search("sequence transduction recurrence")
    assert len(results) == 0


def test_bm25_persistence(tmp_path):
    """Verify BM25 corpus persistence and reconstruction."""
    retriever = BM25Retriever()
    retriever.add_chunks(sample_chunks())
    retriever.save(tmp_path)

    loaded = BM25Retriever.load(tmp_path)
    assert loaded.total_chunks == 3

    results = loaded.search("Residual connections", top_k=1)
    assert len(results) == 1
    assert results[0][0].chunk_id == "chunk_1"
