"""Integration tests for IndexManager lifecycle, persistence, and deletion."""

import numpy as np

from app.schemas import StableChunkMetadata
from rag_engine.bm25_retriever import BM25Retriever
from rag_engine.dense_retriever import DenseRetriever
from rag_engine.embedding_model import EmbeddingService
from rag_engine.index_manager import IndexManager


class DummyLifecycleEncoder:
    def get_sentence_embedding_dimension(self):
        return 8

    def encode(self, texts, **kwargs):
        vecs = []
        for t in texts:
            v = np.full(8, float(len(t)), dtype=np.float32)
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm > 0 else v)
        return np.array(vecs, dtype=np.float32)


def make_chunks():
    return [
        StableChunkMetadata(
            chunk_id="paperA#1",
            document_id="docA",
            text="First chunk of paper A discussing transformers.",
            page_number=1,
            source_filename="paperA.pdf",
        ),
        StableChunkMetadata(
            chunk_id="paperA#2",
            document_id="docA",
            text="Second chunk of paper A discussing self attention.",
            page_number=2,
            source_filename="paperA.pdf",
        ),
        StableChunkMetadata(
            chunk_id="paperB#1",
            document_id="docB",
            text="First chunk of paper B discussing convolutional layers.",
            page_number=1,
            source_filename="paperB.pdf",
        ),
    ]


def test_index_lifecycle_add_persist_reload(tmp_path):
    """Verify add_chunks, save to disk, reload, and integrity."""
    encoder = DummyLifecycleEncoder()
    embedding_service = EmbeddingService(custom_encoder=encoder)
    dense_retriever = DenseRetriever(dimension=8)
    bm25_retriever = BM25Retriever()

    manager = IndexManager(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        embedding_service=embedding_service,
        storage_dir=tmp_path,
    )

    chunks = make_chunks()
    manager.add_chunks(chunks)

    assert manager.dense_retriever.total_chunks == 3
    assert manager.bm25_retriever.total_chunks == 3

    # Persist to disk
    manager.save(tmp_path)

    # Create new manager and load from disk
    new_dense = DenseRetriever(dimension=8)
    new_bm25 = BM25Retriever()
    new_manager = IndexManager(
        dense_retriever=new_dense,
        bm25_retriever=new_bm25,
        embedding_service=embedding_service,
        storage_dir=tmp_path,
    )
    new_manager.load(tmp_path)

    assert new_manager.dense_retriever.total_chunks == 3
    assert new_manager.bm25_retriever.total_chunks == 3


def test_document_deletion_removes_from_all_indexes(tmp_path):
    """Verify deleting docA removes all its chunks from dense and BM25 indexes."""
    encoder = DummyLifecycleEncoder()
    embedding_service = EmbeddingService(custom_encoder=encoder)
    dense_retriever = DenseRetriever(dimension=8)
    bm25_retriever = BM25Retriever()

    manager = IndexManager(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        embedding_service=embedding_service,
        storage_dir=tmp_path,
    )

    chunks = make_chunks()
    manager.add_chunks(chunks)

    # Delete docA (2 chunks)
    purged = manager.delete_document("docA")
    assert purged == 2

    assert manager.dense_retriever.total_chunks == 1
    assert manager.bm25_retriever.total_chunks == 1

    # Remaining chunk must be paperB#1
    remaining_chunk = manager.bm25_retriever.chunks[0]
    assert remaining_chunk.document_id == "docB"
