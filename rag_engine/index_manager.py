"""Index lifecycle management orchestrating dense and keyword index persistence."""

import logging
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

from app.schemas import StableChunkMetadata
from rag_engine.bm25_retriever import BM25Retriever
from rag_engine.dense_retriever import DenseRetriever
from rag_engine.embedding_model import EmbeddingService

logger = logging.getLogger("rag_engine.index_manager")


class IndexManager:
    """Orchestrates synchronized additions, persistence, deletion, and rebuilds across dense and BM25 indexes."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        embedding_service: EmbeddingService,
        storage_dir: Union[str, Path] = "./indexes",
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.embedding_service = embedding_service
        self.storage_dir = Path(storage_dir)

    def add_chunks(self, chunks: List[StableChunkMetadata]) -> None:
        """Embed and synchronously index document chunks in both FAISS and BM25."""
        if not chunks:
            return

        # 1. Compute dense embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        # 2. Add to dense retriever
        self.dense_retriever.add_chunks(chunks=chunks, embeddings=embeddings)

        # 3. Add to BM25 retriever
        self.bm25_retriever.add_chunks(chunks=chunks)

        logger.info("Synchronously indexed %d chunks in dense and BM25 engines", len(chunks))

    def delete_document(self, document_id: str) -> int:
        """Remove a document's chunks from both dense and keyword retrieval indexes."""
        # For dense retriever, collect remaining chunks to rebuild cleanly
        remaining_chunks = [
            c for c in self.dense_retriever.chunk_store.values() if c.document_id != document_id
        ]
        rebuild_embeddings: Optional[np.ndarray] = None
        if remaining_chunks:
            rebuild_embeddings = self.embedding_service.embed_texts([c.text for c in remaining_chunks])

        dense_removed = self.dense_retriever.delete_document(
            document_id=document_id,
            rebuild_embeddings=rebuild_embeddings,
        )
        bm25_removed = self.bm25_retriever.delete_document(document_id=document_id)

        logger.info(
            "Purged document '%s' (Dense removed: %d, BM25 removed: %d)",
            document_id,
            dense_removed,
            bm25_removed,
        )
        return max(dense_removed, bm25_removed)

    def save(self, directory: Optional[Union[str, Path]] = None) -> None:
        """Persist both FAISS and BM25 index states to storage."""
        target_dir = Path(directory or self.storage_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        self.dense_retriever.save(target_dir)
        self.bm25_retriever.save(target_dir)
        logger.info("Successfully persisted all retrieval indexes to %s", target_dir)

    def load(self, directory: Optional[Union[str, Path]] = None) -> None:
        """Restore both FAISS and BM25 index states from storage."""
        target_dir = Path(directory or self.storage_dir)
        if not target_dir.exists():
            raise FileNotFoundError(f"Index storage directory does not exist: {target_dir}")

        self.dense_retriever = DenseRetriever.load(target_dir)
        self.bm25_retriever = BM25Retriever.load(target_dir)
        logger.info("Successfully loaded retrieval indexes from %s", target_dir)

    def rebuild_indexes(self, chunks: List[StableChunkMetadata]) -> None:
        """Reconstruct both indexes from scratch using an authoritative list of chunk records."""
        logger.info("Initiating full index rebuild with %d chunks...", len(chunks))
        dim = self.embedding_service.embedding_dimension
        self.dense_retriever = DenseRetriever(dimension=dim)
        self.bm25_retriever = BM25Retriever()

        if chunks:
            self.add_chunks(chunks)
        logger.info("Full index rebuild complete.")
