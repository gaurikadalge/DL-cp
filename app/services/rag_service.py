"""Singleton RAG Service coordinating IndexManager and RAGPipeline."""

import logging
from pathlib import Path

from app.config import Settings, get_settings
from app.schemas import QueryResponse, StableChunkMetadata
from rag_engine.answer_generator import LLMProvider
from rag_engine.index_manager import IndexManager
from rag_engine.rag_pipeline import RAGPipeline, create_pipeline

logger = logging.getLogger("app.services.rag_service")


class DeterministicHashEncoder:
    """Deterministic fallback encoder when sentence-transformers is not installed in the environment."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, **kwargs):
        import hashlib

        import numpy as np

        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = text.lower().split()
            for w in words:
                idx = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % self.dimension
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


class RAGService:
    """Manages the full lifecycle of the RAG pipeline and retrieval indexes."""

    def __init__(
        self,
        settings: Settings | None = None,
        custom_encoder: object | None = None,
        custom_reranker: object | None = None,
        custom_llm: LLMProvider | None = None,
    ):
        self.settings = settings or get_settings()
        self.index_path = Path(self.settings.index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        encoder = custom_encoder
        if encoder is None:
            try:
                import sentence_transformers  # noqa: F401
            except ImportError:
                logger.info(
                    "sentence_transformers not installed; activating DeterministicHashEncoder (dim=384)."
                )
                encoder = DeterministicHashEncoder(dimension=384)

        # Create configured pipeline
        self.pipeline: RAGPipeline = create_pipeline(
            settings=self.settings,
            custom_encoder=encoder,
            custom_reranker=custom_reranker,
            custom_llm=custom_llm,
        )

        # IndexManager shares the dense, BM25, and embedding components
        self.index_manager = IndexManager(
            dense_retriever=self.pipeline.hybrid_retriever.dense_retriever,
            bm25_retriever=self.pipeline.hybrid_retriever.bm25_retriever,
            embedding_service=self.pipeline.hybrid_retriever.embedding_service,
            storage_dir=self.index_path,
        )

        # Attempt to load persisted indexes from disk if present
        self._load_persisted_indexes()
        self._sync_pipeline_retrievers()

    def _sync_pipeline_retrievers(self) -> None:
        """Keep hybrid retriever in sync with index manager retriever instances."""
        self.pipeline.hybrid_retriever.dense_retriever = self.index_manager.dense_retriever
        self.pipeline.hybrid_retriever.bm25_retriever = self.index_manager.bm25_retriever

    def _load_persisted_indexes(self) -> None:
        """Load FAISS and BM25 indexes from storage if artifacts exist."""
        dense_faiss_file = self.index_path / "dense_index.faiss"
        bm25_meta_file = self.index_path / "bm25_metadata.json"

        if dense_faiss_file.exists() and bm25_meta_file.exists():
            try:
                self.index_manager.load(self.index_path)
                self._sync_pipeline_retrievers()
                logger.info(
                    "Successfully restored indexes from %s (Dense chunks: %d, BM25 chunks: %d)",
                    self.index_path,
                    self.index_manager.dense_retriever.total_chunks,
                    self.index_manager.bm25_retriever.total_chunks,
                )
            except Exception as e:
                logger.warning("Failed to restore existing indexes from %s: %s", self.index_path, e)
        else:
            logger.info(
                "No pre-existing index files found at %s. Initialized fresh in-memory indexes.",
                self.index_path,
            )

    def index_chunks(self, chunks: list[StableChunkMetadata]) -> int:
        """Add chunks to retrieval indexes and persist updated state to disk."""
        if not chunks:
            return 0

        self.index_manager.add_chunks(chunks)
        self._sync_pipeline_retrievers()
        self.save_indexes()
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        """Purge all chunks associated with document_id and persist updated state."""
        removed = self.index_manager.delete_document(document_id)
        self._sync_pipeline_retrievers()
        if removed > 0:
            self.save_indexes()
        return removed

    def save_indexes(self) -> None:
        """Persist current index state to disk."""
        try:
            self.index_manager.save(self.index_path)
            logger.info("Persisted updated indexes to %s", self.index_path)
        except Exception as e:
            logger.error("Failed to persist indexes to %s: %s", self.index_path, e)

    def query(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
        query_id: str | None = None,
    ) -> QueryResponse:
        """Execute question answering over indexed research literature."""
        return self.pipeline.answer_question(
            query=query,
            document_ids=document_ids,
            top_k=top_k,
            query_id=query_id,
        )

    @property
    def total_indexed_chunks(self) -> int:
        return self.index_manager.dense_retriever.total_chunks

    @property
    def is_ready(self) -> bool:
        return True


# Global singleton instance
_RAG_SERVICE: RAGService | None = None


def get_rag_service() -> RAGService:
    """Return the global RAGService instance."""
    global _RAG_SERVICE
    if _RAG_SERVICE is None:
        _RAG_SERVICE = RAGService()
    return _RAG_SERVICE


def set_rag_service(service: RAGService | None) -> None:
    """Explicitly set or reset the singleton instance (primarily for testing)."""
    global _RAG_SERVICE
    _RAG_SERVICE = service
