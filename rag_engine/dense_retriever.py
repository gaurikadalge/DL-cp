"""Dense vector retrieval engine using FAISS and normalized semantic embeddings."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import faiss
import numpy as np

from app.schemas import StableChunkMetadata

logger = logging.getLogger("rag_engine.dense_retriever")


class DenseRetriever:
    """FAISS-based dense semantic retriever.

    Maintains an exact-search IndexFlatIP for cosine similarity on L2-normalized vectors.
    Preserves a strict 1-to-1 mapping between FAISS integer row indices and StableChunkMetadata.
    """

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError(f"Vector dimension must be positive, got {dimension}")
        self.dimension = dimension
        self.index: faiss.Index = faiss.IndexFlatIP(dimension)
        # Internal sequential mapping: position (int) -> chunk_id (str)
        self.id_map: List[str] = []
        # Chunk registry: chunk_id (str) -> StableChunkMetadata
        self.chunk_store: Dict[str, StableChunkMetadata] = {}

    @property
    def total_chunks(self) -> int:
        """Return the number of vectors stored in the index."""
        return self.index.ntotal

    def add_chunks(self, chunks: List[StableChunkMetadata], embeddings: np.ndarray) -> None:
        """Add new chunks and their corresponding dense vectors to the FAISS index."""
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match embeddings count ({len(embeddings)})"
            )

        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding matrix shape {vectors.shape} does not match retriever dimension ({self.dimension})"
            )

        # Check for duplicate chunk_ids already in store
        for chunk in chunks:
            if chunk.chunk_id in self.chunk_store:
                logger.warning("Chunk ID '%s' already indexed. Updating metadata.", chunk.chunk_id)

        # Add to FAISS index
        start_idx = self.index.ntotal
        self.index.add(vectors)

        # Record sequential mapping and chunk data
        for i, chunk in enumerate(chunks):
            self.id_map.append(chunk.chunk_id)
            self.chunk_store[chunk.chunk_id] = chunk

        logger.info("Added %d chunks to FAISS dense index (total: %d)", len(chunks), self.index.ntotal)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        document_ids: Optional[List[str]] = None,
    ) -> List[Tuple[StableChunkMetadata, float]]:
        """Perform top-k similarity search against dense index with optional document filtering.

        Returns:
            List of (StableChunkMetadata, similarity_score) sorted descending by similarity.
        """
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query vector dimension ({query_vector.shape[1]}) does not match index ({self.dimension})"
            )

        if self.index.ntotal == 0:
            logger.debug("Dense index is empty; returning 0 results.")
            return []

        # If document filter applied, search larger candidate pool to ensure top_k filtered matches
        search_k = min(self.index.ntotal, max(top_k * 4, 50)) if document_ids else min(top_k, self.index.ntotal)

        distances, indices = self.index.search(query_vector, search_k)

        results: List[Tuple[StableChunkMetadata, float]] = []
        doc_filter_set = set(document_ids) if document_ids else None

        for score, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.id_map):
                continue
            chunk_id = self.id_map[idx]
            chunk = self.chunk_store.get(chunk_id)
            if not chunk:
                continue

            if doc_filter_set and chunk.document_id not in doc_filter_set:
                continue

            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break

        return results

    def delete_document(self, document_id: str, rebuild_embeddings: Optional[np.ndarray] = None) -> int:
        """Remove all chunks associated with a document_id and rebuild index integrity.

        Returns:
            Number of chunks removed.
        """
        chunks_to_keep: List[StableChunkMetadata] = []
        ids_to_keep: List[str] = []
        removed_count = 0

        for chunk_id in self.id_map:
            chunk = self.chunk_store.get(chunk_id)
            if chunk and chunk.document_id == document_id:
                removed_count += 1
            elif chunk:
                chunks_to_keep.append(chunk)
                ids_to_keep.append(chunk_id)

        if removed_count == 0:
            return 0

        # Purge deleted chunks from store
        for chunk_id in list(self.chunk_store.keys()):
            if self.chunk_store[chunk_id].document_id == document_id:
                del self.chunk_store[chunk_id]

        # Rebuild FAISS index
        new_index = faiss.IndexFlatIP(self.dimension)
        if chunks_to_keep and rebuild_embeddings is not None:
            if len(rebuild_embeddings) != len(chunks_to_keep):
                raise ValueError("Rebuild embeddings length does not match remaining chunks length")
            new_index.add(np.ascontiguousarray(rebuild_embeddings, dtype=np.float32))

        self.index = new_index
        self.id_map = ids_to_keep

        logger.info("Deleted %d chunks for doc '%s'. Remaining total: %d", removed_count, document_id, self.index.ntotal)
        return removed_count

    def save(self, directory: Union[str, Path]) -> None:
        """Persist FAISS binary index and metadata mapping to disk."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        index_file = target_dir / "dense_index.faiss"
        meta_file = target_dir / "dense_metadata.json"

        faiss.write_index(self.index, str(index_file))

        metadata_payload = {
            "dimension": self.dimension,
            "id_map": self.id_map,
            "chunks": [chunk.model_dump(mode="json") for chunk in self.chunk_store.values()],
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)

        logger.info("Persisted dense index (%d vectors) to %s", self.index.ntotal, target_dir)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "DenseRetriever":
        """Load FAISS index and metadata mapping from disk."""
        target_dir = Path(directory)
        index_file = target_dir / "dense_index.faiss"
        meta_file = target_dir / "dense_metadata.json"

        if not index_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"Missing dense index files in {target_dir}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        dimension = meta["dimension"]
        retriever = cls(dimension=dimension)
        retriever.index = faiss.read_index(str(index_file))
        retriever.id_map = meta["id_map"]
        retriever.chunk_store = {
            c["chunk_id"]: StableChunkMetadata(**c) for c in meta["chunks"]
        }

        if retriever.index.ntotal != len(retriever.id_map):
            logger.warning(
                "FAISS ntotal (%d) differs from id_map length (%d)",
                retriever.index.ntotal,
                len(retriever.id_map),
            )

        logger.info("Loaded dense index (%d vectors) from %s", retriever.index.ntotal, target_dir)
        return retriever
