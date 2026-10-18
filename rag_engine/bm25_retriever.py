"""Lexical keyword retrieval using rank-bm25 and deterministic tokenization."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from rank_bm25 import BM25Okapi

from app.schemas import StableChunkMetadata

logger = logging.getLogger("rag_engine.bm25_retriever")

# Deterministic alphanumeric word tokenizer for academic text
WORD_REGEX = re.compile(r"\b[a-zA-Z0-9_\-\./]+\b")


def tokenize(text: str) -> List[str]:
    """Tokenize academic text deterministically, preserving terms, abbreviations, and numbers."""
    if not text:
        return []
    return [token.lower() for token in WORD_REGEX.findall(text)]


class BM25Retriever:
    """Keyword retriever utilizing Okapi BM25 for domain terms and acronym search."""

    def __init__(self):
        self.chunks: List[StableChunkMetadata] = []
        self.corpus_tokens: List[List[str]] = []
        self.chunk_store: Dict[str, StableChunkMetadata] = {}
        self._bm25: Optional[BM25Okapi] = None

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def add_chunks(self, chunks: List[StableChunkMetadata]) -> None:
        """Add new chunks and update BM25 corpus."""
        if not chunks:
            return

        for chunk in chunks:
            # If chunk already exists, update store
            if chunk.chunk_id in self.chunk_store:
                logger.debug("Updating existing chunk '%s' in BM25 index", chunk.chunk_id)
                idx = next(i for i, c in enumerate(self.chunks) if c.chunk_id == chunk.chunk_id)
                self.chunks[idx] = chunk
                self.corpus_tokens[idx] = tokenize(chunk.text)
                self.chunk_store[chunk.chunk_id] = chunk
            else:
                self.chunks.append(chunk)
                self.corpus_tokens.append(tokenize(chunk.text))
                self.chunk_store[chunk.chunk_id] = chunk

        # Re-initialize BM25Okapi with updated corpus
        if self.corpus_tokens:
            self._bm25 = BM25Okapi(self.corpus_tokens)

        logger.info("BM25 index updated with %d chunks (total: %d)", len(chunks), len(self.chunks))

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_ids: Optional[List[str]] = None,
    ) -> List[Tuple[StableChunkMetadata, float]]:
        """Search corpus for matching query keywords.

        Returns:
            List of (StableChunkMetadata, bm25_score) sorted descending by relevance.
        """
        if not self._bm25 or not self.chunks:
            logger.debug("BM25 index is empty; returning 0 results.")
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Pair scores with chunks
        scored_pairs: List[Tuple[StableChunkMetadata, float]] = []
        doc_filter_set = set(document_ids) if document_ids else None

        for chunk, score in zip(self.chunks, scores):
            if score <= 0.0:
                continue
            if doc_filter_set and chunk.document_id not in doc_filter_set:
                continue
            scored_pairs.append((chunk, float(score)))

        # Sort descending by BM25 score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]

    def delete_document(self, document_id: str) -> int:
        """Purge all chunks belonging to document_id and rebuild BM25 model."""
        initial_count = len(self.chunks)
        new_chunks = [c for c in self.chunks if c.document_id != document_id]
        removed_count = initial_count - len(new_chunks)

        if removed_count == 0:
            return 0

        self.chunks = new_chunks
        self.corpus_tokens = [tokenize(c.text) for c in self.chunks]
        self.chunk_store = {c.chunk_id: c for c in self.chunks}

        if self.corpus_tokens:
            self._bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self._bm25 = None

        logger.info("Deleted %d BM25 chunks for doc '%s'. Remaining: %d", removed_count, document_id, len(self.chunks))
        return removed_count

    def save(self, directory: Union[str, Path]) -> None:
        """Persist BM25 chunks and metadata to disk for deterministic recovery."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / "bm25_metadata.json"

        payload = {
            "chunks": [chunk.model_dump(mode="json") for chunk in self.chunks]
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("Persisted BM25 corpus (%d chunks) to %s", len(self.chunks), file_path)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "BM25Retriever":
        """Reconstruct BM25 index from persisted chunk records."""
        target_dir = Path(directory)
        file_path = target_dir / "bm25_metadata.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Missing BM25 metadata file in {target_dir}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = [StableChunkMetadata(**c) for c in data["chunks"]]
        retriever = cls()
        retriever.add_chunks(chunks)
        logger.info("Rebuilt BM25 index from disk (%d chunks)", len(chunks))
        return retriever
