"""Embedding service using Sentence Transformers with validation and lazy loading."""

import logging

import numpy as np

logger = logging.getLogger("rag_engine.embedding_model")


class EmbeddingService:
    """Service for encoding academic documents and queries into dense vector representations.

    Features:
    - Lazy loading of SentenceTransformer model.
    - L2 normalization for cosine similarity search in inner-product indexes.
    - Strict input validation against empty, non-finite, or malformed inputs.
    - Configurable query prefix for asymmetric models (e.g., BAAI/bge-*).
    - Pluggable mock/custom encoder support for fast, deterministic unit testing without downloads.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
        query_instruction: str | None = None,
        custom_encoder: object | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.query_instruction = query_instruction
        self._custom_encoder = custom_encoder
        self._model = None
        self._dimension: int | None = None

    @property
    def model(self):
        """Lazily load the underlying embedding model."""
        if self._custom_encoder is not None:
            return self._custom_encoder

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model '%s' on %s...", self.model_name, self.device)
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                logger.error("Failed to load embedding model '%s': %s", self.model_name, e)
                raise RuntimeError(
                    f"Unable to load SentenceTransformer model '{self.model_name}'. "
                    f"Check model identifier or network connectivity: {e}"
                ) from e
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding vector dimensionality."""
        if self._dimension is not None:
            return self._dimension

        if self._custom_encoder is not None and hasattr(
            self._custom_encoder, "get_sentence_embedding_dimension"
        ):
            self._dimension = self._custom_encoder.get_sentence_embedding_dimension()
            return self._dimension

        if self._model is not None and hasattr(self._model, "get_sentence_embedding_dimension"):
            self._dimension = self._model.get_sentence_embedding_dimension()
            return self._dimension

        # Default dimension for all-MiniLM-L6-v2 is 384
        if "all-MiniLM-L6-v2" in self.model_name:
            return 384
        elif "bge-base" in self.model_name:
            return 768

        # If model is loaded, query it
        sample = self.embed_query("dimension probe")
        self._dimension = len(sample)
        return self._dimension

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Apply L2 normalization to ensure unit length for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def _validate_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Validate numeric integrity of generated embeddings."""
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Generated embeddings contain non-finite values (NaN or Inf).")
        return embeddings

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of document texts into normalized dense embeddings."""
        if not texts:
            raise ValueError("Input text list for embedding cannot be empty.")

        cleaned_texts: list[str] = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"Item at index {i} is not a string: {type(text)}")
            stripped = text.strip()
            if not stripped:
                raise ValueError(f"Text item at index {i} is empty or whitespace only.")
            cleaned_texts.append(stripped)

        # Call underlying encoder
        raw_embeddings = self.model.encode(
            cleaned_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        vectors = np.array(raw_embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = np.expand_dims(vectors, axis=0)

        self._validate_embeddings(vectors)

        if self.normalize:
            vectors = self._normalize_vectors(vectors)

        if self._dimension is None and vectors.shape[1] > 0:
            self._dimension = vectors.shape[1]

        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single search query into a normalized vector."""
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query)}")
        stripped = query.strip()
        if not stripped:
            raise ValueError("Search query cannot be empty or whitespace only.")

        # Apply query prefix/instruction if configured
        formatted_query = (
            f"{self.query_instruction} {stripped}" if self.query_instruction else stripped
        )

        vectors = self.embed_texts([formatted_query])
        return vectors[0]
