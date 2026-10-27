"""Cross-encoder reranking service for candidate refinement."""

import logging

from app.schemas import RetrievedEvidence

logger = logging.getLogger("rag_engine.reranker")


class RerankerService:
    """Optional Cross-Encoder reranker.

    Rescores (query, passage) pairs using a transformer cross-encoder.
    Supports graceful fallback to RRF candidates if disabled or if model fails.
    """

    def __init__(
        self,
        enabled: bool = False,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        custom_encoder: object | None = None,
    ):
        self.enabled = enabled
        self.model_name = model_name
        self.device = device
        self._custom_encoder = custom_encoder
        self._model = None

    @property
    def model(self):
        """Lazily load cross-encoder model."""
        if self._custom_encoder is not None:
            return self._custom_encoder

        if self._model is None and self.enabled:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(
                    "Loading CrossEncoder model '%s' on %s...", self.model_name, self.device
                )
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as e:
                logger.error("Failed to load CrossEncoder model '%s': %s", self.model_name, e)
                raise RuntimeError(
                    f"Unable to load CrossEncoder model '{self.model_name}': {e}"
                ) from e
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedEvidence],
        top_k: int | None = None,
    ) -> list[RetrievedEvidence]:
        """Rerank candidate evidence chunks.

        If disabled or on model error, returns candidates in their original RRF order.
        """
        if not candidates:
            return []

        if not self.enabled:
            logger.debug("Reranker is disabled; preserving RRF rank order.")
            return candidates[:top_k] if top_k else candidates

        try:
            # Prepare query-passage pairs
            pairs = [(query, cand.chunk.text) for cand in candidates]
            scores = self.model.predict(pairs)

            # Assign reranker scores
            scored_candidates: list[RetrievedEvidence] = []
            for cand, score in zip(candidates, scores, strict=False):
                cand.scoring.rerank_score = float(score)
                scored_candidates.append(cand)

            # Sort descending by rerank_score
            scored_candidates.sort(
                key=lambda c: c.scoring.rerank_score or -float("inf"), reverse=True
            )

            logger.info(
                "Successfully reranked %d candidates using %s", len(candidates), self.model_name
            )
            return scored_candidates[:top_k] if top_k else scored_candidates

        except Exception as e:
            logger.warning(
                "Reranker failed with error '%s'. Falling back to RRF candidates.",
                e,
                exc_info=True,
            )
            return candidates[:top_k] if top_k else candidates
