"""Hybrid retrieval combining dense vector similarity and BM25 lexical ranking via RRF."""

import logging

from app.schemas import RetrievalScoring, RetrievedEvidence, StableChunkMetadata
from rag_engine.bm25_retriever import BM25Retriever
from rag_engine.dense_retriever import DenseRetriever
from rag_engine.embedding_model import EmbeddingService

logger = logging.getLogger("rag_engine.hybrid_retriever")


class HybridRetriever:
    """Orchestrates multi-modal dense and sparse retrieval with Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        embedding_service: EmbeddingService,
        rrf_k: int = 60,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.embedding_service = embedding_service
        self.rrf_k = rrf_k

    @staticmethod
    def compute_rrf(
        dense_results: list[tuple[StableChunkMetadata, float]],
        bm25_results: list[tuple[StableChunkMetadata, float]],
        k: int = 60,
    ) -> list[RetrievedEvidence]:
        """Fuse dense and BM25 ranked lists using Reciprocal Rank Fusion (RRF).

        Formula:
            RRF_score(d) = sum_{m in {dense, bm25}} 1 / (k + rank_m(d))
        Where rank_m(d) is the 1-indexed rank position of chunk d in list m.
        """
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, StableChunkMetadata] = {}
        dense_score_map: dict[str, float] = {}
        bm25_score_map: dict[str, float] = {}

        # Process dense results (1-based rank)
        for rank, (chunk, score) in enumerate(dense_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            dense_score_map[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

        # Process BM25 results (1-based rank)
        for rank, (chunk, score) in enumerate(bm25_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            bm25_score_map[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

        # Sort descending by RRF score, breaking ties deterministically by chunk_id
        sorted_cids = sorted(
            rrf_scores.keys(),
            key=lambda cid: (rrf_scores[cid], cid),
            reverse=True,
        )

        evidence_list: list[RetrievedEvidence] = []
        for fusion_rank, cid in enumerate(sorted_cids, start=1):
            evidence = RetrievedEvidence(
                chunk=chunk_map[cid],
                scoring=RetrievalScoring(
                    dense_score=dense_score_map.get(cid),
                    bm25_score=bm25_score_map.get(cid),
                    fusion_score=round(rrf_scores[cid], 6),
                    fusion_rank=fusion_rank,
                    rerank_score=None,
                ),
            )
            evidence_list.append(evidence)

        return evidence_list

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        top_k_dense: int = 10,
        top_k_bm25: int = 10,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedEvidence]:
        """Execute hybrid search pipeline:

        1. Validate query.
        2. Retrieve dense candidates via FAISS.
        3. Retrieve keyword candidates via BM25.
        4. Fuse via RRF and return top_k candidates.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        # 1. Dense retrieval
        dense_results: list[tuple[StableChunkMetadata, float]] = []
        if self.dense_retriever.total_chunks > 0:
            query_vector = self.embedding_service.embed_query(query)
            dense_results = self.dense_retriever.search(
                query_vector=query_vector,
                top_k=top_k_dense,
                document_ids=document_ids,
            )

        # 2. BM25 retrieval
        bm25_results: list[tuple[StableChunkMetadata, float]] = []
        if self.bm25_retriever.total_chunks > 0:
            bm25_results = self.bm25_retriever.search(
                query=query,
                top_k=top_k_bm25,
                document_ids=document_ids,
            )

        # 3. Reciprocal Rank Fusion
        fused_evidence = self.compute_rrf(
            dense_results=dense_results,
            bm25_results=bm25_results,
            k=self.rrf_k,
        )

        return fused_evidence[:top_k]
