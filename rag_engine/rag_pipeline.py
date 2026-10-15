"""Central RAG pipeline orchestrating retrieval, fusion, reranking, generation, and citations."""

import logging
import time
from typing import List, Optional
import uuid

from app.schemas import (
    QueryResponse,
    QueryTimingMetadata,
    RetrievedEvidence,
)
from rag_engine.answer_generator import (
    INSUFFICIENT_EVIDENCE_MARKER,
    AnswerGenerator,
    LLMProvider,
    MockLLMProvider,
    OpenAILLMProvider,
)
from rag_engine.citations import CitationValidator
from rag_engine.context_builder import ContextBuilder
from rag_engine.hybrid_retriever import HybridRetriever
from rag_engine.reranker import RerankerService

logger = logging.getLogger("rag_engine.rag_pipeline")


class RAGPipeline:
    """Full-pipeline orchestrator for the Multimodal RAG Research Assistant."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: RerankerService,
        context_builder: ContextBuilder,
        answer_generator: AnswerGenerator,
        citation_validator: CitationValidator,
        minimum_evidence_threshold: float = 0.0,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.answer_generator = answer_generator
        self.citation_validator = citation_validator
        self.minimum_evidence_threshold = minimum_evidence_threshold

    def answer_question(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        query_id: Optional[str] = None,
    ) -> QueryResponse:
        """Process research question from end-to-end and emit grounded, verifiable response."""
        start_total = time.perf_counter()
        q_id = query_id or f"query_{uuid.uuid4().hex[:8]}"

        stripped_query = query.strip()
        if not stripped_query:
            raise ValueError("Query cannot be empty or whitespace only.")

        # ----------------------------------------------------------------------
        # Stage 1: Hybrid Retrieval (Dense + BM25 + RRF)
        # ----------------------------------------------------------------------
        start_retrieval = time.perf_counter()
        candidates: List[RetrievedEvidence] = self.hybrid_retriever.retrieve(
            query=stripped_query,
            top_k=top_k,
            document_ids=document_ids,
        )
        retrieval_ms = (time.perf_counter() - start_retrieval) * 1000.0

        # ----------------------------------------------------------------------
        # Stage 2: Optional Cross-Encoder Reranking
        # ----------------------------------------------------------------------
        start_rerank = time.perf_counter()
        reranking_ms: Optional[float] = None
        if self.reranker.enabled and candidates:
            candidates = self.reranker.rerank(
                query=stripped_query,
                candidates=candidates,
                top_k=top_k,
            )
            reranking_ms = (time.perf_counter() - start_rerank) * 1000.0

        # ----------------------------------------------------------------------
        # Stage 3: Evidence Selection & Context Construction
        # ----------------------------------------------------------------------
        context_str, citation_map = self.context_builder.build_context(candidates)

        # Check for insufficient evidence
        if not candidates or not context_str.strip():
            total_ms = (time.perf_counter() - start_total) * 1000.0
            return QueryResponse(
                query_id=q_id,
                query=stripped_query,
                answer=(
                    "Insufficient evidence: The provided research documents do not contain sufficient evidence to answer this question."
                ),
                citations=[],
                evidence=[],
                insufficient_evidence=True,
                timing=QueryTimingMetadata(
                    retrieval_ms=round(retrieval_ms, 2),
                    reranking_ms=round(reranking_ms, 2) if reranking_ms is not None else None,
                    generation_ms=0.0,
                    total_ms=round(total_ms, 2),
                ),
            )

        # ----------------------------------------------------------------------
        # Stage 4: Grounded Answer Synthesis
        # ----------------------------------------------------------------------
        start_gen = time.perf_counter()
        raw_answer = self.answer_generator.generate_answer(
            query=stripped_query,
            context_str=context_str,
        )
        generation_ms = (time.perf_counter() - start_gen) * 1000.0

        # Check if provider signaled insufficient evidence
        is_insufficient = INSUFFICIENT_EVIDENCE_MARKER in raw_answer
        if is_insufficient:
            cleaned_answer = raw_answer.replace(f"{INSUFFICIENT_EVIDENCE_MARKER}:", "").strip()
            total_ms = (time.perf_counter() - start_total) * 1000.0
            return QueryResponse(
                query_id=q_id,
                query=stripped_query,
                answer=cleaned_answer or "Insufficient evidence to answer this question.",
                citations=[],
                evidence=candidates,
                insufficient_evidence=True,
                timing=QueryTimingMetadata(
                    retrieval_ms=round(retrieval_ms, 2),
                    reranking_ms=round(reranking_ms, 2) if reranking_ms is not None else None,
                    generation_ms=round(generation_ms, 2),
                    total_ms=round(total_ms, 2),
                ),
            )

        # ----------------------------------------------------------------------
        # Stage 5: Citation Validation & Source Resolution
        # ----------------------------------------------------------------------
        citations, unresolvable = self.citation_validator.validate_citations(
            answer=raw_answer,
            citation_map=citation_map,
        )

        total_ms = (time.perf_counter() - start_total) * 1000.0

        return QueryResponse(
            query_id=q_id,
            query=stripped_query,
            answer=raw_answer,
            citations=citations,
            evidence=candidates,
            insufficient_evidence=False,
            timing=QueryTimingMetadata(
                retrieval_ms=round(retrieval_ms, 2),
                reranking_ms=round(reranking_ms, 2) if reranking_ms is not None else None,
                generation_ms=round(generation_ms, 2),
                total_ms=round(total_ms, 2),
            ),
        )


def create_pipeline(
    settings,
    custom_encoder: Optional[object] = None,
    custom_reranker: Optional[object] = None,
    custom_llm: Optional[LLMProvider] = None,
) -> RAGPipeline:
    """Factory helper creating a fully configured RAGPipeline from Settings."""
    from rag_engine.bm25_retriever import BM25Retriever
    from rag_engine.dense_retriever import DenseRetriever
    from rag_engine.embedding_model import EmbeddingService

    embedding_service = EmbeddingService(
        model_name=settings.embedding_model,
        device=settings.device,
        batch_size=settings.embedding_batch_size,
        custom_encoder=custom_encoder,
    )

    dense_retriever = DenseRetriever(dimension=embedding_service.embedding_dimension)
    bm25_retriever = BM25Retriever()

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        embedding_service=embedding_service,
        rrf_k=settings.rrf_k,
    )

    reranker = RerankerService(
        enabled=settings.reranker_enabled,
        model_name=settings.reranker_model,
        device=settings.device,
        custom_encoder=custom_reranker,
    )

    context_builder = ContextBuilder(token_budget=settings.context_budget_tokens)

    # Determine LLM Provider
    if custom_llm is not None:
        llm_provider = custom_llm
    elif settings.llm_provider == "openai" and settings.llm_api_key:
        llm_provider = OpenAILLMProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model or "gpt-4o-mini",
            base_url=settings.llm_base_url or "https://api.openai.com/v1",
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )
    else:
        llm_provider = MockLLMProvider()

    answer_generator = AnswerGenerator(provider=llm_provider)
    citation_validator = CitationValidator()

    return RAGPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        context_builder=context_builder,
        answer_generator=answer_generator,
        citation_validator=citation_validator,
        minimum_evidence_threshold=settings.minimum_evidence_threshold,
    )
