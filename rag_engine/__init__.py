"""Deep Learning & Retrieval-Augmented Generation (RAG) Engine Package.

Provides:
- EmbeddingService: Dense vector representation with Sentence Transformers.
- DenseRetriever: Semantic search utilizing FAISS IndexFlatIP.
- BM25Retriever: Lexical keyword search utilizing rank-bm25.
- HybridRetriever: Reciprocal Rank Fusion (RRF) search.
- RerankerService: Cross-Encoder candidate refinement.
- ContextBuilder: Evidence selection within token budget with injection protection.
- AnswerGenerator: Grounded answer synthesis across LLM providers.
- CitationValidator: Verifiable source resolution against actual chunks.
- IndexManager: Synchronized index additions, persistence, and deletion.
- RAGPipeline: End-to-end question answering pipeline.
"""

from rag_engine.answer_generator import (
    AnswerGenerator,
    LLMProvider,
    MockLLMProvider,
    OpenAILLMProvider,
)
from rag_engine.bm25_retriever import BM25Retriever
from rag_engine.citations import CitationValidator
from rag_engine.context_builder import ContextBuilder
from rag_engine.dense_retriever import DenseRetriever
from rag_engine.embedding_model import EmbeddingService
from rag_engine.hybrid_retriever import HybridRetriever
from rag_engine.index_manager import IndexManager
from rag_engine.rag_pipeline import RAGPipeline, create_pipeline
from rag_engine.reranker import RerankerService

__all__ = [
    "EmbeddingService",
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "RerankerService",
    "ContextBuilder",
    "AnswerGenerator",
    "LLMProvider",
    "MockLLMProvider",
    "OpenAILLMProvider",
    "CitationValidator",
    "IndexManager",
    "RAGPipeline",
    "create_pipeline",
]
