"""Shared Pydantic data schemas and API models.

Defines schemas for health checks, application metadata, unified error formats,
document ingestion lifecycle, chunk metadata, retrieval evidence, and query responses.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# General API & Health Schemas
# ==============================================================================

class AppInfo(BaseModel):
    """Metadata regarding the running API instance."""
    app_name: str
    version: str
    environment: str
    docs_url: str
    health_url: str


class DependencyStatus(str, Enum):
    """Granular status indicators for external or deferred subsystems."""
    CONFIGURED = "configured"
    NOT_CONFIGURED = "not_configured"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_LOADED_SCAFFOLD = "not_loaded_scaffold"
    DISABLED = "disabled"


class LiteralStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


class HealthCheckResponse(BaseModel):
    """System health check response distinguishing application availability from external dependencies."""
    status: LiteralStatus = Field(default=LiteralStatus.OK, description="Overall service status")
    app_name: str
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dependencies: Dict[str, DependencyStatus] = Field(
        default_factory=dict,
        description="Explicit health/configuration state of external components",
    )


class ErrorDetail(BaseModel):
    """Standardized error response preventing leakage of stack traces or internal paths."""
    error_code: str = Field(description="Machine-readable error identifier")
    message: str = Field(description="Human-readable explanation of the error")
    details: Optional[Any] = Field(default=None, description="Optional structured validation details")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Document Processing Lifecycle Schemas (Contract specification for Phase 2+)
# ==============================================================================

class DocumentProcessingStatus(str, Enum):
    """Allowed document lifecycle status values."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETING = "deleting"


class DocumentUploadResponse(BaseModel):
    """Response returned upon document receipt."""
    document_id: str
    filename: str
    status: DocumentProcessingStatus = DocumentProcessingStatus.UPLOADED
    upload_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    file_size_bytes: int
    message: str


class DocumentMetadata(BaseModel):
    """Stored metadata for an ingested document."""
    document_id: str
    filename: str
    status: DocumentProcessingStatus
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None


# ==============================================================================
# Stable Chunk & Retrieval Schemas (Contract specification for Phase 3+)
# ==============================================================================

class StableChunkMetadata(BaseModel):
    """Standardized chunk representation across ingestion, indexing, and retrieval."""
    chunk_id: str = Field(description="Deterministic chunk identifier (e.g., doc_id#chunk_idx)")
    document_id: str
    text: str
    page_number: int = Field(ge=1, description="1-indexed starting page number")
    page_end: Optional[int] = Field(default=None, ge=1, description="Ending page number for cross-page chunks")
    section: Optional[str] = Field(default=None, description="Extracted section heading")
    content_type: str = Field(default="text", description="Type of content: 'text', 'table', 'caption', 'ocr'")
    source_filename: str
    extraction_method: str = Field(default="pymupdf", description="Extraction tool: 'pymupdf', 'tesseract', etc.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional chunk attributes")


class RetrievalScoring(BaseModel):
    """Granular retrieval scores across hybrid pipeline stages."""
    dense_score: Optional[float] = Field(default=None, description="Cosine similarity or inner product from FAISS")
    bm25_score: Optional[float] = Field(default=None, description="Unnormalized BM25 keyword score")
    fusion_score: Optional[float] = Field(default=None, description="Reciprocal Rank Fusion score")
    fusion_rank: Optional[int] = Field(default=None, description="Rank position after RRF")
    rerank_score: Optional[float] = Field(default=None, description="Cross-encoder relevance score (if reranker enabled)")


class RetrievedEvidence(BaseModel):
    """A retrieved chunk bundled with traceable metadata and scoring breakdown."""
    chunk: StableChunkMetadata
    scoring: RetrievalScoring


# ==============================================================================
# Query & Grounded Answer Schemas (Contract specification for Phase 5+)
# ==============================================================================

class Citation(BaseModel):
    """Fine-grained traceable citation linking generation to source academic text."""
    citation_id: str = Field(description="Reference key matching answer citations e.g. [1]")
    chunk_id: str
    document_id: str
    source_filename: str
    page_number: int
    section: Optional[str] = None
    snippet: str = Field(description="Verbatim excerpt from the source chunk")


class QueryRequest(BaseModel):
    """User inquiry request payload."""
    query: str = Field(min_length=3, max_length=1000, description="Research question to answer")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional filter to specific documents")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of evidence chunks to retrieve")
    stream: bool = Field(default=False, description="Whether to stream response tokens")


class QueryTimingMetadata(BaseModel):
    """Timing breakdown for transparency into latency."""
    retrieval_ms: float
    reranking_ms: Optional[float] = None
    generation_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    """Structured response containing grounded answer, citations, evidence, and safeguards."""
    query_id: Optional[str] = Field(default=None, description="Optional unique identifier for the inquiry")
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    evidence: List[RetrievedEvidence] = Field(default_factory=list)
    insufficient_evidence: bool = Field(
        default=False,
        description="True if evidence retrieved was insufficient to reliably answer the query",
    )
    timing: QueryTimingMetadata

