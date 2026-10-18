"""Document upload, listing, status, and deletion API endpoints.

Implements Phase 1 & 2 Document Lifecycle API contracts:
- POST /documents/upload
- GET /documents
- GET /documents/{document_id}
- DELETE /documents/{document_id}
"""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.config import get_settings
from app.schemas import (
    DocumentMetadata,
    DocumentProcessingStatus,
    DocumentUploadResponse,
)
from ingestion.pipeline import MultimodalIngestionPipeline

logger = logging.getLogger("app.api.documents")

router = APIRouter(prefix="/documents", tags=["Document Lifecycle"])

# In-memory document registry for local prototype persistence
_DOCUMENTS_DB: dict[str, DocumentMetadata] = {}

# Singleton pipeline instance
_PIPELINE: MultimodalIngestionPipeline | None = None


def get_pipeline() -> MultimodalIngestionPipeline:
    """Return configured singleton ingestion pipeline."""
    global _PIPELINE
    if _PIPELINE is None:
        settings = get_settings()
        _PIPELINE = MultimodalIngestionPipeline(
            ocr_enabled=settings.ocr_enabled,
        )
    return _PIPELINE


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Academic Document",
)
async def upload_document(
    file: UploadFile = File(..., description="Academic PDF or research paper binary"),
) -> DocumentUploadResponse:
    """Upload, validate, parse, and extract multimodal research content."""
    settings = get_settings()
    pipeline = get_pipeline()

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    # Read binary content
    file_bytes = await file.read()
    file_size = len(file_bytes)

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file cannot be empty (0 bytes).",
        )

    # Validate and process through pipeline
    try:
        result = pipeline.process(
            file_input=file_bytes,
            filename=file.filename,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Failed to process uploaded file '%s': %s", file.filename, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal document extraction failed: {str(e)}",
        ) from e

    # Save to storage directory safely
    save_path = settings.storage_dir / f"{result.document_id}_{result.filename}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(file_bytes)

    # Record in document registry
    now = datetime.now(UTC)
    metadata = DocumentMetadata(
        document_id=result.document_id,
        filename=result.filename,
        status=DocumentProcessingStatus.INDEXED if result.chunks else DocumentProcessingStatus.UPLOADED,
        page_count=result.page_count,
        chunk_count=len(result.chunks),
        created_at=now,
        updated_at=now,
        error_message=None,
    )
    _DOCUMENTS_DB[result.document_id] = metadata

    return DocumentUploadResponse(
        document_id=result.document_id,
        filename=result.filename,
        status=metadata.status,
        upload_timestamp=now,
        file_size_bytes=file_size,
        message=f"Document successfully extracted into {len(result.chunks)} structured chunks.",
    )


@router.get(
    "",
    response_model=list[DocumentMetadata],
    summary="List Ingested Documents",
)
async def list_documents(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[DocumentMetadata]:
    """Retrieve list of currently ingested academic research documents."""
    docs = list(_DOCUMENTS_DB.values())
    return docs[skip : skip + limit]


@router.get(
    "/{document_id}",
    response_model=DocumentMetadata,
    summary="Get Document Metadata",
)
async def get_document(document_id: str) -> DocumentMetadata:
    """Retrieve metadata and extraction status for a specific document."""
    if document_id not in _DOCUMENTS_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return _DOCUMENTS_DB[document_id]


@router.delete(
    "/{document_id}",
    response_model=dict[str, str],
    summary="Delete Document",
)
async def delete_document(document_id: str) -> dict[str, str]:
    """Remove a document from registry and storage."""
    if document_id not in _DOCUMENTS_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    _DOCUMENTS_DB.pop(document_id, None)
    return {
        "document_id": document_id,
        "status": "deleted",
        "message": f"Document '{document_id}' removed successfully.",
    }
