"""Document upload, listing, status, and deletion API endpoints.

Implements Document Lifecycle API contracts:
- POST /documents/upload: Ingests PDF, extracts multimodal content, indexes chunks into FAISS/BM25
- GET /documents: Lists ingested documents with pagination
- GET /documents/{document_id}: Retrieves metadata for a document
- GET /documents/{document_id}/status: Checks processing status of a document
- DELETE /documents/{document_id}: Removes document from registry and purges vectors from indexes
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
from app.services.document_store import DocumentStore
from app.services.rag_service import get_rag_service
from ingestion.pipeline import MultimodalIngestionPipeline

logger = logging.getLogger("app.api.documents")

router = APIRouter(prefix="/documents", tags=["Document Lifecycle"])

# In-memory document cache for quick lookup & test compatibility
_DOCUMENTS_DB: dict[str, DocumentMetadata] = {}

# Singleton instances
_PIPELINE: MultimodalIngestionPipeline | None = None
_STORE: DocumentStore | None = None


def get_pipeline() -> MultimodalIngestionPipeline:
    """Return configured singleton ingestion pipeline."""
    global _PIPELINE
    if _PIPELINE is None:
        settings = get_settings()
        _PIPELINE = MultimodalIngestionPipeline(
            ocr_enabled=settings.ocr_enabled,
        )
    return _PIPELINE


def get_document_store() -> DocumentStore:
    """Return configured singleton document metadata store."""
    global _STORE
    if _STORE is None:
        settings = get_settings()
        db_path = settings.database_url.replace("sqlite:///", "")
        _STORE = DocumentStore(db_path=db_path)
    return _STORE


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Academic Document",
)
async def upload_document(
    file: UploadFile = File(..., description="Academic PDF or research paper binary"),
) -> DocumentUploadResponse:
    """Upload, validate, parse, extract multimodal content, and index into FAISS & BM25."""
    settings = get_settings()
    pipeline = get_pipeline()
    store = get_document_store()
    rag_service = get_rag_service()

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

    # Validate and process through multimodal ingestion pipeline
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

    # Save PDF copy to storage directory safely
    save_path = settings.storage_dir / f"{result.document_id}_{result.filename}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(file_bytes)

    # Synchronously index extracted chunks into FAISS vector index & BM25 keyword index
    if result.chunks:
        try:
            rag_service.index_chunks(result.chunks)
            logger.info(
                "Indexed %d chunks for document '%s'", len(result.chunks), result.document_id
            )
        except Exception as e:
            logger.error(
                "Failed to index chunks for '%s': %s", result.document_id, e, exc_info=True
            )
            # Record failed status
            now = datetime.now(UTC)
            failed_meta = DocumentMetadata(
                document_id=result.document_id,
                filename=result.filename,
                status=DocumentProcessingStatus.FAILED,
                page_count=result.page_count,
                chunk_count=len(result.chunks),
                created_at=now,
                updated_at=now,
                error_message=f"Indexing failed: {str(e)}",
            )
            _DOCUMENTS_DB[result.document_id] = failed_meta
            store.save_document(failed_meta, file_size_bytes=file_size)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document parsed but indexing failed: {str(e)}",
            ) from e

    # Record in persistent document store and in-memory cache
    now = datetime.now(UTC)
    doc_status = (
        DocumentProcessingStatus.INDEXED if result.chunks else DocumentProcessingStatus.UPLOADED
    )
    metadata = DocumentMetadata(
        document_id=result.document_id,
        filename=result.filename,
        status=doc_status,
        page_count=result.page_count,
        chunk_count=len(result.chunks),
        created_at=now,
        updated_at=now,
        error_message=None,
    )
    _DOCUMENTS_DB[result.document_id] = metadata
    store.save_document(metadata, file_size_bytes=file_size)

    return DocumentUploadResponse(
        document_id=result.document_id,
        filename=result.filename,
        status=metadata.status,
        upload_timestamp=now,
        file_size_bytes=file_size,
        message=f"Document successfully extracted and indexed into {len(result.chunks)} structured chunks.",
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
    store = get_document_store()
    persisted_docs = store.list_documents(skip=skip, limit=limit)
    if persisted_docs:
        return persisted_docs

    # Fallback to in-memory dict
    docs = list(_DOCUMENTS_DB.values())
    return docs[skip : skip + limit]


@router.get(
    "/{document_id}",
    response_model=DocumentMetadata,
    summary="Get Document Metadata",
)
async def get_document(document_id: str) -> DocumentMetadata:
    """Retrieve metadata and extraction status for a specific document."""
    store = get_document_store()
    doc = store.get_document(document_id)
    if doc is not None:
        return doc

    if document_id in _DOCUMENTS_DB:
        return _DOCUMENTS_DB[document_id]

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Document '{document_id}' not found.",
    )


@router.get(
    "/{document_id}/status",
    response_model=DocumentMetadata,
    summary="Get Document Processing Status",
)
async def get_document_status(document_id: str) -> DocumentMetadata:
    """Inspect processing and indexing status for a document."""
    return await get_document(document_id)


@router.delete(
    "/{document_id}",
    response_model=dict[str, str],
    summary="Delete Document",
)
async def delete_document(document_id: str) -> dict[str, str]:
    """Remove a document from registry and purge its vectors from FAISS & BM25 indexes."""
    store = get_document_store()
    rag_service = get_rag_service()

    found = False
    if store.get_document(document_id) is not None:
        found = True
        store.delete_document(document_id)

    if document_id in _DOCUMENTS_DB:
        found = True
        _DOCUMENTS_DB.pop(document_id, None)

    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    # Purge from retrieval indexes
    try:
        rag_service.delete_document(document_id)
    except Exception as e:
        logger.warning("Error purging chunks for '%s' from RAG indexes: %s", document_id, e)

    return {
        "document_id": document_id,
        "status": "deleted",
        "message": f"Document '{document_id}' removed successfully.",
    }
