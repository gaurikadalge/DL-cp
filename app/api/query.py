"""Research question answering and hybrid retrieval API endpoint."""

import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas import QueryRequest, QueryResponse
from app.services.rag_service import get_rag_service

logger = logging.getLogger("app.api.query")

router = APIRouter(prefix="/query", tags=["Retrieval & QA"])


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Research Query & Grounded Synthesis",
    description=(
        "Executes hybrid dense (FAISS) and lexical (BM25) search with Reciprocal Rank Fusion (RRF), "
        "optional cross-encoder reranking, and synthesizes a strictly grounded answer with verifiable citations."
    ),
)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """Answer research inquiry using indexed academic evidence."""
    rag_service = get_rag_service()

    try:
        response = rag_service.query(
            query=request.query,
            document_ids=request.document_ids,
            top_k=request.top_k or 5,
        )
        return response
    except ValueError as e:
        logger.warning("Invalid query request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Internal query synthesis error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate grounded research response due to an internal error.",
        ) from e
