"""Application entrypoint for the Multimodal RAG Research Assistant API.

Provides:
- GET /: Service info and documentation links
- GET /health: Health status distinguishing app availability from external dependency readiness
- Global error handling returning predictable error responses without leaking internal paths
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.config import get_settings
from app.schemas import (
    AppInfo,
    DependencyStatus,
    ErrorDetail,
    HealthCheckResponse,
    LiteralStatus,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown routines."""
    settings = get_settings()
    logger.info(
        "Initializing %s (v%s) in '%s' environment",
        settings.app_name,
        __version__,
        settings.app_env,
    )

    # Ensure runtime data directories exist safely
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.index_path.mkdir(parents=True, exist_ok=True)

    yield

    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Grounded Multimodal RAG Research Assistant backend API for academic PDF extraction, "
            "hybrid dense/keyword retrieval, and traceable citation generation."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS configuration for development & frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --------------------------------------------------------------------------
    # Centralized Error Handlers (Prevents traceback & path leakage)
    # --------------------------------------------------------------------------

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.warning("HTTP %d error on %s: %s", exc.status_code, request.url.path, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorDetail(
                error_code=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
                details=None,
            ).model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.info("Validation error on %s: %s", request.url.path, exc.errors())
        # Clean errors so internal representations are safe
        sanitized_errors = [
            {"loc": [str(loc) for loc in err["loc"]], "msg": err["msg"], "type": err["type"]}
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            content=ErrorDetail(
                error_code="VALIDATION_ERROR",
                message="Request payload validation failed",
                details=sanitized_errors,
            ).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception processing request to %s", request.url.path, exc_info=True
        )
        # Never leak raw traceback or system paths in production/API responses
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorDetail(
                error_code="INTERNAL_SERVER_ERROR",
                message="An unexpected server error occurred. Please contact the administrator.",
                details=None,
            ).model_dump(mode="json"),
        )

    # --------------------------------------------------------------------------
    # Foundational Endpoints
    # --------------------------------------------------------------------------

    @app.get(
        "/",
        response_model=AppInfo,
        summary="Service Information",
        tags=["System"],
    )
    async def root() -> AppInfo:
        """Returns service metadata and interactive API documentation references."""
        return AppInfo(
            app_name=settings.app_name,
            version=__version__,
            environment=settings.app_env,
            docs_url="/docs",
            health_url="/health",
        )

    @app.get(
        "/health",
        response_model=HealthCheckResponse,
        summary="Service Health Check",
        tags=["System"],
    )
    async def health_check() -> HealthCheckResponse:
        """Evaluates basic API health and explicitly discloses external dependency statuses.

        Note: Does not claim vector indexes, OCR engines, or LLMs are healthy unless
        explicitly initialized and verified.
        """
        # Audit dependency readiness honestly
        dependencies = {
            "database": (
                DependencyStatus.CONFIGURED
                if settings.database_url
                else DependencyStatus.NOT_CONFIGURED
            ),
            "vector_index": DependencyStatus.NOT_LOADED_SCAFFOLD,
            "keyword_retrieval": DependencyStatus.NOT_LOADED_SCAFFOLD,
            "llm_provider": (
                DependencyStatus.CONFIGURED
                if settings.llm_provider
                else DependencyStatus.NOT_CONFIGURED
            ),
            "ocr_engine": (
                DependencyStatus.AVAILABLE if settings.ocr_enabled else DependencyStatus.DISABLED
            ),
            "reranker": (
                DependencyStatus.AVAILABLE
                if settings.reranker_enabled
                else DependencyStatus.DISABLED
            ),
        }

        return HealthCheckResponse(
            status=LiteralStatus.OK,
            app_name=settings.app_name,
            version=__version__,
            environment=settings.app_env,
            dependencies=dependencies,
        )

    # Mount Document Lifecycle API routes
    from app.api.documents import router as documents_router

    app.include_router(documents_router)

    # Mount Query & QA API routes
    from app.api.query import router as query_router

    app.include_router(query_router)

    @app.get(
        "/evaluation/benchmark",
        summary="Retrieval Benchmark Results",
        tags=["Evaluation"],
    )
    async def get_benchmark_results() -> dict:
        """Returns verified machine-readable retrieval benchmark metrics."""
        import json
        from pathlib import Path

        results_file = Path("evaluation/retrieval_benchmark_results.json")
        if results_file.exists():
            return json.loads(results_file.read_text(encoding="utf-8"))
        return {"error": "Benchmark results not yet generated"}

    return app


app = create_app()
