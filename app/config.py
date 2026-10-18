"""Centralized application configuration management.

Loads and validates settings from environment variables and optional .env file.
Provides safe string representation preventing credential leakage in logs.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings schema with validation and safe defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Core Application Settings
    app_name: str = Field(
        default="Multimodal RAG Research Assistant",
        description="Public display name of the service",
    )
    app_env: Literal["development", "testing", "production", "staging"] = Field(
        default="development",
        description="Runtime execution environment",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (disables production protections)",
    )
    api_host: str = Field(
        default="127.0.0.1",
        description="Host interface for API server",
    )
    api_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Port for API server (1024-65535)",
    )

    # Storage and Persistence
    database_url: str = Field(
        default="sqlite:///./data/research_assistant.db",
        description="Database connection URI",
    )
    storage_dir: Path = Field(
        default=Path("./data/uploads"),
        description="Directory where uploaded files are stored",
    )
    index_path: Path = Field(
        default=Path("./indexes"),
        description="Directory where vector and BM25 index artifacts are stored",
    )
    max_upload_size_mb: int = Field(
        default=25,
        gt=0,
        le=200,
        description="Maximum allowed PDF upload size in megabytes",
    )

    # Retrieval and Feature Flags
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model ID for dense embeddings",
    )
    embedding_batch_size: int = Field(default=32, gt=0, le=256)
    device: str = Field(default="cpu", description="Compute device: 'cpu', 'cuda', 'mps'")
    ocr_enabled: bool = Field(
        default=False,
        description="Enable Tesseract OCR extraction for image-heavy pages",
    )
    reranker_enabled: bool = Field(
        default=False,
        description="Enable cross-encoder reranking stage in retrieval pipeline",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="HuggingFace model ID for cross-encoder reranking",
    )
    top_k_dense: int = Field(default=10, gt=0, le=100)
    top_k_bm25: int = Field(default=10, gt=0, le=100)
    top_k_final: int = Field(default=5, gt=0, le=50)
    rrf_k: int = Field(default=60, gt=0, le=200, description="RRF rank smoothing constant")
    context_budget_tokens: int = Field(default=2048, gt=100, le=16384)
    minimum_evidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum score threshold for evidence acceptance",
    )

    # LLM Generation Provider Settings
    llm_provider: str | None = Field(
        default=None,
        description="LLM provider: 'openai', 'anthropic', 'ollama', 'groq', or mock",
    )
    llm_model: str | None = Field(
        default=None,
        description="Model name/id to pass to LLM provider",
    )
    llm_api_key: str | None = Field(
        default=None,
        description="API key for selected LLM provider",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Custom base URL for local or proxy LLM endpoints",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for grounded generation",
    )
    llm_timeout_seconds: float = Field(
        default=30.0,
        gt=1.0,
        le=300.0,
        description="Timeout for LLM API requests in seconds",
    )

    @field_validator("api_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {v}")
        return v

    def safe_dict(self) -> dict:
        """Export dictionary masking sensitive secrets for safe logging or debug display."""
        data = self.model_dump()
        if data.get("llm_api_key"):
            data["llm_api_key"] = "****** [REDACTED]"
        return data


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()
