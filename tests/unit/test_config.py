"""Unit tests for configuration loading, validation, and secret masking."""

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_default_settings():
    """Verify default configuration values and types."""
    settings = Settings(_env_file=None)
    assert settings.app_name == "Multimodal RAG Research Assistant"
    assert settings.app_env in ["development", "testing", "production"]
    assert settings.api_port == 8000
    assert settings.api_host == "127.0.0.1"
    assert settings.database_url.startswith("sqlite")
    assert settings.max_upload_size_mb == 25
    assert settings.ocr_enabled is False
    assert settings.reranker_enabled is False


def test_port_validation():
    """Verify invalid port raises validation error."""
    with pytest.raises(ValidationError):
        Settings(api_port=999999)

    with pytest.raises(ValidationError):
        Settings(api_port=-1)


def test_secret_masking():
    """Ensure sensitive credentials are redacted when exported via safe_dict."""
    settings = Settings(llm_api_key="sk-test-super-secret-key-12345")
    assert settings.llm_api_key == "sk-test-super-secret-key-12345"

    safe_export = settings.safe_dict()
    assert "sk-test-super-secret-key-12345" not in safe_export["llm_api_key"]
    assert "[REDACTED]" in safe_export["llm_api_key"]


def test_environment_override(monkeypatch):
    """Verify settings pick up environment variable overrides correctly."""
    monkeypatch.setenv("APP_NAME", "Custom Test RAG")
    monkeypatch.setenv("API_PORT", "9090")
    monkeypatch.setenv("OCR_ENABLED", "true")

    settings = Settings()
    assert settings.app_name == "Custom Test RAG"
    assert settings.api_port == 9090
    assert settings.ocr_enabled is True
