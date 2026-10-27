"""Pytest configuration and session fixtures for the Multimodal RAG Assistant test suite."""

import os
import pytest

# Ensure test mode is active for deterministic, fast offline testing
os.environ["DL_CP_TEST_MODE"] = "true"


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons in app services between tests to prevent state leakage."""
    yield
    # Cleanup any active RAG service singletons if needed
    import app.services.rag_service as rs
    import app.api.documents as doc_api

    if hasattr(rs, "_RAG_SERVICE"):
        rs._RAG_SERVICE = None
    if hasattr(doc_api, "_PIPELINE"):
        doc_api._PIPELINE = None
    if hasattr(doc_api, "_STORE"):
        doc_api._STORE = None
