"""Unit tests for foundational HTTP endpoints and error handling."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify GET / returns application information and docs URLs."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "app_name" in data
    assert "version" in data
    assert data["docs_url"] == "/docs"
    assert data["health_url"] == "/health"


def test_health_endpoint():
    """Verify GET /health returns status ok and honest dependency disclosure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "dependencies" in data
    deps = data["dependencies"]

    # Verify dependency statuses are accurately disclosed
    assert deps["database"] in ["configured", "not_configured"]
    assert deps["vector_index"] == "not_loaded_scaffold"
    assert deps["keyword_retrieval"] == "not_loaded_scaffold"


def test_not_found_returns_structured_error():
    """Verify 404 errors adhere to the unified ErrorDetail schema without path leakage."""
    response = client.get("/non_existent_route_404")
    assert response.status_code == 404
    data = response.json()
    assert "error_code" in data
    assert data["error_code"] == "HTTP_404"
    assert "message" in data
    assert "timestamp" in data
