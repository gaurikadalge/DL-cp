"""Unit tests for Research Query and Status API endpoints."""

import io

import fitz  # PyMuPDF
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_academic_pdf(title: str, text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(50, 50), f"1. {title}", fontsize=14)
    page.insert_text(fitz.Point(50, 80), text, fontsize=10)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_query_validation_errors():
    """Verify short query or invalid parameters trigger validation errors."""
    # Query too short (< 3 chars)
    res = client.post("/query", json={"query": "ab"})
    assert res.status_code == 422
    data = res.json()
    assert data["error_code"] == "VALIDATION_ERROR"

    # Top-K out of bounds (< 1)
    res = client.post("/query", json={"query": "valid query", "top_k": 0})
    assert res.status_code == 422


def test_query_insufficient_evidence():
    """Verify querying an empty corpus or nonexistent document filter returns insufficient evidence."""
    res = client.post(
        "/query",
        json={
            "query": "What are quantum computing qubit algorithms?",
            "document_ids": ["doc_nonexistent_xyz"],
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert data["insufficient_evidence"] is True
    assert data["citations"] == []
    assert data["evidence"] == []
    assert data["timing"]["total_ms"] >= 0.0


def test_upload_and_query_flow():
    """Verify uploading an academic paper and querying it produces grounded response and citations."""
    pdf_bytes = make_academic_pdf(
        title="Deep Residual Learning",
        text=(
            "Deep residual networks introduce shortcut identity connections to address the degradation problem. "
            "By reformulating layers to fit residual mappings F(x) = H(x) - x, deep models optimize easily."
        ),
    )

    # 1. Upload Document
    upload_res = client.post(
        "/documents/upload",
        files={"file": ("residual_paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert upload_res.status_code == 201
    upload_data = upload_res.json()
    doc_id = upload_data["document_id"]
    assert upload_data["status"] == "indexed"

    # 2. Check Document Status endpoint
    status_res = client.get(f"/documents/{doc_id}/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["document_id"] == doc_id
    assert status_data["status"] == "indexed"
    assert status_data["chunk_count"] > 0

    # 3. Query the indexed paper
    query_res = client.post(
        "/query",
        json={
            "query": "How do residual networks address the degradation problem?",
            "document_ids": [doc_id],
            "top_k": 3,
        },
    )
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert q_data["query"] == "How do residual networks address the degradation problem?"
    assert q_data["insufficient_evidence"] is False
    assert len(q_data["evidence"]) > 0
    assert q_data["evidence"][0]["chunk"]["document_id"] == doc_id
    assert len(q_data["citations"]) > 0
    assert q_data["citations"][0]["document_id"] == doc_id
    assert q_data["timing"]["retrieval_ms"] >= 0.0
    assert q_data["timing"]["total_ms"] >= 0.0

    # 4. Clean up
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200
