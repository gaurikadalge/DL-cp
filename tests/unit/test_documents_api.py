"""Unit tests for Document Lifecycle API routes."""

import io

import fitz  # PyMuPDF
from fastapi.testclient import TestClient

from app.main import app


def make_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        fitz.Point(50, 50), "1. Document API Test\nSample academic text for API upload testing."
    )
    data = doc.tobytes()
    doc.close()
    return data


def test_document_upload_and_lifecycle():
    client = TestClient(app)
    pdf_bytes = make_pdf_bytes()

    # 1. Upload Document
    response = client.post(
        "/documents/upload",
        files={"file": ("sample_paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "sample_paper.pdf"
    assert data["status"] in ("indexed", "uploaded")
    doc_id = data["document_id"]

    # 2. List Documents
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    docs = list_res.json()
    assert any(d["document_id"] == doc_id for d in docs)

    # 3. Get Document Details
    detail_res = client.get(f"/documents/{doc_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["document_id"] == doc_id
    assert detail["filename"] == "sample_paper.pdf"

    # 4. Delete Document
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # 5. Verify 404 after deletion
    get_after_del = client.get(f"/documents/{doc_id}")
    assert get_after_del.status_code == 404
