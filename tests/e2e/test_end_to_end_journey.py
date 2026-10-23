"""End-to-End User Journey integration test suite.

Simulates the complete researcher journey:
1. Health audit validation
2. Academic document upload, layout parsing, and chunk indexing
3. Document listing and status inspection
4. Hybrid question answering with strict grounding and page citations
5. Out-of-domain query triggering insufficient evidence safeguard
6. Persistence verification (restoring indexes from disk)
7. Document deletion and complete vector purge
"""

import io

import fitz  # PyMuPDF
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services.rag_service import RAGService, get_rag_service

client = TestClient(app)


def build_academic_paper_pdf() -> bytes:
    """Construct a realistic multi-page academic PDF with sections and technical terminology."""
    doc = fitz.open()

    # Page 1: Abstract & Introduction
    p1 = doc.new_page()
    p1.insert_text(fitz.Point(50, 50), "Deep Residual Learning for Image Recognition", fontsize=16)
    p1.insert_text(fitz.Point(50, 75), "Abstract", fontsize=12)
    p1.insert_text(
        fitz.Point(50, 95),
        "Deeper neural networks are more difficult to train. We present a residual learning framework "
        "to ease the training of networks that are substantially deeper than those used previously. "
        "We explicitly reformulate the layers as learning residual functions with reference to the layer inputs.",
        fontsize=10,
    )
    p1.insert_text(fitz.Point(50, 160), "1. Introduction", fontsize=12)
    p1.insert_text(
        fitz.Point(50, 180),
        "When deeper networks are able to start converging, a degradation problem has been exposed: "
        "with the network depth increasing, accuracy gets saturated and then degrades rapidly. "
        "Unexpectedly, such degradation is not caused by overfitting, and adding more layers leads to higher training error.",
        fontsize=10,
    )

    # Page 2: Formulation & Methodology
    p2 = doc.new_page()
    p2.insert_text(fitz.Point(50, 50), "3. Deep Residual Learning", fontsize=14)
    p2.insert_text(fitz.Point(50, 75), "3.1. Formulation", fontsize=12)
    p2.insert_text(
        fitz.Point(50, 95),
        "Let us consider H(x) as an underlying mapping to be fit by a few stacked layers, with x denoting the inputs. "
        "If one hypothesizes that multiple nonlinear layers can asymptotically approximate complicated functions, "
        "it is equivalent to hypothesize that they can asymptotically approximate the residual functions, F(x) = H(x) - x. "
        "The formulation of F(x) + x can be realized by feedforward neural networks with shortcut connections.",
        fontsize=10,
    )
    p2.insert_text(fitz.Point(50, 180), "3.2. Optimization with AdamW", fontsize=12)
    p2.insert_text(
        fitz.Point(50, 200),
        "We train our deep residual networks using standard stochastic gradient descent with decoupled weight decay (AdamW). "
        "The learning rate starts at 0.1 and is divided by 10 when the error plateaus.",
        fontsize=10,
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_complete_end_to_end_researcher_journey(tmp_path):
    """Execute complete end-to-end user journey across all lifecycle stages."""
    settings = get_settings()

    # Step 1: Health audit check
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "ok"

    # Step 2: Upload paper
    pdf_bytes = build_academic_paper_pdf()
    upload_res = client.post(
        "/documents/upload",
        files={"file": ("resnet_paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert upload_res.status_code == 201
    upload_data = upload_res.json()
    doc_id = upload_data["document_id"]
    assert upload_data["status"] == "indexed"
    assert upload_data["filename"] == "resnet_paper.pdf"

    # Step 3: Document listing and detail
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    docs = list_res.json()
    assert any(d["document_id"] == doc_id for d in docs)

    detail_res = client.get(f"/documents/{doc_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["document_id"] == doc_id
    assert detail["page_count"] == 2
    assert detail["chunk_count"] > 0

    status_res = client.get(f"/documents/{doc_id}/status")
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "indexed"

    # Step 4: Grounded research question answering
    query_res = client.post(
        "/query",
        json={
            "query": "How does the residual mapping F(x) = H(x) - x address degradation?",
            "document_ids": [doc_id],
            "top_k": 3,
        },
    )
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert q_data["insufficient_evidence"] is False
    assert len(q_data["evidence"]) > 0

    # Verify score structures in evidence
    first_ev = q_data["evidence"][0]
    assert first_ev["chunk"]["document_id"] == doc_id
    assert first_ev["scoring"]["fusion_score"] is not None
    assert first_ev["scoring"]["fusion_rank"] == 1

    # Verify traceable citations
    assert len(q_data["citations"]) > 0
    first_cite = q_data["citations"][0]
    assert first_cite["citation_id"] == "[1]"
    assert first_cite["document_id"] == doc_id
    assert first_cite["source_filename"] == "resnet_paper.pdf"
    assert first_cite["page_number"] in [1, 2]
    assert len(first_cite["snippet"]) > 10

    # Verify timing breakdown
    timing = q_data["timing"]
    assert timing["retrieval_ms"] >= 0.0
    assert timing["total_ms"] >= timing["retrieval_ms"]

    # Step 5: Insufficient evidence for out-of-domain query
    irrelevant_res = client.post(
        "/query",
        json={
            "query": "What quantum optical cryogenic sensors were calibrated in this experiment?",
            "document_ids": [doc_id],
            "top_k": 3,
        },
    )
    assert irrelevant_res.status_code == 200
    irr_data = irrelevant_res.json()
    # Out of domain query either has insufficient evidence or no matching evidence
    assert irr_data["insufficient_evidence"] is True or len(irr_data["citations"]) == 0

    # Step 6: Verify index persistence on disk and reload
    rag_service = get_rag_service()
    rag_service.save_indexes()

    # Create fresh RAGService instance simulating server restart
    reloaded_rag = RAGService(settings=settings)
    assert reloaded_rag.total_indexed_chunks >= 1

    # Step 7: Document deletion and purge
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Confirm 404 on get
    assert client.get(f"/documents/{doc_id}").status_code == 404

    # Confirm querying deleted document yields insufficient evidence
    post_del_query = client.post(
        "/query",
        json={
            "query": "How does the residual mapping F(x) = H(x) - x address degradation?",
            "document_ids": [doc_id],
        },
    )
    assert post_del_query.status_code == 200
    assert post_del_query.json()["insufficient_evidence"] is True
