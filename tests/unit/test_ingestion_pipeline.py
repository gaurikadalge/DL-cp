"""Unit tests for the master MultimodalIngestionPipeline."""

import fitz  # PyMuPDF

from ingestion.image_extractor import MockCaptioningEngine
from ingestion.ocr_engine import MockOCREngine
from ingestion.pipeline import MultimodalIngestionPipeline


def create_sample_academic_pdf() -> bytes:
    """Helper creating a valid 2-page in-memory academic PDF with headings and text."""
    doc = fitz.open()

    # Page 1: Abstract and Introduction
    page1 = doc.new_page()
    page1.insert_text(
        fitz.Point(72, 72), "Multimodal Representation Learning in Deep Neural Systems", fontsize=16
    )
    page1.insert_text(fitz.Point(72, 110), "Abstract", fontsize=13)
    page1.insert_text(
        fitz.Point(72, 130),
        "This paper investigates multimodal representations combining vision and language representations. "
        "We introduce cross-attention fusion layers that boost downstream accuracy across vision-language benchmarks.",
        fontsize=10,
    )
    page1.insert_text(fitz.Point(72, 180), "1. Introduction", fontsize=13)
    page1.insert_text(
        fitz.Point(72, 200),
        "Recent advances in transformer architectures have revolutionized natural language understanding and computer vision. "
        "Standard uni-modal models often fail to capture rich cross-modal interactions required for complex academic reasoning.",
        fontsize=10,
    )

    # Page 2: Methodology
    page2 = doc.new_page()
    page2.insert_text(fitz.Point(72, 72), "2. Methodology", fontsize=13)
    page2.insert_text(
        fitz.Point(72, 95),
        "Given an input document containing both textual passages and figures, our framework parses the visual and linguistic streams. "
        "The dense embeddings are aligned using contrastive learning with an InfoNCE loss objective.",
        fontsize=10,
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_multimodal_ingestion_pipeline_end_to_end():
    pipeline = MultimodalIngestionPipeline(
        ocr_engine=MockOCREngine(),
        captioning_engine=MockCaptioningEngine(),
    )

    pdf_bytes = create_sample_academic_pdf()
    result = pipeline.process(
        file_input=pdf_bytes,
        filename="multimodal_paper.pdf",
    )

    assert result.document_id.startswith("doc_")
    assert result.filename == "multimodal_paper.pdf"
    assert result.page_count == 2
    assert len(result.chunks) >= 2
    assert result.execution_time_ms > 0

    # Verify chunk structure adheres strictly to contract
    for chunk in result.chunks:
        assert chunk.document_id == result.document_id
        assert chunk.chunk_id.startswith(f"{result.document_id}#chunk_")
        assert chunk.page_number in (1, 2)
        assert len(chunk.text) > 0
        assert chunk.source_filename == "multimodal_paper.pdf"

    summary = result.summary()
    assert summary["document_id"] == result.document_id
    assert summary["chunks_count"] == len(result.chunks)
    assert summary["page_count"] == 2
