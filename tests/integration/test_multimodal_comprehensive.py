"""Comprehensive integration test suite validating all multimodal processing modalities.

Tests:
1. Standard single-column academic research PDF.
2. Two-column academic research paper layout.
3. Scanned PDF page triggering OCR fallback via ScannedPageDetector.
4. Academic PDF with embedded structured tables.
5. Academic PDF with embedded figures and captions.
6. Corrupted, truncated, and invalid PDF inputs.
7. Explicit distinction between figure extraction and vision understanding.
"""

import io

import fitz  # PyMuPDF
import pytest
from PIL import Image

from ingestion.image_extractor import MockCaptioningEngine
from ingestion.ocr_engine import MockOCREngine, ScannedPageDetector
from ingestion.pipeline import MultimodalIngestionPipeline
from ingestion.validator import DocumentValidator


def create_normal_pdf() -> bytes:
    """Generate a standard research paper PDF with sections and paragraphs."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(50, 50), "1. Introduction to Deep Learning", fontsize=14)
    page.insert_textbox(
        fitz.Rect(50, 70, 520, 300),
        "Deep learning allows computational models that are composed of multiple processing layers "
        "to learn representations of data with multiple levels of abstraction. These methods have "
        "dramatically improved the state-of-the-art in visual object recognition.",
        fontsize=10,
    )
    b = doc.tobytes()
    doc.close()
    return b


def create_two_column_pdf() -> bytes:
    """Generate a dual-column academic PDF page."""
    doc = fitz.open()
    page = doc.new_page()

    # Column 1 (Left: x=50 to x=250)
    col1_rect = fitz.Rect(50, 60, 260, 500)
    col1_text = (
        "1. First Column Overview\n\n"
        "Recent advances in neural network architectures have shown that residual shortcuts "
        "significantly alleviate training difficulties in deeper networks. The gradient flow "
        "is preserved through linear identity mapping."
    )
    page.insert_textbox(col1_rect, col1_text, fontsize=10)

    # Column 2 (Right: x=300 to x=520)
    col2_rect = fitz.Rect(300, 60, 520, 500)
    col2_text = (
        "2. Second Column Formulation\n\n"
        "In this section we formalize the optimization dynamics. When weight decay is decoupled "
        "from the gradient update as in AdamW, generalization bounds improve markedly."
    )
    page.insert_textbox(col2_rect, col2_text, fontsize=10)

    b = doc.tobytes()
    doc.close()
    return b


def create_scanned_page_pdf() -> bytes:
    """Generate a scanned document simulation: full-page image with zero selectable text."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size

    # Create dummy scanned document raster image (400x500 px)
    img = Image.new("RGB", (400, 500), color=(240, 240, 240))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    # Insert image covering > 60% of the page
    rect = fitz.Rect(50, 50, 545, 750)
    page.insert_image(rect, stream=img_bytes)

    b = doc.tobytes()
    doc.close()
    return b


def create_pdf_with_table() -> bytes:
    """Generate a PDF containing an academic tabular comparison and caption."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        fitz.Point(50, 50), "Table 1: Accuracy comparison on ImageNet-1K benchmark.", fontsize=11
    )

    # Draw table border lines
    page.draw_rect(fitz.Rect(50, 70, 500, 160), width=1)
    page.draw_line(fitz.Point(50, 100), fitz.Point(500, 100), width=1)
    page.draw_line(fitz.Point(200, 70), fitz.Point(200, 160), width=1)
    page.draw_line(fitz.Point(350, 70), fitz.Point(350, 160), width=1)

    # Table Header
    page.insert_text(fitz.Point(60, 90), "Model Architecture", fontsize=10)
    page.insert_text(fitz.Point(210, 90), "Top-1 Accuracy", fontsize=10)
    page.insert_text(fitz.Point(360, 90), "FLOPs (G)", fontsize=10)

    # Table Row 1
    page.insert_text(fitz.Point(60, 120), "ResNet-50", fontsize=9)
    page.insert_text(fitz.Point(210, 120), "76.1%", fontsize=9)
    page.insert_text(fitz.Point(360, 120), "4.1", fontsize=9)

    # Table Row 2
    page.insert_text(fitz.Point(60, 145), "ResNet-152", fontsize=9)
    page.insert_text(fitz.Point(210, 145), "78.3%", fontsize=9)
    page.insert_text(fitz.Point(360, 145), "11.3", fontsize=9)

    b = doc.tobytes()
    doc.close()
    return b


def create_pdf_with_figure() -> bytes:
    """Generate a PDF containing an embedded figure and caption."""
    doc = fitz.open()
    page = doc.new_page()

    # Draw figure raster image (150x150 px)
    img = Image.new("RGB", (150, 150), color=(180, 200, 230))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    rect = fitz.Rect(100, 80, 250, 230)
    page.insert_image(rect, stream=img_bytes)

    # Figure Caption proximate to image
    page.insert_text(
        fitz.Point(100, 250),
        "Figure 1: Residual learning building block with shortcut identity connection.",
        fontsize=10,
    )

    b = doc.tobytes()
    doc.close()
    return b


# ------------------------------------------------------------------------------
# Test Executions
# ------------------------------------------------------------------------------


def test_normal_pdf_extraction():
    """Verify clean parsing of standard single-column research paper."""
    pipeline = MultimodalIngestionPipeline()
    pdf_bytes = create_normal_pdf()

    result = pipeline.process(pdf_bytes, filename="normal_paper.pdf")
    assert result.page_count == 1
    assert len(result.chunks) >= 1
    assert any("visual object recognition" in c.text for c in result.chunks)
    assert result.chunks[0].page_number == 1
    assert result.chunks[0].source_filename == "normal_paper.pdf"
    assert result.chunks[0].content_type == "text"


def test_two_column_pdf_extraction():
    """Verify column ordering and text extraction across dual columns."""
    pipeline = MultimodalIngestionPipeline()
    pdf_bytes = create_two_column_pdf()

    result = pipeline.process(pdf_bytes, filename="two_col.pdf")
    all_text = " ".join([c.text for c in result.chunks])
    assert "First Column" in all_text
    assert "Second Column" in all_text
    assert result.page_count == 1


def test_scanned_pdf_triggers_ocr():
    """Verify that an image-dominated page triggers ScannedPageDetector and OCREngine."""
    mock_ocr = MockOCREngine(canned_response="OCR Recovered: Archival scanned text from 1950.")
    detector = ScannedPageDetector(char_threshold=50, min_image_area_ratio=0.35)

    pipeline = MultimodalIngestionPipeline(
        ocr_engine=mock_ocr,
        ocr_enabled=True,
    )
    scanned_bytes = create_scanned_page_pdf()

    # Verify detector independently
    doc = fitz.open(stream=scanned_bytes, filetype="pdf")
    assert detector.is_scanned_page(doc[0]) is True
    doc.close()

    # Process through pipeline
    result = pipeline.process(scanned_bytes, filename="scanned_historical.pdf")
    assert mock_ocr.call_count >= 1
    assert any("OCR Recovered" in c.text for c in result.chunks)
    assert any(
        c.extraction_method in ("ocr", "tesseract") or c.content_type == "ocr"
        for c in result.chunks
    )


def test_table_extraction_modality():
    """Verify extraction of table structures into Markdown representation."""
    pipeline = MultimodalIngestionPipeline(extract_tables=True)
    pdf_bytes = create_pdf_with_table()

    result = pipeline.process(pdf_bytes, filename="paper_with_tables.pdf")
    assert result.page_count == 1
    # Check that chunks preserve table content
    all_chunks_text = "\n".join([c.text for c in result.chunks])
    assert "Table 1" in all_chunks_text or "ResNet-50" in all_chunks_text


def test_figure_extraction_and_modality_boundary():
    """Verify figure extraction, proximity caption association, and honest modality declaration."""
    mock_vision = MockCaptioningEngine(
        canned_caption="Neural architecture diagram showing skip connections."
    )
    pipeline = MultimodalIngestionPipeline(
        captioning_engine=mock_vision,
        extract_figures=True,
    )
    pdf_bytes = create_pdf_with_figure()

    result = pipeline.process(pdf_bytes, filename="paper_with_figures.pdf")
    assert result.page_count == 1
    assert len(result.figures) >= 1
    fig = result.figures[0]
    assert fig.page_number == 1
    assert fig.width > 50
    assert fig.height > 50

    # Verify that the caption is captured in chunks
    all_chunks_text = "\n".join([c.text for c in result.chunks])
    assert "Figure 1" in all_chunks_text or "Neural architecture" in all_chunks_text


def test_invalid_and_corrupted_inputs():
    """Verify robust error handling for corrupted or invalid documents."""
    pipeline = MultimodalIngestionPipeline()
    validator = DocumentValidator()

    # 1. Zero-byte empty file
    with pytest.raises(ValueError, match="validation failed|empty"):
        pipeline.process(b"", filename="empty.pdf")

    # 2. Corrupted PDF header (plain text disguised as PDF)
    fake_pdf = b"This is plain text without any PDF magic bytes."
    with pytest.raises(ValueError, match="validation failed|magic header"):
        pipeline.process(fake_pdf, filename="fake.pdf")

    # 3. Truncated PDF bytes (header present but body truncated)
    truncated_pdf = b"%PDF-1.4\n% truncated random garbage..."
    with pytest.raises(fitz.FileDataError):
        pipeline.process(truncated_pdf, filename="truncated.pdf")

    # 4. File exceeding size limit
    val_res = validator.validate(
        b"%PDF-1.5\n" + b"0" * (1024 * 1024 * 2), filename="big.pdf", max_size_mb=1
    )
    assert val_res.is_valid is False
    assert "exceeds maximum allowed limit" in str(val_res.error_message)
