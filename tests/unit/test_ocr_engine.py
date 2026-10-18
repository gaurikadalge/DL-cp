"""Unit tests for OCR engine abstractions and scanned page detection."""

import fitz  # PyMuPDF

from ingestion.ocr_engine import MockOCREngine, ScannedPageDetector, TesseractOCREngine


def test_mock_ocr_engine_execution():
    engine = MockOCREngine(canned_response="Recognized academic text from scanned table.")
    assert engine.is_available() is True
    res = engine.extract_text(b"fake_image_bytes")
    assert res == "Recognized academic text from scanned table."
    assert engine.call_count == 1


def test_tesseract_engine_availability_check():
    # Test with non-existent binary path
    engine = TesseractOCREngine(tesseract_cmd="/non/existent/path/to/tesseract")
    assert engine.is_available() is False


def test_scanned_page_detector_on_text_page():
    # Create an in-memory PDF with ample text
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        fitz.Point(50, 50),
        "This is an academic research document with plenty of selectable text characters. "
        "Therefore, this page should not be classified as a scanned image.",
    )

    detector = ScannedPageDetector(char_threshold=40)
    is_scanned = detector.is_scanned_page(page)
    doc.close()

    assert is_scanned is False


def test_scanned_page_detector_on_sparse_page_without_images():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(50, 50), "Short text.")

    detector = ScannedPageDetector(char_threshold=50)
    is_scanned = detector.is_scanned_page(page)
    doc.close()

    # Sparse text without images should not be treated as a scanned image requiring OCR
    assert is_scanned is False
