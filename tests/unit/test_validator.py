"""Unit tests for document validation, magic bytes, and size boundaries."""

from ingestion.validator import (
    JPEG_MAGIC_HEADER,
    PDF_MAGIC_HEADER,
    PNG_MAGIC_HEADER,
    DocumentValidator,
)


def test_validator_detects_valid_pdf():
    validator = DocumentValidator(default_max_size_mb=10)
    pdf_content = b"%PDF-1.7\nSample academic PDF content for testing."
    result = validator.validate(pdf_content, filename="paper.pdf")

    assert result.is_valid is True
    assert result.file_type == "pdf"
    assert result.document_id.startswith("doc_")
    assert result.filename == "paper.pdf"
    assert result.error_message is None


def test_validator_detects_image_formats():
    validator = DocumentValidator()

    # PNG
    png_content = PNG_MAGIC_HEADER + b"\x00\x00\x00\rIHDR"
    res_png = validator.validate(png_content, filename="figure.png")
    assert res_png.is_valid is True
    assert res_png.file_type == "image/png"

    # JPEG
    jpeg_content = JPEG_MAGIC_HEADER + b"\xe0\x00\x10JFIF"
    res_jpeg = validator.validate(jpeg_content, filename="plot.jpg")
    assert res_jpeg.is_valid is True
    assert res_jpeg.file_type == "image/jpeg"


def test_validator_rejects_empty_file():
    validator = DocumentValidator()
    result = validator.validate(b"", filename="empty.pdf")

    assert result.is_valid is False
    assert "empty" in result.error_message.lower()


def test_validator_enforces_size_limit():
    validator = DocumentValidator(default_max_size_mb=1)  # 1 MB limit
    oversized = PDF_MAGIC_HEADER + b"x" * (2 * 1024 * 1024)  # 2 MB
    result = validator.validate(oversized, filename="large.pdf")

    assert result.is_valid is False
    assert "exceeds maximum allowed limit" in result.error_message


def test_validator_rejects_corrupted_header():
    validator = DocumentValidator()
    corrupt = b"NOT_A_PDF_OR_IMAGE_JUST_PLAIN_TEXT"
    result = validator.validate(corrupt, filename="corrupt.pdf")

    assert result.is_valid is False
    assert "magic header" in result.error_message.lower()


def test_validator_sanitizes_filename():
    validator = DocumentValidator()
    assert validator.sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert validator.sanitize_filename("my research paper (2024) [final].pdf") == "my_research_paper__2024___final_.pdf"
