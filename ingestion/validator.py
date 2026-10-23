"""File upload and document format validator.

Enforces MIME validation, magic byte verification, file size boundaries,
and path sanitization for academic PDFs and scanned documents.
"""

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

# Standard magic byte signatures
PDF_MAGIC_HEADER = b"%PDF-"
PNG_MAGIC_HEADER = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC_HEADER = b"\xff\xd8\xff"
TIFF_LE_MAGIC_HEADER = b"II*\x00"
TIFF_BE_MAGIC_HEADER = b"MM\x00*"
WEBP_RIFF_HEADER = b"RIFF"
WEBP_TYPE_HEADER = b"WEBP"


@dataclass
class ValidationResult:
    """Outcome of document upload validation."""

    is_valid: bool
    document_id: str
    filename: str
    file_type: str  # "pdf", "image/png", "image/jpeg", "image/tiff", "image/webp"
    file_size_bytes: int
    error_message: str | None = None


class DocumentValidator:
    """Validates uploaded research documents against size, magic header, and security constraints."""

    ALLOWED_EXTENSIONS: set[str] = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp"}

    def __init__(self, default_max_size_mb: int = 25):
        self.default_max_size_mb = default_max_size_mb

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent directory traversal or malicious characters."""
        basename = os.path.basename(filename.strip())
        # Replace dangerous characters with underscores
        sanitized = re.sub(r"[^\w\.\-]", "_", basename)
        # Avoid empty or dot-only filenames
        if not sanitized or sanitized.startswith("."):
            sanitized = f"document_{uuid.uuid4().hex[:8]}.pdf"
        return sanitized

    @staticmethod
    def generate_document_id(file_bytes: bytes, filename: str) -> str:
        """Generate a deterministic, traceable document ID based on SHA-256 hash."""
        digest = hashlib.sha256(file_bytes[:4096] + filename.encode("utf-8")).hexdigest()[:12]
        return f"doc_{digest}"

    @classmethod
    def detect_file_type(cls, file_bytes: bytes) -> str:
        """Inspect binary magic headers to identify the authentic file format."""
        if len(file_bytes) < 4:
            raise ValueError("File content is too small to determine format.")

        # PDF: Header may appear within the first 1024 bytes per PDF 1.7 spec
        first_1024 = file_bytes[:1024]
        if PDF_MAGIC_HEADER in first_1024:
            return "pdf"

        # PNG
        if file_bytes.startswith(PNG_MAGIC_HEADER):
            return "image/png"

        # JPEG
        if file_bytes.startswith(JPEG_MAGIC_HEADER):
            return "image/jpeg"

        # TIFF
        if file_bytes.startswith(TIFF_LE_MAGIC_HEADER) or file_bytes.startswith(
            TIFF_BE_MAGIC_HEADER
        ):
            return "image/tiff"

        # WebP
        if (
            len(file_bytes) >= 12
            and file_bytes.startswith(WEBP_RIFF_HEADER)
            and file_bytes[8:12] == WEBP_TYPE_HEADER
        ):
            return "image/webp"

        raise ValueError(
            "Unsupported or corrupted document format: magic header does not match PDF or supported image types."
        )

    def validate(
        self,
        content: bytes | BinaryIO | Path | str,
        filename: str | None = None,
        max_size_mb: int | None = None,
    ) -> ValidationResult:
        """Execute comprehensive validation on an uploaded document."""
        # 1. Resolve raw bytes and filename
        if isinstance(content, (str, Path)):
            path = Path(content)
            if not path.exists():
                return ValidationResult(
                    is_valid=False,
                    document_id="",
                    filename=str(path.name),
                    file_type="unknown",
                    file_size_bytes=0,
                    error_message=f"File not found: {content}",
                )
            file_bytes = path.read_bytes()
            if filename is None:
                filename = path.name
        elif isinstance(content, bytes):
            file_bytes = content
            if filename is None:
                filename = "uploaded_document.pdf"
        elif hasattr(content, "read"):
            file_bytes = content.read()
            if filename is None:
                filename = getattr(content, "filename", "uploaded_document.pdf")
        else:
            raise TypeError(f"Unsupported content type for validation: {type(content)}")

        # 2. Check empty file
        file_size = len(file_bytes)
        if file_size == 0:
            return ValidationResult(
                is_valid=False,
                document_id="",
                filename=filename or "unnamed",
                file_type="unknown",
                file_size_bytes=0,
                error_message="Uploaded document is empty (0 bytes).",
            )

        # 3. Check file size bounds
        effective_max_mb = max_size_mb if max_size_mb is not None else self.default_max_size_mb
        max_bytes = effective_max_mb * 1024 * 1024
        if file_size > max_bytes:
            return ValidationResult(
                is_valid=False,
                document_id="",
                filename=filename or "unnamed",
                file_type="unknown",
                file_size_bytes=file_size,
                error_message=(
                    f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed limit "
                    f"of {effective_max_mb} MB."
                ),
            )

        # 4. Sanitize filename & check extension
        clean_name = self.sanitize_filename(filename or "document.pdf")
        ext = Path(clean_name).suffix.lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            return ValidationResult(
                is_valid=False,
                document_id="",
                filename=clean_name,
                file_type="unsupported",
                file_size_bytes=file_size,
                error_message=f"File extension '{ext}' is not permitted. Allowed: {sorted(self.ALLOWED_EXTENSIONS)}",
            )

        # 5. Verify magic header
        try:
            detected_type = self.detect_file_type(file_bytes)
        except ValueError as e:
            return ValidationResult(
                is_valid=False,
                document_id="",
                filename=clean_name,
                file_type="unknown",
                file_size_bytes=file_size,
                error_message=str(e),
            )

        # 6. Ensure extension matches detected type
        if detected_type == "pdf" and ext != ".pdf":
            return ValidationResult(
                is_valid=False,
                document_id="",
                filename=clean_name,
                file_type=detected_type,
                file_size_bytes=file_size,
                error_message=f"MIME mismatch: file is PDF but has extension '{ext}'.",
            )

        doc_id = self.generate_document_id(file_bytes, clean_name)

        return ValidationResult(
            is_valid=True,
            document_id=doc_id,
            filename=clean_name,
            file_type=detected_type,
            file_size_bytes=file_size,
            error_message=None,
        )
