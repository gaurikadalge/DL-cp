"""High-fidelity PDF extraction and layout parsing using PyMuPDF (fitz).

Provides:
- Page-by-page block-level text extraction with bounding boxes.
- Section heading detection using regex patterns and font-size heuristics.
- Document-level metadata extraction (title, authors, page count).
- Scanned page detection with automated OCR fallback orchestration.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from ingestion.cleaner import TextCleaner
from ingestion.ocr_engine import OCREngine, ScannedPageDetector

logger = logging.getLogger("ingestion.pdf_extractor")

# Section heading patterns common in academic publications
SECTION_PATTERNS = [
    # Numbered sections e.g., "1. Introduction", "2.1 Problem Formulation", "3.2.1 Loss Function"
    re.compile(r"^(?:[0-9]+(?:\.[0-9]+)*)\s+([A-Z][A-Za-z0-9\s,\-:]{2,60})$"),
    # Roman numeral sections e.g., "I. INTRODUCTION", "II. RELATED WORK"
    re.compile(r"^(?:[IVXLCDM]+)\.\s+([A-Z][A-Za-z0-9\s,\-:]{2,60})$"),
    # Standard academic unnumbered headings
    re.compile(
        r"^(?:Abstract|Introduction|Related Work|Background|Methodology|Method|Architecture|"
        r"Proposed Approach|Experiments|Experimental Setup|Results|Discussion|Conclusion|"
        r"Conclusions|Future Work|Ablation Studies|Limitations|Broader Impacts|References|Appendix)$",
        re.IGNORECASE,
    ),
]


@dataclass
class TextBlock:
    """Atomic layout block of text extracted from a PDF page."""

    text: str
    bbox: tuple[float, float, float, float]
    page_number: int
    is_heading: bool = False
    section_title: str | None = None
    font_size: float = 10.0


@dataclass
class ExtractedPage:
    """Structured extraction output for a single PDF page."""

    page_number: int
    raw_text: str
    cleaned_text: str
    blocks: list[TextBlock] = field(default_factory=list)
    detected_sections: list[str] = field(default_factory=list)
    is_scanned: bool = False
    ocr_applied: bool = False


@dataclass
class ExtractedDocument:
    """Full structured extraction representation of an academic document."""

    document_id: str
    filename: str
    page_count: int
    metadata: dict[str, Any]
    pages: list[ExtractedPage] = field(default_factory=list)

    def get_full_text(self) -> str:
        """Combine cleaned text across all pages."""
        return "\n\n".join(p.cleaned_text for p in self.pages if p.cleaned_text)


class PDFExtractor:
    """Extracts text, sections, and metadata from academic PDF publications."""

    def __init__(
        self,
        cleaner: TextCleaner | None = None,
        ocr_engine: OCREngine | None = None,
        scanned_detector: ScannedPageDetector | None = None,
        ocr_enabled: bool = False,
    ):
        self.cleaner = cleaner or TextCleaner()
        self.ocr_engine = ocr_engine
        self.scanned_detector = scanned_detector or ScannedPageDetector()
        self.ocr_enabled = ocr_enabled

    @staticmethod
    def _is_section_heading(line: str, font_size: float = 10.0, avg_font_size: float = 10.0) -> bool:
        """Evaluate whether a line represents an academic section heading."""
        stripped = line.strip()
        if not stripped or len(stripped) > 80:
            return False

        # 1. Regex check
        for pattern in SECTION_PATTERNS:
            if pattern.match(stripped):
                return True

        # 2. Font size heuristic: heading typically >= 1.2x body font
        if avg_font_size > 0 and font_size >= 1.25 * avg_font_size:
            # Must look like a title (short line, starts with capital)
            if stripped[0].isupper() and len(stripped.split()) <= 8:
                return True

        return False

    def extract_document(
        self,
        file_input: bytes | str | Path,
        filename: str = "document.pdf",
        document_id: str = "doc_unknown",
    ) -> ExtractedDocument:
        """Extract all pages, blocks, and layout elements from a PDF."""
        # 1. Open document via PyMuPDF
        if isinstance(file_input, bytes):
            doc = fitz.open(stream=file_input, filetype="pdf")
        else:
            doc = fitz.open(str(file_input))

        page_count = len(doc)
        doc_metadata = doc.metadata or {}

        # Capture document-level metadata
        clean_metadata = {
            "title": doc_metadata.get("title") or filename,
            "author": doc_metadata.get("author") or "Unknown",
            "subject": doc_metadata.get("subject") or "",
            "keywords": doc_metadata.get("keywords") or "",
            "page_count": page_count,
            "format": doc_metadata.get("format") or "PDF",
        }

        pages: list[ExtractedPage] = []
        current_section: str | None = None

        for page_idx in range(page_count):
            page_num = page_idx + 1
            page = doc[page_idx]

            # 2. Scanned page detection
            is_scanned = self.scanned_detector.is_scanned_page(page)
            ocr_applied = False
            page_text = ""
            text_blocks: list[TextBlock] = []
            page_sections: list[str] = []

            if is_scanned and self.ocr_enabled and self.ocr_engine and self.ocr_engine.is_available():
                # Render page to pixmap image and run OCR
                try:
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    page_text = self.ocr_engine.extract_text(img_bytes)
                    ocr_applied = True
                    text_blocks.append(
                        TextBlock(
                            text=page_text,
                            bbox=(0.0, 0.0, page.rect.width, page.rect.height),
                            page_number=page_num,
                            section_title=current_section,
                        )
                    )
                except Exception as e:
                    logger.warning("OCR failed on scanned page %d: %s", page_num, e)

            if not ocr_applied:
                # 3. Direct PyMuPDF block extraction
                # Compute average font size on page for heading heuristic
                font_sizes: list[float] = []
                page_dict = page.get_text("dict")
                for b in page_dict.get("blocks", []):
                    if b.get("type") == 0:  # Text block
                        for line in b.get("lines", []):
                            for s in line.get("spans", []):
                                font_sizes.append(s.get("size", 10.0))

                avg_font = (sum(font_sizes) / len(font_sizes)) if font_sizes else 10.0

                # Extract blocks with position and heading tag
                blocks = page.get_text("blocks")
                block_texts: list[str] = []

                for b in blocks:
                    # b: (x0, y0, x1, y1, text, block_no, block_type)
                    if len(b) >= 5:
                        raw_block_text = b[4].strip()
                        if not raw_block_text:
                            continue

                        # Check if first line of block is a section heading
                        first_line = raw_block_text.splitlines()[0].strip()
                        is_heading = self._is_section_heading(first_line, avg_font_size=avg_font)

                        if is_heading:
                            current_section = first_line
                            page_sections.append(current_section)

                        tb = TextBlock(
                            text=raw_block_text,
                            bbox=(b[0], b[1], b[2], b[3]),
                            page_number=page_num,
                            is_heading=is_heading,
                            section_title=current_section,
                            font_size=avg_font,
                        )
                        text_blocks.append(tb)
                        block_texts.append(raw_block_text)

                page_text = "\n\n".join(block_texts)

            # 4. Clean page text
            cleaned_page_text = self.cleaner.clean(page_text)

            pages.append(
                ExtractedPage(
                    page_number=page_num,
                    raw_text=page_text,
                    cleaned_text=cleaned_page_text,
                    blocks=text_blocks,
                    detected_sections=page_sections,
                    is_scanned=is_scanned,
                    ocr_applied=ocr_applied,
                )
            )

        doc.close()

        return ExtractedDocument(
            document_id=document_id,
            filename=filename,
            page_count=page_count,
            metadata=clean_metadata,
            pages=pages,
        )
