"""Table extraction and structure recognition for academic PDFs.

Extracts tables from PDF pages using PyMuPDF (fitz) table finder with
pdfplumber fallback. Converts tabular layouts into normalized Markdown
and structured JSON representations with associated table captions.
"""

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger("ingestion.table_extractor")

# Regex to detect academic table caption lines (e.g., "Table 1: Performance comparison", "Table 2. Summary")
TABLE_CAPTION_REGEX = re.compile(
    r"^(?:Table|TABLE)\s+([0-9A-Za-z]+)[:\.\-]?\s*(.*)",
    re.IGNORECASE,
)


@dataclass
class ExtractedTable:
    """Structured representation of an extracted academic table."""

    table_id: str
    page_number: int
    title_or_caption: str | None
    headers: list[str]
    rows: list[list[str]]
    markdown: str
    bbox: tuple[float, float, float, float] | None = None

    def to_chunk_text(self) -> str:
        """Render table as a readable, context-rich text chunk for RAG indexing."""
        header_line = f"[Table: {self.title_or_caption}]" if self.title_or_caption else "[Table]"
        return f"{header_line}\n\n{self.markdown}".strip()

    def to_dict(self) -> dict[str, Any]:
        """Serialize table to dictionary representation."""
        return asdict(self)


class TableExtractor:
    """Extracts tabular data from PDF documents using PyMuPDF and pdfplumber."""

    def __init__(self, min_rows: int = 2, min_cols: int = 2):
        self.min_rows = min_rows
        self.min_cols = min_cols

    @staticmethod
    def _cells_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
        """Convert headers and rows into clean Markdown table format."""
        if not headers and not rows:
            return ""

        # Normalize column count
        col_count = len(headers) if headers else max(len(r) for r in rows)
        norm_headers = [headers[i].strip() if i < len(headers) else f"Col {i+1}" for i in range(col_count)]

        # Header line
        md_lines = ["| " + " | ".join(h if h else "-" for h in norm_headers) + " |"]
        # Separator line
        md_lines.append("| " + " | ".join("---" for _ in range(col_count)) + " |")

        # Data rows
        for row in rows:
            norm_row = [row[i].strip().replace("\n", " ") if i < len(row) else "" for i in range(col_count)]
            md_lines.append("| " + " | ".join(norm_row) + " |")

        return "\n".join(md_lines)

    def _find_nearby_caption(self, page: fitz.Page, table_bbox: fitz.Rect) -> str | None:
        """Search for table caption located immediately above or below the table."""
        # Inspect text blocks on the page
        blocks = page.get_text("blocks")
        candidate_captions: list[tuple[float, str]] = []

        for b in blocks:
            text = b[4].strip()
            if not text:
                continue

            match = TABLE_CAPTION_REGEX.search(text)
            if match:
                b_rect = fitz.Rect(b[0], b[1], b[2], b[3])
                # Compute vertical distance between table and caption
                if b_rect.y1 <= table_bbox.y0:  # Above table
                    dist = table_bbox.y0 - b_rect.y1
                elif b_rect.y0 >= table_bbox.y1:  # Below table
                    dist = b_rect.y0 - table_bbox.y1
                else:
                    dist = 0.0

                if dist < 120.0:  # Within 120 points distance
                    candidate_captions.append((dist, text.splitlines()[0]))

        if candidate_captions:
            # Pick caption with closest vertical distance
            candidate_captions.sort(key=lambda x: x[0])
            return candidate_captions[0][1]

        return None

    def extract_tables_from_page(self, page: fitz.Page, page_number: int) -> list[ExtractedTable]:
        """Extract all tables detected on a specific PDF page."""
        extracted: list[ExtractedTable] = []

        # 1. Primary: PyMuPDF find_tables()
        try:
            tabs = page.find_tables()
            for idx, tab in enumerate(tabs):
                raw_data = tab.extract()
                if not raw_data or len(raw_data) < self.min_rows:
                    continue

                # Header is typically first row
                raw_headers = [str(c or "").strip() for c in raw_data[0]]
                raw_rows = [[str(c or "").strip() for c in row] for row in raw_data[1:]]

                # Check minimum column threshold
                if len(raw_headers) < self.min_cols:
                    continue

                bbox_rect = fitz.Rect(tab.bbox)
                caption = self._find_nearby_caption(page, bbox_rect)
                md = self._cells_to_markdown(raw_headers, raw_rows)

                extracted.append(
                    ExtractedTable(
                        table_id=f"p{page_number}_tbl_{idx+1}",
                        page_number=page_number,
                        title_or_caption=caption,
                        headers=raw_headers,
                        rows=raw_rows,
                        markdown=md,
                        bbox=(tab.bbox[0], tab.bbox[1], tab.bbox[2], tab.bbox[3]),
                    )
                )
        except Exception as e:
            logger.warning("PyMuPDF table extraction encountered an issue on page %d: %s", page_number, e)

        return extracted
