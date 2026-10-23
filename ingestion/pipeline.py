"""Unified Multimodal Document Ingestion Pipeline.

Orchestrates:
1. Document validation & magic header verification.
2. Layout-aware PDF text extraction with PyMuPDF.
3. Scanned page detection with automated OCR fallback.
4. Tabular data extraction & Markdown formatting.
5. Figure extraction & vision model captioning.
6. Academic text cleaning & hyphenation repair.
7. Section-aware semantic chunking into StableChunkMetadata.
8. Direct handoff integration to RAG IndexManager.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from app.schemas import StableChunkMetadata
from ingestion.chunker import SemanticChunker
from ingestion.cleaner import TextCleaner
from ingestion.image_extractor import (
    CaptioningEngine,
    ExtractedFigure,
    FigureExtractor,
    HeuristicCaptioningEngine,
)
from ingestion.ocr_engine import OCREngine, ScannedPageDetector
from ingestion.pdf_extractor import ExtractedDocument, PDFExtractor
from ingestion.table_extractor import ExtractedTable, TableExtractor
from ingestion.validator import DocumentValidator, ValidationResult

logger = logging.getLogger("ingestion.pipeline")


@dataclass
class IngestionResult:
    """Consolidated outcome of the multimodal document ingestion process."""

    document_id: str
    filename: str
    page_count: int
    validation: ValidationResult
    extracted_doc: ExtractedDocument
    tables: list[ExtractedTable] = field(default_factory=list)
    figures: list[ExtractedFigure] = field(default_factory=list)
    chunks: list[StableChunkMetadata] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def summary(self) -> dict[str, Any]:
        """Return executive summary metrics for monitoring."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_count": self.page_count,
            "chunks_count": len(self.chunks),
            "tables_count": len(self.tables),
            "figures_count": len(self.figures),
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


class MultimodalIngestionPipeline:
    """Master orchestrator for processing multimodal academic research documents."""

    def __init__(
        self,
        validator: DocumentValidator | None = None,
        cleaner: TextCleaner | None = None,
        ocr_engine: OCREngine | None = None,
        captioning_engine: CaptioningEngine | None = None,
        chunker: SemanticChunker | None = None,
        ocr_enabled: bool = False,
        extract_tables: bool = True,
        extract_figures: bool = True,
    ):
        self.validator = validator or DocumentValidator()
        self.cleaner = cleaner or TextCleaner()
        self.ocr_engine = ocr_engine
        self.captioning_engine = captioning_engine or HeuristicCaptioningEngine()
        self.chunker = chunker or SemanticChunker()
        self.ocr_enabled = ocr_enabled
        self.extract_tables = extract_tables
        self.extract_figures = extract_figures

        # Sub-modules
        self.scanned_detector = ScannedPageDetector()
        self.pdf_extractor = PDFExtractor(
            cleaner=self.cleaner,
            ocr_engine=self.ocr_engine,
            scanned_detector=self.scanned_detector,
            ocr_enabled=self.ocr_enabled,
        )
        self.table_extractor = TableExtractor()
        self.figure_extractor = FigureExtractor(
            captioning_engine=self.captioning_engine,
        )

    def process(
        self,
        file_input: bytes | str | Path,
        filename: str | None = None,
        document_id: str | None = None,
    ) -> IngestionResult:
        """Process an uploaded document through the complete multimodal ingestion pipeline."""
        start_time = time.perf_counter()

        # 1. Validation
        val_result = self.validator.validate(file_input, filename=filename)
        if not val_result.is_valid:
            raise ValueError(f"Document validation failed: {val_result.error_message}")

        doc_id = document_id or val_result.document_id
        safe_filename = val_result.filename

        # Resolve raw bytes for downstream extraction
        if isinstance(file_input, (str, Path)):
            file_bytes = Path(file_input).read_bytes()
        else:
            file_bytes = file_input

        # 2. PDF text & layout extraction
        extracted_doc = self.pdf_extractor.extract_document(
            file_input=file_bytes,
            filename=safe_filename,
            document_id=doc_id,
        )

        # 3. Multimodal extractions (tables & figures)
        tables: list[ExtractedTable] = []
        figures: list[ExtractedFigure] = []

        if self.extract_tables or self.extract_figures:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            try:
                for page_idx in range(len(doc)):
                    page_num = page_idx + 1
                    page = doc[page_idx]

                    if self.extract_tables:
                        page_tables = self.table_extractor.extract_tables_from_page(page, page_num)
                        tables.extend(page_tables)

                    if self.extract_figures:
                        page_figures = self.figure_extractor.extract_figures_from_page(
                            doc, page, page_num
                        )
                        figures.extend(page_figures)
            finally:
                doc.close()

        # 4. Semantic chunking
        chunks = self.chunker.chunk_document(
            extracted_doc=extracted_doc,
            tables=tables,
            figures=figures,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Ingestion pipeline completed for '%s' in %.1fms: %d pages, %d chunks, %d tables, %d figures",
            safe_filename,
            elapsed_ms,
            extracted_doc.page_count,
            len(chunks),
            len(tables),
            len(figures),
        )

        return IngestionResult(
            document_id=doc_id,
            filename=safe_filename,
            page_count=extracted_doc.page_count,
            validation=val_result,
            extracted_doc=extracted_doc,
            tables=tables,
            figures=figures,
            chunks=chunks,
            execution_time_ms=elapsed_ms,
        )

    def ingest_and_index(
        self,
        file_input: bytes | str | Path,
        index_manager: Any,
        filename: str | None = None,
        document_id: str | None = None,
    ) -> IngestionResult:
        """Process document and immediately pass output chunks to RAG IndexManager."""
        result = self.process(file_input=file_input, filename=filename, document_id=document_id)
        if result.chunks and hasattr(index_manager, "add_chunks"):
            index_manager.add_chunks(result.chunks)
            logger.info(
                "Synchronized %d chunks from '%s' into IndexManager",
                len(result.chunks),
                result.filename,
            )
        return result

    def process_directory(self, directory_path: str | Path) -> list[IngestionResult]:
        """Batch-process all permitted documents found in a directory."""
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Provided path is not a directory: {directory_path}")

        results: list[IngestionResult] = []
        for file_path in sorted(dir_path.iterdir()):
            if file_path.suffix.lower() in DocumentValidator.ALLOWED_EXTENSIONS:
                try:
                    res = self.process(file_path, filename=file_path.name)
                    results.append(res)
                except Exception as e:
                    logger.warning(
                        "Batch processing skipped failed file '%s': %s", file_path.name, e
                    )

        return results
