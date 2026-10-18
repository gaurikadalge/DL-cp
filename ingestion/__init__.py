"""Multimodal Document Processing and Ingestion Package (Person 3).

Provides:
- DocumentValidator: MIME, magic byte, and file integrity validation.
- TextCleaner: Academic hyphenation repair, ligature expansion, header suppression.
- PDFExtractor: Layout-aware extraction using PyMuPDF (fitz) with section detection.
- OCREngine, TesseractOCREngine, EasyOCREngine, MockOCREngine: Dual OCR support.
- ScannedPageDetector: Scanned vs digital PDF page heuristic classification.
- TableExtractor, ExtractedTable: Table structure recognition and Markdown conversion.
- FigureExtractor, ExtractedFigure: Figure extraction, proximity caption association.
- CaptioningEngine, BLIPCaptioningEngine, HeuristicCaptioningEngine, MockCaptioningEngine: Vision captioning.
- SemanticChunker: Section-aware chunking with sliding overlap and stable ID generation.
- MultimodalIngestionPipeline, IngestionResult: Master end-to-end ingestion pipeline.
"""

from ingestion.chunker import SemanticChunker
from ingestion.cleaner import TextCleaner
from ingestion.image_extractor import (
    BLIPCaptioningEngine,
    CaptioningEngine,
    ExtractedFigure,
    FigureExtractor,
    HeuristicCaptioningEngine,
    MockCaptioningEngine,
)
from ingestion.ocr_engine import (
    EasyOCREngine,
    MockOCREngine,
    OCREngine,
    ScannedPageDetector,
    TesseractOCREngine,
)
from ingestion.pdf_extractor import (
    ExtractedDocument,
    ExtractedPage,
    PDFExtractor,
    TextBlock,
)
from ingestion.pipeline import IngestionResult, MultimodalIngestionPipeline
from ingestion.table_extractor import ExtractedTable, TableExtractor
from ingestion.validator import DocumentValidator, ValidationResult

__all__ = [
    "DocumentValidator",
    "ValidationResult",
    "TextCleaner",
    "OCREngine",
    "TesseractOCREngine",
    "EasyOCREngine",
    "MockOCREngine",
    "ScannedPageDetector",
    "PDFExtractor",
    "TextBlock",
    "ExtractedPage",
    "ExtractedDocument",
    "TableExtractor",
    "ExtractedTable",
    "FigureExtractor",
    "ExtractedFigure",
    "CaptioningEngine",
    "BLIPCaptioningEngine",
    "HeuristicCaptioningEngine",
    "MockCaptioningEngine",
    "SemanticChunker",
    "MultimodalIngestionPipeline",
    "IngestionResult",
]
