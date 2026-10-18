# Multimodal Document Ingestion Pipeline Engineering Guide

This guide documents the architecture, implementation details, algorithms, and integration handoffs for the **Multimodal Document Ingestion Subsystem** (Person 3 — Multimodal Processing Lead).

---

## 1. Subsystem Architecture Overview

The ingestion subsystem converts raw academic publications (multi-column PDFs, tables, figures, scanned documents) into clean, structured, and reproducible chunks conforming to `StableChunkMetadata`.

```mermaid
graph TD
    Raw[Uploaded Document: PDF / Image] --> Val[DocumentValidator<br/>Magic Bytes & MIME Check]
    Val --> PDF[PDFExtractor<br/>PyMuPDF Layout & Font Analysis]
    PDF --> ScannedCheck{ScannedPageDetector<br/>Image Coverage > 35% & Chars < 50?}
    ScannedCheck -- Yes --> OCR[OCREngine<br/>Tesseract / EasyOCR / Fallback]
    ScannedCheck -- No --> Blocks[Block & Section Parser<br/>Academic Section Hierarchy]
    PDF --> Tbl[TableExtractor<br/>PyMuPDF find_tables & Markdown]
    PDF --> Fig[FigureExtractor<br/>Image Extraction & Proximity Captions]
    Fig --> Vision[CaptioningEngine<br/>BLIP Vision Model / Heuristics]
    Blocks --> Clean[TextCleaner<br/>Hyphens, Ligatures, Whitespace]
    OCR --> Clean
    Clean --> Chunk[SemanticChunker<br/>Sliding Overlap & Stable IDs]
    Tbl --> Chunk
    Vision --> Chunk
    Chunk --> Out[List[StableChunkMetadata]<br/>Ready for IndexManager]
```

---

## 2. Core Components

### 2.1 File Validation (`ingestion/validator.py`)
- **Magic Header Verification**: Inspects leading bytes to verify authentic file formats, preventing extension spoofing:
  - PDF: `%PDF-` header present in the first 1024 bytes.
  - PNG: `\x89PNG\r\n\x1a\n`
  - JPEG: `\xff\xd8\xff`
  - TIFF: Little-endian `II*\x00` and Big-endian `MM\x00*`
  - WebP: `RIFF` ... `WEBP`
- **File Size Boundaries**: Enforces configurable maximum file sizes (default: 25MB) and rejects empty files (0 bytes).
- **Sanitization & Provenance**: Sanitizes filenames against directory traversal (`../`) and generates deterministic SHA-256 document IDs: `doc_{hash[:12]}`.

### 2.2 Academic Text Cleaning (`ingestion/cleaner.py`)
- **Hyphenation Repair**: Uses regex `(\b[A-Za-z]{2,})-\s*\n\s*([A-Za-z]{2,}\b)` to rejoin words broken across line breaks (e.g. `trans-\nformer` $\to$ `transformer`), while preserving genuine hyphenated compounds (e.g. `state-of-the-art`).
- **Ligature & Typography Expansion**: Normalizes unicode ligatures (`ﬁ` $\to$ `fi`, `ﬂ` $\to$ `fl`, `ﬀ` $\to$ `ff`, `ﬃ` $\to$ `ffi`) and converts typographical quotes/dashes to standard ASCII equivalents.
- **Header & Footer Suppression**: Strips standalone page numbers (`Page 4 of 12`), arXiv stamps (`arXiv:2301.12345v1`), and running publication headers.
- **Intra-Paragraph Whitespace Normalization**: Collapses single erratic newlines within sentences while preserving structural paragraph breaks (`\n\n`).

### 2.3 Scanned Page Detection & OCR (`ingestion/ocr_engine.py`)
- **Heuristic Classifier**: Inspects character density and image bounding area. When selectable character count is below threshold ($<50$ chars) and image area covers $>35\%$ of page dimensions, the page is classified as scanned.
- **Pluggable Engines**:
  - `TesseractOCREngine`: System Tesseract binary wrapper via `pytesseract`.
  - `EasyOCREngine`: Deep-learning OCR with GPU/CPU support.
  - `MockOCREngine`: Deterministic test engine avoiding external dependencies.

### 2.4 Table Extraction (`ingestion/table_extractor.py`)
- **Layout Recognition**: Leverages PyMuPDF `find_tables()` to detect cell coordinates, headers, and data rows.
- **Markdown Conversion**: Formats tables into standard GitHub Flavored Markdown:
  ```markdown
  | Metric | Baseline | Ours |
  | --- | --- | --- |
  | Top-1 Acc | 82.4% | 85.1% |
  ```
- **Spatial Caption Association**: Detects preceding or succeeding table captions matching `r"^Table\s+[0-9]+[:\.\-]"` within 120 points.

### 2.5 Figure Extraction & Vision Captioning (`ingestion/image_extractor.py`)
- **Image Parsing**: Extracts embedded raster and vector illustrations (`fitz.Document.extract_image()`).
- **Noise Filtering**: Eliminates decorative rules, bullets, and icons below minimum dimensions ($80\times 80$ px).
- **Caption Association**: Matches figures with proximate academic text captions (`Figure 1: ...`).
- **Multimodal Captioning**:
  - `BLIPCaptioningEngine`: Zero-shot vision captioning via `Salesforce/blip-image-captioning-base`.
  - `HeuristicCaptioningEngine`: Fast rule-based caption extraction from PDF spatial blocks.
  - `MockCaptioningEngine`: Deterministic test stub.

### 2.6 Section-Aware Semantic Chunker (`ingestion/chunker.py`)
- **Target Size & Overlap**: Default 800 characters (~120 words) with 150 character overlap (~25 words).
- **Boundary Preservation**: Splits along natural paragraph and sentence boundaries, avoiding mid-sentence or mid-word cuts.
- **Section Provenance**: Preserves section headings (`section="3. Methodology"`).
- **Deterministic Stable ID Scheme**: `{document_id}#chunk_{index:04d}` (e.g. `doc_7a8b9c#chunk_0001`).

---

## 3. Integration Hand-Offs

### 3.1 Ingestion $\to$ RAG Retrieval Engine (Person 3 $\to$ Person 1)
```python
from ingestion.pipeline import MultimodalIngestionPipeline
from rag_engine.index_manager import IndexManager

# 1. Process document
pipeline = MultimodalIngestionPipeline()
result = pipeline.process("research_paper.pdf")

# 2. Add chunks directly to IndexManager
index_manager.add_chunks(result.chunks)
```

### 3.2 Ingestion $\to$ FastAPI Lifecycle (Person 3 $\to$ Person 2)
The endpoints in `app/api/documents.py` provide full REST access to the ingestion pipeline:
- `POST /documents/upload`: Multipart upload, validation, parsing, and chunk generation.
- `GET /documents`: List ingested research papers with chunk counts and statuses.
- `GET /documents/{document_id}`: Inspect specific document extraction metadata.
- `DELETE /documents/{document_id}`: Remove document from registry and index.

---

## 4. Verification Evidence

All 74 automated unit and integration tests are verified and passing:
```bash
.venv/bin/pytest -v
======================== 74 passed, 1 warning in 0.84s =========================
```
Code quality verified with zero lint errors:
```bash
.venv/bin/ruff check .
All checks passed!
```
