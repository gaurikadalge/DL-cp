"""OCR engine abstractions, Tesseract/EasyOCR integration, and scanned page detection.

Provides:
- OCREngine protocol/ABC.
- TesseractOCREngine using pytesseract with binary detection.
- EasyOCREngine with optional lazy loading.
- MockOCREngine for deterministic, dependency-free unit testing.
- ScannedPageDetector to identify image-dominated or scanned PDF pages.
"""

import io
import logging
import shutil
from abc import ABC, abstractmethod

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger("ingestion.ocr_engine")


class OCREngine(ABC):
    """Abstract interface for optical character recognition engines."""

    @abstractmethod
    def extract_text(self, image_input: Image.Image | bytes, lang: str = "eng") -> str:
        """Extract textual content from a PIL Image or raw image bytes."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if engine dependencies and binaries are accessible."""
        pass


class TesseractOCREngine(OCREngine):
    """Tesseract OCR implementation wrapping pytesseract."""

    def __init__(self, tesseract_cmd: str | None = None):
        self._cmd = tesseract_cmd or shutil.which("tesseract")
        self._pytesseract = None

    def is_available(self) -> bool:
        """Check if pytesseract library and tesseract system binary exist."""
        if not self._cmd:
            return False
        import os

        if not (os.path.exists(self._cmd) or shutil.which(self._cmd)):
            return False
        try:
            import pytesseract

            self._pytesseract = pytesseract
            self._pytesseract.pytesseract.tesseract_cmd = self._cmd
            return True
        except ImportError:
            return False

    def extract_text(self, image_input: Image.Image | bytes, lang: str = "eng") -> str:
        """Perform OCR on image using Tesseract."""
        if not self.is_available():
            raise RuntimeError(
                "Tesseract OCR is not available. Please install tesseract-ocr system binary "
                "or enable MockOCREngine for test environments."
            )

        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input))
        else:
            image = image_input

        # Convert to RGB if needed (e.g. RGBA or palette)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        try:
            text = self._pytesseract.image_to_string(image, lang=lang)
            return text.strip()
        except Exception as e:
            logger.error("Tesseract OCR extraction failed: %s", e)
            raise RuntimeError(f"Tesseract OCR failed: {e}") from e


class EasyOCREngine(OCREngine):
    """EasyOCR implementation with lazy loading of reader models."""

    def __init__(self, languages: list | None = None, gpu: bool = False):
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader = None

    def is_available(self) -> bool:
        try:
            import easyocr  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_reader(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def extract_text(self, image_input: Image.Image | bytes, lang: str = "eng") -> str:
        if not self.is_available():
            raise RuntimeError("EasyOCR is not installed. Install via: pip install easyocr")

        if isinstance(image_input, bytes):
            image_bytes = image_input
        else:
            buf = io.BytesIO()
            image_input.save(buf, format="PNG")
            image_bytes = buf.getvalue()

        reader = self._get_reader()
        results = reader.readtext(image_bytes, detail=0)
        return "\n".join(results).strip()


class MockOCREngine(OCREngine):
    """Deterministic mock OCR engine for fast unit tests without external models or binaries."""

    def __init__(
        self, canned_response: str = "OCR Extracted Text: Sample content from scanned document."
    ):
        self.canned_response = canned_response
        self.call_count = 0

    def is_available(self) -> bool:
        return True

    def extract_text(self, image_input: Image.Image | bytes, lang: str = "eng") -> str:
        self.call_count += 1
        return self.canned_response


class ScannedPageDetector:
    """Heuristic detector determining if a PDF page is a scanned image requiring OCR."""

    def __init__(self, char_threshold: int = 50, min_image_area_ratio: float = 0.35):
        self.char_threshold = char_threshold
        self.min_image_area_ratio = min_image_area_ratio

    def is_scanned_page(self, page: fitz.Page) -> bool:
        """Evaluate page textual density and embedded image coverage.

        A page is deemed 'scanned' if:
        1. Extracted text character count is below the threshold, AND
        2. The page contains at least one image or raster graphic covering significant area.
        """
        # 1. Measure direct text length
        text = page.get_text("text").strip()
        if len(text) >= self.char_threshold:
            return False

        # 2. Check embedded images
        image_list = page.get_images(full=True)
        if not image_list:
            # Empty page, but no image to OCR
            return False

        # 3. Compute image bounding area vs page rect area
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        if page_area <= 0:
            return False

        # Estimate image area
        total_image_area = 0.0
        for img in image_list:
            xref = img[0]
            try:
                rects = page.get_image_rects(xref)
                for r in rects:
                    total_image_area += r.width * r.height
            except Exception:
                # If PyMuPDF rect lookup fails, count presence as scanned
                return True

        area_ratio = total_image_area / page_area
        return area_ratio >= self.min_image_area_ratio
