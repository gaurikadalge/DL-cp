"""Figure and image extraction and multimodal captioning for academic PDFs.

Extracts embedded figures and diagrams from PDF pages, associates them
with spatial text captions (e.g., 'Figure 1: ...'), and supports vision
model captioning (BLIP / heuristic fallback).
"""

import io
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger("ingestion.image_extractor")

# Regex to detect academic figure captions (e.g., "Figure 1: Transformer model architecture", "Fig. 2. Confusion matrix")
FIGURE_CAPTION_REGEX = re.compile(
    r"^(?:Figure|Fig\.|FIGURE)\s+([0-9A-Za-z]+)[:\.\-]?\s*(.*)",
    re.IGNORECASE,
)


@dataclass
class ExtractedFigure:
    """Extracted figure representation containing image payload and multimodal captions."""

    figure_id: str
    page_number: int
    caption: str | None
    ai_description: str | None
    image_bytes: bytes
    image_format: str  # "png", "jpeg", etc.
    width: int
    height: int
    bbox: tuple[float, float, float, float] | None = None

    def to_chunk_text(self) -> str:
        """Render figure as a semantically rich text chunk for vector indexing."""
        parts = []
        if self.caption:
            parts.append(f"[Figure Caption: {self.caption}]")
        if self.ai_description and self.ai_description != self.caption:
            parts.append(f"[Visual Description: {self.ai_description}]")
        if not parts:
            parts.append(f"[Figure {self.figure_id} on Page {self.page_number}]")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary without heavy image bytes."""
        return {
            "figure_id": self.figure_id,
            "page_number": self.page_number,
            "caption": self.caption,
            "ai_description": self.ai_description,
            "image_format": self.image_format,
            "width": self.width,
            "height": self.height,
            "bbox": self.bbox,
        }


class CaptioningEngine(ABC):
    """Abstract interface for image captioning vision models."""

    @abstractmethod
    def generate_caption(self, image_input: Image.Image | bytes, context_text: str | None = None) -> str:
        """Generate a natural language description of an image."""
        pass


class BLIPCaptioningEngine(CaptioningEngine):
    """Vision-language captioning model using Salesforce/blip-image-captioning-base."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._processor = None
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from transformers import BlipForConditionalGeneration, BlipProcessor
                logger.info("Loading BLIP captioning model '%s' on %s...", self.model_name, self.device)
                self._processor = BlipProcessor.from_pretrained(self.model_name)
                self._model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            except Exception as e:
                logger.error("Failed to load BLIP model '%s': %s", self.model_name, e)
                raise RuntimeError(f"Unable to load BLIP model '{self.model_name}': {e}") from e

    def generate_caption(self, image_input: Image.Image | bytes, context_text: str | None = None) -> str:
        self._load_model()

        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            image = image_input.convert("RGB")

        inputs = self._processor(image, text=context_text, return_tensors="pt").to(self.device)
        out = self._model.generate(**inputs, max_new_tokens=50)
        caption = self._processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()


class HeuristicCaptioningEngine(CaptioningEngine):
    """Lightweight rule-based captioning using spatial PDF caption text and image dimensions."""

    def generate_caption(self, image_input: Image.Image | bytes, context_text: str | None = None) -> str:
        if context_text and context_text.strip():
            return context_text.strip()
        return "Academic diagram or figure illustration extracted from publication."


class MockCaptioningEngine(CaptioningEngine):
    """Deterministic mock captioning engine for testing."""

    def __init__(self, canned_caption: str = "Diagram illustrating neural network architecture and data flow."):
        self.canned_caption = canned_caption
        self.call_count = 0

    def generate_caption(self, image_input: Image.Image | bytes, context_text: str | None = None) -> str:
        self.call_count += 1
        if context_text:
            return f"{context_text.strip()} ({self.canned_caption})"
        return self.canned_caption


class FigureExtractor:
    """Extracts figures from PDF pages and coordinates spatial caption association."""

    def __init__(
        self,
        min_width: int = 80,
        min_height: int = 80,
        captioning_engine: CaptioningEngine | None = None,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.captioning_engine = captioning_engine or HeuristicCaptioningEngine()

    def _find_nearby_caption(self, page: fitz.Page, figure_bbox: fitz.Rect | None) -> str | None:
        """Search text blocks on page for Figure caption in spatial proximity."""
        blocks = page.get_text("blocks")
        candidate_captions: list[tuple[float, str]] = []

        for b in blocks:
            text = b[4].strip()
            if not text:
                continue

            match = FIGURE_CAPTION_REGEX.search(text)
            if match:
                if figure_bbox is not None:
                    b_rect = fitz.Rect(b[0], b[1], b[2], b[3])
                    # Measure distance
                    if b_rect.y0 >= figure_bbox.y1:  # Below figure (standard for academic figures)
                        dist = b_rect.y0 - figure_bbox.y1
                    elif b_rect.y1 <= figure_bbox.y0:  # Above figure
                        dist = figure_bbox.y0 - b_rect.y1
                    else:
                        dist = 0.0

                    if dist < 180.0:
                        candidate_captions.append((dist, text.splitlines()[0]))
                else:
                    candidate_captions.append((0.0, text.splitlines()[0]))

        if candidate_captions:
            candidate_captions.sort(key=lambda x: x[0])
            return candidate_captions[0][1]

        return None

    def extract_figures_from_page(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_number: int,
    ) -> list[ExtractedFigure]:
        """Extract valid figures from a single PDF page."""
        extracted: list[ExtractedFigure] = []
        images = page.get_images(full=True)

        for idx, img_info in enumerate(images):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                img_format = base_image.get("ext", "png").lower()
                img_bytes = base_image.get("image", b"")

                # Filter decorative rules and tiny icons
                if width < self.min_width or height < self.min_height:
                    continue

                # Locate bounding box on page
                bbox_tuple = None
                rects = page.get_image_rects(xref)
                fig_rect = rects[0] if rects else None
                if fig_rect:
                    bbox_tuple = (fig_rect.x0, fig_rect.y0, fig_rect.x1, fig_rect.y1)

                caption = self._find_nearby_caption(page, fig_rect)
                ai_desc = self.captioning_engine.generate_caption(img_bytes, context_text=caption)

                figure = ExtractedFigure(
                    figure_id=f"p{page_number}_fig_{idx+1}",
                    page_number=page_number,
                    caption=caption,
                    ai_description=ai_desc,
                    image_bytes=img_bytes,
                    image_format=img_format,
                    width=width,
                    height=height,
                    bbox=bbox_tuple,
                )
                extracted.append(figure)
            except Exception as e:
                logger.warning("Failed extracting figure xref %d on page %d: %s", xref, page_number, e)

        return extracted
