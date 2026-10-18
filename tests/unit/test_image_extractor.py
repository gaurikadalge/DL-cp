"""Unit tests for figure extraction and multimodal captioning."""

from ingestion.image_extractor import (
    ExtractedFigure,
    HeuristicCaptioningEngine,
    MockCaptioningEngine,
)


def test_heuristic_captioning_engine():
    engine = HeuristicCaptioningEngine()
    caption = engine.generate_caption(b"dummy_bytes", context_text="Figure 1: Transformer encoder architecture.")
    assert caption == "Figure 1: Transformer encoder architecture."


def test_mock_captioning_engine():
    engine = MockCaptioningEngine(canned_caption="Neural architecture diagram.")
    caption = engine.generate_caption(b"dummy_bytes", context_text="Figure 2: Residual Block")
    assert "Figure 2: Residual Block (Neural architecture diagram.)" == caption
    assert engine.call_count == 1


def test_extracted_figure_chunk_text():
    figure = ExtractedFigure(
        figure_id="p1_fig_1",
        page_number=3,
        caption="Figure 3: Attention map heatmaps across heads.",
        ai_description="Heatmap showing self-attention weights correlating tokens.",
        image_bytes=b"dummy_bytes",
        image_format="png",
        width=400,
        height=300,
    )

    chunk_text = figure.to_chunk_text()
    assert "[Figure Caption: Figure 3: Attention map heatmaps across heads.]" in chunk_text
    assert "[Visual Description: Heatmap showing self-attention weights correlating tokens.]" in chunk_text

    meta_dict = figure.to_dict()
    assert meta_dict["width"] == 400
    assert meta_dict["height"] == 300
    assert "image_bytes" not in meta_dict  # Ensures heavy image payload is omitted from metadata dict
