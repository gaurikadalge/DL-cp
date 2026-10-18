"""Unit tests for semantic chunker, section preservation, and deterministic chunk IDs."""

import pytest

from ingestion.chunker import SemanticChunker
from ingestion.image_extractor import ExtractedFigure
from ingestion.pdf_extractor import ExtractedDocument, ExtractedPage, TextBlock
from ingestion.table_extractor import ExtractedTable


def test_chunker_parameter_validation():
    with pytest.raises(ValueError, match="positive"):
        SemanticChunker(target_chunk_size=-10)

    with pytest.raises(ValueError, match="strictly less"):
        SemanticChunker(target_chunk_size=500, chunk_overlap=550)


def test_chunker_deterministic_stable_ids():
    chunker = SemanticChunker(target_chunk_size=200, chunk_overlap=50)

    text = (
        "Academic publications are densely packed with information. "
        "Residual connections help solve the vanishing gradient problem in deep networks. "
        "Attention mechanisms enable parallel sequence processing. "
        "Large language models benefit from instruction tuning and reinforcement learning from human feedback."
    )

    doc = ExtractedDocument(
        document_id="doc_test123",
        filename="test_paper.pdf",
        page_count=1,
        metadata={"title": "Test Paper"},
        pages=[
            ExtractedPage(
                page_number=1,
                raw_text=text,
                cleaned_text=text,
                blocks=[
                    TextBlock(
                        text=text,
                        bbox=(0, 0, 100, 100),
                        page_number=1,
                        is_heading=False,
                        section_title="1. Introduction",
                    )
                ],
            )
        ],
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2

    # Check deterministic naming scheme: {document_id}#chunk_{idx:04d}
    assert chunks[0].chunk_id == "doc_test123#chunk_0001"
    assert chunks[1].chunk_id == "doc_test123#chunk_0002"
    assert chunks[0].document_id == "doc_test123"
    assert chunks[0].source_filename == "test_paper.pdf"
    assert chunks[0].page_number == 1
    assert chunks[0].section == "1. Introduction"


def test_chunker_includes_tables_and_figures():
    chunker = SemanticChunker(target_chunk_size=500, chunk_overlap=100)

    doc = ExtractedDocument(
        document_id="doc_multimodal",
        filename="multimodal.pdf",
        page_count=2,
        metadata={},
        pages=[
            ExtractedPage(
                page_number=1,
                raw_text="Introduction text.",
                cleaned_text="Introduction text.",
                blocks=[
                    TextBlock(
                        text="Introduction text.",
                        bbox=(0, 0, 50, 50),
                        page_number=1,
                        section_title="Intro",
                    )
                ],
            )
        ],
    )

    table = ExtractedTable(
        table_id="p1_tbl_1",
        page_number=1,
        title_or_caption="Table 1: Accuracy Benchmark",
        headers=["Model", "Score"],
        rows=[["ResNet", "92.1"]],
        markdown="| Model | Score |\n| --- | --- |\n| ResNet | 92.1 |",
    )

    figure = ExtractedFigure(
        figure_id="p2_fig_1",
        page_number=2,
        caption="Figure 1: Training curve",
        ai_description="Loss curve descending over epochs",
        image_bytes=b"bytes",
        image_format="png",
        width=200,
        height=150,
    )

    chunks = chunker.chunk_document(doc, tables=[table], figures=[figure])

    # Text chunk + Table chunk + Figure chunk
    assert len(chunks) == 3
    assert chunks[0].content_type == "text"
    assert chunks[1].content_type == "table"
    assert chunks[1].metadata.get("has_table") is True
    assert chunks[2].content_type == "caption"
    assert chunks[2].metadata.get("has_image") is True
