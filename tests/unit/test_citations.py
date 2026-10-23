"""Unit tests for CitationValidator."""

from app.schemas import StableChunkMetadata
from rag_engine.citations import CitationValidator, extract_citation_snippet


def sample_chunk(
    cid: str = "chunk_01", text: str = "A detailed research passage on neural networks."
):
    return StableChunkMetadata(
        chunk_id=cid,
        document_id="doc_res",
        text=text,
        page_number=5,
        section="3. Methodology",
        source_filename="resnet.pdf",
    )


def test_snippet_extraction():
    """Verify snippet extraction cleans whitespace and trims appropriately."""
    long_text = "Sentence one. " * 30
    snippet = extract_citation_snippet(long_text, max_chars=100)
    assert len(snippet) <= 103  # with ellipsis or dot
    assert "Sentence one." in snippet


def test_valid_citation_resolution():
    """Verify inline tags [1] resolve to true source chunk metadata."""
    chunk = sample_chunk()
    citation_map = {"[1]": chunk}
    answer = "The model achieves state of the art results [1]."

    citations, unresolvable = CitationValidator.validate_citations(answer, citation_map)

    assert len(citations) == 1
    assert len(unresolvable) == 0
    c = citations[0]
    assert c.citation_id == "[1]"
    assert c.chunk_id == "chunk_01"
    assert c.source_filename == "resnet.pdf"
    assert c.page_number == 5
    assert c.section == "3. Methodology"
    assert "Sentence" in c.snippet or "passage" in c.snippet


def test_unresolvable_citation_rejection():
    """Verify hallucinated tags like [99] are rejected without fabricating metadata."""
    chunk = sample_chunk()
    citation_map = {"[1]": chunk}
    answer = "Claim supported by real evidence [1], and another unsupported claim [99]."

    citations, unresolvable = CitationValidator.validate_citations(answer, citation_map)

    assert len(citations) == 1
    assert citations[0].citation_id == "[1]"
    assert "[99]" in unresolvable


def test_empty_answer_citations():
    """Verify empty answer returns empty citation list."""
    citations, unresolvable = CitationValidator.validate_citations("", {})
    assert citations == []
    assert unresolvable == set()
