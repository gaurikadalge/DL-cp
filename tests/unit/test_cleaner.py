"""Unit tests for text cleaner, ligature expansion, and hyphenation repair."""

from ingestion.cleaner import TextCleaner


def test_cleaner_expands_ligatures():
    cleaner = TextCleaner()
    # \ufb01 -> fi, \ufb02 -> fl
    sample = "The \ufb01rst \ufb02ow of gradients was stabilized."
    cleaned = cleaner.clean(sample)
    assert "The first flow of gradients was stabilized." == cleaned


def test_cleaner_repairs_line_broken_hyphens():
    cleaner = TextCleaner()
    # trans-\nformer -> transformer
    sample = "The trans-\nformer model achieves supe-\nrior results on machine translation."
    cleaned = cleaner.clean(sample)
    assert "transformer model achieves superior results" in cleaned


def test_cleaner_preserves_genuine_hyphens():
    cleaner = TextCleaner()
    sample = "We propose a state-of-the-art multi-modal architecture with zero-shot capabilities."
    cleaned = cleaner.clean(sample)
    assert "state-of-the-art" in cleaned
    assert "multi-modal" in cleaned
    assert "zero-shot" in cleaned


def test_cleaner_removes_running_page_numbers_and_arxiv():
    cleaner = TextCleaner()
    sample = """arXiv:2301.12345v1 [cs.CL] 15 Jan 2023

1. Introduction
Deep learning has advanced rapidly.

Page 1 of 12

2. Related Work
Attention mechanisms are widely used.

12"""
    cleaned = cleaner.clean(sample)
    assert "arXiv:2301.12345v1" not in cleaned
    assert "Page 1 of 12" not in cleaned
    assert "Deep learning has advanced rapidly." in cleaned
    assert "Attention mechanisms are widely used." in cleaned


def test_cleaner_normalizes_whitespace_preserving_paragraphs():
    cleaner = TextCleaner()
    sample = """First paragraph line one
    and line two of the first paragraph.

    Second paragraph line one    with   extra  spaces."""
    cleaned = cleaner.clean(sample)
    paragraphs = cleaned.split("\n\n")
    assert len(paragraphs) == 2
    assert paragraphs[0] == "First paragraph line one and line two of the first paragraph."
    assert paragraphs[1] == "Second paragraph line one with extra spaces."
