"""Semantic and section-aware chunker with sliding overlap and deterministic IDs.

Generates StableChunkMetadata objects conforming strictly to the repository
data contract for embedding, FAISS indexing, and BM25 retrieval.
"""

import logging
import re
from typing import Any

from app.schemas import StableChunkMetadata
from ingestion.image_extractor import ExtractedFigure
from ingestion.pdf_extractor import ExtractedDocument
from ingestion.table_extractor import ExtractedTable

logger = logging.getLogger("ingestion.chunker")

# Sentence boundary regex for safe splitting
SENTENCE_END_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class SemanticChunker:
    """Chunks academic documents preserving section semantics, paragraph boundaries, and page provenance."""

    def __init__(
        self,
        target_chunk_size: int = 800,
        chunk_overlap: int = 150,
        min_chunk_size: int = 80,
    ):
        if target_chunk_size <= 0:
            raise ValueError(f"target_chunk_size must be positive, got {target_chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= target_chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than target_chunk_size ({target_chunk_size})"
            )

        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences cleanly."""
        paragraphs = text.split("\n\n")
        sentences = []
        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue
            p_sents = SENTENCE_END_REGEX.split(p_clean)
            for s in p_sents:
                s_stripped = s.strip()
                if s_stripped:
                    sentences.append(s_stripped)
        return sentences

    def _chunk_text_sequence(
        self,
        text: str,
        document_id: str,
        source_filename: str,
        start_page: int,
        end_page: int | None = None,
        section: str | None = None,
        content_type: str = "text",
        extraction_method: str = "pymupdf",
        starting_index: int = 1,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[StableChunkMetadata]:
        """Split a string into overlapping chunks honoring sentence boundaries."""
        cleaned_text = text.strip()
        if not cleaned_text:
            return []

        # If text is already smaller than or equal to target chunk size, return single chunk
        if len(cleaned_text) <= self.target_chunk_size:
            chunk_id = f"{document_id}#chunk_{starting_index:04d}"
            return [
                StableChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=cleaned_text,
                    page_number=start_page,
                    page_end=end_page or start_page,
                    section=section,
                    content_type=content_type,
                    source_filename=source_filename,
                    extraction_method=extraction_method,
                    metadata={
                        "char_count": len(cleaned_text),
                        "token_count": len(cleaned_text.split()),
                        **(extra_metadata or {}),
                    },
                )
            ]

        # Break text into sentence units
        sentences = self._split_into_sentences(cleaned_text)
        if not sentences:
            sentences = [cleaned_text]

        chunks: list[StableChunkMetadata] = []
        current_sentences: list[str] = []
        current_len = 0
        chunk_idx = starting_index

        for sent in sentences:
            sent_len = len(sent)

            # If adding this sentence exceeds target and we already have content
            if current_len + sent_len > self.target_chunk_size and current_sentences:
                chunk_str = " ".join(current_sentences).strip()
                if len(chunk_str) >= self.min_chunk_size:
                    cid = f"{document_id}#chunk_{chunk_idx:04d}"
                    chunks.append(
                        StableChunkMetadata(
                            chunk_id=cid,
                            document_id=document_id,
                            text=chunk_str,
                            page_number=start_page,
                            page_end=end_page or start_page,
                            section=section,
                            content_type=content_type,
                            source_filename=source_filename,
                            extraction_method=extraction_method,
                            metadata={
                                "char_count": len(chunk_str),
                                "token_count": len(chunk_str.split()),
                                **(extra_metadata or {}),
                            },
                        )
                    )
                    chunk_idx += 1

                # Calculate overlap: retain tail sentences that fit within chunk_overlap
                overlap_sentences: list[str] = []
                overlap_len = 0
                for prev_sent in reversed(current_sentences):
                    if overlap_len + len(prev_sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev_sent)
                        overlap_len += len(prev_sent)
                    else:
                        break

                current_sentences = overlap_sentences
                current_len = sum(len(s) for s in current_sentences)

            current_sentences.append(sent)
            current_len += sent_len

        # Flush residual content
        if current_sentences:
            residual_str = " ".join(current_sentences).strip()
            if len(residual_str) >= self.min_chunk_size or not chunks:
                cid = f"{document_id}#chunk_{chunk_idx:04d}"
                chunks.append(
                    StableChunkMetadata(
                        chunk_id=cid,
                        document_id=document_id,
                        text=residual_str,
                        page_number=start_page,
                        page_end=end_page or start_page,
                        section=section,
                        content_type=content_type,
                        source_filename=source_filename,
                        extraction_method=extraction_method,
                        metadata={
                            "char_count": len(residual_str),
                            "token_count": len(residual_str.split()),
                            **(extra_metadata or {}),
                        },
                    )
                )

        return chunks

    def chunk_document(
        self,
        extracted_doc: ExtractedDocument,
        tables: list[ExtractedTable] | None = None,
        figures: list[ExtractedFigure] | None = None,
    ) -> list[StableChunkMetadata]:
        """Synthesize all document modalities (text, sections, tables, figures) into stable ordered chunks."""
        all_chunks: list[StableChunkMetadata] = []
        global_idx = 1

        # 1. Chunk text page-by-page, respecting section headings
        for page in extracted_doc.pages:
            page_num = page.page_number
            method = "tesseract" if page.ocr_applied else "pymupdf"
            ctype = "ocr" if page.ocr_applied else "text"

            # Group blocks by section
            current_sec: str | None = None
            sec_texts: list[str] = []

            for block in page.blocks:
                # If block specifies a section and current_sec not yet initialized
                if current_sec is None and block.section_title:
                    current_sec = block.section_title

                if block.is_heading and block.section_title and block.section_title != current_sec:
                    # Flush previous section text
                    if sec_texts:
                        combined = "\n\n".join(sec_texts)
                        sec_chunks = self._chunk_text_sequence(
                            text=combined,
                            document_id=extracted_doc.document_id,
                            source_filename=extracted_doc.filename,
                            start_page=page_num,
                            section=current_sec,
                            content_type=ctype,
                            extraction_method=method,
                            starting_index=global_idx,
                        )
                        all_chunks.extend(sec_chunks)
                        global_idx += len(sec_chunks)
                        sec_texts = []

                    current_sec = block.section_title

                sec_texts.append(block.text)

            # Flush trailing section on page
            if sec_texts:
                combined = "\n\n".join(sec_texts)
                sec_chunks = self._chunk_text_sequence(
                    text=combined,
                    document_id=extracted_doc.document_id,
                    source_filename=extracted_doc.filename,
                    start_page=page_num,
                    section=current_sec,
                    content_type=ctype,
                    extraction_method=method,
                    starting_index=global_idx,
                )
                all_chunks.extend(sec_chunks)
                global_idx += len(sec_chunks)

        # 2. Append tabular chunks
        if tables:
            for tbl in tables:
                tbl_text = tbl.to_chunk_text()
                tbl_chunk = StableChunkMetadata(
                    chunk_id=f"{extracted_doc.document_id}#chunk_{global_idx:04d}",
                    document_id=extracted_doc.document_id,
                    text=tbl_text,
                    page_number=tbl.page_number,
                    page_end=tbl.page_number,
                    section=tbl.title_or_caption or "Table",
                    content_type="table",
                    source_filename=extracted_doc.filename,
                    extraction_method="pymupdf_table",
                    metadata={
                        "table_id": tbl.table_id,
                        "rows": len(tbl.rows),
                        "cols": len(tbl.headers),
                        "has_table": True,
                    },
                )
                all_chunks.append(tbl_chunk)
                global_idx += 1

        # 3. Append figure / multimodal caption chunks
        if figures:
            for fig in figures:
                fig_text = fig.to_chunk_text()
                fig_chunk = StableChunkMetadata(
                    chunk_id=f"{extracted_doc.document_id}#chunk_{global_idx:04d}",
                    document_id=extracted_doc.document_id,
                    text=fig_text,
                    page_number=fig.page_number,
                    page_end=fig.page_number,
                    section=fig.caption or "Figure",
                    content_type="caption",
                    source_filename=extracted_doc.filename,
                    extraction_method="pymupdf_image",
                    metadata={
                        "figure_id": fig.figure_id,
                        "dimensions": f"{fig.width}x{fig.height}",
                        "has_image": True,
                    },
                )
                all_chunks.append(fig_chunk)
                global_idx += 1

        logger.info(
            "Created %d stable chunks for document '%s' (ID: %s)",
            len(all_chunks),
            extracted_doc.filename,
            extracted_doc.document_id,
        )
        return all_chunks
