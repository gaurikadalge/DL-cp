"""Evidence context selection and formatting with prompt-injection defense."""

import logging

from app.schemas import RetrievedEvidence, StableChunkMetadata

logger = logging.getLogger("rag_engine.context_builder")

# Approximate token estimation: ~4 characters per token for English academic prose
CHARS_PER_TOKEN = 4


def sanitize_evidence_text(text: str) -> str:
    """Sanitize retrieved text to prevent prompt injection / instruction hijacking."""
    # Defang common injection phrases while preserving semantic academic content
    sanitized = text.replace("```", "'''")
    # Defang system prompt escape sequences
    sanitized = sanitized.replace("<|im_start|>", "[token_escaped]")
    sanitized = sanitized.replace("<|im_end|>", "[token_escaped]")
    return sanitized.strip()


class ContextBuilder:
    """Builds token-budgeted, structured evidence context blocks for grounded LLM generation."""

    def __init__(self, token_budget: int = 2048):
        self.token_budget = token_budget
        self.char_budget = token_budget * CHARS_PER_TOKEN

    def build_context(
        self,
        evidence_list: list[RetrievedEvidence],
    ) -> tuple[str, dict[str, StableChunkMetadata]]:
        """Select evidence chunks within token budget and format them with numbered citation keys.

        Returns:
            Tuple of:
            - formatted_context_str: Formatted string ready for prompt injection.
            - citation_map: Dict mapping citation tags (e.g., "[1]") to StableChunkMetadata.
        """
        if not evidence_list:
            return "", {}

        formatted_blocks: list[str] = []
        citation_map: dict[str, StableChunkMetadata] = {}
        seen_chunks = set()
        accumulated_chars = 0

        citation_idx = 1
        for item in evidence_list:
            chunk = item.chunk
            if chunk.chunk_id in seen_chunks:
                continue

            clean_text = sanitize_evidence_text(chunk.text)
            if not clean_text:
                continue

            tag = f"[{citation_idx}]"
            section_str = f", Section: {chunk.section}" if chunk.section else ""
            page_str = (
                f"Page {chunk.page_number}"
                if chunk.page_end is None or chunk.page_end == chunk.page_number
                else f"Pages {chunk.page_number}-{chunk.page_end}"
            )

            block = (
                f"--- BEGIN EVIDENCE {tag} ---\n"
                f"ID: {chunk.chunk_id}\n"
                f"Source: {chunk.source_filename} ({page_str}{section_str})\n"
                f"Content:\n{clean_text}\n"
                f"--- END EVIDENCE {tag} ---"
            )

            block_chars = len(block)
            if accumulated_chars + block_chars > self.char_budget and formatted_blocks:
                logger.info(
                    "Context budget reached (%d chars / ~%d tokens). Selected %d chunks.",
                    accumulated_chars,
                    accumulated_chars // CHARS_PER_TOKEN,
                    len(formatted_blocks),
                )
                break

            formatted_blocks.append(block)
            citation_map[tag] = chunk
            seen_chunks.add(chunk.chunk_id)
            accumulated_chars += block_chars
            citation_idx += 1

        context_str = "\n\n".join(formatted_blocks)
        return context_str, citation_map
