"""Academic document text cleaner and normalizer.

Repairs common PDF extraction artifacts:
- Hyphenated words split across line breaks.
- Typographical ligatures (fi, fl, ffi, etc.).
- Running headers, footers, page numbering, and arXiv stamps.
- Erratic intra-paragraph line breaks and redundant whitespace.
- Unicode mathematical and quotation mark normalization.
"""

import re
import unicodedata
from re import Pattern

# Typographical ligature translation table
LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2014": " -- ",
    "\u2013": "-",
    "\u00a0": " ",  # Non-breaking space
    "\u200b": "",  # Zero-width space
    "\ufeff": "",  # Byte-order mark
}

# Regex for line-split hyphenated words: word- \n nextword -> wordnextword
# Ensures genuine hyphenated words (e.g., "state-of-the-art") are preserved if on same line
HYPHENATION_REGEX: Pattern = re.compile(r"(\b[A-Za-z]{2,})-\s*\n\s*([A-Za-z]{2,}\b)")

# Common running headers/footers in research papers
PAGE_NUMBER_FOOTER_REGEX: Pattern = re.compile(
    r"^(?:page\s+[0-9]+(?:\s+of\s+[0-9]+)?|[0-9]+\s*/\s*[0-9]+|\b[0-9]{1,4}\b)$",
    re.IGNORECASE,
)
ARXIV_STAMP_REGEX: Pattern = re.compile(
    r"^arXiv:[0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?\s+\[[a-zA-Z\.\-]+\](?:\s+[0-9]{1,2}\s+[a-zA-Z]{3,9}\s+[0-9]{4})?$",
    re.IGNORECASE,
)
RUNNING_HEADER_REGEX: Pattern = re.compile(
    r"^(?:(?:proceedings of|conference on|workshop on|journal of|ieee|acm|springer|neurips|icml|cvpr|iclr|emnlp|acl)\b.*)$",
    re.IGNORECASE,
)


class TextCleaner:
    """Provides modular sanitization and normalization for extracted academic text."""

    def __init__(
        self,
        repair_hyphens: bool = True,
        expand_ligatures: bool = True,
        remove_running_headers: bool = True,
        normalize_unicode: bool = True,
    ):
        self.repair_hyphens = repair_hyphens
        self.expand_ligatures = expand_ligatures
        self.remove_running_headers = remove_running_headers
        self.normalize_unicode = normalize_unicode

    def clean(self, text: str | None) -> str:
        """Execute full cleaning pipeline on text."""
        if not text:
            return ""

        result = text

        # 1. Unicode NFKC normalization
        if self.normalize_unicode:
            result = unicodedata.normalize("NFKC", result)

        # 2. Expand ligatures & replace typographical quotes/dashes
        if self.expand_ligatures:
            for lig, replacement in LIGATURE_MAP.items():
                result = result.replace(lig, replacement)

        # 3. Repair line-break split hyphenated words (e.g., trans-\nformer -> transformer)
        if self.repair_hyphens:
            result = HYPHENATION_REGEX.sub(r"\1\2", result)

        # 4. Remove running headers / footers line-by-line
        if self.remove_running_headers:
            lines = result.splitlines()
            filtered_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    filtered_lines.append("")
                    continue
                # Skip standalone page numbers or arXiv stamps
                if PAGE_NUMBER_FOOTER_REGEX.match(stripped) or ARXIV_STAMP_REGEX.match(stripped):
                    continue
                # Skip obvious journal headers
                if RUNNING_HEADER_REGEX.match(stripped):
                    continue
                filtered_lines.append(line)
            result = "\n".join(filtered_lines)

        # 5. Normalize whitespace within paragraphs while preserving true paragraph breaks
        result = self._normalize_whitespace(result)

        return result.strip()

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse redundant spaces and intra-sentence line breaks while preserving paragraph breaks."""
        # Split into distinct paragraphs (separated by 2 or more newlines)
        paragraphs = re.split(r"\n\s*\n+", text)
        cleaned_paragraphs = []

        for p in paragraphs:
            # Replace single newlines within a paragraph with a single space
            single_line = re.sub(r"(?<!\n)\n(?!\n)", " ", p)
            # Collapse consecutive spaces and tabs
            collapsed = re.sub(r"[ \t]+", " ", single_line).strip()
            if collapsed:
                cleaned_paragraphs.append(collapsed)

        return "\n\n".join(cleaned_paragraphs)

    @staticmethod
    def clean_table_markdown(md_table: str) -> str:
        """Normalize spacing and alignment markers in Markdown tables."""
        if not md_table:
            return ""
        lines = [line.strip() for line in md_table.splitlines() if line.strip()]
        return "\n".join(lines)
