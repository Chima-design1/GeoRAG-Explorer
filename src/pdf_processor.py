"""
PDF text extraction module for GeoRAG Explorer.

Extracts geological report text page-by-page while preserving
page numbers and basic document metadata.
"""

from pathlib import Path
from typing import Dict, List

import pymupdf


def clean_text(text: str) -> str:
    """Clean extracted PDF text while preserving paragraph structure."""
    lines = []

    for line in text.splitlines():
        line = " ".join(line.split())

        if line:
            lines.append(line)

    return "\n".join(lines)


def extract_pdf(pdf_path: str | Path) -> Dict:
    """
    Extract text from a PDF document.

    Returns:
        Dictionary containing document metadata and page-level text.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path.suffix}")

    pages: List[Dict] = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            raw_text = page.get_text("text")
            text = clean_text(raw_text)

            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "character_count": len(text),
                }
            )

        result = {
            "document_id": pdf_path.stem,
            "filename": pdf_path.name,
            "page_count": len(document),
            "pages": pages,
            "total_characters": sum(
                page["character_count"] for page in pages
            ),
        }

    return result
