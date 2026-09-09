from pathlib import Path
import json
import re

from src.chunker import Chunk


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_map_text(text: str) -> str:
    """Clean PDF/OCR map text while removing obvious extraction artifacts."""
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        # Remove known PDF extraction artifacts.
        line = re.sub(r"\^`+", " ", line)
        line = re.sub(r"\^_+", " ", line)
        line = re.sub(r"!+\[+", " ", line)

        # Remove repeated punctuation-only fragments.
        if re.fullmatch(r"[\W_]+", line):
            continue

        # Collapse whitespace.
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Category detection
# ---------------------------------------------------------------------------

def infer_map_category(filename: str) -> str:
    """Infer map category from filename."""
    name = filename.lower()

    if "corridor" in name:
        return "corridor"

    if "schist" in name:
        return "schist_belt"

    if "state" in name:
        return "state"

    if any(
        term in name
        for term in [
            "north-central",
            "north-east",
            "north-west",
            "south-east",
            "south-south",
            "south-west",
        ]
    ):
        return "regional"

    if any(
        term in name
        for term in [
            "nigeria",
            "mineral",
            "geochemical",
            "carbonate",
            "agric",
            "ngsa",
        ]
    ):
        return "national"

    return "map"


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_map_json(json_path: str | Path) -> dict:
    """Load one processed map JSON file."""
    json_path = Path(json_path)

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ---------------------------------------------------------------------------
# Map text extraction
# ---------------------------------------------------------------------------

def extract_map_text(record: dict) -> str:
    """Extract searchable text from a processed map JSON record."""
    parts = []

    filename = record.get("filename", "")

    category = infer_map_category(
        filename
    )

    parts.append(
        f"Map: {filename}"
    )

    parts.append(
        f"Category: {category}"
    )

    parts.append(
        "Processing type: "
        f"{record.get('processing_type', 'unknown')}"
    )

    for page in record.get(
        "pages",
        [],
    ):
        page_number = page.get(
            "page_number",
            1,
        )

        page_text = clean_map_text(
            page.get("text", "")
        )

        if page_text:
            parts.append(
                f"Page {page_number}: {page_text}"
            )

    ocr_text = clean_map_text(
        record.get(
            "ocr_text",
            "",
        )
    )

    if ocr_text:
        parts.append(
            f"OCR map text: {ocr_text}"
        )

    return "\n\n".join(
        parts
    )


# ---------------------------------------------------------------------------
# Map corpus
# ---------------------------------------------------------------------------

def load_map_corpus(
    map_json_dir: str | Path = r"artifacts\map_ocr",
) -> list[dict]:
    """Load all processed map JSON files."""
    map_json_dir = Path(
        map_json_dir
    )

    documents = []

    for json_path in sorted(
        map_json_dir.glob("*.json")
    ):
        record = load_map_json(
            json_path
        )

        text = extract_map_text(
            record
        )

        if not text.strip():
            continue

        filename = record.get(
            "filename",
            json_path.name,
        )

        document_id = record.get(
            "document_id",
            json_path.stem,
        )

        documents.append(
            {
                "document_id": (
                    f"map_{document_id}"
                ),
                "filename": filename,
                "category": infer_map_category(
                    filename
                ),
                "processing_type": record.get(
                    "processing_type",
                    "unknown",
                ),
                "page_count": record.get(
                    "page_count",
                    1,
                ),
                "text": text,
                "source_json": str(
                    json_path
                ),
            }
        )

    return documents


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_map_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Split map text into coherent chunks.

    Small map OCR documents are kept intact when they fit within
    a 2,000-character limit. This preserves relationships between
    map titles, legends, geographic labels, and state selections.
    Larger maps continue to use the normal chunking strategy.
    """
    text = clean_map_text(text)

    if not text:
        return []

    # Keep small map documents together so geographic information
    # is not separated from the map legend or state list.
    if len(text) <= 2000:
        return [text]

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        if len(paragraph) <= chunk_size:

            if not current:
                current = paragraph

            elif (
                len(current)
                + len(paragraph)
                + 2
                <= chunk_size
            ):
                current += (
                    "\n\n"
                    + paragraph
                )

            else:
                chunks.append(current)

                if chunk_overlap > 0:
                    current = (
                        current[-chunk_overlap:]
                        + "\n\n"
                        + paragraph
                    )
                else:
                    current = paragraph

        else:

            if current:
                chunks.append(current)
                current = ""

            start = 0

            while start < len(paragraph):
                end = min(
                    start + chunk_size,
                    len(paragraph),
                )

                piece = paragraph[start:end].strip()

                if piece:
                    chunks.append(piece)

                if end >= len(paragraph):
                    break

                start = max(
                    end - chunk_overlap,
                    start + 1,
                )

    if current:
        chunks.append(current)

    return chunks

# ---------------------------------------------------------------------------
# Chunk conversion
# ---------------------------------------------------------------------------

def build_map_chunks(
    documents: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Convert map documents into the existing Chunk class."""
    chunks = []

    for document in documents:

        text_chunks = split_map_text(
            document["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk_text in enumerate(
            text_chunks
        ):

            metadata = {
                "title": document[
                    "filename"
                ],
                "source_file": document[
                    "filename"
                ],
                "source_url": "",
                "document_type": "geological_map",
                "source_type": "map",
                "map_category": document[
                    "category"
                ],
                "processing_type": document[
                    "processing_type"
                ],
                "page_count": document[
                    "page_count"
                ],
                "source_json": document[
                    "source_json"
                ],
            }

            chunk = Chunk(
                chunk_id=(
                    f"{document['document_id']}"
                    f"_chunk_{chunk_index}"
                ),
                document_id=document[
                    "document_id"
                ],
                text=normalize_whitespace(
                    chunk_text
                ),
                chunk_index=chunk_index,
                start_char=0,
                end_char=len(chunk_text),
                metadata=metadata,
            )

            chunks.append(
                chunk
            )

    return chunks


def load_map_chunks(
    map_json_dir: str | Path = r"artifacts\map_ocr",
) -> list[Chunk]:
    """Load all processed maps as Chunk objects."""
    documents = load_map_corpus(
        map_json_dir
    )

    return build_map_chunks(
        documents
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    documents = load_map_corpus()

    chunks = build_map_chunks(
        documents
    )

    print(
        f"Map documents: {len(documents)}"
    )

    print(
        f"Map chunks: {len(chunks)}"
    )

    noisy_tokens = [
        "^`",
        "^_",
        "![",
    ]

    total_noise = sum(
        sum(
            chunk.text.count(token)
            for token in noisy_tokens
        )
        for chunk in chunks
    )

    print(
        f"Remaining known noise tokens: "
        f"{total_noise}"
    )

    print("\nSample:")

    if chunks:
        print(
            json.dumps(
                chunks[0].to_dict(),
                indent=2,
                ensure_ascii=False,
            )
        )

