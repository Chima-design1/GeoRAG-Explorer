"""Chunk documents while preserving metadata.

Splits documents into overlapping chunks while maintaining
document and PDF page metadata for retrieval.
"""

from typing import List, Dict, Any

from src.document_loader import Document
from src.logger import get_logger


class Chunk:
    """Represents a document chunk with metadata."""

    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        text: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        metadata: Dict[str, Any],
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.text = text
        self.chunk_index = chunk_index
        self.start_char = start_char
        self.end_char = end_char
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }


class Chunker:
    """Split documents into overlapping chunks with metadata."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = get_logger(__name__)

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents."""
        chunks = []

        for doc in documents:
            doc_chunks = self.chunk_document(doc)
            chunks.extend(doc_chunks)

            self.logger.debug(
                f"Chunked {doc.document_id}: {len(doc_chunks)} chunks"
            )

        self.logger.info(
            f"Total chunks: {len(chunks)} from {len(documents)} documents"
        )

        return chunks

    def _get_page_number(
        self,
        document: Document,
        start_char: int,
        end_char: int,
    ) -> int | None:
        """Determine which PDF page contains the center of a chunk."""
        if not document.page_metadata:
            return None

        target_char = (start_char + end_char) // 2
        running_chars = 0

        for page in document.page_metadata:
            page_text = page.get("text", "")
            page_start = running_chars
            page_end = running_chars + len(page_text)

            if page_start <= target_char <= page_end:
                return page.get("page_number")

            running_chars = page_end + 2  # account for "\n\n"

        return None

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a single document."""
        text = document.content
        chunks = []

        chunk_index = 0
        start_char = 0

        while start_char < len(text):
            end_char = min(
                start_char + self.chunk_size,
                len(text),
            )

            if end_char < len(text):
                for boundary_char in [".", "\n", " "]:
                    last_boundary = text.rfind(
                        boundary_char,
                        start_char,
                        end_char,
                    )

                    if last_boundary > start_char + self.chunk_size // 2:
                        end_char = last_boundary + 1
                        break

            chunk_text = text[start_char:end_char].strip()

            if chunk_text:
                chunk_id = (
                    f"{document.document_id}_chunk_{chunk_index}"
                )

                metadata = {
                    "title": document.title,
                    "source_file": document.source_file,
                    "source_url": document.source_url,
                    "document_type": document.document_type,
                }

                page_number = self._get_page_number(
                    document,
                    start_char,
                    end_char,
                )

                if page_number is not None:
                    metadata["page_number"] = page_number

                chunk = Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=metadata,
                )

                chunks.append(chunk)
                chunk_index += 1

                if end_char >= len(text):
                    break

                next_start = end_char - self.chunk_overlap

                if next_start <= start_char:
                    next_start = end_char

                start_char = next_start

        return chunks
