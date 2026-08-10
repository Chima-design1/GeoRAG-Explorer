"""Chunk documents while preserving metadata.

Splits documents into overlapping chunks, maintaining source information
and metadata for retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
from src.document_loader import Document
from src.logger import get_logger


class Chunk:
    """Represents a document chunk with metadata.
    
    Attributes:
        chunk_id: Unique identifier (document_id + chunk index).
        document_id: Parent document ID.
        text: Chunk text content.
        chunk_index: Index within document.
        start_char: Character offset in original document.
        end_char: Character offset in original document.
        metadata: Dict containing document-level metadata.
    """
    
    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        text: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        metadata: Dict[str, Any]
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
    """Split documents into overlapping chunks with metadata.
    
    Uses simple character-based chunking with configurable size and overlap.
    Preserves document metadata in each chunk for retrieval.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = get_logger(__name__)
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents.
        
        Args:
            documents: List of Document objects.
        
        Returns:
            List of Chunk objects.
        """
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
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a single document.
        
        Args:
            document: Document object.
        
        Returns:
            List of Chunk objects from this document.
        """
        text = document.content
        chunks = []
        
        # Simple character-based chunking
        chunk_index = 0
        start_char = 0
        
        while start_char < len(text):
            # Calculate chunk end, trying to break at sentence/word boundary
            end_char = min(start_char + self.chunk_size, len(text))
            
            # If not at end of text, try to break at a reasonable boundary
            if end_char < len(text):
                # Try to break at the last period, newline, or space
                for boundary_char in [".", "\n", " "]:
                    last_boundary = text.rfind(boundary_char, start_char, end_char)
                    if last_boundary > start_char + self.chunk_size // 2:
                        end_char = last_boundary + 1
                        break
            
            # Extract chunk text
            chunk_text = text[start_char:end_char].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunk_id = f"{document.document_id}_chunk_{chunk_index}"
                
                # Preserve document metadata in chunk
                metadata = {
                    "title": document.title,
                    "source_file": document.source_file,
                    "source_url": document.source_url,
                    "document_type": document.document_type,
                }
                
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
            
            # Move to next chunk with overlap
            start_char = end_char - self.chunk_overlap
        
        return chunks
