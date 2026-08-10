"""Unit tests for document chunking."""

import pytest
from src.document_loader import Document
from src.chunker import Chunker, Chunk


def test_chunker_initialization():
    """Test chunker initialization."""
    chunker = Chunker(chunk_size=500, chunk_overlap=100)
    assert chunker.chunk_size == 500
    assert chunker.chunk_overlap == 100


def test_chunk_single_document():
    """Test chunking a single document."""
    doc = Document(
        document_id="test_doc",
        title="Test Geological Report",
        content="This is a test geological document. " * 100,  # ~3600 chars
        source_file="test.txt",
        source_url="http://example.com/test.txt",
        document_type="geological_report"
    )
    
    chunker = Chunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_document(doc)
    
    # Should create multiple chunks
    assert len(chunks) > 1
    
    # All chunks should be Chunk instances
    assert all(isinstance(c, Chunk) for c in chunks)
    
    # All chunks should have metadata
    assert all(c.metadata["title"] == "Test Geological Report" for c in chunks)
    assert all(c.metadata["source_file"] == "test.txt" for c in chunks)


def test_chunk_metadata_preservation():
    """Test that metadata is preserved in chunks."""
    doc = Document(
        document_id="mineral_report",
        title="Copper Mineral Occurrence",
        content="Copper deposits found in Ogun State. " * 50,
        source_file="copper_report.txt",
        source_url="http://example.com/copper",
        document_type="mineral_report"
    )
    
    chunker = Chunker()
    chunks = chunker.chunk_document(doc)
    
    for chunk in chunks:
        assert chunk.document_id == "mineral_report"
        assert chunk.metadata["title"] == "Copper Mineral Occurrence"
        assert chunk.metadata["source_file"] == "copper_report.txt"
        assert chunk.metadata["source_url"] == "http://example.com/copper"


def test_chunk_empty_document():
    """Test chunking empty document."""
    doc = Document(
        document_id="empty",
        title="Empty",
        content="",
        source_file="empty.txt"
    )
    
    chunker = Chunker()
    chunks = chunker.chunk_document(doc)
    
    assert len(chunks) == 0


def test_chunk_multiple_documents():
    """Test chunking multiple documents."""
    docs = [
        Document(
            document_id=f"doc_{i}",
            title=f"Document {i}",
            content="Test content. " * 100,
            source_file=f"doc_{i}.txt"
        )
        for i in range(3)
    ]
    
    chunker = Chunker()
    all_chunks = chunker.chunk_documents(docs)
    
    # Should have chunks from all documents
    assert len(all_chunks) > 0
    
    # Chunks should have correct document IDs
    doc_ids = {c.document_id for c in all_chunks}
    assert doc_ids == {"doc_0", "doc_1", "doc_2"}
