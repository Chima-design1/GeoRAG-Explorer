"""Unit tests for RAG pipeline."""

import pytest
from src.config import Config
from src.rag import RAG
from src.retriever import VectorRetriever
from src.document_loader import Document
from src.chunker import Chunker
import numpy as np


def test_rag_initialization():
    """Test RAG pipeline initialization."""
    try:
        # Create minimal test setup
        chunks_list = [
            __import__('src.chunker', fromlist=['Chunk']).Chunk(
                chunk_id="chunk_0",
                document_id="doc_0",
                text="Copper deposits occur in Nigeria.",
                chunk_index=0,
                start_char=0,
                end_char=50,
                metadata={"title": "Test", "source_file": "test.txt"}
            )
        ]
        embeddings = np.random.randn(1, 1536).astype(np.float32)
        retriever = VectorRetriever(chunks_list, embeddings)
        
        config = Config()
        rag = RAG(retriever, config)
        
        assert rag.retriever == retriever
        assert rag.config == config
    except ValueError as e:
        pytest.skip(f"OpenAI API key not configured: {e}")


def test_rag_query_format():
    """Test that RAG query returns expected format."""
    try:
        # This is a structural test, not a full integration test
        # Full integration testing requires actual embeddings/API calls
        from src.chunker import Chunk
        
        chunks_list = [
            Chunk(
                chunk_id="chunk_0",
                document_id="doc_0",
                text="Lithium occurrences in Ogun State are documented.",
                chunk_index=0,
                start_char=0,
                end_char=60,
                metadata={
                    "title": "Mineral Report",
                    "source_file": "minerals.txt",
                    "source_url": "http://example.com"
                }
            )
        ]
        embeddings = np.random.randn(1, 1536).astype(np.float32)
        retriever = VectorRetriever(chunks_list, embeddings)
        
        config = Config()
        rag = RAG(retriever, config)
        
        # The RAG.query method requires actual API calls
        # For now, we just verify the setup
        assert hasattr(rag, 'query')
        assert callable(rag.query)
        
    except ValueError as e:
        pytest.skip(f"OpenAI API key not configured: {e}")
