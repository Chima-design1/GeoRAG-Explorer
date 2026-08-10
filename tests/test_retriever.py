"""Unit tests for retriever module."""

import pytest
import numpy as np
from src.retriever import VectorRetriever, RetrievedResult
from src.chunker import Chunk


def create_test_chunks(n: int = 5) -> list:
    """Create test chunks."""
    chunks = []
    for i in range(n):
        chunk = Chunk(
            chunk_id=f"chunk_{i}",
            document_id=f"doc_{i % 2}",
            text=f"Test chunk {i} with content.",
            chunk_index=i,
            start_char=0,
            end_char=100,
            metadata={
                "title": f"Document {i % 2}",
                "source_file": f"doc_{i % 2}.txt"
            }
        )
        chunks.append(chunk)
    return chunks


def create_test_embeddings(n: int = 5, dim: int = 1536) -> np.ndarray:
    """Create random test embeddings."""
    return np.random.randn(n, dim).astype(np.float32)


def test_retriever_initialization():
    """Test VectorRetriever initialization."""
    chunks = create_test_chunks(5)
    embeddings = create_test_embeddings(5, 1536)
    
    retriever = VectorRetriever(chunks, embeddings, top_k=3)
    assert len(retriever.chunks) == 5
    assert retriever.embeddings.shape == (5, 1536)
    assert retriever.top_k == 3


def test_retriever_mismatch():
    """Test that retriever rejects mismatched chunks/embeddings."""
    chunks = create_test_chunks(5)
    embeddings = create_test_embeddings(3, 1536)  # Wrong count
    
    with pytest.raises(ValueError):
        VectorRetriever(chunks, embeddings)


def test_retrieve_top_k():
    """Test retrieval with top_k parameter."""
    chunks = create_test_chunks(10)
    embeddings = create_test_embeddings(10, 1536)
    
    retriever = VectorRetriever(chunks, embeddings, top_k=3)
    
    query_embedding = embeddings[0]  # Use first embedding as query
    results = retriever.retrieve(query_embedding, top_k=3)
    
    assert len(results) == 3
    assert all(isinstance(r, RetrievedResult) for r in results)
    
    # Results should be sorted by score (descending)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_override_top_k():
    """Test overriding default top_k at query time."""
    chunks = create_test_chunks(10)
    embeddings = create_test_embeddings(10, 1536)
    
    retriever = VectorRetriever(chunks, embeddings, top_k=3)
    query_embedding = embeddings[0]
    
    # Override with different top_k
    results = retriever.retrieve(query_embedding, top_k=5)
    assert len(results) == 5


def test_retrieved_result():
    """Test RetrievedResult object."""
    chunk = create_test_chunks(1)[0]
    result = RetrievedResult(chunk, score=0.95, rank=1)
    
    assert result.score == 0.95
    assert result.rank == 1
    
    result_dict = result.to_dict()
    assert result_dict["score"] == 0.95
    assert result_dict["rank"] == 1
    assert result_dict["chunk_id"] == "chunk_0"
