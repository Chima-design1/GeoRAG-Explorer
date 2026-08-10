"""Unit tests for embeddings module."""

import pytest
import numpy as np
from pathlib import Path
from src.config import Config
from src.embeddings import EmbeddingGenerator
from src.document_loader import Document
from src.chunker import Chunker


def test_embedding_generator_initialization():
    """Test EmbeddingGenerator initialization."""
    # This requires OPENAI_API_KEY to be set
    try:
        config = Config()
        gen = EmbeddingGenerator(config)
        assert gen.config == config
        assert gen.client is not None
    except ValueError as e:
        pytest.skip(f"OpenAI API key not configured: {e}")


def test_cosine_similarity():
    """Test cosine similarity computation."""
    from src.retriever import VectorRetriever
    
    # Test vectors
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])  # Identical
    v3 = np.array([0.0, 1.0, 0.0])  # Orthogonal
    
    embeddings = np.vstack([v2, v3])
    
    scores = VectorRetriever._cosine_similarity(v1, embeddings)
    
    # Identical vectors should have high similarity
    assert scores[0] > 0.9
    
    # Orthogonal vectors should have low similarity
    assert scores[1] < 0.6


def test_embedding_cache():
    """Test embedding caching mechanism."""
    try:
        config = Config()
        gen = EmbeddingGenerator(config)
        
        # Create test chunks
        doc = Document(
            document_id="test",
            title="Test",
            content="Test document. " * 50,
            source_file="test.txt"
        )
        chunker = Chunker(chunk_size=200)
        chunks = chunker.chunk_document(doc)
        
        if not chunks:
            pytest.skip("No chunks created")
        
        # Generate embeddings with cache
        cache_path = Path("/tmp/test_embeddings.pkl")
        embeddings = gen.embed_chunks(chunks, cache_path=cache_path, force_regenerate=True)
        
        assert embeddings.shape[0] == len(chunks)
        assert cache_path.exists()
        
        # Load from cache
        embeddings_cached = gen.embed_chunks(chunks, cache_path=cache_path)
        assert embeddings_cached.shape == embeddings.shape
        np.testing.assert_array_almost_equal(embeddings, embeddings_cached)
        
        # Cleanup
        cache_path.unlink()
    except ValueError as e:
        pytest.skip(f"OpenAI API key not configured: {e}")
