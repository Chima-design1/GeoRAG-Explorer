"""Vector retrieval using cosine similarity.

Implements semantic search with local vector index (NumPy + cosine similarity).
Designed to be replaceable with FAISS, Qdrant, or pgvector in future phases.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from src.chunker import Chunk
from src.logger import get_logger


class RetrievedResult:
    """Represents a retrieved chunk with relevance score.
    
    Attributes:
        chunk: The Chunk object.
        score: Similarity score (0-1).
        rank: Rank in retrieval results (1-indexed).
    """
    
    def __init__(self, chunk: Chunk, score: float, rank: int):
        self.chunk = chunk
        self.score = score
        self.rank = rank
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rank": self.rank,
            "score": float(self.score),
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "text": self.chunk.text,
            "metadata": self.chunk.metadata,
        }


class VectorRetriever:
    """Semantic search using cosine similarity on embeddings.
    
    Phase 1 implementation using NumPy. Replaceable with FAISS, Qdrant,
    or pgvector in future phases.
    """
    
    def __init__(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        top_k: int = 5,
    ):
        """Initialize retriever.
        
        Args:
            chunks: List of Chunk objects.
            embeddings: NumPy array of shape (n_chunks, embedding_dim).
            top_k: Default number of top results to return.
        
        Raises:
            ValueError: If chunks and embeddings have mismatched lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
            )
        
        self.chunks = chunks
        self.embeddings = embeddings
        self.top_k = top_k
        self.logger = get_logger(__name__)
        
        self.logger.info(
            f"Initialized VectorRetriever with {len(chunks)} chunks, "
            f"embedding dimension {embeddings.shape[1]}, default top_k={top_k}"
        )
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
    ) -> List[RetrievedResult]:
        """Retrieve top-k most similar chunks.
        
        Args:
            query_embedding: Query embedding vector (shape: (embedding_dim,)).
            top_k: Number of results to return. If None, uses self.top_k.
        
        Returns:
            List of RetrievedResult objects, sorted by score (descending).
        
        Raises:
            ValueError: If query_embedding has wrong shape.
        """
        if top_k is None:
            top_k = self.top_k
        
        if query_embedding.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Query embedding dimension ({query_embedding.shape[0]}) "
                f"!= stored embedding dimension ({self.embeddings.shape[1]})"
            )
        
        # Compute cosine similarity: scores in [0, 1]
        scores = self._cosine_similarity(query_embedding, self.embeddings)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            result = RetrievedResult(
                chunk=self.chunks[idx],
                score=float(scores[idx]),
                rank=rank,
            )
            results.append(result)
        
        return results
    
    @staticmethod
    def _cosine_similarity(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all embeddings.
        
        Args:
            query: Query embedding (shape: (embedding_dim,)).
            embeddings: Stored embeddings (shape: (n_chunks, embedding_dim)).
        
        Returns:
            Similarity scores (shape: (n_chunks,)) in range [0, 1].
        """
        # Normalize vectors
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity: dot product of normalized vectors
        similarities = np.dot(embeddings_norm, query_norm)
        
        # Shift to [0, 1] range (from [-1, 1])
        similarities = (similarities + 1.0) / 2.0
        
        return similarities
