"""Generate and cache embeddings using OpenAI API.

Handles embedding generation with local caching to avoid redundant API calls.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from openai import OpenAI
from src.config import Config
from src.chunker import Chunk
from src.logger import get_logger


class EmbeddingGenerator:
    """Generate embeddings using OpenAI API with local caching."""
    
    def __init__(self, config: Config):
        """Initialize embedding generator.
        
        Args:
            config: Config object with OpenAI API key and model.
        """
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.logger = get_logger(__name__)
    
    def embed_chunks(
        self,
        chunks: List[Chunk],
        cache_path: Optional[Path] = None,
        force_regenerate: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for chunks with optional caching.
        
        Args:
            chunks: List of Chunk objects.
            cache_path: Optional path to cache embeddings. If None, uses config default.
            force_regenerate: If True, regenerate even if cache exists.
        
        Returns:
            NumPy array of shape (n_chunks, embedding_dim).
        
        Raises:
            ValueError: If no chunks provided.
        """
        if not chunks:
            raise ValueError("No chunks provided for embedding")
        
        cache_path = Path(cache_path or self.config.embeddings_cache_path)
        
        # Try to load from cache
        if not force_regenerate and cache_path.exists():
            try:
                embeddings = self._load_cache(cache_path)
                if len(embeddings) == len(chunks):
                    self.logger.info(
                        f"Loaded {len(chunks)} embeddings from cache: {cache_path}"
                    )
                    return embeddings
                else:
                    self.logger.warning(
                        f"Cache has {len(embeddings)} embeddings but {len(chunks)} chunks. "
                        "Regenerating."
                    )
            except Exception as e:
                self.logger.warning(f"Error loading cache: {e}. Regenerating.")
        
        # Generate embeddings
        self.logger.info(
            f"Generating embeddings for {len(chunks)} chunks "
            f"using {self.config.embedding_model}..."
        )
        
        embeddings = self._generate_embeddings(chunks)
        
        # Cache embeddings
        if cache_path:
            try:
                self._save_cache(embeddings, cache_path)
                self.logger.info(f"Cached embeddings to: {cache_path}")
            except Exception as e:
                self.logger.warning(f"Could not cache embeddings: {e}")
        
        return embeddings
    
    def _generate_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """Generate embeddings for chunks using OpenAI API.
        
        Args:
            chunks: List of Chunk objects.
        
        Returns:
            NumPy array of shape (n_chunks, embedding_dim).
        """
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings via API
        try:
            response = self.client.embeddings.create(
                model=self.config.embedding_model,
                input=texts,
            )
            embeddings_list = [item.embedding for item in response.data]
            embeddings = np.array(embeddings_list, dtype=np.float32)
            self.logger.info(
                f"Generated {len(embeddings)} embeddings. "
                f"Dimension: {embeddings.shape[1]}"
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _save_cache(self, embeddings: np.ndarray, cache_path: Path) -> None:
        """Save embeddings to pickle cache.
        
        Args:
            embeddings: NumPy array of embeddings.
            cache_path: Path to save cache.
        """
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
    
    def _load_cache(self, cache_path: Path) -> np.ndarray:
        """Load embeddings from pickle cache.
        
        Args:
            cache_path: Path to cache file.
        
        Returns:
            NumPy array of embeddings.
        """
        with open(cache_path, "rb") as f:
            embeddings = pickle.load(f)
        return embeddings
