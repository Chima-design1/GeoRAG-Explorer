"""Generate and cache local embeddings for geological documents.

Uses Sentence Transformers locally, avoiding OpenAI API costs for
embedding generation during development and testing.
"""

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import Config
from src.chunker import Chunk
from src.logger import get_logger


class EmbeddingGenerator:
    """Generate embeddings locally using Sentence Transformers."""

    def __init__(self, config: Config):
        """Initialize the local embedding model.

        Args:
            config: Config object.
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Local embedding model.
        self.model_name = "all-MiniLM-L6-v2"

        self.logger.info(
            f"Loading local embedding model: {self.model_name}"
        )

        self.model = SentenceTransformer(self.model_name)

        self.logger.info(
            f"Local embedding model loaded. "
            f"Dimension: {self.model.get_embedding_dimension()}"
        )

    def embed_chunks(
        self,
        chunks: List[Chunk],
        cache_path: Optional[Path] = None,
        force_regenerate: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for document chunks.

        Uses a local Sentence Transformer model and caches the
        resulting embeddings to disk.

        Args:
            chunks: List of Chunk objects.
            cache_path: Optional path to embedding cache.
            force_regenerate: Regenerate embeddings even if cache exists.

        Returns:
            NumPy array with shape (n_chunks, embedding_dim).

        Raises:
            ValueError: If no chunks are provided.
        """

        if not chunks:
            raise ValueError("No chunks provided for embedding")

        cache_path = Path(
            cache_path or self.config.embeddings_cache_path
        )

        # Try loading existing cache.
        if not force_regenerate and cache_path.exists():
            try:
                embeddings = self._load_cache(cache_path)

                if len(embeddings) == len(chunks):
                    self.logger.info(
                        f"Loaded {len(chunks)} embeddings from cache: "
                        f"{cache_path}"
                    )
                    return embeddings

                self.logger.warning(
                    f"Cache contains {len(embeddings)} embeddings, "
                    f"but {len(chunks)} chunks were provided. "
                    "Regenerating."
                )

            except Exception as e:
                self.logger.warning(
                    f"Could not load embedding cache: {e}. "
                    "Regenerating."
                )

        # Generate embeddings.
        self.logger.info(
            f"Generating local embeddings for {len(chunks)} chunks..."
        )

        embeddings = self._generate_embeddings(chunks)

        # Save cache.
        try:
            self._save_cache(embeddings, cache_path)
            self.logger.info(
                f"Cached embeddings to: {cache_path}"
            )
        except Exception as e:
            self.logger.warning(
                f"Could not cache embeddings: {e}"
            )

        return embeddings

    def _generate_embeddings(
        self,
        chunks: List[Chunk]
    ) -> np.ndarray:
        """Generate embeddings for document chunks."""

        texts = [chunk.text for chunk in chunks]

        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32
            )

            self.logger.info(
                f"Generated {len(embeddings)} embeddings. "
                f"Dimension: {embeddings.shape[1]}"
            )

            return embeddings

        except Exception as e:
            self.logger.error(
                f"Error generating local embeddings: {e}"
            )
            raise

    def embed_query(
        self,
        query: str
    ) -> np.ndarray:
        """Generate an embedding for a single search query.

        Args:
            query: Natural-language search question.

        Returns:
            NumPy array with shape (embedding_dim,).

        Raises:
            ValueError: If the query is empty.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            embedding = np.asarray(
                embedding,
                dtype=np.float32
            )

            return embedding

        except Exception as e:
            self.logger.error(
                f"Error generating query embedding: {e}"
            )
            raise

    def _save_cache(
        self,
        embeddings: np.ndarray,
        cache_path: Path
    ) -> None:
        """Save embeddings to pickle cache."""

        cache_path = Path(cache_path)

        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)

    def _load_cache(
        self,
        cache_path: Path
    ) -> np.ndarray:
        """Load embeddings from pickle cache."""

        with open(cache_path, "rb") as f:
            embeddings = pickle.load(f)

        return embeddings
