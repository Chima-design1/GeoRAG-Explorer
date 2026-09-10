"""Configuration management for GeoRAG Explorer.

Loads environment variables and provides centralized access to:
- OpenAI API configuration
- Embedding and chat model settings
- File paths
- Logging configuration
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class Config:
    """Configuration container for GeoRAG Explorer.

    Loads configuration from environment variables with sensible defaults.
    """

    def __init__(self, env_path: Optional[str] = None):
        """Initialize configuration.

        Args:
            env_path: Optional path to .env file.
        """

        # ---------------------------------------------------------
        # Load environment variables
        # ---------------------------------------------------------
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        # ---------------------------------------------------------
        # OpenAI API configuration
        # ---------------------------------------------------------
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # ---------------------------------------------------------
        # Model configuration
        # ---------------------------------------------------------

        # Local embedding model used by EmbeddingGenerator
        self.embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small"
        )

        # OpenAI chat model
        self.chat_model = os.getenv(
            "OPENAI_CHAT_MODEL",
            "gpt-4-turbo"
        )

        # Alias required by RAGPipeline
        self.llm_model = os.getenv(
            "OPENAI_CHAT_MODEL",
            "gpt-4-turbo"
        )

        # ---------------------------------------------------------
        # Logging configuration
        # ---------------------------------------------------------
        self.log_level = os.getenv(
            "LOG_LEVEL",
            "INFO"
        )

        # ---------------------------------------------------------
        # Retrieval configuration
        # ---------------------------------------------------------
        self.default_top_k = int(
            os.getenv(
                "DEFAULT_TOP_K",
                "5"
            )
        )

        # ---------------------------------------------------------
        # Project paths
        # ---------------------------------------------------------
        project_root = Path(__file__).parent.parent

        self.reports_dir = Path(
            os.getenv(
                "REPORTS_DIR",
                str(project_root / "data" / "reports")
            )
        )

        self.maps_dir = Path(
            os.getenv(
                "MAPS_DIR",
                str(project_root / "data" / "maps")
            )
        )

        self.embeddings_cache_path = Path(
            os.getenv(
                "EMBEDDINGS_CACHE_PATH",
                str(project_root / "artifacts" / "embeddings.pkl")
            )
        )

        # ---------------------------------------------------------
        # Create required directories
        # ---------------------------------------------------------
        self.reports_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.maps_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.embeddings_cache_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def __repr__(self) -> str:
        """Return a readable configuration representation."""

        return (
            f"Config("
            f"embedding_model={self.embedding_model}, "
            f"chat_model={self.chat_model}, "
            f"llm_model={self.llm_model}, "
            f"log_level={self.log_level}, "
            f"default_top_k={self.default_top_k}"
            f")"
        )
