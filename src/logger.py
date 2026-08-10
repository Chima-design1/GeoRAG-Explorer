"""Logging utilities for GeoRAG Explorer.

Provides structured logging with configurable levels.
"""

import logging
from typing import Optional
from src.config import Config


def get_logger(name: str, config: Optional[Config] = None) -> logging.Logger:
    """Get a configured logger.
    
    Args:
        name: Logger name (typically __name__).
        config: Optional Config object. If None, uses INFO level.
    
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Set level
        level = logging.INFO
        if config:
            level_str = config.log_level.upper()
            level = getattr(logging, level_str, logging.INFO)
        logger.setLevel(level)
        
        # Create console handler with formatting
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
