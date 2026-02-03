"""Logging utilities for LoLaMA."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger: logging.Logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler: logging.StreamHandler = logging.StreamHandler(sys.stderr)
        formatter: logging.Formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(level)
    return logger


def set_verbosity(level: int | str) -> None:
    """Set global verbosity level.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR) or int
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    logging.getLogger("lolama").setLevel(level)


# Convenience loggers for common modules
def get_model_logger() -> logging.Logger:
    """Get logger for model operations."""
    return get_logger("lolama.model")


def get_data_logger() -> logging.Logger:
    """Get logger for data loading operations."""
    return get_logger("lolama.data")


def get_generation_logger() -> logging.Logger:
    """Get logger for generation operations."""
    return get_logger("lolama.generation")
