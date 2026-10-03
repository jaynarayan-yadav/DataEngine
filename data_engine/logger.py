"""
Logging utilities for cleanframe.
Provides cleanly formatted, notebook-friendly log messages.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

LOGGER_NAME = "cleanframe"


def get_logger(name: str = LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    """
    Get or configure a logger for cleanframe.
    
    Parameters
    ----------
    name : str, default "cleanframe"
        Logger name.
    level : int, default logging.INFO
        Logging level.
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(name)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class CleanerLogger:
    """
    Lightweight logging wrapper that respects a verbose flag and records log entries.
    """
    def __init__(self, verbose: bool = True, logger: Optional[logging.Logger] = None):
        self.verbose = verbose
        self.logger = logger or get_logger()
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        """Log an informational message if verbose is True, and record it in history."""
        self.messages.append(message)
        if self.verbose:
            self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message if verbose is True, and record it in history."""
        self.messages.append(f"WARNING: {message}")
        if self.verbose:
            self.logger.warning(message)

    def clear(self) -> None:
        """Clear recorded log messages."""
        self.messages.clear()
