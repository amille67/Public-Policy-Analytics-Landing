"""Structured logging utilities for PPA pipelines."""

from __future__ import annotations

import logging
import sys
from typing import Any


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create a configured logger for a pipeline module.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Configured logging.Logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def log_step(
    logger: logging.Logger,
    chapter: str,
    step: str,
    **extra: Any,
) -> None:
    """Log a pipeline step with structured fields.

    Args:
        logger: Logger instance.
        chapter: Chapter identifier (e.g., "ch01").
        step: Step name (e.g., "load", "feature_engineering").
        **extra: Additional key-value fields to include in the message.
    """
    parts = [f"chapter={chapter}", f"step={step}"]
    parts.extend(f"{k}={v}" for k, v in extra.items())
    logger.info(" | ".join(parts))
