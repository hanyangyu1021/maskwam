"""Lightweight logging helpers."""

from __future__ import annotations

import logging

_DEF_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"


def get_logger(name: str = "maskwam", level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEF_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
