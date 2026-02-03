from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import Iterator, Optional


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_rhime_logger(
    name: str = "rhime",
    level: Optional[str] = None,
    stream=None,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> logging.Logger:
    """
    Configure and return a logger suitable for Slurm streaming logs.

    - StreamHandler -> stdout by default (goes to Slurm .out)
    - Level from RHIME_LOG_LEVEL if not provided
    - Idempotent: won't add multiple handlers if called repeatedly
    """
    if stream is None:
        stream = sys.stdout

    if level is None:
        level = os.environ.get("RHIME_LOG_LEVEL", "INFO")

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # prevent double logging via root handlers

    # Idempotent handler setup
    if not any(isinstance(h, logging.StreamHandler) and h.stream is stream for h in logger.handlers):
        handler = logging.StreamHandler(stream)
        handler.setLevel(logger.level)
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(handler)

    return logger


@contextlib.contextmanager
def log_step(logger: logging.Logger, message: str, level: int = logging.INFO) -> Iterator[None]:
    """
    Context manager to log start/end of a step + timing.
    """
    t0 = time.perf_counter()
    logger.log(level, f"START: {message}")
    try:
        yield
    except Exception:
        dt = time.perf_counter() - t0
        logger.exception(f"FAIL:  {message} (dt={dt:.2f}s)")
        raise
    else:
        dt = time.perf_counter() - t0
        logger.log(level, f"END:   {message} (dt={dt:.2f}s)")
