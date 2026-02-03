"""
Lightweight logging helpers for Slurm-friendly debugging.

Environment variables
---------------------
RHIME_LOG_LEVEL
    Logging level for the logger (default: "INFO").
    Examples: "DEBUG", "INFO", "WARNING", "ERROR".

RHIME_LOG_STYLE
    Formatting style for the *console* StreamHandler (default: "pretty").

    Accepted values:
      - "pretty" (default): compact, human-readable for Slurm .out
      - "psv" / "pipe": pipe-separated values (machine-parseable)

RHIME_LOG_FILE
    Optional path to a file that will receive *PSV-formatted* logs in addition to
    console output. If set, a FileHandler is attached with PSV formatting.

    Notes:
      - The file output is always PSV regardless of RHIME_LOG_STYLE, so you can
        keep console logs readable while still collecting parseable logs.
      - The file is opened in append mode.

Pandas parsing snippet (for RHIME_LOG_FILE output or console with RHIME_LOG_STYLE=psv)
-------------------------------------------------------------------------------------
import pandas as pd

df = pd.read_csv(
    "rhime.log.psv",
    sep="|",
    names=["time", "level", "logger", "file", "line", "func", "msg"],
    engine="python",
)

Tip: avoid using the '|' character in log messages if you rely on delimiter parsing.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import Iterator, Optional


# Console styles
# _FMT_PRETTY = "%(asctime)s %(levelname)-8s %(name)s " "%(filename)s:%(lineno)d:%(funcName)s - %(message)s"
_FMT_PRETTY = "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d:%(funcName)s - %(message)s"

# Machine-parseable: stable columns, no extra ':' parsing required
_FMT_PSV = "%(asctime)s|%(levelname)s|%(name)s|%(filename)s|%(lineno)d|%(funcName)s|%(message)s"

_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_rhime_logger(
    name: str = "rhime",
    level: Optional[str] = None,
    stream=None,
    fmt: Optional[str] = None,
    datefmt: str = _DEFAULT_DATEFMT,
) -> logging.Logger:
    """
    Configure and return a logger suitable for Slurm streaming logs.

    - Console handler -> stdout by default (goes to Slurm .out)
    - Level from RHIME_LOG_LEVEL if not provided
    - Console style from RHIME_LOG_STYLE if fmt is not provided
    - Optional PSV file logging via RHIME_LOG_FILE
    - Idempotent-ish: avoids adding duplicate handlers; updates format/level if re-called
    """
    if stream is None:
        stream = sys.stdout

    if level is None:
        level = os.environ.get("RHIME_LOG_LEVEL", "INFO")

    # Choose console format if not explicitly provided
    if fmt is None:
        style = os.environ.get("RHIME_LOG_STYLE", "pretty").strip().lower()
        if style in {"psv", "pipe", "pipes"}:
            fmt = _FMT_PSV
        else:
            fmt = _FMT_PRETTY

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # prevent double logging via root handlers

    # --- Console handler (StreamHandler) ---
    stream_handler = None
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is stream:
            stream_handler = h
            break

    if stream_handler is None:
        stream_handler = logging.StreamHandler(stream)
        logger.addHandler(stream_handler)

    stream_handler.setLevel(logger.level)
    stream_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    # --- Optional file handler (always PSV) ---
    log_file = os.environ.get("RHIME_LOG_FILE")
    if log_file:
        file_handler = None
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(
                log_file
            ):
                file_handler = h
                break

        if file_handler is None:
            # Delay open until first emit to be resilient in some environments.
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8", delay=True)
            logger.addHandler(file_handler)

        file_handler.setLevel(logger.level)
        file_handler.setFormatter(logging.Formatter(fmt=_FMT_PSV, datefmt=datefmt))

    return logger


_LOG_STEP_STACKLEVEL = 3  # 2 shows contextlib.py; 3 typically points to your `with log_step(...)` line


@contextlib.contextmanager
def log_step(logger: logging.Logger, message: str, level: int = logging.INFO) -> Iterator[None]:
    """
    Context manager to log start/end of a step + timing.

    Uses stacklevel=_LOG_STEP_STACKLEVEL so filename/lineno/funcName point at the caller site
    (the `with log_step(...):` line), not inside this helper.
    """
    t0 = time.perf_counter()
    logger.log(level, f"START: {message}", stacklevel=_LOG_STEP_STACKLEVEL)
    try:
        yield
    except Exception:
        dt = time.perf_counter() - t0
        logger.exception(f"FAIL:  {message} (dt={dt:.2f}s)", stacklevel=_LOG_STEP_STACKLEVEL)
        raise
    else:
        dt = time.perf_counter() - t0
        logger.log(level, f"END:   {message} (dt={dt:.2f}s)", stacklevel=_LOG_STEP_STACKLEVEL)
