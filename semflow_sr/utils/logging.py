"""Minimal logging setup."""
from __future__ import annotations
import logging, sys

_FMT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "semflow_sr", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger
