"""
src/utils/logging_utils.py
--------------------------
Structured experiment logging for the UNSW-NB15 IDS project.

All experiment code must use this logger rather than bare print() calls.
Logs are written to:
    results/logs/<experiment_id>/run.log

Usage:
    from src.utils.logging_utils import get_experiment_logger
    logger = get_experiment_logger("EXP_DATA_ACQUISITION_AUDIT", "results/logs")
    logger.info("...")
"""

import logging
import os
from pathlib import Path


def get_experiment_logger(
    experiment_id: str,
    log_dir: str = "results/logs",
    level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Create (or retrieve) a structured logger for the given experiment.

    The logger writes to:
        <log_dir>/<experiment_id>/run.log
    and also streams to the console.

    Parameters
    ----------
    experiment_id : str
        Stable experiment identifier (e.g., "EXP_DATA_ACQUISITION_AUDIT").
    log_dir : str
        Base directory for log files. Defaults to "results/logs".
    level : int
        Logging level. Defaults to logging.DEBUG.

    Returns
    -------
    logging.Logger
        A configured Logger instance named after experiment_id.
    """
    # Resolve the log directory relative to the caller's CWD.
    log_path = Path(log_dir) / experiment_id
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "run.log"

    logger = logging.getLogger(experiment_id)

    # Avoid duplicate handlers if the logger was already configured.
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # File handler — always append so reruns are traceable.
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler.
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("Logger initialised | log_file=%s", log_file)
    return logger
