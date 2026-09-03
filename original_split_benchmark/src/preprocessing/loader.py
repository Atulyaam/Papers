"""
src/preprocessing/loader.py
----------------------------
Raw CSV loader for the UNSW-NB15 IDS project.

Responsibilities:
- Load raw CSV files exactly as stored (no column renaming, no value changes).
- Validate file presence before attempting to read.
- Log file paths, sizes, and shapes.
- Return DataFrames keyed by split name.

LEAKAGE POLICY:
- This module performs NO transforms (no fit, no fit_transform, no scaling,
  no encoding, no imputation, no column dropping).
- The returned DataFrames are raw, unmodified reads of the source CSVs.

data/raw/ is treated as IMMUTABLE after download.
"""

import logging
from pathlib import Path

import pandas as pd


def load_raw_unswnb15(config: dict, logger: logging.Logger | None = None) -> dict[str, pd.DataFrame]:
    """
    Load the official UNSW-NB15 pre-split CSV files.

    The file paths are read from the 'data' section of the project config dict.
    Expected config keys:
        config["data"]["raw_dir"]        — directory containing raw CSVs
        config["data"]["train_file"]     — training CSV filename
        config["data"]["test_file"]      — testing CSV filename
        config["data"]["header"]         — bool: True if CSVs have embedded headers
        config["data"]["encoding"]       — file encoding (default "utf-8")

    Parameters
    ----------
    config : dict
        Parsed project_config.yaml as a dictionary.
    logger : logging.Logger | None
        Optional logger. If None, a module-level logger is used.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: "train", "test"
        Values: raw DataFrames with original column names intact.

    Raises
    ------
    FileNotFoundError
        If any expected raw file is absent.
    KeyError
        If required config keys are missing.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    raw_dir = Path(config["data"]["raw_dir"])
    train_file = raw_dir / config["data"]["train_file"]
    test_file = raw_dir / config["data"]["test_file"]
    header = config["data"].get("header", True)
    encoding = config["data"].get("encoding", "utf-8")

    # --- Validate presence before reading ---
    for fpath in (train_file, test_file):
        if not fpath.exists():
            raise FileNotFoundError(
                f"Expected raw file not found: {fpath}\n"
                f"Place the official UNSW-NB15 CSV files in: {raw_dir.resolve()}"
            )

    header_arg = 0 if header else None  # pandas header parameter

    splits = {}
    for split_name, fpath in [("train", train_file), ("test", test_file)]:
        file_size = fpath.stat().st_size
        logger.info(
            "Loading | split=%s | file=%s | size_bytes=%d",
            split_name,
            fpath.name,
            file_size,
        )
        df = pd.read_csv(fpath, header=header_arg, encoding=encoding, low_memory=False)
        logger.info(
            "Loaded  | split=%s | shape=%s | columns=%d",
            split_name,
            df.shape,
            len(df.columns),
        )
        splits[split_name] = df

    return splits
