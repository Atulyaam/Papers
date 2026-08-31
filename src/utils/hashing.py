"""
src/utils/hashing.py
--------------------
SHA-256 fingerprinting utilities for the UNSW-NB15 IDS project.

Used to:
- fingerprint raw source files (immutability verification)
- fingerprint derived DataFrames (provenance)
- record file hashes in data/audit/file_hashes.json

Design:
- Files are hashed by streaming in 8 MB chunks (safe for large CSVs).
- DataFrames are hashed via a deterministic serialisation to avoid memory
  issues with in-memory copies.
"""

import hashlib
import io
from pathlib import Path

import pandas as pd


_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def sha256_file(path: Path | str) -> str:
    """
    Compute the SHA-256 hex digest of a file, streaming in chunks.

    Parameters
    ----------
    path : Path | str
        Path to the file to hash.

    Returns
    -------
    str
        Lowercase hex digest string (64 characters).

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash — file not found: {path}")

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_dataframe(df: pd.DataFrame) -> str:
    """
    Compute a deterministic SHA-256 fingerprint of a pandas DataFrame.

    Serialisation strategy:
    - Convert to CSV bytes in memory (deterministic column ordering).
    - Hash the bytes.

    This avoids hashing Python object internals and is reproducible across
    processes as long as the column order and content are identical.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to fingerprint.

    Returns
    -------
    str
        Lowercase hex digest string.
    """
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    raw_bytes = buf.getvalue().encode("utf-8")
    return hashlib.sha256(raw_bytes).hexdigest()
