"""
original_split_benchmark/tests/test_benchmark_isolation.py
Tests to ensure the benchmark code does not import from the main project.
"""
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent.parent

def test_no_main_project_imports():
    """
    Scans all Python files in original_split_benchmark/ to ensure they don't contain
    'from src.' or 'import src.'.
    """
    benchmark_dir = ROOT / "original_split_benchmark"
    violating_files = []

    for py_file in benchmark_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        if re.search(r'^from src\.', content, re.MULTILINE) or re.search(r'^import src\.', content, re.MULTILINE):
            violating_files.append(str(py_file))

    assert len(violating_files) == 0, f"Found main project imports in: {violating_files}"

def test_data_hash_consistency():
    """
    Checks if the files exist and their hashes match the known hashes.
    """
    import hashlib
    train_path = ROOT / "data/raw/UNSW_NB15_training-set.csv"
    test_path = ROOT / "data/raw/UNSW_NB15_testing-set.csv"

    assert train_path.exists()
    assert test_path.exists()

    def get_hash(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1048576), b''):
                h.update(chunk)
        return h.hexdigest()

    assert get_hash(train_path) == "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa"
    assert get_hash(test_path) == "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559"
