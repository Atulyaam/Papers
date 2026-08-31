"""tests/test_schema_audit.py — Unit tests for src/preprocessing/schema_audit.py"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.schema_audit import (
    audit_attack_cat_distribution,
    audit_attack_cat_raw_strings,
    audit_dataframe,
    audit_label_distribution,
    check_overlap,
)


@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "label": [0, 1, 1, 0],
        "attack_cat": ["Normal", "DoS", "Backdoor", "Normal"],
        "feat1": [1.0, 2.0, 3.0, 4.0],
    })


class TestAuditDataframe:
    def test_returns_correct_shape(self, simple_df):
        result = audit_dataframe(simple_df, "test")
        assert result["shape"]["rows"] == 4
        assert result["shape"]["cols"] == 3

    def test_null_count_correct(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]})
        result = audit_dataframe(df, "train")
        assert result["null_counts"]["a"] == 1
        assert result["null_counts"]["b"] == 1

    def test_null_count_zero_when_clean(self, simple_df):
        result = audit_dataframe(simple_df, "train")
        assert all(v == 0 for v in result["null_counts"].values())

    def test_inf_count_detected(self):
        df = pd.DataFrame({"x": [1.0, np.inf, -np.inf, 2.0]})
        result = audit_dataframe(df, "test")
        assert result["inf_counts"]["x"]["pos_inf"] == 1
        assert result["inf_counts"]["x"]["neg_inf"] == 1

    def test_inf_count_zero_when_clean(self, simple_df):
        result = audit_dataframe(simple_df, "test")
        assert result["inf_counts"]["feat1"]["pos_inf"] == 0

    def test_duplicate_count(self):
        df = pd.DataFrame({"a": [1, 2, 1], "b": [3, 4, 3]})
        result = audit_dataframe(df, "train")
        assert result["duplicate_row_count"] == 1

    def test_no_duplicates_when_clean(self, simple_df):
        result = audit_dataframe(simple_df, "train")
        assert result["duplicate_row_count"] == 0

    def test_split_name_recorded(self, simple_df):
        result = audit_dataframe(simple_df, "my_split")
        assert result["split"] == "my_split"

    def test_columns_listed(self, simple_df):
        result = audit_dataframe(simple_df, "test")
        assert "label" in result["columns"]
        assert "attack_cat" in result["columns"]


class TestAuditLabelDistribution:
    def test_correct_counts(self):
        df = pd.DataFrame({"label": [0, 0, 1, 1, 1]})
        result = audit_label_distribution(df, "label")
        assert result["counts"][0] == 2
        assert result["counts"][1] == 3

    def test_missing_label_col_returns_error_dict(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = audit_label_distribution(df, "label")
        assert "error" in result

    def test_total_correct(self):
        df = pd.DataFrame({"label": [0, 1, 0, 1, 0]})
        result = audit_label_distribution(df, "label")
        assert result["total"] == 5

    def test_no_unexpected_values_for_clean_data(self):
        df = pd.DataFrame({"label": [0, 1, 0]})
        result = audit_label_distribution(df, "label")
        assert result["unexpected_values"] == []


class TestAuditAttackCatRawStrings:
    def test_preserves_whitespace(self):
        """Raw strings with leading/trailing whitespace must be preserved."""
        df = pd.DataFrame({"attack_cat": [" Backdoor ", "Normal", "DoS"]})
        result = audit_attack_cat_raw_strings(df, "attack_cat")
        assert " Backdoor " in result

    def test_sorted(self):
        df = pd.DataFrame({"attack_cat": ["DoS", "Analysis", "Backdoor"]})
        result = audit_attack_cat_raw_strings(df, "attack_cat")
        assert result == sorted(result)

    def test_unique(self):
        df = pd.DataFrame({"attack_cat": ["DoS", "DoS", "Normal"]})
        result = audit_attack_cat_raw_strings(df, "attack_cat")
        assert len(result) == len(set(result))


class TestAuditAttackCatDistribution:
    def test_counts_correct(self):
        df = pd.DataFrame({"attack_cat": ["Backdoor", "Normal", "Backdoor", "DoS"]})
        result = audit_attack_cat_distribution(df, "attack_cat")
        assert result["Backdoor"] == 2
        assert result["Normal"] == 1

    def test_preserves_raw_strings(self):
        """Exact raw strings (with spaces) are preserved as dict keys."""
        df = pd.DataFrame({"attack_cat": [" Backdoor ", " Backdoor "]})
        result = audit_attack_cat_distribution(df, "attack_cat")
        assert " Backdoor " in result


class TestCheckOverlap:
    def test_no_overlap(self):
        df_a = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df_b = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
        result = check_overlap(df_a, df_b)
        assert result["overlap_count"] == 0

    def test_one_row_overlap(self):
        df_a = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df_b = pd.DataFrame({"a": [2, 5], "b": [4, 6]})
        result = check_overlap(df_a, df_b)
        assert result["overlap_count"] == 1

    def test_full_overlap(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = check_overlap(df, df.copy())
        assert result["overlap_count"] == 2

    def test_row_counts_recorded(self):
        df_a = pd.DataFrame({"x": [1, 2, 3]})
        df_b = pd.DataFrame({"x": [4, 5]})
        result = check_overlap(df_a, df_b)
        assert result["train_row_count"] == 3
        assert result["test_row_count"] == 2
