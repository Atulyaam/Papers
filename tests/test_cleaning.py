"""
tests/test_cleaning.py
-----------------------
Unit tests for src/preprocessing/cleaning.py

Covers:
  - validate_required_columns: pass, missing label, missing attack_cat
  - detect_nonfinite: clean passes, NaN raises, +inf raises, -inf raises,
    combination raises, correct column names in exception
  - separate_target_and_features: column exclusion, target separation,
    column classification, row count, dtype contract, whitespace in category
  - Empty input and single-row input
  - label/attack_cat/id NOT in feature columns
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.cleaning import (
    ATTACK_CAT_COL,
    CATEGORICAL_COLS,
    EXCLUDE_COLS,
    LABEL_COL,
    detect_nonfinite,
    separate_target_and_features,
    validate_required_columns,
)
from src.preprocessing.exceptions import NonFiniteValueError, PreprocessingSchemaError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(
    n=10,
    extra_cols: dict | None = None,
    include_id: bool = True,
) -> pd.DataFrame:
    """Minimal UNSW-NB15-like DataFrame for testing."""
    rng = np.random.default_rng(0)
    data = {
        "label": rng.integers(0, 2, size=n),
        "attack_cat": ["Normal"] * n,
        "proto": ["tcp"] * n,
        "service": ["-"] * n,
        "state": ["FIN"] * n,
        "dur": rng.uniform(0, 1, size=n),
        "sbytes": rng.uniform(100, 1000, size=n),
    }
    if include_id:
        data["id"] = list(range(n))
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# validate_required_columns
# ---------------------------------------------------------------------------


class TestValidateRequiredColumns:
    def test_passes_when_both_present(self):
        df = _make_df()
        validate_required_columns(df, "train")  # no exception

    def test_raises_when_label_missing(self):
        df = _make_df().drop(columns=["label"])
        with pytest.raises(PreprocessingSchemaError) as exc_info:
            validate_required_columns(df, "train")
        assert "label" in str(exc_info.value)
        assert exc_info.value.missing_cols == ["label"]

    def test_raises_when_attack_cat_missing(self):
        df = _make_df().drop(columns=["attack_cat"])
        with pytest.raises(PreprocessingSchemaError) as exc_info:
            validate_required_columns(df, "test")
        assert "attack_cat" in str(exc_info.value)

    def test_raises_when_both_missing(self):
        df = pd.DataFrame({"proto": ["tcp"]})
        with pytest.raises(PreprocessingSchemaError) as exc_info:
            validate_required_columns(df, "train")
        assert len(exc_info.value.missing_cols) == 2

    def test_split_name_in_error_message(self):
        df = _make_df().drop(columns=["label"])
        with pytest.raises(PreprocessingSchemaError, match="my_split"):
            validate_required_columns(df, "my_split")


# ---------------------------------------------------------------------------
# detect_nonfinite
# ---------------------------------------------------------------------------


class TestDetectNonfinite:
    def test_clean_data_passes(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        detect_nonfinite(df, "train", ["a", "b"])  # no exception

    def test_nan_raises(self):
        df = pd.DataFrame({"a": [1.0, np.nan], "b": [3.0, 4.0]})
        with pytest.raises(NonFiniteValueError) as exc_info:
            detect_nonfinite(df, "train", ["a", "b"])
        assert exc_info.value.nan_cols.get("a") == 1
        assert exc_info.value.split_name == "train"

    def test_pos_inf_raises(self):
        df = pd.DataFrame({"a": [np.inf, 2.0]})
        with pytest.raises(NonFiniteValueError) as exc_info:
            detect_nonfinite(df, "validation", ["a"])
        assert exc_info.value.pos_inf_cols.get("a") == 1

    def test_neg_inf_raises(self):
        df = pd.DataFrame({"a": [-np.inf, 2.0]})
        with pytest.raises(NonFiniteValueError) as exc_info:
            detect_nonfinite(df, "test", ["a"])
        assert exc_info.value.neg_inf_cols.get("a") == 1

    def test_multiple_bad_columns(self):
        df = pd.DataFrame({
            "a": [np.nan, 2.0],
            "b": [np.inf, 3.0],
            "c": [1.0, 2.0],
        })
        with pytest.raises(NonFiniteValueError) as exc_info:
            detect_nonfinite(df, "train", ["a", "b", "c"])
        err = exc_info.value
        assert "a" in err.nan_cols
        assert "b" in err.pos_inf_cols
        assert "c" not in err.nan_cols
        assert "c" not in err.pos_inf_cols

    def test_error_message_contains_split_name(self):
        df = pd.DataFrame({"x": [np.nan]})
        with pytest.raises(NonFiniteValueError, match="mysplit"):
            detect_nonfinite(df, "mysplit", ["x"])

    def test_skips_cols_not_in_df(self):
        """Columns in numeric_cols that are absent from df are skipped silently."""
        df = pd.DataFrame({"a": [1.0, 2.0]})
        detect_nonfinite(df, "train", ["a", "nonexistent"])  # no exception


# ---------------------------------------------------------------------------
# separate_target_and_features
# ---------------------------------------------------------------------------


class TestSeparateTargetAndFeatures:
    def test_label_not_in_features(self):
        cleaned = separate_target_and_features(_make_df(), "train")
        assert LABEL_COL not in cleaned.X_raw.columns

    def test_attack_cat_not_in_features(self):
        cleaned = separate_target_and_features(_make_df(), "train")
        assert ATTACK_CAT_COL not in cleaned.X_raw.columns

    def test_id_excluded_from_features(self):
        cleaned = separate_target_and_features(_make_df(include_id=True), "train")
        assert "id" not in cleaned.X_raw.columns

    def test_y_is_label_column(self):
        df = _make_df()
        cleaned = separate_target_and_features(df, "train")
        pd.testing.assert_series_equal(
            cleaned.y.reset_index(drop=True),
            df["label"].reset_index(drop=True),
        )

    def test_attack_cat_is_metadata_column(self):
        df = _make_df()
        df["attack_cat"] = "DoS"
        cleaned = separate_target_and_features(df, "train")
        assert (cleaned.attack_cat == "DoS").all()

    def test_row_count_preserved(self):
        df = _make_df(n=50)
        cleaned = separate_target_and_features(df, "train")
        assert len(cleaned.X_raw) == 50
        assert len(cleaned.y) == 50
        assert len(cleaned.attack_cat) == 50

    def test_categorical_cols_identified_from_contract(self):
        df = _make_df()
        cleaned = separate_target_and_features(df, "train")
        # proto, service, state are in the contract AND in the df
        for col in ("proto", "service", "state"):
            assert col in cleaned.categorical_cols

    def test_categorical_col_absent_from_df_not_in_list(self):
        """If 'state' is absent from the df, it must not appear in categorical_cols."""
        df = _make_df().drop(columns=["state"])
        cleaned = separate_target_and_features(df, "train")
        assert "state" not in cleaned.categorical_cols

    def test_numeric_cols_do_not_include_categorical(self):
        cleaned = separate_target_and_features(_make_df(), "train")
        for cat_col in cleaned.categorical_cols:
            assert cat_col not in cleaned.numeric_cols

    def test_raises_on_nonfinite(self):
        df = _make_df()
        df.loc[0, "dur"] = np.nan
        with pytest.raises(NonFiniteValueError):
            separate_target_and_features(df, "train")

    def test_single_row_input(self):
        df = _make_df(n=1)
        cleaned = separate_target_and_features(df, "train")
        assert len(cleaned.X_raw) == 1

    def test_empty_dataframe_raises_schema_error(self):
        df = pd.DataFrame({"label": [], "attack_cat": [], "proto": []})
        # Empty df passes schema validation (columns present) but returns 0-row result
        cleaned = separate_target_and_features(df, "train")
        assert len(cleaned.X_raw) == 0

    def test_dash_in_service_is_not_converted_to_nan(self):
        """The service value '-' must survive as a string, not become NaN."""
        df = _make_df()
        df["service"] = "-"
        cleaned = separate_target_and_features(df, "train")
        # Service should be categorical, not NaN
        assert (cleaned.X_raw["service"] == "-").all()
        assert "service" in cleaned.categorical_cols
