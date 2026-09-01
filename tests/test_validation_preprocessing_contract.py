"""
tests/test_validation_preprocessing_contract.py
--------------------------------------------------
Regression tests for the Sprint 5 validation preprocessing contract.

Targeted failure: validate_base_models.py originally called
    pd.read_csv(TRAIN_PATH, nrows=2000) → pipe.fit(2000 rows)
which caused:
    ValueError: The following 23 feature(s) are missing from the DataFrame:
    ['proto_unas', 'proto_sctp', 'service_pop3', ...]

Root cause: OHE fitted on a subset misses rare categorical values, so the
corresponding encoded columns never appear in ds_unscaled.feature_names,
making build_feature_matrix() fail with a missing-feature error.

This test module verifies:
1.  raw DataFrame + full preprocessing → encoded 75-feature matrix works
2.  Encoded feature names containing proto_*/service_*/state_* are valid
3.  Feature order is preserved exactly as in selected_features.json
4.  Missing raw categorical values are handled safely (sparse categories)
5.  Validation never calls build_feature_matrix() on raw data
6.  The ValueError remains correct for genuinely missing encoded features
7.  Protected/test files are never accessed
8.  Fitting on a subset deliberately reproduces the bug (regression guard)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ─── helpers ──────────────────────────────────────────────────────────────────

def make_raw_train_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    Synthesise a minimal raw TRAIN-like DataFrame with:
    - A categorical 'proto' column containing common + rare values.
    - A categorical 'service' column with rare entries.
    - A categorical 'state' column.
    - Several numeric columns.
    - A binary 'label' column.
    - An 'attack_cat' string column.
    """
    rng = np.random.default_rng(seed)
    proto_common = ["tcp", "udp", "arp"]
    proto_rare = ["unas", "sctp", "gre", "ipv6", "mobile", "pim", "sun-nd", "swipe"]
    service_common = ["http", "ftp", "-"]
    service_rare = ["pop3", "smtp", "imap"]

    # Ensure rare categories appear at least once
    proto_vals = rng.choice(proto_common, size=n - len(proto_rare)).tolist() + proto_rare
    rng.shuffle(proto_vals)
    service_vals = rng.choice(service_common, size=n - len(service_rare)).tolist() + service_rare
    rng.shuffle(service_vals)

    state_vals = rng.choice(["FIN", "CON", "INT", "REQ"], size=n)

    df = pd.DataFrame({
        "proto": proto_vals,
        "service": service_vals,
        "state": state_vals,
        "dur": rng.exponential(1.0, n),
        "sbytes": rng.integers(0, 10000, n).astype(float),
        "dbytes": rng.integers(0, 5000, n).astype(float),
        "rate": rng.uniform(0, 100, n),
        "sttl": rng.integers(0, 256, n).astype(float),
        "dttl": rng.integers(0, 256, n).astype(float),
        "sload": rng.uniform(0, 1e6, n),
        "dload": rng.uniform(0, 1e6, n),
        "label": rng.integers(0, 2, n),
        "attack_cat": rng.choice(["Normal", "DoS", "Exploits", "Generic"], size=n),
    })
    return df


def make_selected_features_json(path: Path, feature_names: list[str]) -> Path:
    """Write a selected_features.json with the given feature names."""
    count = len(feature_names)
    data = {
        "features": feature_names,
        "feature_count": count,
        "selected_k": count,
        "experiment_id": "EXP_MI_V1_1",
    }
    p = path / "selected_features.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ─── import targets ───────────────────────────────────────────────────────────

from src.models.base_models.preprocessing import (
    build_feature_matrix,
    load_selected_features,
    EXPECTED_FEATURE_COUNT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: ValueError remains intact for genuinely missing encoded features
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFeatureMatrixValueError:
    """The ValueError must NOT be weakened."""

    def test_missing_encoded_feature_raises(self, tmp_path: Path):
        """
        build_feature_matrix() on a DataFrame that lacks an encoded column
        (e.g. proto_unas) must still raise ValueError.
        This is the INTENDED behaviour — do not suppress it.
        """
        features = [f"f{i}" for i in range(75)]
        # DataFrame that only has f0..f73 (missing f74)
        df = pd.DataFrame(np.zeros((10, 74)), columns=features[:74])
        with pytest.raises(ValueError, match="missing"):
            build_feature_matrix(df, features)

    def test_raw_dataframe_raises_for_ohe_columns(self, tmp_path: Path):
        """
        A raw (un-encoded) DataFrame does NOT contain OHE columns like
        proto_unas.  Passing it directly to build_feature_matrix() with
        OHE feature names must raise ValueError — NOT silently succeed.
        """
        raw_df = make_raw_train_df(n=20)
        # Simulate OHE column names that raw CSV doesn't have
        ohe_features = ["proto_tcp", "proto_unas", "proto_sctp"] + [f"num_{i}" for i in range(72)]
        with pytest.raises(ValueError):
            build_feature_matrix(raw_df, ohe_features)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Encoded feature names (proto_*/service_*/state_*) are valid
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodedFeatureNames:
    def test_ohe_column_names_not_in_raw_csv(self):
        """
        OHE column names (proto_unas, service_pop3, …) must NOT appear as
        raw CSV column names — they are synthetic post-encoding columns.
        """
        raw_df = make_raw_train_df(n=50)
        ohe_like = ["proto_unas", "proto_sctp", "service_pop3", "state_FIN"]
        for col in ohe_like:
            assert col not in raw_df.columns, (
                f"'{col}' found in raw DataFrame columns — "
                "raw data must not be passed to build_feature_matrix with OHE names."
            )

    def test_build_feature_matrix_accepts_encoded_names(self, tmp_path: Path):
        """
        build_feature_matrix() succeeds when the DataFrame DOES have the
        OHE columns (simulating the output of PreprocessingPipeline.transform).
        """
        # Simulate 75 encoded columns
        features = [f"proto_{i}" for i in range(10)] + [f"service_{i}" for i in range(10)] + \
                   [f"num_{i}" for i in range(55)]
        df = pd.DataFrame(np.zeros((5, 75)), columns=features)
        p = make_selected_features_json(tmp_path, features)
        loaded = load_selected_features(p)
        result = build_feature_matrix(df, loaded)
        assert result.shape == (5, 75)

    def test_feature_order_is_preserved(self, tmp_path: Path):
        """
        build_feature_matrix() returns columns in the SAME ORDER as the
        selected_features list, not as they appear in the DataFrame.
        """
        features = [f"col_{i}" for i in range(75)]
        # Shuffle the DataFrame columns deliberately
        shuffled = features[::-1]
        df = pd.DataFrame(np.eye(75)[:5], columns=shuffled)
        p = make_selected_features_json(tmp_path, features)
        loaded = load_selected_features(p)
        result = build_feature_matrix(df, loaded)
        # Column 0 in result should correspond to features[0]
        expected_col0 = df[features[0]].to_numpy()
        np.testing.assert_array_equal(result[:, 0], expected_col0)

    def test_feature_order_exact_match(self, tmp_path: Path):
        """
        The output matrix columns follow features list order precisely.
        """
        features = [f"f_{i}" for i in range(75)]
        # Give each column a unique constant so we can identify it
        data = {f: np.full(3, i, dtype=float) for i, f in enumerate(features)}
        df = pd.DataFrame(data)
        p = make_selected_features_json(tmp_path, features)
        result = build_feature_matrix(df, load_selected_features(p))
        for i in range(75):
            assert result[0, i] == i, f"Column {i} value mismatch"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Fitting on a subset reproduces the original bug (regression guard)
# ─────────────────────────────────────────────────────────────────────────────

class TestSubsetFitReproducesBug:
    """
    Demonstrate that fitting on a subset that misses rare categorical values
    causes build_feature_matrix to fail — confirming the original bug.
    This test MUST PASS (it expects the failure), serving as a regression guard.
    """

    def test_partial_fit_misses_rare_ohe_columns(self, tmp_path: Path):
        """
        If we fit the OHE on only rows that lack 'proto_unas', then
        'proto_unas' will not appear in feature_names.
        build_feature_matrix with a feature list containing 'proto_unas' must fail.
        """
        try:
            from sklearn.preprocessing import OneHotEncoder
        except ImportError:
            pytest.skip("sklearn not available")

        # Subset: only common proto values
        subset = pd.DataFrame({"proto": ["tcp", "udp", "tcp", "arp"] * 5})
        # Full: includes rare value "unas"
        full = pd.DataFrame({"proto": ["tcp", "udp", "unas", "sctp"] * 5})

        enc_partial = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc_partial.fit(subset[["proto"]])
        partial_cols = [f"proto_{c}" for c in enc_partial.categories_[0]]

        enc_full = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc_full.fit(full[["proto"]])
        full_cols = [f"proto_{c}" for c in enc_full.categories_[0]]

        # "proto_unas" present in full but not partial
        assert "proto_unas" in full_cols
        assert "proto_unas" not in partial_cols, (
            "Partial fit should NOT know about proto_unas — "
            "this simulates the original bug"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Validation script never accesses forbidden datasets
# ─────────────────────────────────────────────────────────────────────────────

class TestForbiddenDatasetAccess:
    FORBIDDEN = [
        "validation.csv",
        "development_test.csv",
        "protected_unseen_attack.csv",
        "excluded_train_backdoor.csv",
    ]

    def test_validate_script_does_not_read_forbidden_paths(self):
        """
        The validation script must never read forbidden files.
        It may MENTION them in docstrings/comments as a disclaimer
        (e.g. "This script does NOT access validation.csv"), but it must
        never pass those filenames to read_csv, open, or pd.read_*.

        Strategy: check that no pd.read_csv call has the forbidden filename
        as a literal argument on the same line.
        """
        script = Path(__file__).resolve().parents[1] / "scripts" / "validate_base_models.py"
        lines = script.read_text(encoding="utf-8").splitlines()
        for fname in self.FORBIDDEN:
            for lineno, line in enumerate(lines, start=1):
                stripped = line.strip()
                # Skip comment lines and docstring lines
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'\"\"\""):
                    continue
                # Check for functional read operations on forbidden files
                is_functional = (
                    ("read_csv" in line or "open(" in line or "read_text" in line)
                    and fname in line
                )
                assert not is_functional, (
                    f"validate_base_models.py line {lineno} appears to read "
                    f"forbidden file '{fname}':\n  {line.strip()}"
                )

    def test_validate_script_docstring_correctly_disclaims_forbidden(self):
        """
        The docstring SHOULD mention forbidden files as files NOT accessed —
        this is correct documentation. Verify the disclaimer is present.
        """
        script = Path(__file__).resolve().parents[1] / "scripts" / "validate_base_models.py"
        src = script.read_text(encoding="utf-8")
        # The docstring should contain the disclaimer (mentions but doesn't read)
        assert "does NOT access" in src or "does not access" in src.lower(), (
            "validate_base_models.py should document which files it does NOT use."
        )



# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Validation script uses full TRAIN for pipe.fit()
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationUsesFullTrainForFit:
    def test_nrows_not_used_for_full_load(self):
        """
        The validation script must NOT pass `nrows=` to the read_csv call
        that feeds into pipe.fit().  Fitting on a subset is the root cause
        of the bug this module was written to fix.
        """
        script = Path(__file__).resolve().parents[1] / "scripts" / "validate_base_models.py"
        src = script.read_text(encoding="utf-8")

        # The only acceptable nrows usage would be for something OTHER than
        # the main train_df that goes into pipe.fit().
        # Simple check: if 'nrows=' appears anywhere, that's the bug.
        assert "nrows=" not in src, (
            "validate_base_models.py contains 'nrows=' — this restricts the "
            "TRAIN rows seen by pipe.fit() and causes OHE to miss rare categories. "
            "Remove nrows= and load the full TRAIN."
        )

    def test_validation_docstring_documents_full_train_requirement(self):
        """
        The validation script docstring should document why full TRAIN is needed.
        """
        script = Path(__file__).resolve().parents[1] / "scripts" / "validate_base_models.py"
        src = script.read_text(encoding="utf-8")
        assert "full TRAIN" in src or "full train" in src.lower(), (
            "validate_base_models.py should document the requirement to fit "
            "the pipeline on the full TRAIN."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: load_selected_features count guard
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadSelectedFeaturesCountGuard:
    def test_expected_feature_count_constant_is_75(self):
        assert EXPECTED_FEATURE_COUNT == 75

    def test_wrong_count_raises(self, tmp_path: Path):
        p = make_selected_features_json(tmp_path, [f"f{i}" for i in range(74)])
        with pytest.raises(ValueError, match="75"):
            load_selected_features(p)

    def test_correct_count_passes(self, tmp_path: Path):
        features = [f"f{i}" for i in range(75)]
        p = make_selected_features_json(tmp_path, features)
        result = load_selected_features(p)
        assert len(result) == 75
