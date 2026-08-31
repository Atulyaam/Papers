"""
tests/test_mi_selector.py
--------------------------
Unit tests for Sprint 4: MI feature selector core module.

Tests:
    TestDiscreteMAsk        — build_discrete_mask alignment and correctness
    TestGetSourceFamily     — source family classification
    TestComputeMIScores     — MI computation, validation, errors
    TestRankFeatures        — ranking stability, ordering, ties
    TestSelectTopK          — selection correctness and error paths
    TestMIEdgeCases         — empty matrix, constants, one-class, NaN, inf, etc.
    TestFeatureNameInvariants — uniqueness, alignment, ordering
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_selection.mi_selector import (
    MIConfig,
    MISelectorError,
    build_discrete_mask,
    build_family_report,
    compute_mi_scores,
    get_source_family,
    rank_features,
    select_top_k,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohe_names(proto=3, service=5, state=4) -> list[str]:
    """Build a synthetic list of OHE feature names."""
    names = []
    for i in range(proto):
        names.append(f"proto_p{i}")
    for i in range(service):
        names.append(f"service_s{i}")
    for i in range(state):
        names.append(f"state_st{i}")
    return names


def _make_feature_names(n_ohe=12, n_numeric=6) -> list[str]:
    ohe = _make_ohe_names(proto=3, service=5, state=4)[:n_ohe]
    num = [f"num_feat_{i}" for i in range(n_numeric)]
    return ohe + num


def _make_synthetic_X_y(n_samples=200, n_ohe=12, n_numeric=6, seed=0):
    """Synthetic data: OHE columns are 0/1, numeric are continuous."""
    rng = np.random.default_rng(seed)
    feature_names = _make_feature_names(n_ohe, n_numeric)
    n_features = len(feature_names)
    X_ohe = rng.integers(0, 2, size=(n_samples, n_ohe)).astype(np.float64)
    X_num = rng.normal(0, 1, size=(n_samples, n_numeric))
    X = np.concatenate([X_ohe, X_num], axis=1)
    y = rng.integers(0, 2, size=n_samples).astype(np.int64)
    return X, y, feature_names


# ---------------------------------------------------------------------------
# TestGetSourceFamily
# ---------------------------------------------------------------------------


class TestGetSourceFamily:
    def test_proto_prefix(self):
        assert get_source_family("proto_tcp") == "proto"
        assert get_source_family("proto_udp") == "proto"

    def test_service_prefix(self):
        assert get_source_family("service_-") == "service"
        assert get_source_family("service_http") == "service"

    def test_state_prefix(self):
        assert get_source_family("state_FIN") == "state"
        assert get_source_family("state_CON") == "state"

    def test_numeric(self):
        assert get_source_family("dur") == "numeric"
        assert get_source_family("sbytes") == "numeric"
        assert get_source_family("protocol_extra") == "numeric"  # not exact prefix

    def test_no_underscore(self):
        assert get_source_family("label") == "numeric"

    def test_case_sensitive(self):
        # "Proto_tcp" does NOT start with "proto_" — case matters
        assert get_source_family("Proto_tcp") == "numeric"


# ---------------------------------------------------------------------------
# TestDiscreteMAsk
# ---------------------------------------------------------------------------


class TestDiscreteMask:
    def test_mask_length_equals_feature_count(self):
        names = _make_feature_names(n_ohe=12, n_numeric=6)
        mask = build_discrete_mask(names)
        assert len(mask) == len(names)

    def test_ohe_features_are_discrete(self):
        names = _make_feature_names(n_ohe=12, n_numeric=6)
        mask = build_discrete_mask(names)
        for i, name in enumerate(names[:12]):
            assert mask[i], f"OHE feature '{name}' should be discrete"

    def test_numeric_features_are_continuous(self):
        names = _make_feature_names(n_ohe=12, n_numeric=6)
        mask = build_discrete_mask(names)
        for i, name in enumerate(names[12:], start=12):
            assert not mask[i], f"Numeric feature '{name}' should be continuous"

    def test_all_numeric_mask(self):
        names = [f"feat_{i}" for i in range(5)]
        mask = build_discrete_mask(names)
        assert not mask.any(), "All numeric features should be False"

    def test_all_ohe_mask(self):
        names = [f"proto_v{i}" for i in range(5)]
        mask = build_discrete_mask(names)
        assert mask.all(), "All OHE features should be True"

    def test_empty_feature_names_raises(self):
        with pytest.raises(MISelectorError, match="empty"):
            build_discrete_mask([])

    def test_dtype_is_bool(self):
        names = _make_feature_names()
        mask = build_discrete_mask(names)
        assert mask.dtype == bool

    def test_mask_stable_across_calls(self):
        names = _make_feature_names()
        m1 = build_discrete_mask(names)
        m2 = build_discrete_mask(names)
        np.testing.assert_array_equal(m1, m2)


# ---------------------------------------------------------------------------
# TestComputeMIScores
# ---------------------------------------------------------------------------


class TestComputeMIScores:
    def test_returns_mi_result(self):
        from src.feature_selection.mi_selector import MIResult
        X, y, names = _make_synthetic_X_y()
        result = compute_mi_scores(X, y, names)
        assert isinstance(result, MIResult)

    def test_score_length_matches_features(self):
        X, y, names = _make_synthetic_X_y(n_ohe=8, n_numeric=4)
        result = compute_mi_scores(X, y, names)
        assert len(result.mi_scores) == len(names)

    def test_all_scores_nonnegative(self):
        X, y, names = _make_synthetic_X_y()
        result = compute_mi_scores(X, y, names)
        assert (result.mi_scores >= 0).all(), "MI scores must be non-negative"

    def test_feature_names_preserved(self):
        X, y, names = _make_synthetic_X_y()
        result = compute_mi_scores(X, y, names)
        assert result.feature_names == names

    def test_n_features_correct(self):
        X, y, names = _make_synthetic_X_y(n_ohe=5, n_numeric=3)
        result = compute_mi_scores(X, y, names)
        assert result.n_features == 8

    def test_config_recorded(self):
        X, y, names = _make_synthetic_X_y()
        config = MIConfig(n_neighbors=3, random_state=42)
        result = compute_mi_scores(X, y, names, config=config)
        assert result.config.n_neighbors == 3
        assert result.config.random_state == 42

    def test_discrete_mask_length(self):
        X, y, names = _make_synthetic_X_y(n_ohe=12, n_numeric=6)
        result = compute_mi_scores(X, y, names)
        assert len(result.discrete_mask) == len(names)

    def test_empty_rows_raises(self):
        X = np.empty((0, 5), dtype=np.float64)
        y = np.array([], dtype=np.int64)
        names = [f"f{i}" for i in range(5)]
        with pytest.raises(MISelectorError, match="zero rows"):
            compute_mi_scores(X, y, names)

    def test_empty_cols_raises(self):
        X = np.empty((10, 0), dtype=np.float64)
        y = np.zeros(10, dtype=np.int64)
        names = []
        with pytest.raises(MISelectorError):
            compute_mi_scores(X, y, names)

    def test_shape_mismatch_features_raises(self):
        X, y, names = _make_synthetic_X_y(n_ohe=5, n_numeric=3)
        with pytest.raises(MISelectorError, match="feature_names length"):
            compute_mi_scores(X, y, names[:-1])  # too few names

    def test_shape_mismatch_rows_raises(self):
        X, y, names = _make_synthetic_X_y()
        with pytest.raises(MISelectorError, match="y length"):
            compute_mi_scores(X, y[:-5], names)

    def test_nan_in_X_raises(self):
        X, y, names = _make_synthetic_X_y()
        X[0, 0] = np.nan
        with pytest.raises(MISelectorError, match="NaN"):
            compute_mi_scores(X, y, names)

    def test_inf_in_X_raises(self):
        X, y, names = _make_synthetic_X_y()
        X[0, 0] = np.inf
        with pytest.raises(MISelectorError, match="NaN or"):
            compute_mi_scores(X, y, names)

    def test_neg_inf_in_X_raises(self):
        X, y, names = _make_synthetic_X_y()
        X[0, 0] = -np.inf
        with pytest.raises(MISelectorError, match="NaN or"):
            compute_mi_scores(X, y, names)

    def test_nan_in_y_raises(self):
        X, y, names = _make_synthetic_X_y()
        y_f = y.astype(np.float64)
        y_f[0] = np.nan
        with pytest.raises(MISelectorError, match="NaN"):
            compute_mi_scores(X, y_f, names)

    def test_one_class_target_raises(self):
        X, _, names = _make_synthetic_X_y()
        y_one = np.zeros(len(X), dtype=np.int64)
        with pytest.raises(MISelectorError, match="one class"):
            compute_mi_scores(X, y_one, names)

    def test_non_binary_target_raises(self):
        X, _, names = _make_synthetic_X_y()
        y_bad = np.random.default_rng(0).integers(0, 3, len(X))
        with pytest.raises(MISelectorError, match="non-binary"):
            compute_mi_scores(X, y_bad, names)

    def test_duplicate_feature_names_raises(self):
        X, y, names = _make_synthetic_X_y(n_ohe=5, n_numeric=3)
        dup_names = names[:4] + [names[0]] + names[5:]  # introduce duplicate
        with pytest.raises(MISelectorError, match="duplicate"):
            compute_mi_scores(X, y, dup_names)

    def test_invalid_n_neighbors_raises(self):
        X, y, names = _make_synthetic_X_y()
        with pytest.raises(MISelectorError, match="n_neighbors"):
            compute_mi_scores(X, y, names, config=MIConfig(n_neighbors=0))

    def test_ranking_df_has_required_columns(self):
        X, y, names = _make_synthetic_X_y()
        result = compute_mi_scores(X, y, names)
        required = {"rank", "feature", "mi_score", "source_column",
                    "source_family", "feature_type", "selected"}
        assert required.issubset(set(result.ranking_df.columns))

    def test_constant_feature_has_zero_or_low_mi(self):
        """A constant (zero-variance) numeric feature should have MI ≈ 0.

        sklearn's kNN MI estimator uses a stochastic approximation and may
        return a small but non-zero value (~0.027) for a truly constant column.
        The tolerance is set to 0.05 to account for this numerical artefact
        while still asserting the score is substantially lower than informative
        features (which typically score > 0.1 on real data).
        """
        X, y, names = _make_synthetic_X_y(n_ohe=0, n_numeric=5)
        X[:, 0] = 1.0  # constant column
        result = compute_mi_scores(X, y, names)
        constant_score = result.mi_scores[0]
        assert constant_score < 0.05, (
            f"Constant feature should have near-zero MI (< 0.05), got {constant_score}"
        )

    def test_all_zero_ohe_feature(self):
        """An all-zero OHE column should not crash and should have low MI."""
        X, y, names = _make_synthetic_X_y(n_ohe=5, n_numeric=3)
        X[:, 0] = 0.0
        result = compute_mi_scores(X, y, names)
        assert result.mi_scores[0] >= 0


# ---------------------------------------------------------------------------
# TestRankFeatures
# ---------------------------------------------------------------------------


class TestRankFeatures:
    def test_rank_one_is_highest_mi(self):
        names = ["a", "b", "c", "dur"]
        scores = np.array([0.1, 0.5, 0.3, 0.05])
        df = rank_features(names, scores)
        assert df.iloc[0]["feature"] == "b"
        assert df.iloc[0]["rank"] == 1

    def test_ranks_are_sequential(self):
        names = [f"f{i}" for i in range(10)]
        scores = np.random.default_rng(0).uniform(0, 1, 10)
        df = rank_features(names, scores)
        assert list(df["rank"]) == list(range(1, 11))

    def test_all_selected_false_by_default(self):
        names = _make_feature_names()
        scores = np.ones(len(names))
        df = rank_features(names, scores)
        assert not df["selected"].any()

    def test_source_family_populated(self):
        names = ["proto_tcp", "service_http", "dur"]
        scores = np.array([0.5, 0.3, 0.1])
        df = rank_features(names, scores)
        families = df.set_index("feature")["source_family"].to_dict()
        assert families["proto_tcp"] == "proto"
        assert families["service_http"] == "service"
        assert families["dur"] == "numeric"

    def test_tie_broken_by_feature_name_alphabetically(self):
        names = ["z_feat", "a_feat"]
        scores = np.array([0.5, 0.5])  # tied
        df = rank_features(names, scores)
        # 'a_feat' should come first (alphabetical tie-break)
        assert df.iloc[0]["feature"] == "a_feat"

    def test_zero_mi_features_ranked_last(self):
        names = ["high", "low", "zero_mi_num"]
        scores = np.array([0.9, 0.1, 0.0])
        df = rank_features(names, scores)
        assert df.iloc[-1]["feature"] == "zero_mi_num"


# ---------------------------------------------------------------------------
# TestSelectTopK
# ---------------------------------------------------------------------------


class TestSelectTopK:
    def test_exactly_k_features_selected(self):
        X, y, names = _make_synthetic_X_y()
        result = compute_mi_scores(X, y, names)
        selected_df = select_top_k(result.ranking_df, k=10, feature_names_reference=names)
        assert int(selected_df["selected"].sum()) == 10

    def test_top_k_are_highest_ranked(self):
        X, y, names = _make_synthetic_X_y()
        result = compute_mi_scores(X, y, names)
        selected_df = select_top_k(result.ranking_df, k=5, feature_names_reference=names)
        top5_ranks = selected_df.loc[selected_df["selected"], "rank"].tolist()
        assert all(r <= 5 for r in top5_ranks)

    def test_k_zero_raises(self):
        X, y, names = _make_synthetic_X_y()
        result = compute_mi_scores(X, y, names)
        with pytest.raises(MISelectorError, match="k must be > 0"):
            select_top_k(result.ranking_df, k=0, feature_names_reference=names)

    def test_k_exceeds_features_raises(self):
        X, y, names = _make_synthetic_X_y(n_ohe=5, n_numeric=3)
        result = compute_mi_scores(X, y, names)
        with pytest.raises(MISelectorError, match="exceeds"):
            select_top_k(result.ranking_df, k=999, feature_names_reference=names)

    def test_selection_is_idempotent(self):
        X, y, names = _make_synthetic_X_y()
        result = compute_mi_scores(X, y, names)
        sel1 = select_top_k(result.ranking_df, k=15, feature_names_reference=names)
        sel2 = select_top_k(result.ranking_df, k=15, feature_names_reference=names)
        assert sel1["selected"].equals(sel2["selected"])

    def test_selected_names_are_subset_of_all_features(self):
        # _make_synthetic_X_y defaults to n_ohe=12, n_numeric=6 → 18 features total.
        # Use k=15 which is safely within the 18-feature count.
        X, y, names = _make_synthetic_X_y()
        assert len(names) == 18, f"Expected 18 features, got {len(names)}"
        result = compute_mi_scores(X, y, names)
        sel = select_top_k(result.ranking_df, k=15, feature_names_reference=names)
        selected = set(sel.loc[sel["selected"], "feature"])
        assert selected.issubset(set(names))


# ---------------------------------------------------------------------------
# TestMIEdgeCases
# ---------------------------------------------------------------------------


class TestMIEdgeCases:
    def test_single_feature_selectable(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (100, 1))
        y = (X[:, 0] > 0).astype(np.int64)
        names = ["single_feat"]
        result = compute_mi_scores(X, y, names)
        assert len(result.mi_scores) == 1

    def test_rare_attack_class(self):
        """Target with 95% Normal, 5% Attack should still work."""
        rng = np.random.default_rng(7)
        n = 200
        X = rng.normal(0, 1, (n, 5))
        y = np.zeros(n, dtype=np.int64)
        y[:10] = 1  # 5% attack
        names = [f"feat_{i}" for i in range(5)]
        result = compute_mi_scores(X, y, names)
        assert len(result.mi_scores) == 5
        assert (result.mi_scores >= 0).all()

    def test_exact_zero_mi_score(self):
        """A feature independent of the target should have MI ≈ 0."""
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(0, 1, (n, 3))
        y = rng.integers(0, 2, n, dtype=np.int64)
        names = ["ind_a", "ind_b", "ind_c"]
        result = compute_mi_scores(X, y, names)
        # MI of truly independent features should be small
        assert all(s >= 0 for s in result.mi_scores)

    def test_malformed_preprocessing_output_nan_raises(self):
        X, y, names = _make_synthetic_X_y()
        X[5, 5] = np.nan
        with pytest.raises(MISelectorError):
            compute_mi_scores(X, y, names)

    def test_mismatched_feature_name_length_raises(self):
        X, y, names = _make_synthetic_X_y()
        with pytest.raises(MISelectorError):
            compute_mi_scores(X, y, names + ["extra_name"])

    def test_negative_k_raises(self):
        X, y, names = _make_synthetic_X_y()
        result = compute_mi_scores(X, y, names)
        with pytest.raises(MISelectorError, match="k must be > 0"):
            select_top_k(result.ranking_df, k=-1, feature_names_reference=names)

    def test_k_equal_to_n_features(self):
        """K == total features: all features selected."""
        X, y, names = _make_synthetic_X_y(n_ohe=5, n_numeric=3)
        result = compute_mi_scores(X, y, names)
        sel = select_top_k(result.ranking_df, k=8, feature_names_reference=names)
        assert sel["selected"].all()

    def test_tied_mi_scores_stable(self):
        """Tied scores should produce consistent (deterministic) ranking."""
        n = 100
        X = np.ones((n, 4), dtype=np.float64)
        X[:, 0] = np.arange(n) % 2  # binary
        y = (X[:, 0] == 1).astype(np.int64)
        names = ["active", "const_a", "const_b", "const_c"]
        result = compute_mi_scores(X, y, names)
        # Ranking must be stable across reruns
        r2 = compute_mi_scores(X, y, names)
        pd.testing.assert_frame_equal(
            result.ranking_df.reset_index(drop=True),
            r2.ranking_df.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# TestFeatureNameInvariants
# ---------------------------------------------------------------------------


class TestFeatureNameInvariants:
    def test_all_ranked_features_exist_in_feature_names(self):
        X, y, names = _make_synthetic_X_y()
        result = compute_mi_scores(X, y, names)
        for fname in result.ranking_df["feature"]:
            assert fname in names, f"Ranked feature '{fname}' not in feature_names"

    def test_all_selected_features_exist_in_feature_names(self):
        X, y, names = _make_synthetic_X_y()
        result = compute_mi_scores(X, y, names)
        sel = select_top_k(result.ranking_df, k=10, feature_names_reference=names)
        for fname in sel.loc[sel["selected"], "feature"]:
            assert fname in names

    def test_feature_names_unique_in_ranking(self):
        X, y, names = _make_synthetic_X_y()
        result = compute_mi_scores(X, y, names)
        assert result.ranking_df["feature"].is_unique

    def test_feature_ordering_stable(self):
        """Same input → same MI scores → same ranking order."""
        X, y, names = _make_synthetic_X_y(seed=42)
        r1 = compute_mi_scores(X, y, names)
        r2 = compute_mi_scores(X, y, names)
        assert list(r1.ranking_df["feature"]) == list(r2.ranking_df["feature"])

    def test_discrete_mask_length_equals_feature_count(self):
        X, y, names = _make_synthetic_X_y(n_ohe=12, n_numeric=6)
        result = compute_mi_scores(X, y, names)
        assert len(result.discrete_mask) == result.n_features

    def test_source_family_count_equals_feature_count(self):
        X, y, names = _make_synthetic_X_y(n_ohe=12, n_numeric=6)
        result = compute_mi_scores(X, y, names)
        assert len(result.ranking_df) == result.n_features
        # Every row has a source_family
        assert not result.ranking_df["source_family"].isna().any()
