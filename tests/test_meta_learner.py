"""
tests/test_meta_learner.py
----------------------------
Sprint 6 meta-learner unit tests.

Covers:
- Fixed configuration enforcement
- No hyperparameter search
- Correct input (4 columns, no row_id)
- Binary output contract
- Limitation texts present
- H1 summary correctness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_oof_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "row_id": np.arange(n),
        "dt_attack_probability": rng.rand(n),
        "rf_attack_probability": rng.rand(n),
        "svm_decision_score": rng.randn(n),
        "nn_attack_probability": rng.rand(n),
        "label": np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)]),
    })


# ---------------------------------------------------------------------------
# Configuration enforcement
# ---------------------------------------------------------------------------

class TestMetaConfig:

    def test_meta_config_solver_is_lbfgs(self):
        from src.models.stacking.meta_learner import META_CONFIG
        assert META_CONFIG["solver"] == "lbfgs"

    def test_meta_config_C_is_1(self):
        from src.models.stacking.meta_learner import META_CONFIG
        assert META_CONFIG["C"] == 1.0

    def test_meta_config_class_weight_is_balanced(self):
        from src.models.stacking.meta_learner import META_CONFIG
        assert META_CONFIG["class_weight"] == "balanced"

    def test_meta_config_max_iter_is_1000(self):
        from src.models.stacking.meta_learner import META_CONFIG
        assert META_CONFIG["max_iter"] == 1000

    def test_meta_config_random_state_not_in_config(self):
        """random_state is passed at training time per H1 seed, not in META_CONFIG."""
        from src.models.stacking.meta_learner import META_CONFIG
        assert "random_state" not in META_CONFIG

    def test_trained_clf_uses_meta_config(self):
        """Trained meta-learner must use all frozen META_CONFIG values."""
        from src.models.stacking.meta_learner import train_meta_learner, META_CONFIG
        df = _make_oof_df()
        clf = train_meta_learner(df, h1_seed=42)
        assert clf.solver == META_CONFIG["solver"]
        assert clf.C == META_CONFIG["C"]
        assert clf.class_weight == META_CONFIG["class_weight"]
        assert clf.max_iter == META_CONFIG["max_iter"]

    def test_trained_clf_random_state_matches_h1_seed(self):
        """random_state of trained clf must match the H1 seed passed."""
        from src.models.stacking.meta_learner import train_meta_learner
        for seed in [42, 123, 2024]:
            df = _make_oof_df(seed=seed)
            clf = train_meta_learner(df, h1_seed=seed)
            assert clf.random_state == seed


# ---------------------------------------------------------------------------
# No hyperparameter search
# ---------------------------------------------------------------------------

class TestNoHyperparamSearch:

    def test_no_gridsearchcv_in_meta_learner_module(self):
        """GridSearchCV must not be imported or used in meta_learner.py."""
        import src.models.stacking.meta_learner as module
        source = module.__file__
        with open(source, "r", encoding="utf-8") as fh:
            content = fh.read()
        assert "GridSearchCV" not in content, \
            "GridSearchCV found in meta_learner.py — forbidden"
        assert "RandomizedSearchCV" not in content, \
            "RandomizedSearchCV found in meta_learner.py — forbidden"

    def test_no_hyperparameter_search_in_oof_runner(self):
        """oof_runner.py must not use GridSearchCV or RandomizedSearchCV."""
        import src.models.stacking.oof_runner as module
        with open(module.__file__, "r", encoding="utf-8") as fh:
            content = fh.read()
        assert "GridSearchCV" not in content
        assert "RandomizedSearchCV" not in content


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------

class TestMetaLearnerInput:

    def test_trained_on_4_columns(self):
        """Meta-learner must be trained on exactly 4 features."""
        from src.models.stacking.meta_learner import train_meta_learner
        df = _make_oof_df()
        clf = train_meta_learner(df, h1_seed=42)
        assert clf.n_features_in_ == 4

    def test_row_id_absent_from_meta_features(self):
        """META_FEATURE_COLS must not include row_id."""
        from src.models.stacking.meta_learner import META_FEATURE_COLS
        assert "row_id" not in META_FEATURE_COLS
        assert len(META_FEATURE_COLS) == 4

    def test_label_is_not_a_feature(self):
        """label must not be in META_FEATURE_COLS."""
        from src.models.stacking.meta_learner import META_FEATURE_COLS
        assert "label" not in META_FEATURE_COLS

    def test_meta_feature_col_names_correct(self):
        """Exact column names must match specification."""
        from src.models.stacking.meta_learner import META_FEATURE_COLS
        expected = [
            "dt_attack_probability",
            "rf_attack_probability",
            "svm_decision_score",
            "nn_attack_probability",
        ]
        assert META_FEATURE_COLS == expected

    def test_missing_meta_column_raises(self):
        """Training with missing meta-feature columns raises ValueError."""
        from src.models.stacking.meta_learner import train_meta_learner
        df = _make_oof_df()
        df = df.drop(columns=["dt_attack_probability"])
        with pytest.raises(ValueError, match="missing"):
            train_meta_learner(df, h1_seed=42)

    def test_nan_in_meta_features_raises(self):
        """NaN in OOF matrix raises ValueError."""
        from src.models.stacking.meta_learner import train_meta_learner
        df = _make_oof_df()
        df.loc[0, "rf_attack_probability"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            train_meta_learner(df, h1_seed=42)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class TestMetaLearnerOutput:

    def test_predict_meta_returns_binary_labels(self):
        """predict_meta must return labels in {0, 1}."""
        from src.models.stacking.meta_learner import train_meta_learner, predict_meta, META_FEATURE_COLS
        df = _make_oof_df()
        clf = train_meta_learner(df, h1_seed=42)
        X_meta = df[META_FEATURE_COLS].to_numpy()
        y_pred, proba = predict_meta(clf, X_meta)
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_predict_meta_returns_probabilities_in_01(self):
        """attack_proba must be in [0, 1]."""
        from src.models.stacking.meta_learner import train_meta_learner, predict_meta, META_FEATURE_COLS
        df = _make_oof_df()
        clf = train_meta_learner(df, h1_seed=42)
        X_meta = df[META_FEATURE_COLS].to_numpy()
        _, proba = predict_meta(clf, X_meta)
        assert proba.min() >= 0.0 and proba.max() <= 1.0

    def test_compute_oof_metrics_returns_macro_f1(self):
        """compute_oof_metrics must return macro_f1."""
        from src.models.stacking.meta_learner import train_meta_learner, compute_oof_metrics
        df = _make_oof_df()
        clf = train_meta_learner(df, h1_seed=42)
        metrics = compute_oof_metrics(clf, df, h1_seed=42)
        assert "macro_f1" in metrics
        assert 0.0 <= metrics["macro_f1"] <= 1.0

    def test_compute_oof_metrics_in_sample_warning_true(self):
        """in_sample_evaluation_warning must be True."""
        from src.models.stacking.meta_learner import train_meta_learner, compute_oof_metrics
        df = _make_oof_df()
        clf = train_meta_learner(df, h1_seed=42)
        metrics = compute_oof_metrics(clf, df, h1_seed=42)
        assert metrics["in_sample_evaluation_warning"] is True

    def test_compute_oof_metrics_contains_limitation_text(self):
        """Metrics dict must include meta_evaluation_limitation and scaling_limitation."""
        from src.models.stacking.meta_learner import train_meta_learner, compute_oof_metrics
        df = _make_oof_df()
        clf = train_meta_learner(df, h1_seed=42)
        metrics = compute_oof_metrics(clf, df, h1_seed=42)
        assert "meta_evaluation_limitation" in metrics
        assert "scaling_limitation" in metrics
        assert len(metrics["meta_evaluation_limitation"]) > 0
        assert len(metrics["scaling_limitation"]) > 0


# ---------------------------------------------------------------------------
# Limitation texts
# ---------------------------------------------------------------------------

class TestLimitationTexts:

    def test_meta_evaluation_limitation_text_content(self):
        """META_EVALUATION_LIMITATION_TEXT must contain key phrases."""
        from src.models.stacking.meta_learner import META_EVALUATION_LIMITATION_TEXT
        text = META_EVALUATION_LIMITATION_TEXT.lower()
        assert "in-sample" in text or "in sample" in text, \
            f"Expected 'in-sample' in limitation text: {META_EVALUATION_LIMITATION_TEXT}"
        assert "not" in text
        assert "generalisation" in text or "generalization" in text

    def test_scaling_limitation_text_content(self):
        """SCALING_LIMITATION_TEXT must reference the bounded leakage."""
        from src.models.stacking.oof_runner import SCALING_LIMITATION_TEXT
        text = SCALING_LIMITATION_TEXT.lower()
        assert "bounded" in text
        assert "label-independent" in text

    def test_sprint5_reference_value_correct(self):
        """SPRINT5_RF_REFERENCE must be 0.9508532..."""
        from src.models.stacking.meta_learner import SPRINT5_RF_REFERENCE
        assert abs(SPRINT5_RF_REFERENCE - 0.9508532) < 1e-5

    def test_sprint5_reference_label_content(self):
        """Label must say 'single-CV reference', not 'H1 baseline'."""
        from src.models.stacking.meta_learner import SPRINT5_RF_REFERENCE_LABEL
        assert "single-CV reference" in SPRINT5_RF_REFERENCE_LABEL
        assert "3-seed H1 baseline" in SPRINT5_RF_REFERENCE_LABEL or \
               "matched 3-seed" in SPRINT5_RF_REFERENCE_LABEL


# ---------------------------------------------------------------------------
# H1 summary
# ---------------------------------------------------------------------------

class TestH1Summary:

    def _seed_results(self, f1s=(0.95, 0.94, 0.96)):
        return [
            {
                "h1_seed": s,
                "macro_f1": f,
                "in_sample_evaluation_warning": True,
                "meta_evaluation_limitation": "meta lim text",
                "scaling_limitation": "scaling lim text",
            }
            for s, f in zip([42, 123, 2024], f1s)
        ]

    def test_h1_summary_mean_correct(self):
        from src.models.stacking.meta_learner import compute_h1_summary
        f1s = (0.90, 0.92, 0.94)
        summary = compute_h1_summary(self._seed_results(f1s))
        assert abs(summary["mean_macro_f1"] - np.mean(f1s)) < 1e-10

    def test_h1_summary_std_correct(self):
        from src.models.stacking.meta_learner import compute_h1_summary
        f1s = (0.90, 0.92, 0.94)
        summary = compute_h1_summary(self._seed_results(f1s))
        assert abs(summary["std_macro_f1"] - np.std(f1s, ddof=1)) < 1e-10

    def test_h1_summary_three_seeds_present(self):
        from src.models.stacking.meta_learner import compute_h1_summary
        summary = compute_h1_summary(self._seed_results())
        assert set(summary["h1_seeds"]) == {42, 123, 2024}

    def test_h1_summary_has_sprint5_ref(self):
        from src.models.stacking.meta_learner import compute_h1_summary, SPRINT5_RF_REFERENCE
        summary = compute_h1_summary(self._seed_results())
        ref = summary["sprint5_reference"]
        assert abs(ref["macro_f1"] - SPRINT5_RF_REFERENCE) < 1e-8
        assert "single-CV reference" in ref["label"]

    def test_h1_summary_empty_raises(self):
        from src.models.stacking.meta_learner import compute_h1_summary
        with pytest.raises(ValueError):
            compute_h1_summary([])

    def test_h1_summary_has_two_reporting_units_statement(self):
        from src.models.stacking.meta_learner import compute_h1_summary
        summary = compute_h1_summary(self._seed_results())
        stmt = summary.get("two_reporting_units_statement", "")
        assert len(stmt) > 20

    def test_h1_summary_no_stat_significance_language(self):
        import json
        from src.models.stacking.meta_learner import compute_h1_summary
        summary = compute_h1_summary(self._seed_results())
        text = json.dumps(summary).lower()
        assert "statistically significant" not in text

    def test_h1_summary_per_seed_f1_matches_input(self):
        from src.models.stacking.meta_learner import compute_h1_summary
        f1s = (0.91, 0.93, 0.95)
        summary = compute_h1_summary(self._seed_results(f1s))
        for seed, f1 in zip([42, 123, 2024], f1s):
            assert abs(summary["per_seed_macro_f1"][str(seed)] - f1) < 1e-10
