"""
src/preprocessing/processed_dataset.py
-----------------------------------------
ProcessedDataset: the typed result object produced by the preprocessing pipeline.

Design decisions:
    - Two model views are produced from ONE fitted preprocessing state.
    - VIEW 1 (view_type="unscaled"): encoded-but-unscaled  -> DT, RF
    - VIEW 2 (view_type="scaled"):   encoded-and-scaled     -> SVM, NN, AE
    - Both views share: same feature_names, same row ordering.
    - X is a NumPy array (model-ready).
    - y and attack_cat preserve original row alignment.
    - feature_names survive for SHAP, MI, and evaluation.
    - encoder_metadata and scaler_metadata are JSON-serialisable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

VIEW_UNSCALED: Literal["unscaled"] = "unscaled"
VIEW_SCALED: Literal["scaled"] = "scaled"
ViewType = Literal["unscaled", "scaled"]


@dataclass
class ProcessedDataset:
    """
    Typed result object returned by the preprocessing pipeline's transform().

    Attributes
    ----------
    X : np.ndarray
        Model-ready feature matrix. Shape: (n_rows, n_features).
        For view_type="unscaled": encoded but NOT StandardScaled.
        For view_type="scaled":   encoded AND StandardScaled.

    y : np.Series
        Binary target series aligned with X rows.

    attack_cat : pd.Series
        Attack category metadata series aligned with X rows.
        Must NOT appear in X. Used for evaluation and protocol logic.

    feature_names : list[str]
        Ordered feature names corresponding to X columns.
        Identical for both view types.

    view_type : ViewType
        "unscaled" or "scaled". Explicitly identifies which model this
        representation is intended for.

    split_name : str
        Identifier for the data split (e.g., "train", "development_test").

    n_rows : int
        Number of rows in X. Must equal len(y) == len(attack_cat).

    n_features : int
        Number of columns in X. Must equal len(feature_names).

    encoder_metadata : dict[str, Any]
        JSON-serialisable dict from get_encoder_metadata().

    scaler_metadata : dict[str, Any]
        JSON-serialisable dict from get_scaler_metadata().
        For unscaled view: records "not_applied" flag.

    categorical_cols : list[str]
        Categorical columns that were one-hot encoded.

    numeric_cols : list[str]
        Numeric columns that were included in the feature matrix.
    """

    X: np.ndarray
    y: pd.Series
    attack_cat: pd.Series
    feature_names: list[str]
    view_type: ViewType
    split_name: str
    n_rows: int
    n_features: int
    encoder_metadata: dict[str, Any] = field(default_factory=dict)
    scaler_metadata: dict[str, Any] = field(default_factory=dict)
    categorical_cols: list[str] = field(default_factory=list)
    numeric_cols: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Integrity checks on construction."""
        if self.X.shape != (self.n_rows, self.n_features):
            raise ValueError(
                f"X.shape {self.X.shape} does not match "
                f"(n_rows={self.n_rows}, n_features={self.n_features})."
            )
        if len(self.y) != self.n_rows:
            raise ValueError(
                f"y length {len(self.y)} does not match n_rows={self.n_rows}."
            )
        if len(self.attack_cat) != self.n_rows:
            raise ValueError(
                f"attack_cat length {len(self.attack_cat)} does not match "
                f"n_rows={self.n_rows}."
            )
        if len(self.feature_names) != self.n_features:
            raise ValueError(
                f"feature_names length {len(self.feature_names)} does not match "
                f"n_features={self.n_features}."
            )

    def to_summary_dict(self) -> dict[str, Any]:
        """
        Return a compact JSON-serialisable summary (for logging and metadata).

        Does NOT include the raw arrays (X, y) — only descriptive info.
        """
        return {
            "split_name": self.split_name,
            "view_type": self.view_type,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "categorical_cols": self.categorical_cols,
            "numeric_cols": self.numeric_cols,
            "feature_names_head": self.feature_names[:10],
            "y_value_counts": {str(k): int(v) for k, v in self.y.value_counts().items()},
            "attack_cat_counts": {
                str(k): int(v) for k, v in self.attack_cat.value_counts().items()
            },
            "encoder_metadata": self.encoder_metadata,
            "scaler_metadata": self.scaler_metadata,
        }
