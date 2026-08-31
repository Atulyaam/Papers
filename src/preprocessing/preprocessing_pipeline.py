"""
src/preprocessing/preprocessing_pipeline.py
----------------------------------------------
Leakage-safe preprocessing pipeline for the UNSW-NB15 IDS project.

Architecture: 11 phases (per project spec).

    Phase 1:  Input/schema validation
    Phase 2:  Non-finite value validation
    Phase 3:  Target/metadata separation
    Phase 4:  Categorical/numeric column identification (from contract)
    Phase 5:  Encoder fitting          (TRAIN only)
    Phase 6:  Encoder transformation
    Phase 7:  Full encoded matrix assembly (OHE + numeric)
    Phase 8:  Scaler fitting           (TRAIN only)
    Phase 9:  Scaler transformation
    Phase 10: Model-view construction  (unscaled / scaled)
    Phase 11: Structured result generation (ProcessedDataset)

Leakage guarantees:
    - fit() only calls encoder.fit() and scaler.fit() on TRAIN data.
    - transform() ONLY calls .transform() — never .fit() or .fit_transform().
    - Calling transform() before fit() raises PreprocessingNotFittedError.
    - The fitted state is immutable after fit() (no refitting on transform).

One fitted preprocessing instance serves all splits:
    preprocessor.fit(train_df)
    preprocessor.transform(val_df, view="scaled")
    preprocessor.transform(dev_test_df, view="unscaled")
    preprocessor.transform(protected_df, view="scaled")
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.preprocessing.cleaning import (
    CleanedSplit,
    separate_target_and_features,
)
from src.preprocessing.encoding import (
    FittedEncoder,
    fit_encoder,
    get_encoder_metadata,
    get_feature_names,
    transform_encoder,
)
from src.preprocessing.exceptions import PreprocessingNotFittedError
from src.preprocessing.processed_dataset import (
    ProcessedDataset,
    ViewType,
    VIEW_SCALED,
    VIEW_UNSCALED,
)
from src.preprocessing.scaling import (
    FittedScaler,
    fit_scaler,
    get_scaler_metadata,
    transform_scaler,
)

logger = logging.getLogger(__name__)

ViewLiteral = Literal["unscaled", "scaled"]


class PreprocessingPipeline:
    """
    Leakage-safe preprocessing pipeline.

    One instance per experiment run. Call fit(train_df) once, then
    transform(df, view=...) for any number of splits.

    Parameters
    ----------
    experiment_id : str
        Identifier for logging and metadata (e.g. "EXP_PREPROCESSING_V1").
    """

    def __init__(self, experiment_id: str = "PREPROCESSING") -> None:
        self.experiment_id = experiment_id
        self._fitted: bool = False
        self._fitted_encoder: FittedEncoder | None = None
        self._fitted_scaler: FittedScaler | None = None
        self._feature_names: list[str] = []
        self._categorical_cols: list[str] = []
        self._numeric_cols: list[str] = []
        self._log = logging.getLogger(f"{__name__}.{experiment_id}")

    # ------------------------------------------------------------------
    # Public: fit
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "PreprocessingPipeline":
        """
        Fit the encoder and scaler on TRAIN data.

        This is the ONLY method that may call sklearn .fit() methods.
        After this call, the pipeline state is frozen — transform() will
        only use these fitted parameters.

        Parameters
        ----------
        train_df : pd.DataFrame
            Raw TRAIN DataFrame as loaded by loader.py.

        Returns
        -------
        self
            Fluent API: preprocessor.fit(train_df).transform(val_df).

        Raises
        ------
        PreprocessingSchemaError
            If required columns are missing.
        NonFiniteValueError
            If non-finite values are detected in numeric features.
        """
        self._log.info(
            "=== PREPROCESSING FIT START | experiment=%s | input_shape=%s ===",
            self.experiment_id,
            train_df.shape,
        )

        # --- Phase 1-4: validate, clean, separate ---
        cleaned: CleanedSplit = separate_target_and_features(train_df, split_name="train")

        self._categorical_cols = cleaned.categorical_cols
        self._numeric_cols = cleaned.numeric_cols

        self._log.info(
            "Fit | categorical_cols=%s | numeric_cols_count=%d",
            self._categorical_cols,
            len(self._numeric_cols),
        )

        # --- Phase 5: fit encoder ---
        X_cat_train = cleaned.X_raw[cleaned.categorical_cols]
        self._fitted_encoder = fit_encoder(X_cat_train, cleaned.categorical_cols)

        # --- Phase 6-7: build encoded TRAIN matrix ---
        X_encoded_train = self._build_encoded_matrix(
            cleaned.X_raw,
            cleaned.categorical_cols,
            cleaned.numeric_cols,
        )

        self._log.info(
            "Fit | encoded_matrix_shape=%s", X_encoded_train.shape
        )

        # --- Phase 8: fit scaler ---
        self._fitted_scaler = fit_scaler(X_encoded_train, self._feature_names)

        self._fitted = True
        self._log.info(
            "=== PREPROCESSING FIT COMPLETE | n_features=%d ===",
            len(self._feature_names),
        )

        return self

    # ------------------------------------------------------------------
    # Public: transform
    # ------------------------------------------------------------------

    def transform(
        self,
        df: pd.DataFrame,
        view: ViewLiteral = "scaled",
        split_name: str = "unknown",
    ) -> ProcessedDataset:
        """
        Transform a DataFrame using the TRAIN-fitted preprocessing state.

        This method NEVER calls .fit() or .fit_transform(). It only applies
        the frozen TRAIN-fitted encoder and scaler.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame to transform. May be TRAIN, validation,
            development_test, or protected_unseen_attack.
        view : {"scaled", "unscaled"}
            Which model view to produce.
            "unscaled" -> encoded but NOT scaled (for DT/RF).
            "scaled"   -> encoded AND scaled (for SVM/NN/AE).
        split_name : str
            Identifier for logging and the ProcessedDataset result.

        Returns
        -------
        ProcessedDataset
            Typed result with X, y, attack_cat, feature_names, and metadata.

        Raises
        ------
        PreprocessingNotFittedError
            If called before fit().
        PreprocessingSchemaError
            If required columns are missing.
        NonFiniteValueError
            If non-finite values are detected in numeric features.
        ValueError
            If view is not "scaled" or "unscaled".
        """
        if not self._fitted:
            raise PreprocessingNotFittedError(
                "transform() called before fit(). "
                "Call preprocessor.fit(train_df) first."
            )

        if view not in (VIEW_SCALED, VIEW_UNSCALED):
            raise ValueError(
                f"Unknown view '{view}'. Must be 'scaled' or 'unscaled'."
            )

        self._log.info(
            "Transform | split=%s | view=%s | input_shape=%s",
            split_name,
            view,
            df.shape,
        )

        # --- Phase 1-4: validate, clean, separate ---
        cleaned: CleanedSplit = separate_target_and_features(df, split_name=split_name)

        # --- Phase 6-7: apply encoder (NO refit) ---
        X_encoded = self._build_encoded_matrix(
            cleaned.X_raw,
            self._categorical_cols,  # use TRAIN column list
            self._numeric_cols,      # use TRAIN column list
            is_transform=True,
        )

        self._log.info(
            "Transform | split=%s | encoded_shape=%s", split_name, X_encoded.shape
        )

        # --- Phase 9-10: apply scaler or not ---
        if view == VIEW_SCALED:
            X_out = transform_scaler(self._fitted_scaler, X_encoded)
            scaler_meta = get_scaler_metadata(self._fitted_scaler)
        else:
            X_out = X_encoded
            scaler_meta = {
                "scaler_type": "StandardScaler",
                "applied": False,
                "note": "Unscaled view for DT/RF — scaler NOT applied.",
            }

        # --- Phase 11: assemble result ---
        result = ProcessedDataset(
            X=X_out,
            y=cleaned.y,
            attack_cat=cleaned.attack_cat,
            feature_names=self._feature_names,
            view_type=view,
            split_name=split_name,
            n_rows=X_out.shape[0],
            n_features=X_out.shape[1],
            encoder_metadata=get_encoder_metadata(self._fitted_encoder),
            scaler_metadata=scaler_meta,
            categorical_cols=self._categorical_cols,
            numeric_cols=self._numeric_cols,
        )

        self._log.info(
            "Transform complete | split=%s | view=%s | shape=%s",
            split_name,
            view,
            X_out.shape,
        )

        return result

    # ------------------------------------------------------------------
    # Public: convenience — transform both views at once
    # ------------------------------------------------------------------

    def transform_both_views(
        self,
        df: pd.DataFrame,
        split_name: str = "unknown",
    ) -> tuple[ProcessedDataset, ProcessedDataset]:
        """
        Return both model views for a single split.

        Returns
        -------
        (unscaled_dataset, scaled_dataset)
            Tuple in order: (DT/RF view, SVM/NN/AE view).
        """
        unscaled = self.transform(df, view=VIEW_UNSCALED, split_name=split_name)
        scaled = self.transform(df, view=VIEW_SCALED, split_name=split_name)
        return unscaled, scaled

    # ------------------------------------------------------------------
    # Properties (read-only access to fitted state)
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True if fit() has been called successfully."""
        return self._fitted

    @property
    def feature_names(self) -> list[str]:
        """Ordered feature names (available after fit)."""
        if not self._fitted:
            raise PreprocessingNotFittedError(
                "feature_names is not available before fit()."
            )
        return list(self._feature_names)

    @property
    def fitted_encoder(self) -> FittedEncoder:
        """The fitted encoder (available after fit)."""
        if not self._fitted:
            raise PreprocessingNotFittedError(
                "fitted_encoder is not available before fit()."
            )
        return self._fitted_encoder

    @property
    def fitted_scaler(self) -> FittedScaler:
        """The fitted scaler (available after fit)."""
        if not self._fitted:
            raise PreprocessingNotFittedError(
                "fitted_scaler is not available before fit()."
            )
        return self._fitted_scaler

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_encoded_matrix(
        self,
        X_raw: pd.DataFrame,
        categorical_cols: list[str],
        numeric_cols: list[str],
        is_transform: bool = False,
    ) -> np.ndarray:
        """
        Assemble the full encoded feature matrix: OHE columns + numeric columns.

        The feature_names list is set on the first call (during fit) and
        reused on subsequent transform calls to guarantee identical ordering.

        Column ordering: OHE columns first, numeric columns second.
        This is stable across all splits because the column lists come from
        the TRAIN-fitted state.

        Parameters
        ----------
        X_raw : pd.DataFrame
            Feature DataFrame (label and attack_cat already removed).
        categorical_cols : list[str]
            Ordered categorical column names.
        numeric_cols : list[str]
            Ordered numeric column names.
        is_transform : bool
            If True (transform call), uses frozen feature_names rather than
            rebuilding them. Raises if feature count mismatches.
        """
        # --- OHE block ---
        if categorical_cols:
            X_cat = X_raw[categorical_cols]
            X_ohe: np.ndarray = transform_encoder(self._fitted_encoder, X_cat)
            ohe_names: list[str] = get_feature_names(self._fitted_encoder)
        else:
            X_ohe = np.empty((len(X_raw), 0), dtype=np.float64)
            ohe_names = []

        # --- Numeric block ---
        if numeric_cols:
            # Only include numeric_cols that are present in X_raw
            present_num_cols = [c for c in numeric_cols if c in X_raw.columns]
            X_num: np.ndarray = X_raw[present_num_cols].to_numpy(dtype=np.float64)
            num_names: list[str] = present_num_cols
        else:
            X_num = np.empty((len(X_raw), 0), dtype=np.float64)
            num_names = []

        # --- Concatenate ---
        if X_ohe.shape[1] > 0 and X_num.shape[1] > 0:
            X_full = np.concatenate([X_ohe, X_num], axis=1)
        elif X_ohe.shape[1] > 0:
            X_full = X_ohe
        else:
            X_full = X_num

        # --- Set or verify feature names ---
        combined_names = ohe_names + num_names
        if not is_transform:
            # First call (fit path): record the canonical feature names
            self._feature_names = combined_names
        else:
            # Subsequent calls (transform path): verify names match exactly.
            # Count check alone is insufficient — a reordering of columns would
            # produce the same count but different feature positions.
            if combined_names != self._feature_names:
                # Find first mismatch for a useful error message
                mismatch_idx = next(
                    (i for i, (a, b) in enumerate(zip(combined_names, self._feature_names)) if a != b),
                    len(self._feature_names),
                )
                raise ValueError(
                    f"Feature name mismatch during transform. "
                    f"Expected {len(self._feature_names)} features matching the "
                    f"TRAIN-fitted order; got {len(combined_names)} features. "
                    f"First mismatch at index {mismatch_idx}: "
                    f"got '{combined_names[mismatch_idx] if mismatch_idx < len(combined_names) else '<missing>'}' "
                    f"expected '{self._feature_names[mismatch_idx] if mismatch_idx < len(self._feature_names) else '<missing>'}'. "
                    f"Ensure the same column schema is used across all splits."
                )

        return X_full
