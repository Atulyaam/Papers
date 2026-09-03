"""
src/preprocessing/exceptions.py
---------------------------------
Project-specific exceptions for the UNSW-NB15 IDS preprocessing pipeline.

All preprocessing errors must raise one of these typed exceptions — never
re-raise raw sklearn internal errors without a clear project-level message.
"""

from __future__ import annotations


class PreprocessingNotFittedError(RuntimeError):
    """
    Raised when transform() is called before fit().

    The preprocessing pipeline maintains a fitted state; this exception
    signals an incorrect call ordering.
    """


class NonFiniteValueError(ValueError):
    """
    Raised when NaN, +inf, or -inf values are detected in a DataFrame
    that has been declared ready for preprocessing.

    Policy: fail loudly rather than silently impute or drop.
    A future fallback (TRAIN-only median imputation) is documented but
    NOT active in Sprint 2.
    """

    def __init__(
        self,
        split_name: str,
        nan_cols: dict[str, int],
        pos_inf_cols: dict[str, int],
        neg_inf_cols: dict[str, int],
    ) -> None:
        self.split_name = split_name
        self.nan_cols = nan_cols
        self.pos_inf_cols = pos_inf_cols
        self.neg_inf_cols = neg_inf_cols
        affected = {**nan_cols, **pos_inf_cols, **neg_inf_cols}
        summary = "; ".join(
            f"{col}(nan={nan_cols.get(col,0)}, +inf={pos_inf_cols.get(col,0)}, "
            f"-inf={neg_inf_cols.get(col,0)})"
            for col in affected
        )
        super().__init__(
            f"Non-finite values detected in split='{split_name}'. "
            f"Affected columns: [{summary}]. "
            f"Policy: fail loudly. Do not impute silently. "
            f"Correct the data or adopt an approved imputation policy."
        )


class PreprocessingSchemaError(ValueError):
    """
    Raised when the DataFrame presented to the preprocessor is missing
    required columns or violates the expected schema contract.
    """

    def __init__(self, message: str, missing_cols: list[str] | None = None) -> None:
        self.missing_cols = missing_cols or []
        super().__init__(message)
