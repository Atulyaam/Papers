"""tests/test_schema_validator.py — Unit tests for src/preprocessing/schema_validator.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.schema_validator import (
    SchemaViolationError,
    assert_schema_valid,
    validate_schema,
)


def _make_observed(columns):
    return {"columns": columns, "shape": {"rows": 100, "cols": len(columns)}}


def _make_contract(label_col="label", cat_col="attack_cat"):
    return {
        "target": {"column": label_col},
        "attack_category": {"column": cat_col},
        "candidate_categorical_columns": ["proto", "service"],
        "candidate_exclude_columns": ["id", "srcip"],
    }


class TestValidateSchema:
    def test_passes_with_valid_schema(self):
        obs = _make_observed(["label", "attack_cat", "proto", "service", "feat1"])
        contract = _make_contract()
        violations = validate_schema(obs, contract)
        assert violations == []

    def test_fails_missing_target_col(self):
        obs = _make_observed(["attack_cat", "feat1"])
        contract = _make_contract()
        violations = validate_schema(obs, contract)
        assert any("label" in v for v in violations)

    def test_fails_missing_attack_cat_col(self):
        obs = _make_observed(["label", "feat1"])
        contract = _make_contract()
        violations = validate_schema(obs, contract)
        assert any("attack_cat" in v for v in violations)

    def test_candidate_cat_missing_is_warning_not_violation(self):
        """Candidate categorical columns absent from data are NOT hard violations."""
        obs = _make_observed(["label", "attack_cat"])  # no proto, service
        contract = _make_contract()
        violations = validate_schema(obs, contract)
        # proto/service absence should not appear in violations list
        assert violations == []

    def test_empty_contract_no_violations(self):
        obs = _make_observed(["a", "b"])
        violations = validate_schema(obs, {})
        assert violations == []


class TestAssertSchemaValid:
    def test_no_raise_on_empty_violations(self):
        assert_schema_valid([])  # should not raise

    def test_raises_on_non_empty_violations(self):
        with pytest.raises(SchemaViolationError):
            assert_schema_valid(["Column 'label' not found."])

    def test_violation_error_is_runtime_error(self):
        assert issubclass(SchemaViolationError, RuntimeError)
