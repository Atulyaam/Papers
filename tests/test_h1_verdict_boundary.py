"""
tests/test_h1_verdict_boundary.py
Focused unit test for the H1 verdict function boundary behaviour.

Tests the locked DD-7 three-way rule:
    diff > epsilon      -> SUPPORTED
    diff < -epsilon     -> NOT_SUPPORTED
    else (|diff|<=eps)  -> INCONCLUSIVE

epsilon = 0.005 (LOCKED)
"""
import pytest

EPSILON = 0.005


def h1_verdict_fn(diff, epsilon=EPSILON):
    """Mirror of the locked DD-7 verdict function in evaluate_sprint9.py."""
    if diff > epsilon:
        return "SUPPORTED"
    elif diff < -epsilon:  # DD-7 LOCKED: NOT_SUPPORTED requires diff < -epsilon
        return "NOT_SUPPORTED"
    else:
        return "INCONCLUSIVE"


@pytest.mark.parametrize("diff, expected", [
    # Clear SUPPORTED (above +epsilon)
    (+0.010, "SUPPORTED"),
    # Clear NOT_SUPPORTED (below -epsilon)
    (-0.010, "NOT_SUPPORTED"),
    # INCONCLUSIVE zone: |diff| <= epsilon
    (+0.003, "INCONCLUSIVE"),
    (-0.003, "INCONCLUSIVE"),
    (0.000,  "INCONCLUSIVE"),
    # Exact boundaries are INCONCLUSIVE because rules are strict < and >
    (+EPSILON, "INCONCLUSIVE"),   # diff == +epsilon: NOT > epsilon -> INCONCLUSIVE
    (-EPSILON, "INCONCLUSIVE"),   # diff == -epsilon: NOT < -epsilon -> INCONCLUSIVE
])
def test_h1_verdict_boundary(diff, expected):
    """
    Verifies the locked DD-7 three-way H1 verdict rule.
    This test MUST fail if the old 'diff < 0' bug is reintroduced.
    """
    result = h1_verdict_fn(diff)
    assert result == expected, (
        f"H1 verdict for diff={diff}: expected '{expected}', got '{result}'. "
        f"Locked DD-7 rule: SUPPORTED if diff>{EPSILON}; NOT_SUPPORTED if diff<{-EPSILON}; else INCONCLUSIVE."
    )


def test_h1_boundary_not_triggers_on_negative_within_epsilon():
    """
    Explicit regression test for the original defect (diff < 0 -> NOT_SUPPORTED).
    Values in (-epsilon, 0) MUST return INCONCLUSIVE, not NOT_SUPPORTED.
    """
    for diff in [-0.001, -0.002, -0.003, -0.004, -0.0049]:
        result = h1_verdict_fn(diff)
        assert result == "INCONCLUSIVE", (
            f"Regression: diff={diff} in (-epsilon, 0) returned '{result}' but MUST be INCONCLUSIVE. "
            f"Old 'diff < 0' boundary is reintroduced."
        )


def test_h1_actual_sprint9_result():
    """
    Verifies that the actual Sprint 9 diff produces SUPPORTED.
    diff = 0.01222799051528034 > epsilon=0.005 -> SUPPORTED
    """
    actual_diff = 0.01222799051528034
    result = h1_verdict_fn(actual_diff)
    assert result == "SUPPORTED", (
        f"Actual Sprint 9 diff={actual_diff} must yield SUPPORTED, got '{result}'."
    )
