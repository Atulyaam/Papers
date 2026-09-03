"""
src/preprocessing/withheld_candidate.py
-----------------------------------------
Withheld-attack eligibility computation for the UNSW-NB15 IDS project.

This module:
- Computes TEST counts per canonicalized attack subclass.
- Returns eligible candidates (count >= min_count).
- Applies the project's fixed selection rule.

LEAKAGE POLICY:
- This function reads TEST label counts ONLY.
- It does not influence any model, threshold, or feature selection.
- The selected subclass is fixed BEFORE any model training/evaluation.
- Selection uses no model performance metrics.

Fixed target: Backdoor
Selection rule: fixed_named_target
Eligibility threshold: >= 50 TEST instances

Historical note:
    The project initially considered alphabetical selection of the first
    eligible subclass. Before any model training, this was revised to
    fixed_named_target: Backdoor. The revision_history field in
    experiments/pre_registration.json preserves this decision transparently.
"""

import logging
from typing import Any


_ELIGIBLE_THRESHOLD_DEFAULT = 50
_SELECTION_RULE_DEFAULT = "fixed_named_target"
_TARGET_DEFAULT = "Backdoor"


def compute_eligible_candidates(
    counts: dict[str, int],
    min_count: int = _ELIGIBLE_THRESHOLD_DEFAULT,
) -> dict[str, int]:
    """
    Filter attack-subclass TEST counts to those meeting the eligibility threshold.

    Parameters
    ----------
    counts : dict[str, int]
        Canonical attack category → TEST count mapping.
        Should NOT include "Normal" (benign) rows.
    min_count : int
        Minimum TEST instances for eligibility. Default 50.

    Returns
    -------
    dict[str, int]
        Eligible subclasses → count, sorted alphabetically.

    Raises
    ------
    ValueError
        If counts is empty or all subclasses are below the threshold.
    """
    if not counts:
        raise ValueError("Attack-category count dict is empty. Cannot determine eligible candidates.")

    eligible = {cat: cnt for cat, cnt in counts.items() if cnt >= min_count}

    if not eligible:
        raise ValueError(
            f"No attack subclass has >= {min_count} TEST instances. "
            f"Cannot proceed with withheld-attack protocol. "
            f"Observed counts: {counts}"
        )

    return dict(sorted(eligible.items()))


def select_withheld_subclass(
    eligible: dict[str, int],
    rule: str = _SELECTION_RULE_DEFAULT,
    target: str = _TARGET_DEFAULT,
    logger: logging.Logger | None = None,
) -> str:
    """
    Apply the project's fixed selection rule to choose the withheld subclass.

    The only supported rule is "fixed_named_target".

    Parameters
    ----------
    eligible : dict[str, int]
        Eligible subclasses → count (output of compute_eligible_candidates).
    rule : str
        Selection rule. Must be "fixed_named_target".
    target : str
        The fixed target subclass name. Default "Backdoor".
    logger : logging.Logger | None
        Optional logger.

    Returns
    -------
    str
        The selected withheld subclass name.

    Raises
    ------
    ValueError
        If the target is not in the eligible dict, or if rule is unknown,
        or if eligible dict is empty.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not eligible:
        raise ValueError("Eligible subclass dict is empty. Cannot select withheld subclass.")

    if rule == "fixed_named_target":
        if target not in eligible:
            raise ValueError(
                f"Fixed target '{target}' is not in the eligible subclasses "
                f"(count < {_ELIGIBLE_THRESHOLD_DEFAULT} or not present in TEST).\n"
                f"Eligible candidates: {eligible}\n"
                f"STOP: Do not silently select another attack category. "
                f"Report this conflict."
            )
        logger.info(
            "Withheld subclass selected | rule=%s | target=%s | count=%d",
            rule,
            target,
            eligible[target],
        )
        return target

    raise ValueError(
        f"Unknown selection rule: '{rule}'. "
        f"Only 'fixed_named_target' is supported in this protocol version."
    )


def build_candidate_report(
    canonical_test_counts: dict[str, int],
    min_count: int = _ELIGIBLE_THRESHOLD_DEFAULT,
    exclude_normal: bool = True,
    normal_canonical: str = "Normal",
) -> dict[str, Any]:
    """
    Build the complete withheld-candidate audit report.

    Parameters
    ----------
    canonical_test_counts : dict[str, int]
        All canonical attack categories → TEST count (including Normal).
    min_count : int
        Eligibility threshold.
    exclude_normal : bool
        Whether to exclude the "Normal" (benign) class from candidate list.
    normal_canonical : str
        The canonical string for normal/benign traffic.

    Returns
    -------
    dict
        Full candidate report for data/audit/withheld_candidates.json.
    """
    attack_counts = {
        cat: cnt
        for cat, cnt in canonical_test_counts.items()
        if not (exclude_normal and cat == normal_canonical)
    }

    all_subclasses = {cat: {"count": cnt, "eligible": cnt >= min_count}
                      for cat, cnt in sorted(attack_counts.items())}

    eligible = {cat: cnt for cat, cnt in attack_counts.items() if cnt >= min_count}

    return {
        "eligibility_threshold": min_count,
        "all_attack_subclasses": all_subclasses,
        "eligible_count": len(eligible),
        "eligible_candidates": dict(sorted(eligible.items())),
    }
