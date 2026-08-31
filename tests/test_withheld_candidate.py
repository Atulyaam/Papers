"""tests/test_withheld_candidate.py — Unit tests for withheld_candidate.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.withheld_candidate import (
    compute_eligible_candidates,
    select_withheld_subclass,
    build_candidate_report,
)


class TestComputeEligibleCandidates:
    def test_below_threshold_excluded(self):
        counts = {"Backdoor": 30, "DoS": 200, "Generic": 10}
        result = compute_eligible_candidates(counts, min_count=50)
        assert "Backdoor" not in result
        assert "Generic" not in result
        assert "DoS" in result

    def test_exactly_at_threshold_included(self):
        counts = {"Backdoor": 50}
        result = compute_eligible_candidates(counts, min_count=50)
        assert "Backdoor" in result

    def test_alphabetical_order(self):
        counts = {"Worms": 100, "Backdoor": 100, "Analysis": 100}
        result = compute_eligible_candidates(counts, min_count=50)
        assert list(result.keys()) == sorted(result.keys())

    def test_empty_counts_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_eligible_candidates({})

    def test_all_below_threshold_raises(self):
        counts = {"Backdoor": 10, "DoS": 20}
        with pytest.raises(ValueError):
            compute_eligible_candidates(counts, min_count=50)


class TestSelectWithheldSubclass:
    def test_backdoor_selected_when_eligible(self):
        eligible = {"Backdoor": 100, "DoS": 200}
        result = select_withheld_subclass(eligible, rule="fixed_named_target", target="Backdoor")
        assert result == "Backdoor"

    def test_backdoor_not_in_eligible_raises(self):
        eligible = {"DoS": 200, "Generic": 300}
        with pytest.raises(ValueError, match="Backdoor"):
            select_withheld_subclass(eligible, rule="fixed_named_target", target="Backdoor")

    def test_empty_eligible_raises(self):
        with pytest.raises(ValueError):
            select_withheld_subclass({}, rule="fixed_named_target", target="Backdoor")

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="Unknown selection rule"):
            select_withheld_subclass({"Backdoor": 100}, rule="random", target="Backdoor")


class TestBuildCandidateReport:
    def test_structure(self):
        counts = {"Normal": 5000, "Backdoor": 100, "DoS": 200, "Worms": 10}
        report = build_candidate_report(counts, min_count=50)
        assert "eligibility_threshold" in report
        assert "eligible_candidates" in report
        assert "all_attack_subclasses" in report

    def test_normal_excluded_from_attack_subclasses(self):
        counts = {"Normal": 5000, "Backdoor": 100}
        report = build_candidate_report(counts, min_count=50)
        assert "Normal" not in report["eligible_candidates"]
        assert "Normal" not in report["all_attack_subclasses"]

    def test_eligible_count_correct(self):
        counts = {"Backdoor": 100, "DoS": 200, "Worms": 10}
        report = build_candidate_report(counts, min_count=50)
        assert report["eligible_count"] == 2  # Backdoor and DoS
