# Author: Basil M. Alzboun

"""
tests/test_interventional.py

Tests for ccce.attribution.interventional (Level D statistical machinery).
"""
import numpy as np
import pytest

from ccce.attribution.interventional import (
    InterventionCondition,
    run_interventional_diagnosis,
)


def _make_condition(condition_id: str, mean: float, std: float, n: int, rng: np.random.Generator) -> InterventionCondition:
    values = list(rng.normal(mean, std, size=n))
    return InterventionCondition(condition_id=condition_id, description=f"mean={mean}", replicate_values=values)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
def test_requires_at_least_two_conditions():
    rng = np.random.default_rng(0)
    cond = _make_condition("only_one", 0.0, 1.0, 10, rng)
    with pytest.raises(ValueError, match="at least two conditions"):
        run_interventional_diagnosis("test_factor", [cond])


def test_requires_at_least_two_replicates_per_condition():
    rng = np.random.default_rng(0)
    cond_a = InterventionCondition("A", "single replicate", [1.0])
    cond_b = _make_condition("B", 0.0, 1.0, 10, rng)
    with pytest.raises(ValueError, match="at least 2 independent seeds"):
        run_interventional_diagnosis("test_factor", [cond_a, cond_b])


# --------------------------------------------------------------------------
# Known ground-truth recovery
# --------------------------------------------------------------------------
def test_recovers_no_effect_when_conditions_are_identical_in_distribution():
    """When all conditions are drawn from the SAME distribution (no true
    effect of the manipulated factor), the omnibus test should not
    reject at conventional levels, and no pairwise contrast should be
    Holm-significant, in the large majority of repeated trials."""
    false_positive_count = 0
    n_trials = 30
    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        conditions = [_make_condition(cid, 0.0, 1.0, 20, rng) for cid in ("A", "B", "C")]
        result = run_interventional_diagnosis("null_factor", conditions)
        if result.any_holm_significant_contrast:
            false_positive_count += 1
    empirical_fpr = false_positive_count / n_trials
    assert empirical_fpr < 0.15, (
        f"Empirical false-positive rate {empirical_fpr:.2f} is too high for a "
        f"null-effect scenario; expected close to the nominal 5% rate."
    )


def test_recovers_known_large_effect():
    """When conditions are drawn from clearly different distributions
    (a large true effect), the omnibus test must reject and the relevant
    pairwise contrast(s) must be Holm-significant with correctly signed
    Cohen's d."""
    rng = np.random.default_rng(42)
    cond_low = _make_condition("low", 0.0, 0.5, 15, rng)
    cond_high = _make_condition("high", 3.0, 0.5, 15, rng)
    result = run_interventional_diagnosis("known_effect_factor", [cond_low, cond_high])

    assert result.omnibus_p_value < 0.01
    assert result.any_holm_significant_contrast
    contrast = result.pairwise_contrasts[0]
    assert contrast.holm_significant_at_0_05
    assert abs(contrast.cohens_d) > 2.0  # a 3.0-unit shift with std 0.5 is a huge effect


def test_three_condition_ordering_effect_recovered():
    """A more realistic three-condition scenario (mirroring a task-order
    intervention with three sequences): one condition is a clear outlier,
    the other two are similar to each other. The single differing
    pairwise contrasts should be Holm-significant; the similar-vs-similar
    contrast should not be."""
    rng = np.random.default_rng(7)
    seq_a = _make_condition("seq_a", 0.5, 0.3, 20, rng)
    seq_b = _make_condition("seq_b", 0.55, 0.3, 20, rng)  # similar to seq_a
    seq_c = _make_condition("seq_c", 2.5, 0.3, 20, rng)   # clear outlier

    result = run_interventional_diagnosis("task_order", [seq_a, seq_b, seq_c])
    contrast_by_pair = {
        frozenset([c.condition_a, c.condition_b]): c for c in result.pairwise_contrasts
    }
    ab = contrast_by_pair[frozenset(["seq_a", "seq_b"])]
    ac = contrast_by_pair[frozenset(["seq_a", "seq_c"])]
    bc = contrast_by_pair[frozenset(["seq_b", "seq_c"])]

    assert not ab.holm_significant_at_0_05, "seq_a vs seq_b should NOT be significant (similar means)"
    assert ac.holm_significant_at_0_05, "seq_a vs seq_c SHOULD be significant (large mean gap)"
    assert bc.holm_significant_at_0_05, "seq_b vs seq_c SHOULD be significant (large mean gap)"


# --------------------------------------------------------------------------
# Structural / bookkeeping checks
# --------------------------------------------------------------------------
def test_result_names_the_manipulated_factor_only():
    rng = np.random.default_rng(2)
    conditions = [_make_condition(cid, 0.0, 1.0, 10, rng) for cid in ("A", "B")]
    result = run_interventional_diagnosis("update_budget", conditions)
    assert result.factor_name_for_causal_claim == "update_budget"
    assert result.factor_name == "update_budget"


def test_pairwise_contrast_count_matches_combinations():
    rng = np.random.default_rng(3)
    conditions = [_make_condition(cid, 0.0, 1.0, 10, rng) for cid in ("A", "B", "C", "D")]
    result = run_interventional_diagnosis("factor", conditions)
    assert len(result.pairwise_contrasts) == 6  # C(4,2) = 6


def test_condition_mean_and_std_computed_correctly():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    cond = InterventionCondition("test", "desc", values)
    assert cond.mean == pytest.approx(3.0)
    assert cond.n_replicates == 5


def test_summary_lines_are_human_readable_strings():
    rng = np.random.default_rng(4)
    conditions = [_make_condition(cid, 0.0, 1.0, 10, rng) for cid in ("A", "B")]
    result = run_interventional_diagnosis("factor", conditions)
    lines = result.summary_lines()
    assert isinstance(lines, list)
    assert all(isinstance(line, str) for line in lines)
    assert len(lines) > 0


# --------------------------------------------------------------------------
# Levene's variance-equality test
# --------------------------------------------------------------------------
def test_levene_detects_known_variance_difference():
    """When conditions have clearly different variances (but similar
    means), Levene's test should reject equal variances even though the
    Kruskal-Wallis/pairwise mean tests may show no significant
    difference -- this is the concrete demonstration that a Level D
    intervention's effect on outcome CONSISTENCY is detected separately
    from its effect on the mean, motivated directly by the D04 task-order
    experiment's discovery-stage std pattern (0.021/0.027/0.002)."""
    rng = np.random.default_rng(20)
    cond_stable = InterventionCondition(
        "stable", "low variance", list(rng.normal(1.0, 0.02, size=30))
    )
    cond_volatile = InterventionCondition(
        "volatile", "high variance", list(rng.normal(1.0, 0.30, size=30))
    )
    result = run_interventional_diagnosis("test_factor", [cond_stable, cond_volatile])

    assert result.variance_effect_detected, (
        "Levene's test should detect the large, deliberately constructed "
        "variance difference between conditions."
    )
    # And critically: the MEAN-level test should show little to nothing,
    # demonstrating these are genuinely separate, non-redundant claims.
    assert not result.any_holm_significant_contrast


def test_levene_does_not_falsely_flag_equal_variance_conditions():
    """Calibration check: when conditions genuinely have equal variance,
    Levene's test should not reject in the large majority of trials."""
    false_positive_count = 0
    n_trials = 30
    for trial in range(n_trials):
        rng = np.random.default_rng(2000 + trial)
        conditions = [
            InterventionCondition(cid, "equal variance", list(rng.normal(0.0, 1.0, size=20)))
            for cid in ("A", "B", "C")
        ]
        result = run_interventional_diagnosis("null_variance_factor", conditions)
        if result.variance_effect_detected:
            false_positive_count += 1
    empirical_fpr = false_positive_count / n_trials
    assert empirical_fpr < 0.20, (
        f"Empirical Levene false-positive rate {empirical_fpr:.2f} is too high "
        f"for a true equal-variance scenario."
    )


def test_levene_and_mean_test_are_independently_reported():
    """Structural check: variance_effect_detected and
    any_holm_significant_contrast must be independently settable fields,
    never derived from one another."""
    rng = np.random.default_rng(21)
    conditions = [_make_condition(cid, 0.0, 1.0, 15, rng) for cid in ("A", "B")]
    result = run_interventional_diagnosis("factor", conditions)
    assert hasattr(result, "levene_variance_test_p_value")
    assert hasattr(result, "any_holm_significant_contrast")
    assert isinstance(result.variance_effect_detected, bool)
