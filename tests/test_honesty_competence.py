"""
tests/test_honesty_competence.py

Author: Basil M. Alzboun

Tests for Killer Experiment 2 (Honesty and Competence), exercised against
the reference threshold-based attribution procedure and the two naive
baselines (always-attribute, always-abstain).
"""
import numpy as np
import pytest

from ccce.attribution.honesty_competence import (
    always_abstain_baseline,
    always_attribute_baseline,
    evaluate_competence,
    evaluate_honesty,
    run_killer_experiment_two,
    threshold_attribution_procedure,
)


def test_always_attribute_baseline_never_abstains_on_world_d():
    """Sanity check on the baseline itself: it must score ~0% Honesty,
    since it is defined to always attribute to something."""
    rng = np.random.default_rng(0)
    rate, n = evaluate_honesty(always_attribute_baseline, rng, n_batches=30, batch_size=30)
    assert rate < 0.05, "always_attribute_baseline should almost never abstain"


def test_always_abstain_baseline_never_correct_on_worlds_abc():
    """Sanity check: this baseline must score exactly 0% Competence,
    since it never names any factor."""
    rng = np.random.default_rng(0)
    rate, n, per_world = evaluate_competence(always_abstain_baseline, rng, ("A", "B", "C"), n_batches=20, batch_size=30)
    assert rate == 0.0
    assert all(v == 0.0 for v in per_world.values())


def test_threshold_procedure_honesty_on_world_d():
    """The reference procedure should abstain (correctly) on World D
    substantially more often than the always-attribute baseline, since
    World D's outcome carries no correlation with any measured factor
    by construction."""
    rng = np.random.default_rng(1)
    rate, n = evaluate_honesty(threshold_attribution_procedure, rng, n_batches=40, batch_size=40)
    baseline_rate, _ = evaluate_honesty(always_attribute_baseline, rng, n_batches=40, batch_size=40)
    assert rate > baseline_rate


def test_threshold_procedure_competence_on_world_a():
    """World A has a single, simple associational cause; the reference
    procedure should identify it correctly a clear majority of the time."""
    rng = np.random.default_rng(2)
    rate, n, per_world = evaluate_competence(threshold_attribution_procedure, rng, ("A",), n_batches=40, batch_size=40)
    assert per_world["A"] > 0.5


def test_threshold_procedure_fails_competence_on_world_c_by_design():
    """World C's true cause (task_order) is an intervenable factor with
    NO associational signal; a correlation-only procedure like the
    reference one should systematically fail to identify it, precisely
    demonstrating the necessity of Level D intervention."""
    rng = np.random.default_rng(3)
    rate, n, per_world = evaluate_competence(threshold_attribution_procedure, rng, ("C",), n_batches=40, batch_size=40)
    assert per_world["C"] < 0.3, (
        "An associational-only procedure should largely fail on World C by "
        "construction; a high success rate here would indicate a leak in "
        "World C's non-circularity guarantee."
    )


def test_killer_two_result_structure():
    rng = np.random.default_rng(4)
    result = run_killer_experiment_two(threshold_attribution_procedure, rng, n_batches=20, batch_size=30)
    assert 0.0 <= result.honesty_rate <= 1.0
    assert 0.0 <= result.competence_rate <= 1.0
    assert set(result.competence_by_world.keys()) == {"A", "B", "C"}
    assert isinstance(result.killer_two_passed, bool)


def test_killer_two_always_abstain_fails_because_competence_is_zero():
    """The always-abstain baseline trivially passes Honesty (100%) but
    must fail Competence (0%), so killer_two_passed must be False --
    this is the concrete demonstration of main_v4.tex's warning that
    'passing Honesty without Competence describes a framework that
    abstains indiscriminately.'"""
    rng = np.random.default_rng(5)

    def always_abstain_everywhere(samples):
        return ()

    honesty_rate, _ = evaluate_honesty(always_abstain_everywhere, rng, n_batches=20, batch_size=30)
    competence_rate, _, _ = evaluate_competence(always_abstain_everywhere, rng, ("A", "B", "C"), n_batches=20, batch_size=30)
    assert honesty_rate == 1.0
    assert competence_rate == 0.0
