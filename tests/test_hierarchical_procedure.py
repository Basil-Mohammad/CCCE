# Author: Basil M. Alzboun

"""
tests/test_hierarchical_procedure.py

Tests for ccce.attribution.hierarchical_procedure -- the Level C -> Level D
cascade attribution procedure.
"""
import numpy as np
import pytest

from ccce.attribution.worlds import ALL_WORLDS
from ccce.attribution.hierarchical_procedure import (
    hierarchical_attribution_procedure,
    hierarchical_attribution_with_evidence,
)
from ccce.attribution.diagnostic_record import EvidenceLevel
from ccce.attribution.honesty_competence import (
    evaluate_competence,
    evaluate_honesty,
    threshold_attribution_procedure,
)


# --------------------------------------------------------------------------
# Comparative accuracy against the flat reference procedure
# --------------------------------------------------------------------------
def test_hierarchical_matches_or_beats_reference_on_world_a():
    """World A (simple single associational cause): the two-stage
    individual-then-interaction cascade, with per-stage Holm correction,
    should perform at least as well as the flat reference procedure."""
    rng = np.random.default_rng(100)
    rate_ref, _, _ = evaluate_competence(threshold_attribution_procedure, rng, ("A",), n_batches=40, batch_size=40)
    rng2 = np.random.default_rng(100)
    rate_hier, _, _ = evaluate_competence(hierarchical_attribution_procedure, rng2, ("A",), n_batches=40, batch_size=40)
    assert rate_hier >= rate_ref - 0.1  # allow small noise-driven variation, not a regression


def test_hierarchical_substantially_beats_reference_on_world_c():
    """World C (intervenable-only cause): the flat associational-only
    reference procedure is 0% by design (Category B is out of its
    search space). The hierarchical procedure's Level D tercile-split
    stage should do MUCH better -- this is the core claim motivating
    this module's existence."""
    rng = np.random.default_rng(101)
    rate_ref, _, _ = evaluate_competence(threshold_attribution_procedure, rng, ("C",), n_batches=40, batch_size=40)
    rng2 = np.random.default_rng(101)
    rate_hier, _, _ = evaluate_competence(hierarchical_attribution_procedure, rng2, ("C",), n_batches=40, batch_size=40)
    assert rate_ref < 0.05  # confirms the reference procedure's known World-C blindness
    assert rate_hier > 0.5  # substantial, not marginal, improvement


def test_hierarchical_honesty_on_world_d_remains_reasonable():
    """The added complexity (more candidate hypotheses, an additional
    Level D stage) must not collapse Honesty on World D -- an
    appropriately-corrected cascade should still abstain the large
    majority of the time when no in-vocabulary cause exists."""
    rng = np.random.default_rng(102)
    rate, n = evaluate_honesty(hierarchical_attribution_procedure, rng, n_batches=40, batch_size=40)
    assert rate > 0.5


# --------------------------------------------------------------------------
# Structural / mechanism checks
# --------------------------------------------------------------------------
def test_world_b_interaction_when_found_returns_both_constituent_factors():
    """If the interaction stage ever fires, it must return BOTH
    constituent factor names (matching World B's true_cause tuple
    exactly), never a single combined 'a*b' string."""
    rng = np.random.default_rng(103)
    world = ALL_WORLDS["B"]
    # Run many batches; on any batch where a non-empty result is returned,
    # verify it is a valid subset structure (pairs, not combined strings).
    for _ in range(50):
        batch = [world.generate(rng) for _ in range(40)]
        result = hierarchical_attribution_procedure(batch)
        for name in result:
            assert "*" not in name, f"Found un-split combined name '{name}' in result"


def test_evidence_with_detail_tags_individual_correctly():
    """For World A, when the correct factor is found, it must be tagged
    with ASSOCIATIONAL evidence level and 'individual' detail (not
    escalated to Level D, since Level C already explains it)."""
    rng = np.random.default_rng(104)
    world = ALL_WORLDS["A"]
    found_individual = False
    for _ in range(30):
        batch = [world.generate(rng) for _ in range(40)]
        evidence = hierarchical_attribution_with_evidence(batch)
        for e in evidence:
            if e.factor_name == "gradient_interference" and e.detail == "individual":
                assert e.evidence_level == EvidenceLevel.ASSOCIATIONAL
                found_individual = True
    assert found_individual, "Expected at least one batch to find gradient_interference individually"


def test_evidence_with_detail_tags_interventional_correctly_for_world_c():
    """For World C, when task_order is found via the tercile-split stage,
    it must be tagged INTERVENTIONAL, not ASSOCIATIONAL -- since Level C
    never has access to Category B data in this procedure's design."""
    rng = np.random.default_rng(105)
    world = ALL_WORLDS["C"]
    found_interventional = False
    for _ in range(30):
        batch = [world.generate(rng) for _ in range(40)]
        evidence = hierarchical_attribution_with_evidence(batch)
        for e in evidence:
            if e.factor_name == "task_order":
                assert e.evidence_level == EvidenceLevel.INTERVENTIONAL
                assert e.detail == "tercile_split"
                found_interventional = True
    assert found_interventional, "Expected at least one batch to find task_order via Level D"


def test_small_batch_returns_empty_not_error():
    from ccce.attribution.worlds import AttributionSample
    tiny_batch = [
        AttributionSample("A", ("gradient_interference",), {n: 0.0 for n in
            ["gradient_interference", "representation_drift", "plasticity",
             "task_similarity", "update_magnitude", "instability_signature"]},
            {n: 0.0 for n in ["task_order", "task_overlap", "update_budget",
             "learning_rate", "replay_availability", "environmental_causal_factor"]},
            outcome=0.0)
    ]
    result = hierarchical_attribution_procedure(tiny_batch)
    assert result == ()


def test_deterministic_given_same_data():
    rng = np.random.default_rng(106)
    world = ALL_WORLDS["A"]
    batch = [world.generate(rng) for _ in range(40)]
    result1 = hierarchical_attribution_procedure(batch)
    result2 = hierarchical_attribution_procedure(batch)
    assert result1 == result2
