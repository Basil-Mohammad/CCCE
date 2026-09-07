# Author: Basil M. Alzboun

"""
tests/test_killer_experiments.py

Tests for ccce.attribution.killer_experiments (Killer Experiments 1 and 3).
"""
import numpy as np
import pytest

from ccce.attribution.factors import OBSERVATIONAL_FACTOR_NAMES, INTERVENABLE_FACTOR_NAMES
from ccce.attribution.killer_experiments import (
    WORLD_IDS,
    build_killer1_dataset,
    run_killer_experiment_one,
    run_killer_experiment_three,
)


# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------
def test_killer1_dataset_shapes_are_consistent():
    rng = np.random.default_rng(0)
    data = build_killer1_dataset(rng, n_per_world=20)
    n_expected = 20 * len(WORLD_IDS)
    n_obs = len(OBSERVATIONAL_FACTOR_NAMES)
    n_interv = len(INTERVENABLE_FACTOR_NAMES)

    assert data.X0.shape == (n_expected, n_obs)
    assert data.X1.shape == (n_expected, n_obs + n_interv + 1)  # +1 for cce_probe
    assert data.X2.shape == (n_expected, n_obs + 2)  # +2 for (q_t, q_c)
    assert data.y.shape == (n_expected,)
    assert data.world_labels.shape == (n_expected,)


def test_killer1_dataset_covers_all_worlds_equally():
    rng = np.random.default_rng(0)
    data = build_killer1_dataset(rng, n_per_world=15)
    unique, counts = np.unique(data.world_labels, return_counts=True)
    assert set(unique) == set(WORLD_IDS)
    assert all(c == 15 for c in counts)


def test_killer1_dataset_all_finite():
    rng = np.random.default_rng(1)
    data = build_killer1_dataset(rng, n_per_world=30)
    assert np.all(np.isfinite(data.X0))
    assert np.all(np.isfinite(data.X1))
    assert np.all(np.isfinite(data.X2))
    assert np.all(np.isfinite(data.y))


def test_world_e_target_uses_paired_contrast_not_marginal():
    """The severity target for World E samples must be the paired
    contrast, not the (deliberately uninformative) marginal outcome --
    otherwise Killer 1 would be unwinnable for the wrong reason."""
    rng = np.random.default_rng(2)
    data = build_killer1_dataset(rng, n_per_world=50)
    world_e_mask = data.world_labels == "E"
    world_e_targets = data.y[world_e_mask]
    # The paired contrast is driven by h_i=[0,3,...] and should have
    # meaningfully larger spread than the near-zero marginal outcome would.
    assert np.std(world_e_targets) > 0.1


# --------------------------------------------------------------------------
# Killer Experiment 1
# --------------------------------------------------------------------------
def test_killer1_result_structure():
    rng = np.random.default_rng(3)
    result = run_killer_experiment_one(rng, n_per_world=100, n_folds=5)
    assert isinstance(result.r2_m0, float)
    assert isinstance(result.r2_m1, float)
    assert isinstance(result.r2_m2, float)
    assert result.n_total == 500
    assert result.n_folds == 5


def test_killer1_is_deterministic_given_seed():
    r1 = run_killer_experiment_one(np.random.default_rng(42), n_per_world=100, n_folds=5)
    r2 = run_killer_experiment_one(np.random.default_rng(42), n_per_world=100, n_folds=5)
    assert r1.r2_m0 == r2.r2_m0
    assert r1.r2_m1 == r2.r2_m1
    assert r1.r2_m2 == r2.r2_m2


def test_killer1_m1_outperforms_m0_on_this_benchmark():
    """M1 (with Category B + CCE probe) should outperform M0 (Category A
    only) on this benchmark, because Worlds C and E specifically encode
    causes invisible to Category A alone -- if M1 did not outperform M0
    here, that would indicate a real problem with the feature
    construction, not merely an interesting negative result, since the
    benchmark is deliberately constructed to make this gain available."""
    rng = np.random.default_rng(7)
    result = run_killer_experiment_one(rng, n_per_world=200, n_folds=5)
    assert result.incremental_gain_m1_over_m0 > 0.0


def test_killer1_failure_a_property_is_boolean():
    rng = np.random.default_rng(8)
    result = run_killer_experiment_one(rng, n_per_world=100, n_folds=5)
    assert isinstance(result.failure_a_triggered, bool)


# --------------------------------------------------------------------------
# Killer Experiment 3
# --------------------------------------------------------------------------
def test_killer3_result_structure():
    rng = np.random.default_rng(9)
    result = run_killer_experiment_three(rng, n_samples=200)
    assert 0.0 <= result.best_category_a_correlation_in_world_e <= 1.0
    assert 0.0 <= result.cce_probe_correlation_in_world_e <= 1.0
    assert result.best_category_a_factor in OBSERVATIONAL_FACTOR_NAMES
    assert result.n_samples == 200


def test_killer3_cce_probe_dominates_category_a_in_world_e():
    """The defining result of Killer 3: within World E, the CCE probe's
    correlation with the true signal must be dramatically higher than
    the best Category A observational signal's correlation -- this is
    the concrete demonstration that Level E is NECESSARY for World E,
    not merely one option among several equally good ones."""
    rng = np.random.default_rng(10)
    result = run_killer_experiment_three(rng, n_samples=300)
    assert result.cce_probe_correlation_in_world_e > 0.9
    assert result.best_category_a_correlation_in_world_e < 0.3
    assert result.necessity_demonstrated is True


def test_killer3_category_a_correlation_is_near_chance_level():
    """More specifically: the best Category-A correlation in World E
    should be consistent with pure noise (no true association), since
    World E's non-circularity guarantee (tested independently in
    test_attribution_worlds.py) means no Category-A latent enters World
    E's outcome equation at all."""
    rng = np.random.default_rng(11)
    result = run_killer_experiment_three(rng, n_samples=500)
    assert result.best_category_a_correlation_in_world_e < 0.15
