"""
tests/test_attribution_worlds.py

Author: Basil M. Alzboun

Tests for ccce.attribution.worlds and ccce.attribution.factors.

These tests verify the single most important property of the diagnostic
framework's attribution ground-truth construction (main_v4.tex, Section
7.2): that each world's declared true cause is provably, not merely
statistically, the sole driver of its outcome. Both the structural check
(source-code inspection) and the statistical check (closure test) are
exercised for every world, plus a source-level check confirming that no
two Category A/B measurement functions in ccce.attribution.factors read
the same latent.
"""
import numpy as np
import pytest

from ccce.attribution.factors import (
    ALL_FACTOR_NAMES,
    LatentBundle,
    measure_intervenable_factors,
    measure_observational_factors,
    verify_latent_independence,
)
from ccce.attribution.worlds import (
    ALL_WORLDS,
    OUTCOME_NOISE_SCALE,
    closure_test,
    verify_world_non_circularity,
)


# --------------------------------------------------------------------------
# factors.py: latent bundle and measurement functions
# --------------------------------------------------------------------------
def test_latent_bundle_reproducible_given_seed():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    l1 = LatentBundle.sample(rng1)
    l2 = LatentBundle.sample(rng2)
    assert l1 == l2 or all(
        getattr(l1, f.name) == getattr(l2, f.name) for f in l1.__dataclass_fields__.values()
    )


def test_latent_bundle_fields_are_independent_draws():
    rng = np.random.default_rng(0)
    latents = LatentBundle.sample(rng)
    values = [getattr(latents, f.name) for f in latents.__dataclass_fields__.values()]
    # No two fields should be exactly equal (would indicate a shared draw / bug).
    assert len(set(values)) == len(values)


def test_observational_factors_cover_all_category_a_names():
    rng = np.random.default_rng(0)
    latents = LatentBundle.sample(rng)
    obs = measure_observational_factors(latents, rng=rng)
    from ccce.attribution.factors import OBSERVATIONAL_FACTOR_NAMES
    assert set(obs.keys()) == set(OBSERVATIONAL_FACTOR_NAMES)


def test_intervenable_factors_cover_all_category_b_names():
    rng = np.random.default_rng(0)
    latents = LatentBundle.sample(rng)
    interv = measure_intervenable_factors(latents)
    from ccce.attribution.factors import INTERVENABLE_FACTOR_NAMES
    assert set(interv.keys()) == set(INTERVENABLE_FACTOR_NAMES)


def test_verify_latent_independence_passes_on_unmodified_module():
    # Must not raise.
    verify_latent_independence()


def test_verify_latent_independence_detects_a_planted_violation(monkeypatch):
    """Sanity-check the checker itself: if a measurement function is
    planted to read a latent it should not, the check must fail."""
    import ccce.attribution.factors as factors_module

    def _broken_measure_observational_factors(latents, noise_scale=0.05, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        # Deliberately (and incorrectly) reads task_order, a category-B latent.
        _ = latents.task_order
        return factors_module.measure_observational_factors.__wrapped__(latents, noise_scale, rng) \
            if hasattr(factors_module.measure_observational_factors, "__wrapped__") \
            else {}

    monkeypatch.setattr(factors_module, "measure_observational_factors", _broken_measure_observational_factors)
    with pytest.raises(AssertionError):
        factors_module.verify_latent_independence()


# --------------------------------------------------------------------------
# worlds.py: structural non-circularity
# --------------------------------------------------------------------------
@pytest.mark.parametrize("world_id", ["A", "B", "C", "D", "E"])
def test_structural_non_circularity_for_every_world(world_id):
    world = ALL_WORLDS[world_id]
    verify_world_non_circularity(world)  # must not raise


def test_world_a_true_cause_is_single_factor():
    assert ALL_WORLDS["A"].true_cause == ("gradient_interference",)


def test_world_b_true_cause_is_two_factor_interaction():
    assert ALL_WORLDS["B"].true_cause == ("representation_drift", "update_magnitude")


def test_world_c_true_cause_is_intervenable_only():
    assert ALL_WORLDS["C"].true_cause == ("task_order",)


def test_world_d_true_cause_is_empty():
    assert ALL_WORLDS["D"].true_cause == ()


def test_world_e_true_cause_is_environmental_factor():
    assert ALL_WORLDS["E"].true_cause == ("environmental_causal_factor",)


# --------------------------------------------------------------------------
# worlds.py: statistical (closure-test) non-circularity
# --------------------------------------------------------------------------
@pytest.mark.parametrize("world_id", ["A", "B", "C", "D", "E"])
def test_closure_test_cause_fixed_is_near_noise_floor(world_id):
    """Holding the true cause fixed and resampling every other latent
    must leave the outcome within a small multiple of the irreducible
    noise floor -- i.e., no other latent moves the outcome."""
    world = ALL_WORLDS[world_id]
    rng = np.random.default_rng(123)
    result = closure_test(world, rng, n_samples=300)
    assert result["std_cause_fixed"] < 5 * OUTCOME_NOISE_SCALE, (
        f"World {world_id}: outcome varies too much ({result['std_cause_fixed']:.4f}) "
        f"when the declared cause is held fixed and all other latents are resampled -- "
        f"this indicates a non-circularity violation not caught by the structural check."
    )


@pytest.mark.parametrize("world_id", ["A", "B", "C", "D", "E"])
def test_closure_test_cause_varied_shows_real_sensitivity(world_id):
    """Holding every other latent fixed and resampling only the declared
    cause must produce a materially larger outcome spread than the
    cause-fixed condition -- i.e., the declared cause is not vacuous."""
    world = ALL_WORLDS[world_id]
    rng = np.random.default_rng(456)
    result = closure_test(world, rng, n_samples=300)
    assert result["std_cause_varied"] > 3 * result["std_cause_fixed"], (
        f"World {world_id}: varying the declared cause ({result['std_cause_varied']:.4f}) "
        f"does not produce meaningfully more outcome variance than holding it fixed "
        f"({result['std_cause_fixed']:.4f}) -- the declared cause may not actually "
        f"drive the outcome."
    )


# --------------------------------------------------------------------------
# worlds.py: sample generation end-to-end
# --------------------------------------------------------------------------
@pytest.mark.parametrize("world_id", ["A", "B", "C", "D"])
def test_generate_sample_has_no_paired_outcome_for_non_e_worlds(world_id):
    rng = np.random.default_rng(0)
    sample = ALL_WORLDS[world_id].generate(rng)
    assert sample.outcome_paired is None
    assert isinstance(sample.outcome, float)


def test_generate_world_e_sample_has_paired_outcome():
    rng = np.random.default_rng(0)
    sample = ALL_WORLDS["E"].generate(rng)
    assert sample.outcome_paired is not None
    assert len(sample.outcome_paired) == 2


def test_world_e_marginal_outcome_is_small_relative_to_paired_contrast():
    """The defining property of World E: the marginal (single-arm)
    outcome must be small relative to the paired T-vs-C contrast,
    across many draws, on average -- this is what makes Level E
    necessary rather than merely sufficient."""
    rng = np.random.default_rng(789)
    marginal_abs = []
    contrast_abs = []
    for _ in range(100):
        sample = ALL_WORLDS["E"].generate(rng)
        marginal_abs.append(abs(sample.outcome))
        contrast_abs.append(abs(sample.outcome_paired[0] - sample.outcome_paired[1]))
    assert np.mean(contrast_abs) > 2 * np.mean(marginal_abs), (
        "World E's paired contrast should be substantially larger in magnitude "
        "than its marginal (single-arm) outcome, on average, to genuinely test "
        "the necessity of the Level E counterfactual instrument."
    )


def test_all_worlds_generate_without_error_across_many_seeds():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        for world in ALL_WORLDS.values():
            sample = world.generate(rng)
            assert np.isfinite(sample.outcome)
            for v in sample.observational.values():
                assert np.isfinite(v)
            for v in sample.intervenable.values():
                assert np.isfinite(v)
