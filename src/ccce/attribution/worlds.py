"""
ccce.attribution.worlds
=========================

Author: Basil M. Alzboun

Attribution ground-truth worlds A-E (main_v4.tex, Section 7.2, Table 4).

Scientific purpose
-------------------
Level 1 analytical validation in the original CCCE specification tested
only whether an estimator recovers a known *numerical* ground-truth
value. This module implements a strictly stronger test: for each world,
the *identity of the true causal factor* is fixed by design, and an
attribution procedure is scored not on predictive accuracy but on whether
it names the correct factor (Competence, Worlds A-C), correctly declines
to name any factor when none is genuinely responsible (Honesty, World D),
or correctly requires the Level E counterfactual contrast to detect an
effect that no associational or interventional analysis can find
(Necessity, World E).

Each world's ``generate`` method returns an ``AttributionSample``
containing the full observational and intervenable diagnostic vector,
the ground-truth outcome (or paired outcomes, for World E), and the
world's declared true cause. The declared true cause is never inferred
from the sample; it is a property of the world's construction, known to
the experiment harness and withheld from any attribution procedure under
test.

Non-circularity is the load-bearing guarantee of this module: the
equation generating the ground-truth outcome for every world reads only
from the latent(s) explicitly designated as that world's true cause, and
no other latent. This is enforced in two independent ways:

    1. Structurally, via ``verify_world_non_circularity``, which inspects
       each world's outcome-generating source code and asserts it
       references no ``LatentBundle`` field other than its declared
       cause(s).
    2. Statistically, via ``closure_test``, which holds the declared
       cause fixed across many resamples of every other latent and
       verifies the outcome does not move beyond irreducible noise, then
       holds every other latent fixed and resamples the declared cause
       and verifies the outcome *does* move systematically.

Both checks are exercised in ``tests/test_attribution_worlds.py``.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ccce.attribution.factors import (
    ALL_FACTOR_NAMES,
    LatentBundle,
    measure_intervenable_factors,
    measure_observational_factors,
)
from ccce.scm.analytical import AnalyticalSCM, CompetenceLinearForm, LearningProtocolParams, NoUpdateProtocol

OUTCOME_NOISE_SCALE = 0.05


@dataclass
class AttributionSample:
    """One drawn instance of an attribution ground-truth world.

    For Worlds A-D, ``outcome`` is the single ground-truth Competence
    Change severity and ``outcome_paired`` is ``None``. For World E,
    ``outcome`` is the *marginal* (single-arm) outcome, computed as if
    only the target protocol had been observed, while ``outcome_paired``
    holds the (target, control) pair needed for the Level E counterfactual
    contrast; the true effect is visible only in
    ``outcome_paired[0] - outcome_paired[1]``, not in ``outcome`` alone.
    """

    world_id: str
    true_cause: Tuple[str, ...]
    observational: Dict[str, float]
    intervenable: Dict[str, float]
    outcome: float
    outcome_paired: Optional[Tuple[float, float]] = None


@dataclass
class AttributionWorld:
    """A named, documented attribution ground-truth world."""

    world_id: str
    description: str
    true_cause: Tuple[str, ...]
    diagnostic_role: str
    generate_fn: Callable[[np.random.Generator], AttributionSample]
    outcome_fn: Callable[..., float]

    def generate(self, rng: np.random.Generator) -> AttributionSample:
        return self.generate_fn(rng)

    def outcome_equation_source(self) -> str:
        return inspect.getsource(self.outcome_fn)


# ----------------------------------------------------------------------
# World A: single-factor cause
# ----------------------------------------------------------------------
def _world_a_outcome(latents: LatentBundle) -> float:
    """True cause: gradient_interference's latent, alone. No other
    latent appears in this function's body."""
    return 2.5 * np.tanh(latents.gradient_interference)


def _generate_world_a(rng: np.random.Generator) -> AttributionSample:
    latents = LatentBundle.sample(rng)
    obs = measure_observational_factors(latents, rng=rng)
    interv = measure_intervenable_factors(latents)
    noise = rng.normal(0.0, OUTCOME_NOISE_SCALE)
    outcome = _world_a_outcome(latents) + noise
    return AttributionSample(
        world_id="A", true_cause=("gradient_interference",),
        observational=obs, intervenable=interv, outcome=float(outcome),
    )


WORLD_A = AttributionWorld(
    world_id="A",
    description=(
        "A single designated factor (gradient_interference) is the sole causal "
        "term for the outcome; every other diagnostic-vector component is "
        "constructed to be non-causal by design."
    ),
    true_cause=("gradient_interference",),
    diagnostic_role="Competence test (simple case)",
    generate_fn=_generate_world_a,
    outcome_fn=_world_a_outcome,
)


# ----------------------------------------------------------------------
# World B: two-factor interaction
# ----------------------------------------------------------------------
def _world_b_outcome(latents: LatentBundle) -> float:
    """True cause: a genuine interaction between representation_drift's
    and update_magnitude's latents (a product term, not two independent
    additive terms) -- neither latent alone is sufficient; only their
    product drives the outcome."""
    return 1.8 * np.tanh(latents.representation_drift * latents.update_magnitude)


def _generate_world_b(rng: np.random.Generator) -> AttributionSample:
    latents = LatentBundle.sample(rng)
    obs = measure_observational_factors(latents, rng=rng)
    interv = measure_intervenable_factors(latents)
    noise = rng.normal(0.0, OUTCOME_NOISE_SCALE)
    outcome = _world_b_outcome(latents) + noise
    return AttributionSample(
        world_id="B", true_cause=("representation_drift", "update_magnitude"),
        observational=obs, intervenable=interv, outcome=float(outcome),
    )


WORLD_B = AttributionWorld(
    world_id="B",
    description=(
        "Two factors (representation_drift, update_magnitude) interact "
        "causally through a genuine product term; neither is individually "
        "sufficient to explain the outcome."
    ),
    true_cause=("representation_drift", "update_magnitude"),
    diagnostic_role="Competence test (harder, compound case)",
    generate_fn=_generate_world_b,
    outcome_fn=_world_b_outcome,
)


# ----------------------------------------------------------------------
# World C: intervenable-only cause, invisible to observational signals
# ----------------------------------------------------------------------
def _world_c_outcome(latents: LatentBundle) -> float:
    """True cause: task_order's latent alone. No category-A latent
    appears in this function's body, so no observational signal can be
    associated with the outcome in this world -- by construction, not by
    chance in a finite sample."""
    return 2.0 * np.sign(latents.task_order) * np.sqrt(np.abs(latents.task_order))


def _generate_world_c(rng: np.random.Generator) -> AttributionSample:
    latents = LatentBundle.sample(rng)
    obs = measure_observational_factors(latents, rng=rng)
    interv = measure_intervenable_factors(latents)
    noise = rng.normal(0.0, OUTCOME_NOISE_SCALE)
    outcome = _world_c_outcome(latents) + noise
    return AttributionSample(
        world_id="C", true_cause=("task_order",),
        observational=obs, intervenable=interv, outcome=float(outcome),
    )


WORLD_C = AttributionWorld(
    world_id="C",
    description=(
        "An intervenable factor (task_order) is the true cause; no "
        "category-A observational signal is associated with the outcome "
        "in this world, testing whether Level D intervention is necessary "
        "to detect a cause that Level C associational analysis will miss."
    ),
    true_cause=("task_order",),
    diagnostic_role="Tests necessity of Level D (interventional diagnosis)",
    generate_fn=_generate_world_c,
    outcome_fn=_world_c_outcome,
)


# ----------------------------------------------------------------------
# World D: no measured factor explains the outcome (Honesty test)
# ----------------------------------------------------------------------
def _world_d_outcome(latents: LatentBundle) -> float:
    """True cause: `unmeasured`, a latent deliberately excluded from
    ALL_FACTOR_NAMES and therefore from every measurement function in
    `ccce.attribution.factors`. No factor in the full diagnostic
    vocabulary (category A, B, or C) can, even in principle, be
    correlated with this outcome above noise level, because none of
    their measurement functions read `latents.unmeasured`."""
    return 2.2 * np.tanh(latents.unmeasured)


def _generate_world_d(rng: np.random.Generator) -> AttributionSample:
    latents = LatentBundle.sample(rng)
    obs = measure_observational_factors(latents, rng=rng)
    interv = measure_intervenable_factors(latents)
    noise = rng.normal(0.0, OUTCOME_NOISE_SCALE)
    outcome = _world_d_outcome(latents) + noise
    return AttributionSample(
        world_id="D", true_cause=(),  # empty: no in-vocabulary cause
        observational=obs, intervenable=interv, outcome=float(outcome),
    )


WORLD_D = AttributionWorld(
    world_id="D",
    description=(
        "No factor in the full diagnostic vocabulary explains the outcome; "
        "the true generating latent lies outside the measured vocabulary "
        "entirely, by construction. A correct attribution procedure must "
        "report UNEXPLAINED here, not force an attribution to the nearest "
        "available factor."
    ),
    true_cause=(),
    diagnostic_role="Honesty test",
    generate_fn=_generate_world_d,
    outcome_fn=_world_d_outcome,
)


# ----------------------------------------------------------------------
# World E: counterfactual-only cause (Level E necessity / Killer 3)
# ----------------------------------------------------------------------
# World E is built on the existing Level-1 analytical SCM's nominal/
# intervention-interaction separation (b_i, h_i), the same structural
# pattern used for the earlier "hidden counterfactual vulnerability"
# construction: a moderator latent that is orthogonal to the marginal
# (single-arm) outcome but fully drives the paired target-vs-control
# contrast.
_WORLD_E_DIM = 4


def _world_e_scm() -> AnalyticalSCM:
    comp = CompetenceLinearForm(
        competence_id="K_E",
        a_i=0.0,
        b_i=np.array([1.0, 0.0, 0.0, 0.0]),   # marginal-sensitive direction
        h_i=np.array([0.0, 3.0, 0.0, 0.0]),   # counterfactual-contrast-sensitive direction, orthogonal to b_i
        sigma_y=0.05,
        valid_intervention_support=(0.0, 1.0),
    )
    return AnalyticalSCM(dim=_WORLD_E_DIM, sigma_u=0.02, theta0_mean=np.zeros(_WORLD_E_DIM), competences={"K_E": comp})


def _world_e_outcome(scm: AnalyticalSCM, proto_t: LearningProtocolParams, proto_c: LearningProtocolParams,
                      c_marginal: float, c_paired: float) -> Tuple[float, float, float]:
    """Returns (marginal_outcome_under_T_only, outcome_T_at_c_paired,
    outcome_C_at_c_paired). The marginal outcome (evaluated at the
    nominal condition, using only the target arm, as an observational
    procedure that never instantiates the control arm would see it) is
    constructed, via b_i and h_i's orthogonality, to carry no signal
    about the moderator that only the paired contrast reveals."""
    means = scm.exact_theta_mean_trajectory([])
    marginal = scm.exact_ccce("K_E", c_marginal, means, proto_t, NoUpdateProtocol())
    q_t = scm.exact_ccce("K_E", c_paired, means, proto_t, NoUpdateProtocol())
    q_c = scm.exact_ccce("K_E", c_paired, means, proto_c, NoUpdateProtocol())
    return marginal, q_t, q_c


def _world_e_protocols(latents: LatentBundle) -> Tuple[LearningProtocolParams, LearningProtocolParams]:
    """Builds the (target, control) protocol pair from ONLY the declared
    cause latent (environmental_causal_factor). This function is the
    object of World E's non-circularity check: it must reference no
    latent other than the declared cause.

    Design note (corrected): the control protocol's gain is held FIXED
    at a reference value rather than mirrored as (0.6 - 0.1*env). An
    earlier version mirrored both gains symmetrically around 0.6, which
    is mathematically inert: because the target and control directions
    are exact negatives of one another (`target` vs. `-target`), the
    quantity that drives the SCM's mean trajectory is proportional to
    `(eta_T + eta_C)`, and a symmetric +delta/-delta perturbation on the
    two gains cancels exactly in that sum, making `env` have *zero*
    effect on the outcome regardless of its value -- a genuine
    non-circularity failure caught by `closure_test`
    (`std_cause_varied` came out numerically zero). Holding `gain_c`
    fixed at the reference value 0.6 and letting only `gain_t` respond to
    `env` avoids this cancellation while preserving the intended
    qualitative structure: a target protocol whose aggressiveness depends
    on the causal factor, contrasted against a stable control reference.
    """
    gain_t = float(np.clip(0.6 + 0.15 * latents.environmental_causal_factor, 0.1, 0.95))
    gain_c = 0.6  # fixed reference, independent of env by design
    target = np.array([0.0, 1.0, 0.0, 0.0])
    proto_t = LearningProtocolParams("world_e_T", target=target, gain=gain_t, budget=5.0)
    proto_c = LearningProtocolParams("world_e_C", target=-target, gain=gain_c, budget=5.0)
    return proto_t, proto_c


def _generate_world_e(rng: np.random.Generator) -> AttributionSample:
    latents = LatentBundle.sample(rng)
    obs = measure_observational_factors(latents, rng=rng)
    interv = measure_intervenable_factors(latents)

    scm = _world_e_scm()
    proto_t, proto_c = _world_e_protocols(latents)
    marginal, q_t, q_c = _world_e_outcome(scm, proto_t, proto_c, c_marginal=0.0, c_paired=1.0)
    noise = rng.normal(0.0, OUTCOME_NOISE_SCALE)
    return AttributionSample(
        world_id="E", true_cause=("environmental_causal_factor",),
        observational=obs, intervenable=interv,
        outcome=float(marginal + noise),
        outcome_paired=(float(q_t + noise), float(q_c + noise)),
    )


WORLD_E = AttributionWorld(
    world_id="E",
    description=(
        "The true cause (environmental_causal_factor) is visible only as a "
        "counterfactual contrast between a target and a control learning "
        "protocol; it produces no detectable marginal (single-arm) signal "
        "and no detectable association with any category-A or category-B "
        "factor under direct manipulation alone."
    ),
    true_cause=("environmental_causal_factor",),
    diagnostic_role="Tests necessity of Level E (Killer 3: counterfactual necessity)",
    generate_fn=_generate_world_e,
    outcome_fn=_world_e_protocols,
)


ALL_WORLDS: Dict[str, AttributionWorld] = {
    "A": WORLD_A, "B": WORLD_B, "C": WORLD_C, "D": WORLD_D, "E": WORLD_E,
}


# ----------------------------------------------------------------------
# Non-circularity verification
# ----------------------------------------------------------------------
def verify_world_non_circularity(world: AttributionWorld) -> None:
    """Structural check: the world's outcome-generating function (or, for
    World E, its protocol-construction function) must reference no
    ``LatentBundle`` field other than its declared ``true_cause``, with
    one documented exception: World D's outcome equation legitimately
    reads ``unmeasured`` -- the latent deliberately excluded from the
    measured diagnostic vocabulary (``ALL_FACTOR_NAMES``) -- since
    ``unmeasured`` being outside that vocabulary, not outside World D's
    own generating equation, is precisely what World D is constructed to
    test (Honesty). The check still guarantees, via
    ``test_verify_latent_independence_*`` in the test suite, that
    ``unmeasured`` is never read by any of the actual measurement
    functions in ``ccce.attribution.factors``.
    """
    all_latent_names = set(ALL_FACTOR_NAMES) | {"unmeasured"}
    source = world.outcome_equation_source()
    referenced = {name for name in all_latent_names if f"latents.{name}" in source}
    allowed = set(world.true_cause) | ({"unmeasured"} if world.world_id == "D" else set())
    unexpected = referenced - allowed
    assert not unexpected, (
        f"World {world.world_id} non-circularity violation: outcome/protocol "
        f"construction references latent(s) {unexpected}, which are not in "
        f"the declared true_cause {world.true_cause}."
    )


def _scalar_outcome_for_closure(world: AttributionWorld, latents: LatentBundle) -> float:
    """Reduces any world (including E) to a single scalar outcome for the
    closure test, so Worlds A-E can be tested with one uniform procedure."""
    if world.world_id == "E":
        scm = _world_e_scm()
        proto_t, proto_c = _world_e_protocols(latents)
        _, q_t, q_c = _world_e_outcome(scm, proto_t, proto_c, c_marginal=0.0, c_paired=1.0)
        return q_t - q_c
    return world.outcome_fn(latents)


def closure_test(
    world: AttributionWorld, rng: np.random.Generator, n_samples: int = 500,
) -> Dict[str, float]:
    """Statistical (runtime) non-circularity check, complementing the
    structural check above. Two sub-tests:

    1. **Cause-fixed closure**: fixing the declared true-cause latent(s)
       and resampling every other latent should leave the outcome
       constant up to the irreducible noise floor (``OUTCOME_NOISE_SCALE``).
    2. **Cause-varied sensitivity**: fixing every other latent and
       resampling only the declared cause (or, for World D, the
       `unmeasured` latent) should move the outcome systematically.

    Returns a dict with the empirical standard deviation of the outcome
    under each condition, for direct comparison against
    ``OUTCOME_NOISE_SCALE``.
    """
    base_latents = LatentBundle.sample(rng)
    field_names = [f.name for f in base_latents.__dataclass_fields__.values()]
    cause_fields = world.true_cause if world.true_cause else ("unmeasured",)

    outcomes_cause_fixed = []
    for _ in range(n_samples):
        resampled = LatentBundle.sample(rng)
        merged_dict = {name: getattr(resampled, name) for name in field_names}
        for cause_name in cause_fields:
            merged_dict[cause_name] = getattr(base_latents, cause_name)
        merged = LatentBundle(**merged_dict)
        outcomes_cause_fixed.append(_scalar_outcome_for_closure(world, merged))

    outcomes_cause_varied = []
    for _ in range(n_samples):
        resampled_cause = LatentBundle.sample(rng)
        merged_dict = {name: getattr(base_latents, name) for name in field_names}
        for cause_name in cause_fields:
            merged_dict[cause_name] = getattr(resampled_cause, cause_name)
        merged = LatentBundle(**merged_dict)
        outcomes_cause_varied.append(_scalar_outcome_for_closure(world, merged))

    return dict(
        std_cause_fixed=float(np.std(outcomes_cause_fixed)),
        std_cause_varied=float(np.std(outcomes_cause_varied)),
    )
