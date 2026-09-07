"""
ccce.attribution.honesty_competence
======================================

Author: Basil M. Alzboun

Killer Experiment 2 (main_v4.tex, Section 6.4): the Honesty and
Competence tests, evaluated separately and jointly required.

    Honesty rate  = P(report UNEXPLAINED | World D: no in-vocabulary cause)
    Competence rate = P(correct factor identified | Worlds A-C: genuine cause present)

This module also provides a REFERENCE attribution procedure
(``threshold_attribution_procedure``) implementing a simple, explicit,
auditable correlation-threshold rule. It exists for two purposes:

    1. To give ``evaluate_honesty`` and ``evaluate_competence`` something
       concrete to evaluate in this module's own test suite, demonstrating
       the evaluation machinery end-to-end.
    2. To serve as an explicit, pre-registered BASELINE against which any
       more sophisticated attribution procedure developed later must be
       compared -- consistent with the specification's insistence
       (main_v4.tex, Section 6.2, Killer 1) that predictive/explanatory
       power be judged relative to a stated baseline, never in isolation.

The reference procedure is deliberately simple (correlation magnitude
against a fixed, pre-registered threshold) so that its own Honesty and
Competence rates are informative about the *difficulty* of each world,
not merely about the sophistication of whichever procedure is under test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np

from ccce.attribution.factors import ALL_FACTOR_NAMES, OBSERVATIONAL_FACTOR_NAMES
from ccce.attribution.worlds import ALL_WORLDS, AttributionSample

AttributionProcedure = Callable[[Sequence[AttributionSample]], Tuple[str, ...]]
"""An attribution procedure receives a batch of samples drawn from ONE
world (the world identity is NOT revealed to the procedure, only the
samples' diagnostic vectors and outcomes) and returns the tuple of
factor names it identifies as responsible, or an empty tuple if it
declines to attribute (the UNEXPLAINED/abstention case)."""


@dataclass
class KillerTwoResult:
    honesty_rate: float
    honesty_baseline: float
    honesty_n: int
    competence_rate: float
    competence_baseline: float
    competence_n: int
    competence_by_world: Dict[str, float]

    @property
    def honesty_exceeds_baseline(self) -> bool:
        return self.honesty_rate > self.honesty_baseline

    @property
    def competence_exceeds_baseline(self) -> bool:
        return self.competence_rate > self.competence_baseline

    @property
    def killer_two_passed(self) -> bool:
        """Both tests must pass; passing only one is a failure of Killer 2
        (main_v4.tex, Section 6.4: 'Passing only one of the two rates is
        explicitly treated as a failure of Killer 2, not a partial
        success.')."""
        return self.honesty_exceeds_baseline and self.competence_exceeds_baseline


def _factor_values(samples: Sequence[AttributionSample], factor_name: str) -> np.ndarray:
    if factor_name in samples[0].observational:
        return np.array([s.observational[factor_name] for s in samples])
    return np.array([s.intervenable[factor_name] for s in samples])


def threshold_attribution_procedure(
    samples: Sequence[AttributionSample], correlation_threshold: float = 0.3
) -> Tuple[str, ...]:
    """Reference ASSOCIATIONAL (Level C) attribution procedure: computes
    the Pearson correlation between each CATEGORY-A (observational-only)
    factor's value and the outcome across the batch, and attributes to
    every factor whose absolute correlation exceeds
    ``correlation_threshold``. Returns an empty tuple (abstains) if no
    factor exceeds the threshold.

    Deliberately restricted to ``OBSERVATIONAL_FACTOR_NAMES`` (Category A)
    ONLY, never ``INTERVENABLE_FACTOR_NAMES`` (Category B). This is not an
    arbitrary implementation choice: main_v4.tex Section 4.3 defines
    associational diagnosis as operating on "a vector of observational
    diagnostic signals... category A" specifically. A Category B factor's
    causal role is established only through an actual deliberate
    intervention (Level D, main_v4.tex Section 4.4) -- passively observing
    an intervenable factor's naturally occurring value and correlating it
    with the outcome is not a Level D intervention, even when that
    passive correlation happens to be large, and must not be treated as
    one. Including Category B factors here was an earlier implementation
    error, caught by ``test_threshold_procedure_fails_competence_on_world_c_by_design``:
    with Category B factors included, the procedure could "solve" World C
    (whose true cause, task_order, is a Category B factor) via bare
    correlation, which would silently erase the very Level C/Level D
    distinction the framework is built around. Restricting the search
    space to Category A alone is the fix, not a threshold adjustment.

    This procedure therefore cannot, by construction, detect World C's
    intervenable-only cause (no Category A signal correlates with it) or
    World E's counterfactual-only cause (the marginal outcome carries no
    signal at all). Its Competence rate on Worlds C and E is expected to
    be at chance level; this is a demonstration of exactly the gap Levels
    D and E are designed to close, not a defect in the evaluation.
    """
    if len(samples) < 3:
        return ()

    outcomes = np.array([s.outcome for s in samples])
    identified = []
    for factor_name in OBSERVATIONAL_FACTOR_NAMES:
        values = _factor_values(samples, factor_name)
        if np.std(values) < 1e-9 or np.std(outcomes) < 1e-9:
            continue
        corr = float(np.corrcoef(values, outcomes)[0, 1])
        if abs(corr) >= correlation_threshold:
            identified.append(factor_name)
    return tuple(identified)


def evaluate_honesty(
    procedure: AttributionProcedure,
    rng: np.random.Generator,
    n_batches: int = 50,
    batch_size: int = 30,
) -> Tuple[float, int]:
    """Honesty rate: fraction of World D batches for which the procedure
    correctly returns an empty tuple (abstains / UNEXPLAINED)."""
    world = ALL_WORLDS["D"]
    correct = 0
    for _ in range(n_batches):
        batch = [world.generate(rng) for _ in range(batch_size)]
        result = procedure(batch)
        if result == ():
            correct += 1
    return correct / n_batches, n_batches


def evaluate_competence(
    procedure: AttributionProcedure,
    rng: np.random.Generator,
    worlds: Sequence[str] = ("A", "B", "C"),
    n_batches: int = 50,
    batch_size: int = 30,
) -> Tuple[float, int, Dict[str, float]]:
    """Competence rate: fraction of Worlds A-C batches for which the
    procedure's identified factor set exactly matches (as a set) the
    world's declared true cause. Also returns the per-world breakdown,
    since Worlds A-C are deliberately of increasing difficulty and
    collapsing them into one rate would hide exactly the pattern the
    hierarchy is designed to reveal (main_v4.tex, Section 7.2)."""
    per_world_correct: Dict[str, int] = {w: 0 for w in worlds}
    per_world_total: Dict[str, int] = {w: 0 for w in worlds}

    for world_id in worlds:
        world = ALL_WORLDS[world_id]
        for _ in range(n_batches):
            batch = [world.generate(rng) for _ in range(batch_size)]
            result = set(procedure(batch))
            per_world_total[world_id] += 1
            if result == set(world.true_cause):
                per_world_correct[world_id] += 1

    per_world_rate = {w: per_world_correct[w] / per_world_total[w] for w in worlds}
    total_correct = sum(per_world_correct.values())
    total_n = sum(per_world_total.values())
    return total_correct / total_n, total_n, per_world_rate


def always_attribute_baseline(samples: Sequence[AttributionSample]) -> Tuple[str, ...]:
    """Naive baseline for the Honesty test: attributes to the single
    factor most correlated with the outcome, regardless of whether the
    correlation is meaningful. This baseline should score 0% Honesty by
    design (it never abstains), calibrating what "exceeding baseline"
    means for a real procedure."""
    if not samples:
        return ()
    outcomes = np.array([s.outcome for s in samples])
    best_factor, best_corr = None, 0.0
    for factor_name in ALL_FACTOR_NAMES:
        values = _factor_values(samples, factor_name)
        if np.std(values) < 1e-9 or np.std(outcomes) < 1e-9:
            continue
        corr = abs(float(np.corrcoef(values, outcomes)[0, 1]))
        if corr > best_corr:
            best_corr, best_factor = corr, factor_name
    return (best_factor,) if best_factor is not None else ()


def always_abstain_baseline(samples: Sequence[AttributionSample]) -> Tuple[str, ...]:
    """Naive baseline for the Competence test: never attributes to
    anything. This baseline should score 0% Competence by design (it
    never names the correct factor, even when one exists)."""
    return ()


def run_killer_experiment_two(
    procedure: AttributionProcedure,
    rng: np.random.Generator,
    n_batches: int = 50,
    batch_size: int = 30,
) -> KillerTwoResult:
    """Run the complete Killer Experiment 2 evaluation: Honesty rate vs.
    the always-attribute baseline, and Competence rate vs. the
    always-abstain baseline, for the given attribution procedure."""
    honesty_rate, honesty_n = evaluate_honesty(procedure, rng, n_batches, batch_size)
    honesty_baseline_rate, _ = evaluate_honesty(always_attribute_baseline, rng, n_batches, batch_size)

    competence_rate, competence_n, per_world = evaluate_competence(
        procedure, rng, ("A", "B", "C"), n_batches, batch_size
    )
    competence_baseline_rate, _, _ = evaluate_competence(
        always_abstain_baseline, rng, ("A", "B", "C"), n_batches, batch_size
    )

    return KillerTwoResult(
        honesty_rate=honesty_rate, honesty_baseline=honesty_baseline_rate, honesty_n=honesty_n,
        competence_rate=competence_rate, competence_baseline=competence_baseline_rate, competence_n=competence_n,
        competence_by_world=per_world,
    )
