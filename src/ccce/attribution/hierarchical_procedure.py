# Author: Basil M. Alzboun

"""
ccce.attribution.hierarchical_procedure
==========================================

A hierarchical attribution procedure that correctly cascades through
Level C (associational) and Level D (interventional) diagnosis, as
opposed to a single flat correlational check.

Motivation
------------
`ccce.attribution.honesty_competence.threshold_attribution_procedure`
(the project's original reference procedure) is a pure Level C
associational check, restricted to Category A observational signals. It
was shown (D03_failure_modes) to correctly abstain on World C (its true
cause, task_order, is a Category B factor with no Category A
correlate) -- but "correctly abstains" is only half the story: main_v4.tex's
own hierarchy says that when Level C fails, Level D should be *attempted
next*, not simply given up on. A real diagnostic pipeline that stops at
Level C and reports UNEXPLAINED whenever no Category A correlate exists
is behaving too conservatively for factors that ARE actually testable at
Level D.

This module implements that cascade properly, with two genuine
methodological improvements over the flat reference procedure -- not by
loosening Level C's Category-A restriction (which would reintroduce the
exact leakage bug this project already found and fixed in
`honesty_competence.py`), but by:

    1. **Extending Level C to detect two-factor interactions**, still
       using ONLY Category A signals: pairwise products of Category A
       factor values are also correlated against the outcome. This is
       still a purely associational computation (no Category B data
       involved) and is what World B specifically requires: neither
       individual factor correlates with the outcome, but their PRODUCT
       does.
    2. **Adding a genuine Level D stage**, run only when Level C (step 1)
       finds nothing: each Category B factor's natural batch-level
       variation is discretized into terciles (low/mid/high), and
       `ccce.attribution.interventional.run_interventional_diagnosis` is
       run treating the terciles as three intervention conditions. This
       is a real, if simplified, interventional test -- it uses the same
       Kruskal-Wallis-omnibus-plus-Holm-corrected-pairwise-contrasts
       machinery this project already validated for the real task-order
       experiment (D04), rather than a bare correlation coefficient. A
       genuinely manipulated, pre-registered multi-condition experiment
       (as in D04's `run_intervention.py`) remains the gold-standard form
       of Level D evidence; the tercile-split analysis here is the best
       available Level D-style test from a single passively-collected
       batch, and is documented as such, not oversold as equivalent to
       an actual controlled experiment.

The returned attribution result distinguishes WHICH evidence level
supported each identified factor, exactly matching
`ccce.attribution.diagnostic_record.EvidenceLevel` -- an interaction term
identified at step 1 is tagged ASSOCIATIONAL; a factor identified at
step 2 is tagged INTERVENTIONAL.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Sequence, Tuple

import numpy as np

from ccce.attribution.diagnostic_record import EvidenceLevel
from ccce.attribution.factors import INTERVENABLE_FACTOR_NAMES, OBSERVATIONAL_FACTOR_NAMES
from ccce.attribution.interventional import InterventionCondition, run_interventional_diagnosis
from ccce.attribution.worlds import AttributionSample
from ccce.statistics.inference import holm_correction

CORRELATION_THRESHOLD = 0.3
LEVEL_D_ALPHA = 0.05
LEVEL_C_ALPHA = 0.05


@dataclass
class HierarchicalAttribution:
    factor_name: str
    evidence_level: EvidenceLevel
    detail: str  # e.g. "individual" | "interaction:factor_a*factor_b" | "tercile_split"


def _factor_values(samples: Sequence[AttributionSample], factor_name: str) -> np.ndarray:
    if factor_name in samples[0].observational:
        return np.array([s.observational[factor_name] for s in samples])
    return np.array([s.intervenable[factor_name] for s in samples])


def _level_c_individual_candidates(
    samples: Sequence[AttributionSample], outcomes: np.ndarray
) -> List[Tuple[str, float, float, str]]:
    """Returns (name, correlation, p_value, detail) candidates for every
    Category-A factor, WITHOUT applying any significance threshold --
    thresholding happens once, centrally, after all candidates (both
    individual and interaction terms) are pooled, so the multiple-testing
    correction below accounts for the true size of the search space."""
    from scipy import stats as _stats

    candidates = []
    for name in OBSERVATIONAL_FACTOR_NAMES:
        values = _factor_values(samples, name)
        if np.std(values) < 1e-9 or np.std(outcomes) < 1e-9:
            continue
        corr, p_value = _stats.pearsonr(values, outcomes)
        candidates.append((name, float(corr), float(p_value), "individual"))
    return candidates


def _level_c_interaction_candidates(
    samples: Sequence[AttributionSample], outcomes: np.ndarray
) -> List[Tuple[str, float, float, str]]:
    """Returns (combined_name, correlation, p_value, detail) candidates
    for every pairwise PRODUCT of Category-A factors, still purely
    associational (no Category B data used). Each candidate's
    `combined_name` encodes both constituent factor names, resolved back
    to two separate attributions only after the significance filter is
    applied (see `_resolve_interaction_candidate`)."""
    from scipy import stats as _stats

    candidates = []
    for name_a, name_b in combinations(OBSERVATIONAL_FACTOR_NAMES, 2):
        values_a = _factor_values(samples, name_a)
        values_b = _factor_values(samples, name_b)
        product = values_a * values_b
        if np.std(product) < 1e-9 or np.std(outcomes) < 1e-9:
            continue
        corr, p_value = _stats.pearsonr(product, outcomes)
        candidates.append((f"{name_a}*{name_b}", float(corr), float(p_value), f"interaction:{name_a}*{name_b}"))
    return candidates


def _level_c_combined(samples: Sequence[AttributionSample], outcomes: np.ndarray) -> List[HierarchicalAttribution]:
    """Step 1: a genuine TWO-STAGE cascade within Level C itself, each
    stage independently Holm-corrected -- not a single family of all 21
    candidates pooled together.

    Stage 1a tests the 6 individual Category-A correlations, Holm-
    corrected across those 6 only. If any survive, they are returned
    immediately and Stage 1b is never run.

    Stage 1b (run only if Stage 1a found nothing) tests the 15 pairwise-
    product interaction candidates, Holm-corrected across those 15 only.

    This two-stage design is a direct, load-bearing fix for TWO real
    bugs found in sequence during this module's development:

      Bug 1 (uncorrected multiple testing): an initial version applied a
      fixed |r|>=0.3 threshold independently to each of 21 candidates
      with no correction at all, which let a strong true single-factor
      effect (World A) be reported ALONGSIDE spurious co-attributions
      from chance-inflated interaction correlations, failing the exact-
      match Competence test (World A accuracy: 0.65 -> 0.00).

      Bug 2 (over-correction from a single pooled family): fixing Bug 1
      by pooling ALL 21 candidates into one Holm-corrected family
      improved World C (0.00 -> 0.775, via the separate Level D stage)
      and World D (0.66 -> 0.70), but did NOT fix World A -- pooling
      still required surviving a family-wise correction sized for 21
      tests (Holm's strictest threshold, alpha/21 ~ 0.0024), which is
      needlessly conservative for a hypothesis (a single individual
      correlation) that could have been tested in a much smaller,
      appropriately-sized family of 6.

    The two-stage cascade implemented here is the methodologically
    correct resolution: each stage's correction is sized to the
    hypothesis space ACTUALLY being searched at that stage, and later,
    more complex stages are only reached (and only then contribute to the
    multiple-testing burden) when earlier, simpler stages have already
    failed to explain the outcome.
    """
    individual = _level_c_individual_candidates(samples, outcomes)
    if individual:
        p_values = [c[2] for c in individual]
        reject_flags = holm_correction(p_values, alpha=LEVEL_C_ALPHA)
        found = [
            HierarchicalAttribution(name, EvidenceLevel.ASSOCIATIONAL, detail)
            for (name, corr, p_value, detail), rejected in zip(individual, reject_flags)
            if rejected
        ]
        if found:
            return found

    interactions = _level_c_interaction_candidates(samples, outcomes)
    if not interactions:
        return []
    p_values = [c[2] for c in interactions]
    reject_flags = holm_correction(p_values, alpha=LEVEL_C_ALPHA)

    found = []
    for (name, corr, p_value, detail), rejected in zip(interactions, reject_flags):
        if not rejected:
            continue
        name_a, name_b = detail.split(":", 1)[1].split("*")
        found.append(HierarchicalAttribution(name_a, EvidenceLevel.ASSOCIATIONAL, detail))
        found.append(HierarchicalAttribution(name_b, EvidenceLevel.ASSOCIATIONAL, detail))
    return found


def _level_d_tercile_split(samples: Sequence[AttributionSample], outcomes: np.ndarray) -> List[HierarchicalAttribution]:
    """Step 2: for each Category B factor, split the batch into terciles
    by that factor's naturally occurring value and run
    `run_interventional_diagnosis` treating the three terciles as
    intervention conditions. A factor is attributed at the INTERVENTIONAL
    evidence level if this test finds any Holm-significant pairwise
    contrast between terciles.

    Documented limitation: this is a tercile split of PASSIVELY OBSERVED
    variation, not a genuinely pre-registered, actively manipulated
    experiment (contrast with D04's `run_intervention.py`, which trains
    real policies under actually-assigned, pre-specified task orders).
    It is the best available Level D-style signal extractable from a
    single batch and is reported with that caveat attached by the
    caller, never presented as equivalent in evidentiary strength to a
    real controlled experiment.
    """
    found = []
    if len(samples) < 9:  # need at least 3 per tercile to run the omnibus test at all
        return found

    for name in INTERVENABLE_FACTOR_NAMES:
        values = _factor_values(samples, name)
        if np.std(values) < 1e-9:
            continue
        terciles = np.quantile(values, [1 / 3, 2 / 3])
        low_mask = values <= terciles[0]
        mid_mask = (values > terciles[0]) & (values <= terciles[1])
        high_mask = values > terciles[1]

        conditions = []
        for label, mask in (("low", low_mask), ("mid", mid_mask), ("high", high_mask)):
            vals = outcomes[mask].tolist()
            if len(vals) >= 2:
                conditions.append(InterventionCondition(f"{name}_{label}", label, vals))
        if len(conditions) < 2:
            continue

        try:
            result = run_interventional_diagnosis(name, conditions, alpha=LEVEL_D_ALPHA)
        except ValueError:
            continue

        if result.any_holm_significant_contrast:
            found.append(HierarchicalAttribution(name, EvidenceLevel.INTERVENTIONAL, "tercile_split"))
    return found


def hierarchical_attribution_procedure(samples: Sequence[AttributionSample]) -> Tuple[str, ...]:
    """The full Level C -> Level D cascade. Returns the tuple of factor
    names identified (matching the `AttributionProcedure` interface in
    `ccce.attribution.honesty_competence`, so this procedure is a drop-in
    replacement for `threshold_attribution_procedure` in
    `evaluate_honesty`/`evaluate_competence`), or an empty tuple if
    nothing is found at either level (UNEXPLAINED).
    """
    if len(samples) < 3:
        return ()
    outcomes = np.array([s.outcome for s in samples])

    level_c_results = _level_c_combined(samples, outcomes)
    if level_c_results:
        return tuple(sorted({r.factor_name for r in level_c_results}))

    level_d_results = _level_d_tercile_split(samples, outcomes)
    if level_d_results:
        return tuple(sorted({r.factor_name for r in level_d_results}))

    return ()


def hierarchical_attribution_with_evidence(
    samples: Sequence[AttributionSample],
) -> List[HierarchicalAttribution]:
    """Same cascade as `hierarchical_attribution_procedure`, but returns
    the full evidence-level-tagged results rather than collapsing to a
    bare tuple of names -- for callers that need to build a
    `ccce.attribution.diagnostic_record.DiagnosticRecord` with properly
    tagged `Attribution` entries."""
    if len(samples) < 3:
        return []
    outcomes = np.array([s.outcome for s in samples])

    level_c_results = _level_c_combined(samples, outcomes)
    if level_c_results:
        return level_c_results

    return _level_d_tercile_split(samples, outcomes)
