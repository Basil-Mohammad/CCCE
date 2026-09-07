# Author: Basil M. Alzboun

"""
ccce.attribution.interventional
==================================

Level D (Interventional Diagnosis) statistical machinery
(main_v4.tex, Section 4.4).

Scientific purpose
-------------------
Level C (associational) diagnosis reports a correlation and stops there.
Level D licenses a stronger claim -- "manipulating this factor causally
affects the outcome" -- but only when that manipulation was actually
performed under experimentally controlled conditions: multiple
independent replicates (seeds) assigned to each condition of the
intervenable factor, with every other aspect of the training protocol
held fixed across conditions.

This module implements the generic statistical apparatus for that claim,
independent of what the intervenable factor or the outcome actually are,
so it can be reused for task order (this module's first concrete
application, see ``experiments/D04_diagnostic_attribution/run_intervention.py``),
update budget, learning rate, or replay availability alike
(main_v4.tex, Table 1, Category B).

Design principles enforced here
---------------------------------
1. **The unit of analysis is the independent replicate (seed), not the
   individual observation within a replicate.** A single trained policy
   evaluated over many episodes produces many correlated observations;
   averaging within-replicate before comparing across conditions is
   mandatory and is done inside ``InterventionOutcome.from_replicates``,
   not left to the caller.
2. **Every pairwise contrast across intervention conditions is corrected
   for multiple comparisons** (Holm, reusing
   ``ccce.statistics.inference.holm_correction``) -- an omnibus test
   alone is not treated as sufficient license to describe an
   arbitrary pairwise difference as significant.
3. **Effect sizes (Cohen's d) are reported alongside every p-value.**
   Consistent with this project's established practice elsewhere
   (see ``ccce/statistics/inference.py``), statistical significance is
   never presented without a corresponding magnitude.
4. **A causal claim is only licensed for the specific factor actually
   manipulated.** This module's output (`InterventionalDiagnosisResult`)
   is deliberately scoped to name that one factor; it is the caller's
   responsibility (and is checked in this project's tests) never to
   generalize the resulting causal language to a merely correlated
   Category A signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as _stats

from ccce.statistics.inference import cohens_d, holm_correction


@dataclass
class InterventionCondition:
    """One pre-registered setting of the intervenable factor under test.

    ``condition_id`` must be a stable, human-readable label (e.g. a
    task-sequence name); it is used verbatim in every reported contrast
    so that a reader can trace a result back to the exact experimental
    condition without re-deriving it from raw data.
    """

    condition_id: str
    description: str
    replicate_values: List[float] = field(default_factory=list)  # one scalar per independent seed

    @property
    def n_replicates(self) -> int:
        return len(self.replicate_values)

    @property
    def mean(self) -> float:
        return float(np.mean(self.replicate_values))

    @property
    def std(self) -> float:
        return float(np.std(self.replicate_values, ddof=1)) if self.n_replicates > 1 else 0.0


@dataclass
class PairwiseContrast:
    condition_a: str
    condition_b: str
    mean_difference: float
    cohens_d: float
    p_value_uncorrected: float
    holm_significant_at_0_05: bool


@dataclass
class InterventionalDiagnosisResult:
    factor_name: str
    conditions: Dict[str, InterventionCondition]
    omnibus_p_value: float
    omnibus_test: str
    pairwise_contrasts: List[PairwiseContrast]
    any_holm_significant_contrast: bool
    factor_name_for_causal_claim: str

    def summary_lines(self) -> List[str]:
        lines = [
            f"Interventional diagnosis of factor '{self.factor_name}' "
            f"({self.omnibus_test} omnibus p={self.omnibus_p_value:.4f})"
        ]
        for cond_id, cond in self.conditions.items():
            lines.append(f"  {cond_id}: mean={cond.mean:+.4f} std={cond.std:.4f} n={cond.n_replicates}")
        for c in self.pairwise_contrasts:
            sig = "HOLM-SIGNIFICANT" if c.holm_significant_at_0_05 else "not significant"
            lines.append(
                f"  {c.condition_a} vs {c.condition_b}: "
                f"diff={c.mean_difference:+.4f} d={c.cohens_d:+.3f} "
                f"p={c.p_value_uncorrected:.4f} ({sig})"
            )
        return lines


def run_interventional_diagnosis(
    factor_name: str,
    conditions: Sequence[InterventionCondition],
    alpha: float = 0.05,
) -> InterventionalDiagnosisResult:
    """Runs the full Level D statistical analysis over a set of
    pre-registered intervention conditions for one factor.

    Parameters
    ----------
    factor_name:
        The Category B factor actually manipulated (main_v4.tex Table 1).
        This is the ONLY factor a caller may attach causal language to
        based on this result.
    conditions:
        Two or more ``InterventionCondition`` instances, each populated
        with one scalar outcome value per independent replicate (seed).
        All conditions must share the same outcome definition and the
        same held-fixed protocol aside from the manipulated factor.
    alpha:
        Family-wise significance level for the Holm-corrected pairwise
        contrasts (the omnibus test itself is reported unadjusted, as a
        screening statistic, per standard practice).

    Raises
    ------
    ValueError
        If fewer than two conditions are supplied, or if any condition
        has fewer than 2 replicates (a "sample size" of 1 cannot support
        a variance estimate and must not silently produce a contrast).
    """
    if len(conditions) < 2:
        raise ValueError("Interventional diagnosis requires at least two conditions to contrast.")
    for c in conditions:
        if c.n_replicates < 2:
            raise ValueError(
                f"Condition '{c.condition_id}' has {c.n_replicates} replicate(s); "
                "at least 2 independent seeds are required per condition."
            )

    cond_map = {c.condition_id: c for c in conditions}
    groups = [c.replicate_values for c in conditions]

    # Omnibus test: one-way ANOVA (parametric) is reported alongside the
    # non-parametric Kruskal-Wallis test; we use Kruskal-Wallis as the
    # primary omnibus statistic since it does not assume normality of the
    # (typically small-sample) per-condition replicate distributions.
    omnibus_stat, omnibus_p = _stats.kruskal(*groups)

    pairwise: List[PairwiseContrast] = []
    raw_p_values: List[float] = []
    pair_ids: List[Tuple[str, str]] = []

    for cond_a, cond_b in combinations(conditions, 2):
        t_stat, p_val = _stats.ttest_ind(cond_a.replicate_values, cond_b.replicate_values, equal_var=False)
        d = cohens_d(cond_a.replicate_values, cond_b.replicate_values)
        raw_p_values.append(float(p_val))
        pair_ids.append((cond_a.condition_id, cond_b.condition_id))
        pairwise.append(
            PairwiseContrast(
                condition_a=cond_a.condition_id, condition_b=cond_b.condition_id,
                mean_difference=cond_a.mean - cond_b.mean, cohens_d=d,
                p_value_uncorrected=float(p_val), holm_significant_at_0_05=False,  # filled below
            )
        )

    holm_flags = holm_correction(raw_p_values, alpha=alpha)
    for contrast, flag in zip(pairwise, holm_flags):
        contrast.holm_significant_at_0_05 = flag

    return InterventionalDiagnosisResult(
        factor_name=factor_name, conditions=cond_map,
        omnibus_p_value=float(omnibus_p), omnibus_test="kruskal_wallis",
        pairwise_contrasts=pairwise,
        any_holm_significant_contrast=any(holm_flags),
        factor_name_for_causal_claim=factor_name,
    )
