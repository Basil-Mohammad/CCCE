"""
Paired CCCE estimator (Master Prompt V3, Section 22; main_v2.tex Section 4.7).

    Delta_r = Y_i(L_T, xi_r) - Y_i(L_C, xi_r)
    CCCE_hat = (1/N) sum_r Delta_r

This module calls only `ccce.scm.analytical`'s *sampling* methods
(`sample_theta_trajectory`, `sample_outcome`) -- never the oracle's exact
closed form. It is therefore a genuinely independent estimate of the same
quantity the oracle computes exactly, which is what makes the ground-truth
recovery experiment (E01) a real test rather than a tautology.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ccce.interventions.validity import InterventionMetadata, only_valid_or_borderline_may_certify
from ccce.scm.analytical import AnalyticalSCM, LearningProtocolParams
from ccce.estimator.certification import CertificationResult, CertificationState, certify
from ccce.statistics.inference import EstimateWithCI, paired_mean_ci


@dataclass
class CCCEEstimate:
    competence_id: str
    c: float
    q_target: float
    q_control: float
    ccce: EstimateWithCI
    identification: CertificationResult


def estimate_ccce_paired(
    scm: AnalyticalSCM,
    competence_id: str,
    c: float,
    theta_prev_mean: np.ndarray,
    protocol_T: LearningProtocolParams,
    protocol_C: LearningProtocolParams,
    intervention_meta: InterventionMetadata,
    n_replicates: int,
    training_rng: np.random.Generator,
    evaluation_rng: np.random.Generator,
    envelope_check_fn=None,
) -> CCCEEstimate:
    """Paired Monte Carlo estimate of CCCE_i(c), with common evaluation
    randomness across the T and C arms (Section 59) and independent
    *training* randomness draws for theta (each replicate re-draws the
    stochastic parameter-update noise, as it should -- training noise is
    part of what the paired evaluation-noise design controls for on the
    *evaluation* side, not the training side)."""

    deltas: List[float] = []
    q_t_samples: List[float] = []
    q_c_samples: List[float] = []

    for r in range(n_replicates):
        # Independent training-noise draw per replicate (Section 59: training_rng)
        theta_T = scm.sample_theta_trajectory([protocol_T], theta_prev_mean, training_rng)[-1]
        theta_C = scm.sample_theta_trajectory([protocol_C], theta_prev_mean, training_rng)[-1]

        # Common evaluation randomness across T/C arms for this replicate
        eval_state = evaluation_rng.bit_generator.state
        y_t = scm.sample_outcome(competence_id, theta_T, c, evaluation_rng)
        evaluation_rng.bit_generator.state = eval_state  # rewind -> shared eval noise
        y_c = scm.sample_outcome(competence_id, theta_C, c, evaluation_rng)

        q_t_samples.append(y_t)
        q_c_samples.append(y_c)
        deltas.append(y_t - y_c)

    ccce_ci = paired_mean_ci(deltas, method="t")
    q_target = float(np.mean(q_t_samples))
    q_control = float(np.mean(q_c_samples))

    envelope_ok = envelope_check_fn(q_target) if envelope_check_fn is not None else None
    if not only_valid_or_borderline_may_certify(intervention_meta):
        identification = certify(intervention_meta, envelope_ok)
    else:
        identification = certify(intervention_meta, envelope_ok)

    return CCCEEstimate(
        competence_id=competence_id,
        c=c,
        q_target=q_target,
        q_control=q_control,
        ccce=ccce_ci,
        identification=identification,
    )
