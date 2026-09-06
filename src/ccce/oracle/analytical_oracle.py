"""
Level 1 ground-truth oracle (Master Prompt V3, Section 23).

CRITICAL ARCHITECTURAL RULE (Reviewer-2 attack #11 in main_v2.tex):
this module MUST NOT import from `ccce.estimator`. It computes CCCE^GT
purely from the closed-form expressions in `ccce.scm.analytical`. This is
enforced both by convention (see the import list below -- there is no
import of `ccce.estimator`) and by a static check in
`tests/test_no_oracle_estimator_coupling.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ccce.scm.analytical import AnalyticalSCM, LearningProtocolParams

# NOTE: no "from ccce.estimator import ..." anywhere in this file. This is
# intentional and load-bearing; see module docstring.


@dataclass
class OracleResult:
    competence_id: str
    c: float
    ccce_gt: float
    rc_gt: float


def compute_ground_truth(
    scm: AnalyticalSCM,
    competence_id: str,
    c: float,
    history_means: Sequence,
    protocol_T: LearningProtocolParams,
    protocol_C: LearningProtocolParams,
) -> OracleResult:
    """Exact CCCE^GT and RC^GT for one (competence, c, protocols) query."""
    ccce_gt = scm.exact_ccce(competence_id, c, history_means, protocol_T, protocol_C)
    rc_gt = scm.exact_rc(competence_id, history_means, protocol_T)
    return OracleResult(competence_id=competence_id, c=c, ccce_gt=ccce_gt, rc_gt=rc_gt)
