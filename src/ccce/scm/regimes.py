# Author: Basil M. Alzboun

"""
Constructs the analytical-SCM configurations for the required regimes
(Master Prompt V3, Section 24 and Section 13 of main_v2.tex):

    beneficial   : CCCE^GT(c*) > +delta
    neutral      : |CCCE^GT(c*)| <= delta
    harmful      : CCCE^GT(c*) < -delta
    unidentifiable: c* outside the competence's valid intervention support

plus the hidden-vulnerability case (Section 25 / EXP E03):

    RC_i ~ 0 at c=0, while CCCE_i(c*) < -delta at a valid c* != 0

plus negative and positive controls (Sections 33-34).

All numeric constants here are pre-specified (not tuned post hoc) and are
written once to `configs/experiments/analytical_regimes.yaml` at
generation time so that "the same values used to build the benchmark" and
"the values that were pre-registered" are provably identical (Section 57).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ccce.scm.analytical import AnalyticalSCM, CompetenceLinearForm, LearningProtocolParams

DELTA = 0.05  # pre-registered practical-significance tolerance (Section 37 / 6.6 of main_v2.tex)
INTERVENTION_GRID: Tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
DIM = 4  # dimensionality of the analytical parameter vector theta


@dataclass
class RegimeCase:
    case_id: str
    ground_truth_category: str  # beneficial | neutral | harmful | unidentifiable | hidden_vulnerability | negative_control | positive_control
    scm: AnalyticalSCM
    protocol_T: LearningProtocolParams
    protocol_C: LearningProtocolParams
    competence_id: str
    query_c: float
    expected_sign: str  # "+", "-", "0", or "UNIDENTIFIABLE"


def _base_scm(b_i: np.ndarray, h_i: np.ndarray, support=INTERVENTION_GRID) -> AnalyticalSCM:
    comp = CompetenceLinearForm(
        competence_id="K1",
        a_i=0.0,
        b_i=b_i,
        h_i=h_i,
        sigma_y=0.1,
        valid_intervention_support=support,
    )
    return AnalyticalSCM(
        dim=DIM,
        sigma_u=0.02,
        theta0_mean=np.zeros(DIM),
        competences={"K1": comp},
    )


def build_all_regime_cases() -> Dict[str, RegimeCase]:
    cases: Dict[str, RegimeCase] = {}

    # ---- Beneficial: target pulls strongly in the +b_i direction --------
    b = np.array([1.0, 0.0, 0.0, 0.0])
    h = np.array([0.0, 0.0, 0.0, 0.0])
    scm = _base_scm(b, h)
    proto_T = LearningProtocolParams("L_T_beneficial", target=np.array([1.0, 0, 0, 0]), gain=0.9, budget=5.0)
    proto_C = LearningProtocolParams("L_C_beneficial", target=np.array([-1.0, 0, 0, 0]), gain=0.9, budget=5.0)
    cases["beneficial"] = RegimeCase("beneficial", "beneficial", scm, proto_T, proto_C, "K1", 0.0, "+")

    # ---- Neutral: T and C protocols pull toward (nearly) the same target
    b = np.array([1.0, 0.0, 0.0, 0.0])
    h = np.array([0.0, 0.0, 0.0, 0.0])
    scm = _base_scm(b, h)
    proto_T = LearningProtocolParams("L_T_neutral", target=np.array([0.5, 0, 0, 0]), gain=0.9, budget=5.0)
    proto_C = LearningProtocolParams("L_C_neutral", target=np.array([0.5, 0, 0, 0]), gain=0.9, budget=5.0)
    cases["neutral"] = RegimeCase("neutral", "neutral", scm, proto_T, proto_C, "K1", 0.0, "0")

    # ---- Harmful: target pulls strongly in the -b_i direction ------------
    b = np.array([1.0, 0.0, 0.0, 0.0])
    h = np.array([0.0, 0.0, 0.0, 0.0])
    scm = _base_scm(b, h)
    proto_T = LearningProtocolParams("L_T_harmful", target=np.array([-1.0, 0, 0, 0]), gain=0.9, budget=5.0)
    proto_C = LearningProtocolParams("L_C_harmful", target=np.array([1.0, 0, 0, 0]), gain=0.9, budget=5.0)
    cases["harmful"] = RegimeCase("harmful", "harmful", scm, proto_T, proto_C, "K1", 0.0, "-")

    # ---- Unidentifiable: query c* outside the declared support -----------
    b = np.array([1.0, 0.0, 0.0, 0.0])
    h = np.array([0.5, 0.0, 0.0, 0.0])
    scm = _base_scm(b, h, support=(0.0,))  # only c=0 is in-support
    proto_T = LearningProtocolParams("L_T_unident", target=np.array([1.0, 0, 0, 0]), gain=0.9, budget=5.0)
    proto_C = LearningProtocolParams("L_C_unident", target=np.array([-1.0, 0, 0, 0]), gain=0.9, budget=5.0)
    cases["unidentifiable"] = RegimeCase(
        "unidentifiable", "unidentifiable", scm, proto_T, proto_C, "K1", 3.0, "UNIDENTIFIABLE"
    )

    # ---- Hidden vulnerability: b orthogonal-ish to Delta, h aligned ------
    # Delta_theta (T - C) will point mainly along dim 1 (index 1), so make
    # b_i sensitive to dim 0 only (nominal blind to the T/C difference) and
    # h_i sensitive to dim 1 (revealed only under intervention).
    b = np.array([1.0, 0.0, 0.0, 0.0])
    h = np.array([0.0, 3.0, 0.0, 0.0])
    scm = _base_scm(b, h)
    proto_T = LearningProtocolParams("L_T_hidden", target=np.array([0.0, 1.0, 0, 0]), gain=0.9, budget=5.0)
    proto_C = LearningProtocolParams("L_C_hidden", target=np.array([0.0, -1.0, 0, 0]), gain=0.9, budget=5.0)
    cases["hidden_vulnerability"] = RegimeCase(
        "hidden_vulnerability", "hidden_vulnerability", scm, proto_T, proto_C, "K1", -2.0, "-"
    )

    # ---- Negative control: identical protocols -> CCCE^GT must be exactly 0
    b = np.array([1.0, 0.5, 0.0, 0.0])
    h = np.array([0.2, 0.0, 0.0, 0.0])
    scm = _base_scm(b, h)
    proto_T = LearningProtocolParams("L_T_negctrl", target=np.array([0.7, 0.3, 0, 0]), gain=0.8, budget=4.0)
    proto_C = LearningProtocolParams("L_C_negctrl", target=np.array([0.7, 0.3, 0, 0]), gain=0.8, budget=4.0)
    cases["negative_control"] = RegimeCase(
        "negative_control", "negative_control", scm, proto_T, proto_C, "K1", 1.0, "0"
    )

    # ---- Positive control: large, unambiguous, known-sign effect ---------
    b = np.array([2.0, 0.0, 0.0, 0.0])
    h = np.array([0.0, 0.0, 0.0, 0.0])
    scm = _base_scm(b, h)
    proto_T = LearningProtocolParams("L_T_posctrl", target=np.array([5.0, 0, 0, 0]), gain=0.95, budget=10.0)
    proto_C = LearningProtocolParams("L_C_posctrl", target=np.array([-5.0, 0, 0, 0]), gain=0.95, budget=10.0)
    cases["positive_control"] = RegimeCase(
        "positive_control", "positive_control", scm, proto_T, proto_C, "K1", 0.0, "+"
    )

    return cases
