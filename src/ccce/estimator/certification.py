"""
Identification / abstention state machine (Master Prompt V3, Section 32;
main_v2.tex Section 4.8).

States: CERTIFIED, VIOLATED, UNIDENTIFIABLE.
Never force a point estimate when the intervention is invalid, support is
insufficient, task semantics are altered, or the competence-contract
envelope is violated.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ccce.interventions.validity import InterventionMetadata, ValidityStatus


class CertificationState(str, Enum):
    CERTIFIED = "CERTIFIED"
    VIOLATED = "VIOLATED"
    UNIDENTIFIABLE = "UNIDENTIFIABLE"


@dataclass
class CertificationResult:
    state: CertificationState
    reason: str


def certify(
    intervention_meta: InterventionMetadata,
    envelope_satisfied: Optional[bool],
) -> CertificationResult:
    """Decide CERTIFIED / VIOLATED / UNIDENTIFIABLE for one CCCE query.

    `envelope_satisfied` is None when the envelope check is not
    applicable (e.g., we are only checking whether *estimation* is
    possible, prior to computing Q_i(pi_T, c) at all).
    """
    if intervention_meta.validity_status in (
        ValidityStatus.INVALID,
        ValidityStatus.TASK_ALTERING,
        ValidityStatus.UNIDENTIFIABLE,
    ):
        return CertificationResult(
            state=CertificationState.UNIDENTIFIABLE,
            reason=f"Intervention classified as {intervention_meta.validity_status.value}; "
            "abstaining rather than forcing an estimate (Section 32).",
        )

    if envelope_satisfied is False:
        return CertificationResult(
            state=CertificationState.VIOLATED,
            reason="Competence envelope E_i violated at Q_i(pi_T, c).",
        )

    return CertificationResult(
        state=CertificationState.CERTIFIED,
        reason=f"Intervention is {intervention_meta.validity_status.value} and in-support; "
        "estimate is reported.",
    )
