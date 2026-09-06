"""
Intervention validity classification (Master Prompt V3, Section 18;
main_v2.tex Section 3.4).

Each candidate intervention is classified as one of:
    VALID | BORDERLINE | TASK_ALTERING | INVALID | UNIDENTIFIABLE

Only VALID interventions with sufficient counterfactual support may
contribute to a CERTIFIED CCCE estimate (Section 32).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence


class ValidityStatus(str, Enum):
    VALID = "VALID"
    BORDERLINE = "BORDERLINE"
    TASK_ALTERING = "TASK_ALTERING"
    INVALID = "INVALID"
    UNIDENTIFIABLE = "UNIDENTIFIABLE"


@dataclass
class InterventionMetadata:
    """Exact schema required by Section 18."""

    intervention_id: str
    causal_variable: str
    value: float
    validity_status: ValidityStatus
    feasibility: bool
    contract_preserving: bool
    safety_valid: bool
    task_preserving: bool
    support_available: bool


def classify_intervention(
    intervention_id: str,
    causal_variable: str,
    value: float,
    is_designated_causal_parent: bool,
    feasible: bool,
    contract_preserving: bool,
    safety_valid: bool,
    task_preserving: bool,
    valid_support: Sequence[float],
    borderline_margin: float = 0.15,
) -> InterventionMetadata:
    """Apply the classification rule.

    Order of checks matters and mirrors main_v2.tex Table 15:
      1. Not acting on a designated SCM causal parent -> INVALID (never
         "UNIDENTIFIABLE" for this reason -- that label is reserved for
         *insufficient support*, not for a categorically wrong target).
      2. Support unavailable (outside the pre-registered grid) -> UNIDENTIFIABLE.
      3. Not task-preserving -> TASK_ALTERING.
      4. Not feasible, not contract-preserving, or not safety-valid -> INVALID.
      5. Otherwise VALID, with BORDERLINE if `value` sits within
         `borderline_margin` of the edge of `valid_support`'s range.
    """
    support_available = any(abs(value - s) < 1e-9 for s in valid_support)

    if not is_designated_causal_parent:
        status = ValidityStatus.INVALID
    elif not support_available:
        status = ValidityStatus.UNIDENTIFIABLE
    elif not task_preserving:
        status = ValidityStatus.TASK_ALTERING
    elif not (feasible and contract_preserving and safety_valid):
        status = ValidityStatus.INVALID
    else:
        if valid_support:
            lo, hi = min(valid_support), max(valid_support)
            span = max(hi - lo, 1e-9)
            dist_to_edge = min(abs(value - lo), abs(value - hi)) / span
            status = ValidityStatus.BORDERLINE if dist_to_edge < borderline_margin else ValidityStatus.VALID
        else:
            status = ValidityStatus.VALID

    return InterventionMetadata(
        intervention_id=intervention_id,
        causal_variable=causal_variable,
        value=value,
        validity_status=status,
        feasibility=feasible,
        contract_preserving=contract_preserving,
        safety_valid=safety_valid,
        task_preserving=task_preserving,
        support_available=support_available,
    )


def only_valid_or_borderline_may_certify(meta: InterventionMetadata) -> bool:
    """Section 18: 'Only VALID interventions may contribute to certified
    CCCE estimates.' We are slightly more permissive by design choice
    (documented here, not silently): BORDERLINE interventions may
    contribute but are always flagged in output, never silently merged
    with VALID. TASK_ALTERING, INVALID, and UNIDENTIFIABLE never certify.
    """
    return meta.validity_status in (ValidityStatus.VALID, ValidityStatus.BORDERLINE)
