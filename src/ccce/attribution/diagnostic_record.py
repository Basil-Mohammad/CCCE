"""
ccce.attribution.diagnostic_record
====================================

Author: Basil M. Alzboun

The formal diagnostic record D_i (main_v4.tex, Section 4.6.3, Equation 14):

    D_i = (Status, Profile, Attributions, Counterfactual, Confidence)

This module implements the record as a strongly typed, immutable
dataclass and enforces, in ``build_diagnostic_record``, the single most
important structural constraint the specification imposes on it: a
``Confidence`` label may only be assigned after the relevant Honesty
and Competence criteria (``ccce.attribution.honesty_competence``) have
actually been evaluated for the case category the record belongs to. A
``Confidence`` value is never derived from a single test statistic or
from the Attributions list alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class CompetenceChangeStatus(str, Enum):
    """Level A/B taxonomy (main_v4.tex, Section 3.3)."""

    NO_CHANGE = "No Change"
    DRIFT = "Drift"
    DEGRADATION = "Degradation"


class EvidenceLevel(str, Enum):
    """The two evidence levels an Attributions entry may carry. Never
    merged into a single field (main_v4.tex, Section 4.6.3)."""

    ASSOCIATIONAL = "Associational"
    INTERVENTIONAL = "Interventional"


class DiagnosticConfidence(str, Enum):
    """Final confidence label. Assigned only after both Honesty and
    Competence criteria are evaluated for the relevant case category
    (main_v4.tex, Section 4.6.2)."""

    EXPLAINED = "EXPLAINED"
    PARTIALLY_EXPLAINED = "PARTIALLY_EXPLAINED"
    UNEXPLAINED = "UNEXPLAINED"
    UNIDENTIFIABLE = "UNIDENTIFIABLE"
    NO_CHANGE_DETECTED = "NO_CHANGE_DETECTED"


@dataclass(frozen=True)
class CharacterizationProfile:
    """Level B output: where and how a Competence Change Event manifests."""

    affected_outcome_dimension: str  # e.g. one component of Phi_i, or "aggregate"
    evaluation_condition: str  # e.g. "nominal" or the specific intervention value
    checkpoint: str  # training-history location of the observation


@dataclass(frozen=True)
class Attribution:
    """One (factor, evidence-level) pair. A factor may appear at most
    once per evidence level within a single record's Attributions list;
    the same factor may legitimately appear at both Associational and
    Interventional evidence levels if it was tested at both."""

    factor_name: str
    evidence_level: EvidenceLevel
    effect_estimate: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class CounterfactualResult:
    """Level E output for one Competence Change Event, if computed."""

    cross_competence_effect: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    identification_status: str  # "CERTIFIED" | "VIOLATED" | "UNIDENTIFIABLE"


@dataclass(frozen=True)
class DiagnosticRecord:
    """The complete diagnostic record D_i."""

    event_id: str
    status: CompetenceChangeStatus
    profile: Optional[CharacterizationProfile]
    attributions: Tuple[Attribution, ...]
    counterfactual: Optional[CounterfactualResult]
    confidence: DiagnosticConfidence


def build_diagnostic_record(
    event_id: str,
    status: CompetenceChangeStatus,
    profile: Optional[CharacterizationProfile],
    attributions: List[Attribution],
    counterfactual: Optional[CounterfactualResult],
    honesty_criterion_evaluated: bool,
    competence_criterion_evaluated: bool,
    honesty_passed: Optional[bool],
    competence_passed: Optional[bool],
) -> DiagnosticRecord:
    """Construct a DiagnosticRecord, enforcing that Confidence is derived
    only from the (evaluated) Honesty and Competence outcomes, never
    assigned directly by a caller.

    Rules (main_v4.tex, Section 4.6.2-4.6.3):
      - If status is NO_CHANGE: Confidence is always NO_CHANGE_DETECTED,
        regardless of any attribution attempt (there is nothing to explain).
      - If no attribution was attempted because no valid intervention or
        counterfactual support exists: Confidence is UNIDENTIFIABLE.
      - Otherwise, both honesty_criterion_evaluated and
        competence_criterion_evaluated must be True (the caller must have
        actually run both checks) before EXPLAINED, PARTIALLY_EXPLAINED,
        or UNEXPLAINED may be assigned.
    """
    if status == CompetenceChangeStatus.NO_CHANGE:
        confidence = DiagnosticConfidence.NO_CHANGE_DETECTED
        return DiagnosticRecord(
            event_id=event_id, status=status, profile=profile,
            attributions=tuple(attributions), counterfactual=counterfactual,
            confidence=confidence,
        )

    if counterfactual is not None and counterfactual.identification_status == "UNIDENTIFIABLE" and not attributions:
        confidence = DiagnosticConfidence.UNIDENTIFIABLE
        return DiagnosticRecord(
            event_id=event_id, status=status, profile=profile,
            attributions=tuple(attributions), counterfactual=counterfactual,
            confidence=confidence,
        )

    if not (honesty_criterion_evaluated and competence_criterion_evaluated):
        raise ValueError(
            "Cannot assign a Confidence label of EXPLAINED/PARTIALLY_EXPLAINED/"
            "UNEXPLAINED without BOTH the Honesty and Competence criteria having "
            "been evaluated for this case (main_v4.tex, Section 4.6.2). This is "
            "enforced structurally, not left to caller discretion, to prevent a "
            "Confidence label being assigned on the strength of only one criterion."
        )

    if not attributions:
        confidence = DiagnosticConfidence.UNEXPLAINED
    elif honesty_passed and competence_passed:
        confidence = DiagnosticConfidence.EXPLAINED
    elif competence_passed and not honesty_passed:
        # Competence identified a factor, but the honesty check for this
        # case category failed elsewhere in the batch: report partial
        # confidence rather than full EXPLAINED, since a framework that
        # sometimes over-attributes cannot be fully trusted even in cases
        # where it happens to name the right factor.
        confidence = DiagnosticConfidence.PARTIALLY_EXPLAINED
    else:
        confidence = DiagnosticConfidence.UNEXPLAINED

    return DiagnosticRecord(
        event_id=event_id, status=status, profile=profile,
        attributions=tuple(attributions), counterfactual=counterfactual,
        confidence=confidence,
    )
