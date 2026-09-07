"""
tests/test_diagnostic_record.py

Author: Basil M. Alzboun
"""
import numpy as np
import pytest

from ccce.attribution.diagnostic_record import (
    Attribution,
    CharacterizationProfile,
    CompetenceChangeStatus,
    CounterfactualResult,
    DiagnosticConfidence,
    EvidenceLevel,
    build_diagnostic_record,
)


def test_no_change_status_always_yields_no_change_detected():
    record = build_diagnostic_record(
        event_id="ev1", status=CompetenceChangeStatus.NO_CHANGE, profile=None,
        attributions=[], counterfactual=None,
        honesty_criterion_evaluated=False, competence_criterion_evaluated=False,
        honesty_passed=None, competence_passed=None,
    )
    assert record.confidence == DiagnosticConfidence.NO_CHANGE_DETECTED


def test_cannot_assign_explained_without_evaluating_both_criteria():
    with pytest.raises(ValueError):
        build_diagnostic_record(
            event_id="ev2", status=CompetenceChangeStatus.DEGRADATION, profile=None,
            attributions=[Attribution("gradient_interference", EvidenceLevel.ASSOCIATIONAL)],
            counterfactual=None,
            honesty_criterion_evaluated=True, competence_criterion_evaluated=False,
            honesty_passed=True, competence_passed=None,
        )


def test_explained_requires_both_honesty_and_competence_passed():
    record = build_diagnostic_record(
        event_id="ev3", status=CompetenceChangeStatus.DEGRADATION, profile=None,
        attributions=[Attribution("task_order", EvidenceLevel.INTERVENTIONAL)],
        counterfactual=None,
        honesty_criterion_evaluated=True, competence_criterion_evaluated=True,
        honesty_passed=True, competence_passed=True,
    )
    assert record.confidence == DiagnosticConfidence.EXPLAINED


def test_no_attributions_yields_unexplained():
    record = build_diagnostic_record(
        event_id="ev4", status=CompetenceChangeStatus.DRIFT, profile=None,
        attributions=[], counterfactual=None,
        honesty_criterion_evaluated=True, competence_criterion_evaluated=True,
        honesty_passed=True, competence_passed=True,
    )
    assert record.confidence == DiagnosticConfidence.UNEXPLAINED


def test_attributions_never_merge_evidence_levels():
    a1 = Attribution("gradient_interference", EvidenceLevel.ASSOCIATIONAL)
    a2 = Attribution("gradient_interference", EvidenceLevel.INTERVENTIONAL)
    record = build_diagnostic_record(
        event_id="ev5", status=CompetenceChangeStatus.DEGRADATION, profile=None,
        attributions=[a1, a2], counterfactual=None,
        honesty_criterion_evaluated=True, competence_criterion_evaluated=True,
        honesty_passed=True, competence_passed=True,
    )
    levels = {a.evidence_level for a in record.attributions}
    assert levels == {EvidenceLevel.ASSOCIATIONAL, EvidenceLevel.INTERVENTIONAL}


def test_counterfactual_result_fields():
    cf = CounterfactualResult(cross_competence_effect=-0.05, confidence_interval=(-0.09, -0.01), identification_status="CERTIFIED")
    record = build_diagnostic_record(
        event_id="ev6", status=CompetenceChangeStatus.DRIFT, profile=None,
        attributions=[Attribution("environmental_causal_factor", EvidenceLevel.INTERVENTIONAL)],
        counterfactual=cf,
        honesty_criterion_evaluated=True, competence_criterion_evaluated=True,
        honesty_passed=True, competence_passed=True,
    )
    assert record.counterfactual.cross_competence_effect == -0.05
    assert record.counterfactual.identification_status == "CERTIFIED"


def test_characterization_profile_fields():
    profile = CharacterizationProfile(
        affected_outcome_dimension="Safety", evaluation_condition="c=2.0", checkpoint="T5_end",
    )
    record = build_diagnostic_record(
        event_id="ev7", status=CompetenceChangeStatus.DEGRADATION, profile=profile,
        attributions=[Attribution("plasticity", EvidenceLevel.ASSOCIATIONAL)],
        counterfactual=None,
        honesty_criterion_evaluated=True, competence_criterion_evaluated=True,
        honesty_passed=True, competence_passed=True,
    )
    assert record.profile.affected_outcome_dimension == "Safety"
    assert record.profile.checkpoint == "T5_end"
