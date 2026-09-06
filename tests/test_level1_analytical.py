"""
Unit + integration tests for the Level-1 analytical SCM pipeline
(Master Prompt V3, Sections 35, 43).
"""
import numpy as np
import pytest

from ccce.scm.analytical import AnalyticalSCM, CompetenceLinearForm, LearningProtocolParams, NoUpdateProtocol
from ccce.scm.regimes import build_all_regime_cases, DELTA, INTERVENTION_GRID
from ccce.oracle.analytical_oracle import compute_ground_truth
from ccce.estimator.paired_estimator import estimate_ccce_paired
from ccce.interventions.validity import classify_intervention, ValidityStatus
from ccce.estimator.certification import certify, CertificationState
from ccce.utils.rng import RNGBundle, paired_evaluation_seed
from ccce.utils.checkpoint import save_checkpoint, load_checkpoint, restore_rng_bundle
from ccce.statistics.inference import (
    paired_mean_ci,
    bootstrap_ci,
    holm_correction,
    benjamini_hochberg,
    rmse,
    bias as bias_fn,
    coverage_rate,
    sign_accuracy,
    cohens_d,
)


# --------------------------------------------------------------------------
# SCM correctness
# --------------------------------------------------------------------------
def test_exact_mean_recursion_matches_manual_computation():
    scm = AnalyticalSCM(
        dim=2,
        sigma_u=0.1,
        theta0_mean=np.array([0.0, 0.0]),
        competences={},
    )
    proto = LearningProtocolParams("p1", target=np.array([1.0, 0.0]), gain=0.5, budget=100.0)
    means = scm.exact_theta_mean_trajectory([proto])
    eta = proto.effective_rate
    expected = (1 - eta) * np.array([0.0, 0.0]) + eta * np.array([1.0, 0.0])
    assert np.allclose(means[-1], expected)


def test_effective_rate_saturates_with_budget():
    small_budget = LearningProtocolParams("p", target=np.zeros(1), gain=1.0, budget=0.01)
    large_budget = LearningProtocolParams("p", target=np.zeros(1), gain=1.0, budget=1000.0)
    assert small_budget.effective_rate < 0.1
    assert large_budget.effective_rate > 0.99


def test_no_update_protocol_has_zero_rate():
    assert NoUpdateProtocol().effective_rate == 0.0


def test_exact_ccce_zero_for_identical_protocols():
    """Sanity Test 2 (Section 35): L_T == L_C must give CCCE^GT == 0 exactly."""
    comp = CompetenceLinearForm("K1", 0.0, np.array([1.0, 0.0]), np.array([0.5, 0.0]), 0.1, (0.0, 1.0))
    scm = AnalyticalSCM(dim=2, sigma_u=0.05, theta0_mean=np.zeros(2), competences={"K1": comp})
    proto = LearningProtocolParams("same", target=np.array([1.0, 1.0]), gain=0.7, budget=5.0)
    means = scm.exact_theta_mean_trajectory([])
    ccce = scm.exact_ccce("K1", 1.0, means, proto, proto)
    assert ccce == pytest.approx(0.0, abs=1e-12)


def test_exact_rc_zero_for_no_update():
    """Sanity Test 1 (Section 35): L_T == L_0 must give RC == 0 exactly."""
    comp = CompetenceLinearForm("K1", 0.0, np.array([1.0]), np.array([0.0]), 0.1, (0.0,))
    scm = AnalyticalSCM(dim=1, sigma_u=0.05, theta0_mean=np.zeros(1), competences={"K1": comp})
    no_update = NoUpdateProtocol()
    means = scm.exact_theta_mean_trajectory([])
    rc = scm.exact_rc("K1", means, no_update)
    assert rc == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Regime construction: verify each regime has the claimed sign/property
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case_id", ["beneficial", "neutral", "harmful", "positive_control", "negative_control"])
def test_regime_ground_truth_matches_declared_sign(case_id):
    cases = build_all_regime_cases()
    case = cases[case_id]
    means = case.scm.exact_theta_mean_trajectory([])
    gt = compute_ground_truth(case.scm, case.competence_id, case.query_c, means, case.protocol_T, case.protocol_C)

    if case.expected_sign == "+":
        assert gt.ccce_gt > DELTA
    elif case.expected_sign == "-":
        assert gt.ccce_gt < -DELTA
    elif case.expected_sign == "0":
        assert abs(gt.ccce_gt) <= DELTA


def test_hidden_vulnerability_regime_has_rc_near_zero_and_ccce_negative():
    cases = build_all_regime_cases()
    case = cases["hidden_vulnerability"]
    means = case.scm.exact_theta_mean_trajectory([])
    gt_nominal = compute_ground_truth(case.scm, case.competence_id, 0.0, means, case.protocol_T, case.protocol_C)
    gt_intervened = compute_ground_truth(
        case.scm, case.competence_id, case.query_c, means, case.protocol_T, case.protocol_C
    )
    assert abs(gt_nominal.rc_gt) <= DELTA, f"RC should be ~0, got {gt_nominal.rc_gt}"
    assert gt_intervened.ccce_gt < -DELTA, f"CCCE(c*) should be < -delta, got {gt_intervened.ccce_gt}"


def test_unidentifiable_regime_query_is_outside_support():
    cases = build_all_regime_cases()
    case = cases["unidentifiable"]
    comp = case.scm.competences[case.competence_id]
    assert not comp.is_supported(case.query_c)


# --------------------------------------------------------------------------
# Intervention validity classification
# --------------------------------------------------------------------------
def test_intervention_missing_causal_parent_is_invalid():
    meta = classify_intervention(
        "iv1", "observation_noise", 0.5,
        is_designated_causal_parent=False,
        feasible=True, contract_preserving=True, safety_valid=True, task_preserving=True,
        valid_support=(-2, -1, 0, 1, 2),
    )
    assert meta.validity_status == ValidityStatus.INVALID


def test_intervention_outside_support_is_unidentifiable():
    meta = classify_intervention(
        "iv2", "z1", 10.0,
        is_designated_causal_parent=True,
        feasible=True, contract_preserving=True, safety_valid=True, task_preserving=True,
        valid_support=(-2, -1, 0, 1, 2),
    )
    assert meta.validity_status == ValidityStatus.UNIDENTIFIABLE


def test_intervention_not_task_preserving_is_task_altering():
    meta = classify_intervention(
        "iv3", "z3", 2.0,
        is_designated_causal_parent=True,
        feasible=True, contract_preserving=True, safety_valid=True, task_preserving=False,
        valid_support=(-2, -1, 0, 1, 2),
    )
    assert meta.validity_status == ValidityStatus.TASK_ALTERING


def test_valid_intervention_certifies():
    meta = classify_intervention(
        "iv4", "z1", 0.0,
        is_designated_causal_parent=True,
        feasible=True, contract_preserving=True, safety_valid=True, task_preserving=True,
        valid_support=(-2, -1, 0, 1, 2),
    )
    assert meta.validity_status == ValidityStatus.VALID
    result = certify(meta, envelope_satisfied=True)
    assert result.state == CertificationState.CERTIFIED


def test_unidentifiable_intervention_never_certifies():
    meta = classify_intervention(
        "iv5", "z1", 99.0,
        is_designated_causal_parent=True,
        feasible=True, contract_preserving=True, safety_valid=True, task_preserving=True,
        valid_support=(-2, -1, 0, 1, 2),
    )
    result = certify(meta, envelope_satisfied=None)
    assert result.state == CertificationState.UNIDENTIFIABLE


def test_envelope_violation_yields_violated_not_certified():
    meta = classify_intervention(
        "iv6", "z1", 0.0,
        is_designated_causal_parent=True,
        feasible=True, contract_preserving=True, safety_valid=True, task_preserving=True,
        valid_support=(-2, -1, 0, 1, 2),
    )
    result = certify(meta, envelope_satisfied=False)
    assert result.state == CertificationState.VIOLATED


# --------------------------------------------------------------------------
# Estimator vs oracle: recovers known ground truth (Sanity Test 3)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case_id", ["beneficial", "harmful", "hidden_vulnerability"])
def test_estimator_recovers_known_sign(case_id):
    cases = build_all_regime_cases()
    case = cases[case_id]
    means = case.scm.exact_theta_mean_trajectory([])
    gt = compute_ground_truth(case.scm, case.competence_id, case.query_c, means, case.protocol_T, case.protocol_C)

    bundle = RNGBundle(top_level_seed=12345)
    meta = classify_intervention(
        "iv_test", "z_synth", case.query_c,
        is_designated_causal_parent=True, feasible=True, contract_preserving=True,
        safety_valid=True, task_preserving=True,
        valid_support=case.scm.competences[case.competence_id].valid_intervention_support,
    )
    est = estimate_ccce_paired(
        case.scm, case.competence_id, case.query_c, means[-1],
        case.protocol_T, case.protocol_C, meta,
        n_replicates=400, training_rng=bundle.training, evaluation_rng=bundle.evaluation,
    )
    # Estimator should recover the correct *sign* with 400 paired replicates.
    if gt.ccce_gt > DELTA:
        assert est.ccce.mean > 0
    elif gt.ccce_gt < -DELTA:
        assert est.ccce.mean < 0
    # And should be numerically close to ground truth.
    assert abs(est.ccce.mean - gt.ccce_gt) < 0.15


def test_estimator_ci_contains_ground_truth_at_reasonable_rate():
    """Not a strict per-case guarantee (finite-sample CIs can miss), but
    across all five signed regimes with N=300 replicates each, at least
    the majority of 95% CIs should contain the true value."""
    cases = build_all_regime_cases()
    hits = 0
    total = 0
    for i, (cid, case) in enumerate(cases.items()):
        means = case.scm.exact_theta_mean_trajectory([])
        gt = compute_ground_truth(case.scm, case.competence_id, case.query_c, means, case.protocol_T, case.protocol_C)
        bundle = RNGBundle(top_level_seed=1000 + i)
        meta = classify_intervention(
            f"iv_{cid}", "z_synth", case.query_c,
            is_designated_causal_parent=True, feasible=True, contract_preserving=True,
            safety_valid=True, task_preserving=True,
            valid_support=case.scm.competences[case.competence_id].valid_intervention_support,
        )
        est = estimate_ccce_paired(
            case.scm, case.competence_id, case.query_c, means[-1],
            case.protocol_T, case.protocol_C, meta,
            n_replicates=300, training_rng=bundle.training, evaluation_rng=bundle.evaluation,
        )
        if case.expected_sign != "UNIDENTIFIABLE":
            total += 1
            if est.ccce.ci_low <= gt.ccce_gt <= est.ccce.ci_high:
                hits += 1
    assert total > 0
    assert hits / total >= 0.6  # loose bound; this is a sanity check, not a calibration claim


# --------------------------------------------------------------------------
# RNG stream independence
# --------------------------------------------------------------------------
def test_rng_streams_are_independent_objects():
    bundle = RNGBundle(top_level_seed=42)
    draws = {name: bundle.stream(name).standard_normal(5) for name in
             ("training_rng", "environment_rng", "evaluation_rng", "oracle_rng", "bootstrap_rng")}
    values = list(draws.values())
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            assert not np.allclose(values[i], values[j])


def test_rng_bundle_reproducible_from_same_seed():
    b1 = RNGBundle(top_level_seed=7)
    b2 = RNGBundle(top_level_seed=7)
    assert np.allclose(b1.training.standard_normal(10), b2.training.standard_normal(10))


def test_unknown_stream_name_raises():
    bundle = RNGBundle(top_level_seed=1)
    with pytest.raises(KeyError):
        bundle.stream("not_a_real_stream")


def test_paired_evaluation_seed_deterministic():
    s1 = paired_evaluation_seed(base_seed=5, replicate=3)
    s2 = paired_evaluation_seed(base_seed=5, replicate=3)
    s3 = paired_evaluation_seed(base_seed=5, replicate=4)
    assert s1 == s2
    assert s1 != s3


# --------------------------------------------------------------------------
# Checkpoint save/load/resume (Sanity Test 7)
# --------------------------------------------------------------------------
def test_checkpoint_roundtrip(tmp_path):
    bundle = RNGBundle(top_level_seed=99)
    _ = bundle.training.standard_normal(3)  # advance state so it's non-trivial
    model_state = {"weights": np.array([1.0, 2.0, 3.0])}
    save_checkpoint(
        tmp_path / "ckpt1",
        model_state=model_state,
        optimizer_state={"step": 10},
        environment_state={"pos": [0, 0]},
        rng_bundle=bundle,
        metadata={"experiment_id": "TEST", "seed": 99},
        metrics={"return": 1.23},
    )
    loaded = load_checkpoint(tmp_path / "ckpt1")
    assert np.allclose(loaded["model_state"]["weights"], model_state["weights"])
    assert loaded["metadata"]["experiment_id"] == "TEST"

    restored_bundle = restore_rng_bundle(99, loaded["rng_state"])
    # Continuing from the restored bundle must match continuing from the
    # original (post-advance) bundle -- this is the "resume reproducibility" check.
    expected_next = bundle.training.standard_normal(3)
    actual_next = restored_bundle.training.standard_normal(3)
    assert np.allclose(expected_next, actual_next)


# --------------------------------------------------------------------------
# Statistics utilities
# --------------------------------------------------------------------------
def test_paired_mean_ci_basic():
    deltas = [1.0, 1.1, 0.9, 1.05, 0.95]
    est = paired_mean_ci(deltas)
    assert est.mean == pytest.approx(1.0, abs=0.01)
    assert est.ci_low < est.mean < est.ci_high


def test_bootstrap_ci_reasonable():
    rng = np.random.default_rng(0)
    data = rng.normal(2.0, 1.0, size=200)
    est = bootstrap_ci(data, rng=rng, n_boot=500)
    assert 1.5 < est.mean < 2.5
    assert est.ci_low < est.mean < est.ci_high


def test_holm_correction_controls_family_wise_error_conservatively():
    p_values = [0.001, 0.01, 0.03, 0.04, 0.5]
    rejected = holm_correction(p_values, alpha=0.05)
    assert rejected[0] is True  # smallest p almost certainly rejected
    assert rejected[4] is False  # largest p almost certainly not rejected


def test_benjamini_hochberg_rejects_at_least_as_many_as_holm():
    p_values = [0.001, 0.008, 0.02, 0.03, 0.2]
    holm = holm_correction(p_values)
    bh = benjamini_hochberg(p_values)
    assert sum(bh) >= sum(holm)


def test_rmse_and_bias_zero_for_perfect_estimator():
    est = [1.0, 2.0, 3.0]
    gt = [1.0, 2.0, 3.0]
    assert rmse(est, gt) == pytest.approx(0.0)
    assert bias_fn(est, gt) == pytest.approx(0.0)


def test_coverage_rate_basic():
    gt = [1.0, 2.0, 3.0]
    lo = [0.5, 2.5, 2.5]
    hi = [1.5, 3.5, 3.5]
    # case1: 1.0 in [0.5,1.5] yes; case2: 2.0 in [2.5,3.5] no; case3: 3.0 in [2.5,3.5] yes
    assert coverage_rate(gt, lo, hi) == pytest.approx(2 / 3)


def test_sign_accuracy_basic():
    est = [1.0, -1.0, 0.01]
    gt = [1.0, -1.0, 0.0]
    assert sign_accuracy(est, gt, delta=0.05) == pytest.approx(1.0)


def test_cohens_d_one_sample():
    x = [1.0, 1.0, 1.0, 1.0]
    d = cohens_d(x)
    assert d == 0.0  # zero variance -> defined as 0 by this implementation
