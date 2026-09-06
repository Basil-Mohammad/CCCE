"""
E11_NEGATIVE_CONTROLS -- Master Prompt V3 Section 33.

Generates many independent negative-control scenarios (L_T == L_C by
construction, so CCCE^GT == 0 exactly) with randomized competence
directions and protocol targets, and checks that the *estimator* does not
systematically produce large effects -- i.e., checks the empirical
false-positive rate of "CI excludes 0" against the nominal 5% rate.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ccce.scm.analytical import AnalyticalSCM, CompetenceLinearForm, LearningProtocolParams
from ccce.oracle.analytical_oracle import compute_ground_truth
from ccce.estimator.paired_estimator import estimate_ccce_paired
from ccce.interventions.validity import classify_intervention
from ccce.utils.rng import RNGBundle
from ccce.utils.provenance import build_manifest

EXPERIMENT_ID = "E11_NEGATIVE_CONTROLS"
N_SCENARIOS = 80
N_REPLICATES = 100
DIM = 4


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    false_positives = 0

    for i in range(N_SCENARIOS):
        seed_seq = np.random.SeedSequence([3000, i])
        setup_rng = np.random.default_rng(seed_seq)

        b_i = setup_rng.normal(0, 1, size=DIM)
        h_i = setup_rng.normal(0, 1, size=DIM)
        target = setup_rng.normal(0, 1, size=DIM)
        gain = float(setup_rng.uniform(0.3, 0.95))
        budget = float(setup_rng.uniform(1.0, 10.0))

        comp = CompetenceLinearForm("K1", 0.0, b_i, h_i, sigma_y=0.1, valid_intervention_support=(0.0, 1.0))
        scm = AnalyticalSCM(dim=DIM, sigma_u=0.05, theta0_mean=np.zeros(DIM), competences={"K1": comp})
        proto_T = LearningProtocolParams("L_T", target=target, gain=gain, budget=budget)
        proto_C = LearningProtocolParams("L_C", target=target, gain=gain, budget=budget)  # IDENTICAL -> GT is exactly 0

        means = scm.exact_theta_mean_trajectory([])
        gt = compute_ground_truth(scm, "K1", 1.0, means, proto_T, proto_C)
        assert abs(gt.ccce_gt) < 1e-9, "negative control construction failed: GT should be exactly 0"

        meta = classify_intervention(
            f"negctrl_{i}", "z_synth", 1.0, is_designated_causal_parent=True,
            feasible=True, contract_preserving=True, safety_valid=True, task_preserving=True,
            valid_support=comp.valid_intervention_support,
        )
        bundle = RNGBundle(top_level_seed=int(seed_seq.generate_state(1, dtype=np.uint32)[0]))
        est = estimate_ccce_paired(
            scm, "K1", 1.0, means[-1], proto_T, proto_C, meta,
            n_replicates=N_REPLICATES, training_rng=bundle.training, evaluation_rng=bundle.evaluation,
        )
        ci_excludes_zero = not (est.ccce.ci_low <= 0.0 <= est.ccce.ci_high)
        false_positives += int(ci_excludes_zero)

        rows.append(dict(
            experiment_id=EXPERIMENT_ID, scenario_id=i, ccce_gt=gt.ccce_gt, ccce_hat=est.ccce.mean,
            ci_low=est.ccce.ci_low, ci_high=est.ccce.ci_high, ci_excludes_zero=ci_excludes_zero,
        ))

    csv_path = output_dir / "ccce_estimates.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    empirical_fp_rate = false_positives / N_SCENARIOS
    summary = dict(
        experiment_id=EXPERIMENT_ID, n_scenarios=N_SCENARIOS, n_replicates_per_scenario=N_REPLICATES,
        false_positive_count=false_positives, empirical_false_positive_rate=empirical_fp_rate,
        nominal_rate=0.05,
        assessment=(
            "consistent with nominal 5% rate" if 0.0 <= empirical_fp_rate <= 0.15
            else "ELEVATED false-positive rate -- possible estimator or CI-calibration issue"
        ),
    )
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    manifest = build_manifest(
        experiment_id=EXPERIMENT_ID, run_id=EXPERIMENT_ID, seed=3000,
        algorithm="analytical_paired_monte_carlo_estimator", environment="level1_analytical_scm",
        task_sequence="single_step_negative_control", intervention_grid=[1.0],
        control_protocol="identical_to_target_by_construction",
        hyperparameters={"n_scenarios": N_SCENARIOS, "n_replicates": N_REPLICATES},
    )
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"[{EXPERIMENT_ID}] empirical false-positive rate: {empirical_fp_rate:.3f} "
          f"({false_positives}/{N_SCENARIOS}), nominal=0.05 -> {summary['assessment']}")


if __name__ == "__main__":
    out = Path(__file__).parent / "output"
    run(out)
