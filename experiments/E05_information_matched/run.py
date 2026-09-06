"""
E04_INCREMENTAL_INFORMATION and E05_INFORMATION_MATCHED
(Master Prompt V3, Sections 27-28; main_v2.tex Section 4.4 Circularity Audit).

THESE ARE THE MOST IMPORTANT EXPERIMENTS IN THE PROGRAM. They directly
test whether CCCE is scientifically redundant with (a) conventional
diagnostics or (b) the raw quantities (Q_T, Q_C) from which it is
literally computed.

Dataset construction: we generate many independent random scenarios at
Level 1 (random competence directions, random protocol targets/gains),
compute for each scenario:
    - conventional-diagnostic-like features: here, since Level 1 has no
      gradients in the RL sense, we use the *analytical analogues* defined
      directly from the SCM: BWT_proxy = RC_i, "task similarity" = cosine
      similarity of the T/C target vectors, "update magnitude" = norm of
      the parameter update, "GI proxy" = cosine similarity of (target_T -
      theta_prev) and (target_C - theta_prev). This is a documented
      simplification of Section 26's RL-specific diagnostics, appropriate
      to what Level 1 can express (see docs/DEVIATIONS.md).
    - CCCE (from the same closed-form apparatus as the oracle, for a
      noise-free "best case" test of whether CCCE is informative even
      before considering estimator noise)
    - Q_T, Q_C (the raw quantities)
    - a genuinely independent PREDICTION TARGET: whether the competence
      envelope is violated after a further, independently randomized
      *held-out* continuation task (so the target is not algebraically
      determined by the features being compared -- it requires actually
      generalizing).

We then compare, via 5-fold cross-validation (never random single-split,
per Section 28's "do not use random trajectory splitting" -- here the
natural unit is the scenario, and scenarios are IID by construction so a
grouped/sequence-level split is unnecessary, unlike the real Level-2
setting where task-sequence identity would require group-aware CV):

    M0 = f(bwt_proxy, gi_proxy, representation_drift_proxy, task_similarity, update_magnitude)
    M1 = f(M0_features, CCCE)
    M2 = f(M0_features, Q_T, Q_C)               [[the information-matched model]]

using logistic regression (the target is binary: envelope violated y/n).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

from ccce.scm.analytical import AnalyticalSCM, CompetenceLinearForm, LearningProtocolParams
from ccce.oracle.analytical_oracle import compute_ground_truth
from ccce.utils.provenance import build_manifest

EXPERIMENT_ID_04 = "E04_INCREMENTAL_INFORMATION"
EXPERIMENT_ID_05 = "E05_INFORMATION_MATCHED"
N_SCENARIOS = 400
DIM = 4
DELTA_ENVELOPE = -0.3  # a scenario "violates the envelope" if post-continuation competence < this


def generate_scenario(rng: np.random.Generator) -> dict:
    b_i = rng.normal(0, 1, size=DIM)
    h_i = rng.normal(0, 1, size=DIM)
    comp = CompetenceLinearForm("K1", 0.0, b_i, h_i, sigma_y=0.1, valid_intervention_support=(0.0, 1.0))
    scm = AnalyticalSCM(dim=DIM, sigma_u=0.05, theta0_mean=rng.normal(0, 0.3, size=DIM), competences={"K1": comp})

    target_T = rng.normal(0, 1, size=DIM)
    # L_C is a "partially protected" variant: same direction but damped gain,
    # occasionally a different direction, to get a realistic spread of contrasts.
    if rng.random() < 0.5:
        target_C = target_T
        gain_C = float(rng.uniform(0.1, 0.5))
    else:
        target_C = rng.normal(0, 1, size=DIM)
        gain_C = float(rng.uniform(0.3, 0.9))
    gain_T = float(rng.uniform(0.5, 0.95))
    budget = float(rng.uniform(2.0, 10.0))

    proto_T = LearningProtocolParams("L_T", target=target_T, gain=gain_T, budget=budget)
    proto_C = LearningProtocolParams("L_C", target=target_C, gain=gain_C, budget=budget)

    means = scm.exact_theta_mean_trajectory([])
    theta_prev = means[-1]
    gt = compute_ground_truth(scm, "K1", 1.0, means, proto_T, proto_C)

    mean_T = (1 - proto_T.effective_rate) * theta_prev + proto_T.effective_rate * target_T
    mean_C = (1 - proto_C.effective_rate) * theta_prev + proto_C.effective_rate * target_C
    q_target = float(comp.b_i @ mean_T + 1.0 * (comp.h_i @ mean_T))
    q_control = float(comp.b_i @ mean_C + 1.0 * (comp.h_i @ mean_C))

    # ---- Conventional-diagnostic-like features (documented analogues) -----
    bwt_proxy = gt.rc_gt
    dir_T = target_T - theta_prev
    dir_C = target_C - theta_prev
    gi_proxy = float(np.dot(dir_T, dir_C) / (np.linalg.norm(dir_T) * np.linalg.norm(dir_C) + 1e-9))
    rep_drift_proxy = float(np.linalg.norm(mean_T - theta_prev) / (np.linalg.norm(theta_prev) + 1e-6))
    task_sim_proxy = float(np.dot(target_T, target_C) / (np.linalg.norm(target_T) * np.linalg.norm(target_C) + 1e-9))
    update_mag_proxy = float(np.linalg.norm(mean_T - theta_prev))

    # ---- Independent held-out future continuation -> binary label ---------
    future_target = rng.normal(0, 1, size=DIM)
    future_proto = LearningProtocolParams("future", target=future_target, gain=float(rng.uniform(0.5, 0.95)), budget=budget)
    means_future = scm.exact_theta_mean_trajectory([proto_T, future_proto])
    q_after_future = float(comp.b_i @ means_future[-1])
    label_violation = int(q_after_future < DELTA_ENVELOPE)

    return dict(
        bwt_proxy=bwt_proxy, gi_proxy=gi_proxy, representation_drift_proxy=rep_drift_proxy,
        task_similarity_proxy=task_sim_proxy, update_magnitude_proxy=update_mag_proxy,
        ccce=gt.ccce_gt, q_target=q_target, q_control=q_control, label_violation=label_violation,
    )


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(555)
    scenarios = [generate_scenario(rng) for _ in range(N_SCENARIOS)]

    csv_path = output_dir / "incremental_dataset.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scenarios[0].keys()))
        writer.writeheader()
        writer.writerows(scenarios)

    M0_cols = ["bwt_proxy", "gi_proxy", "representation_drift_proxy", "task_similarity_proxy", "update_magnitude_proxy"]
    X0 = np.array([[s[c] for c in M0_cols] for s in scenarios])
    X1 = np.hstack([X0, np.array([[s["ccce"]] for s in scenarios])])
    X2 = np.hstack([X0, np.array([[s["q_target"], s["q_control"]] for s in scenarios])])
    y = np.array([s["label_violation"] for s in scenarios])

    n_pos = int(y.sum())
    print(f"[{EXPERIMENT_ID_04}/{EXPERIMENT_ID_05}] label balance: {n_pos}/{len(y)} positive "
          f"({n_pos/len(y):.1%}) envelope violations")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def cv_auc(X):
        aucs = []
        for train_idx, test_idx in cv.split(X, y):
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[test_idx])[:, 1]
            if len(np.unique(y[test_idx])) < 2:
                continue
            aucs.append(roc_auc_score(y[test_idx], probs))
        return float(np.mean(aucs)), float(np.std(aucs))

    auc_M0_mean, auc_M0_std = cv_auc(X0)
    auc_M1_mean, auc_M1_std = cv_auc(X1)
    auc_M2_mean, auc_M2_std = cv_auc(X2)

    results = dict(
        n_scenarios=N_SCENARIOS,
        label_positive_rate=n_pos / len(y),
        M0_conventional_diagnostics_only=dict(auc_mean=auc_M0_mean, auc_std=auc_M0_std, features=M0_cols),
        M1_M0_plus_CCCE=dict(auc_mean=auc_M1_mean, auc_std=auc_M1_std, features=M0_cols + ["ccce"]),
        M2_M0_plus_raw_QT_QC=dict(auc_mean=auc_M2_mean, auc_std=auc_M2_std, features=M0_cols + ["q_target", "q_control"]),
        incremental_gain_M1_over_M0=auc_M1_mean - auc_M0_mean,
        information_matched_gap_M1_minus_M2=auc_M1_mean - auc_M2_mean,
    )

    # ---- Pre-specified interpretation rule (fixed BEFORE this ran) --------
    E04_verdict = (
        "SUPPORTED" if results["incremental_gain_M1_over_M0"] > 0.02
        else "NOT_SUPPORTED (Falsification Condition A candidate)"
    )
    E05_verdict = (
        "SUPPORTED" if results["information_matched_gap_M1_minus_M2"] > -0.01
        else "NOT_SUPPORTED (Falsification Condition B candidate: M0+(Q_T,Q_C) outperforms M0+CCCE)"
    )
    results["E04_verdict"] = E04_verdict
    results["E05_verdict"] = E05_verdict

    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    manifest = build_manifest(
        experiment_id=f"{EXPERIMENT_ID_04}+{EXPERIMENT_ID_05}", run_id="combined", seed=555,
        algorithm="logistic_regression_5fold_cv", environment="level1_analytical_scm_random_scenarios",
        task_sequence="random_history+target+held_out_future", intervention_grid=[1.0],
        control_protocol="varied_per_scenario_pre_specified_generator",
        hyperparameters={"n_scenarios": N_SCENARIOS, "delta_envelope": DELTA_ENVELOPE, "cv_folds": 5},
    )
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"[{EXPERIMENT_ID_04}] M0 AUC={auc_M0_mean:.4f}+-{auc_M0_std:.4f}  "
          f"M1(M0+CCCE) AUC={auc_M1_mean:.4f}+-{auc_M1_std:.4f}  "
          f"gain={results['incremental_gain_M1_over_M0']:+.4f} -> {E04_verdict}")
    print(f"[{EXPERIMENT_ID_05}] M1(M0+CCCE) AUC={auc_M1_mean:.4f}  "
          f"M2(M0+Q_T,Q_C) AUC={auc_M2_mean:.4f}  "
          f"gap={results['information_matched_gap_M1_minus_M2']:+.4f} -> {E05_verdict}")


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    run(out_dir / ".." / "E04_incremental_information" / "output")
