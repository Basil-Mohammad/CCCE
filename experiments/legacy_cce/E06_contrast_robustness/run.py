# Author: Basil M. Alzboun

"""
E06_CONTRAST_ROBUSTNESS -- multiple pre-specified counterfactual controls
(Master Prompt V3, Section 31).

Using the Level-1 analytical SCM (chosen because it gives exact ground
truth for each contrast, isolating "does the conclusion change with the
control" from "is the estimator noisy"), we fix one target protocol L_T
and evaluate CCCE against three DIFFERENT, pre-specified control
protocols L_C1, L_C2, L_C3, each representing a distinct realistic
counterfactual choice:

    L_C1: "protected update"    -- small effective learning rate (EWC-like)
    L_C2: "alternative task"    -- pulls toward an unrelated target
    L_C3: "no update"           -- the L_0 baseline (included as a boundary
                                   case; not a genuine alternative "learning"
                                   protocol, but often the naive default
                                   comparison a less careful analysis would use)

We report sign consistency, magnitude range, and classify the case as
robust / contrast-sensitive / unstable per main_v2.tex Section 5.5.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ccce.scm.analytical import AnalyticalSCM, CompetenceLinearForm, LearningProtocolParams, NoUpdateProtocol
from ccce.oracle.analytical_oracle import compute_ground_truth
from ccce.utils.provenance import build_manifest

EXPERIMENT_ID = "E06_CONTRAST_ROBUSTNESS"
DELTA = 0.05


def build_scenario():
    comp = CompetenceLinearForm(
        "K1", a_i=0.0, b_i=np.array([1.0, 0.3, 0.0, 0.0]), h_i=np.array([0.2, 0.0, 0.0, 0.0]),
        sigma_y=0.1, valid_intervention_support=(-2, -1, 0, 1, 2),
    )
    scm = AnalyticalSCM(dim=4, sigma_u=0.02, theta0_mean=np.zeros(4), competences={"K1": comp})
    proto_T = LearningProtocolParams("L_T", target=np.array([1.5, 0.5, 0, 0]), gain=0.9, budget=5.0)
    proto_C1 = LearningProtocolParams("L_C1_protected", target=np.array([1.5, 0.5, 0, 0]), gain=0.2, budget=5.0)
    proto_C2 = LearningProtocolParams("L_C2_alt_task", target=np.array([-1.0, 1.0, 0, 0]), gain=0.9, budget=5.0)
    proto_C3 = NoUpdateProtocol()
    return scm, proto_T, {"L_C1_protected": proto_C1, "L_C2_alt_task": proto_C2, "L_C3_no_update": proto_C3}


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scm, proto_T, controls = build_scenario()
    means = scm.exact_theta_mean_trajectory([])

    rows = []
    for c in (-2.0, -1.0, 0.0, 1.0, 2.0):
        for control_name, proto_C in controls.items():
            gt = compute_ground_truth(scm, "K1", c, means, proto_T, proto_C)
            rows.append(dict(
                experiment_id=EXPERIMENT_ID, c=c, control=control_name, ccce_gt=gt.ccce_gt, rc_gt=gt.rc_gt,
            ))

    csv_path = output_dir / "ccce_estimates_multi_control.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- Robustness classification per c ----------------------------------
    classification_rows = []
    for c in (-2.0, -1.0, 0.0, 1.0, 2.0):
        vals = [r["ccce_gt"] for r in rows if r["c"] == c]
        signs = set(np.sign(v) if abs(v) > DELTA else 0 for v in vals)
        magnitude_range = max(vals) - min(vals)
        if len(signs) == 1:
            classification = "robust"
        elif len(signs) <= 2 and 0 not in signs:
            classification = "unstable"  # sign flips between + and -
        else:
            classification = "contrast_sensitive"
        classification_rows.append(dict(
            c=c, values=vals, sign_set=list(signs), magnitude_range=magnitude_range,
            classification=classification,
        ))

    with open(output_dir / "robustness_classification.json", "w") as f:
        json.dump(classification_rows, f, indent=2, default=str)

    manifest = build_manifest(
        experiment_id=EXPERIMENT_ID, run_id=EXPERIMENT_ID, seed=-1,
        algorithm="analytical_exact_oracle", environment="level1_analytical_scm",
        task_sequence="single_step", intervention_grid=[-2, -1, 0, 1, 2],
        control_protocol="L_C1_protected,L_C2_alt_task,L_C3_no_update",
        hyperparameters={"delta": DELTA},
    )
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"[{EXPERIMENT_ID}] Per-c robustness classification:")
    for row in classification_rows:
        print(f"  c={row['c']:+.1f}: values={[f'{v:+.3f}' for v in row['values']]} "
              f"-> {row['classification']}")


if __name__ == "__main__":
    out = Path(__file__).parent / "output"
    run(out)
