# Author: Basil M. Alzboun

"""
D05_COUNTERFACTUAL_VALIDATION -- main_v4.tex Section 8.5.

Runs Killer Experiment 3 (counterfactual necessity, main_v4.tex Section
6.4): within World E, compares the best Category A observational
correlation against the Level E CCE probe's correlation with the true
signal, and writes real, non-fabricated results.

This experiment directly tests Failure D (main_v4.tex Section 6.6): if
the CCE probe does not substantially dominate every Category A signal in
World E, the counterfactual instrument provides no demonstrated
diagnostic necessity anywhere in this benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ccce.attribution.killer_experiments import run_killer_experiment_three
from ccce.utils.provenance import build_manifest

EXPERIMENT_ID = "D05_COUNTERFACTUAL_VALIDATION"
N_SAMPLES = 500
SEED = 2024


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    result = run_killer_experiment_three(rng, n_samples=N_SAMPLES)

    summary = dict(
        experiment_id=EXPERIMENT_ID,
        n_samples=N_SAMPLES,
        best_category_a_correlation_in_world_e=result.best_category_a_correlation_in_world_e,
        best_category_a_factor=result.best_category_a_factor,
        cce_probe_correlation_in_world_e=result.cce_probe_correlation_in_world_e,
        necessity_demonstrated=result.necessity_demonstrated,
        failure_d_triggered=not result.necessity_demonstrated,
        note=(
            "A CCE-probe correlation of exactly 1.0 at this Level-1 analytical "
            "stage is expected and correct, not overfitting: World E's paired "
            "outcomes share a single common noise draw (common random numbers), "
            "which cancels exactly in the paired difference used as the "
            "ground-truth target, leaving a deterministic function of the true "
            "cause. See killer_experiments.py docstring for the full explanation."
        ),
    )

    with open(output_dir / "killer3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    manifest = build_manifest(
        experiment_id=EXPERIMENT_ID, run_id=EXPERIMENT_ID, seed=SEED,
        algorithm="pearson_correlation_comparison", environment="attribution_ground_truth_world_E",
        task_sequence="n/a", intervention_grid="n/a", control_protocol="n/a",
        hyperparameters={"n_samples": N_SAMPLES},
    )
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"[{EXPERIMENT_ID}] Best Category-A correlation in World E: "
          f"{result.best_category_a_correlation_in_world_e:.4f} ({result.best_category_a_factor})")
    print(f"[{EXPERIMENT_ID}] Level-E CCE probe correlation in World E: "
          f"{result.cce_probe_correlation_in_world_e:.4f}")
    print(f"[{EXPERIMENT_ID}] KILLER 3 VERDICT: "
          f"{'NECESSITY DEMONSTRATED' if result.necessity_demonstrated else 'FAILURE D TRIGGERED'}")


if __name__ == "__main__":
    out = Path(__file__).parent / "output"
    run(out)
