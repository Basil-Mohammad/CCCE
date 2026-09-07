# Author: Basil M. Alzboun

"""
D03_FAILURE_MODES -- main_v4.tex Section 8.3.

Runs Killer Experiment 2 (Honesty and Competence, main_v4.tex Section
6.4) against the reference threshold-based associational attribution
procedure and the two naive calibration baselines, and writes real,
non-fabricated results to disk.

This experiment directly tests Failures B and C (main_v4.tex Section
6.6): the Honesty rate must exceed the always-attribute baseline, and
the Competence rate must exceed the always-abstain baseline, and BOTH
must hold simultaneously for Killer 2 to be considered passed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ccce.attribution.honesty_competence import (
    always_abstain_baseline,
    always_attribute_baseline,
    evaluate_competence,
    evaluate_honesty,
    run_killer_experiment_two,
    threshold_attribution_procedure,
)
from ccce.utils.provenance import build_manifest

EXPERIMENT_ID = "D03_FAILURE_MODES"
N_BATCHES = 100
BATCH_SIZE = 40
SEED = 2024


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    result = run_killer_experiment_two(
        threshold_attribution_procedure, rng, n_batches=N_BATCHES, batch_size=BATCH_SIZE
    )

    summary = dict(
        experiment_id=EXPERIMENT_ID,
        procedure="threshold_attribution_procedure (reference associational-only, Category A)",
        n_batches=N_BATCHES, batch_size=BATCH_SIZE,
        honesty_rate=result.honesty_rate,
        honesty_baseline_always_attribute=result.honesty_baseline,
        honesty_exceeds_baseline=result.honesty_exceeds_baseline,
        competence_rate=result.competence_rate,
        competence_baseline_always_abstain=result.competence_baseline,
        competence_exceeds_baseline=result.competence_exceeds_baseline,
        competence_by_world=result.competence_by_world,
        killer_two_passed=result.killer_two_passed,
        failure_b_triggered=not result.honesty_exceeds_baseline,
        failure_c_triggered=not result.competence_exceeds_baseline,
    )

    with open(output_dir / "killer2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    manifest = build_manifest(
        experiment_id=EXPERIMENT_ID, run_id=EXPERIMENT_ID, seed=SEED,
        algorithm="threshold_attribution_procedure", environment="attribution_ground_truth_worlds_A-D",
        task_sequence="n/a", intervention_grid="n/a", control_protocol="n/a",
        hyperparameters={"n_batches": N_BATCHES, "batch_size": BATCH_SIZE, "correlation_threshold": 0.3},
    )
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"[{EXPERIMENT_ID}] Honesty rate (World D):    {result.honesty_rate:.3f} "
          f"(baseline: {result.honesty_baseline:.3f}) -> "
          f"{'EXCEEDS' if result.honesty_exceeds_baseline else 'DOES NOT EXCEED'} baseline")
    print(f"[{EXPERIMENT_ID}] Competence rate (A/B/C):   {result.competence_rate:.3f} "
          f"(baseline: {result.competence_baseline:.3f}) -> "
          f"{'EXCEEDS' if result.competence_exceeds_baseline else 'DOES NOT EXCEED'} baseline")
    for world_id, rate in result.competence_by_world.items():
        print(f"    World {world_id}: {rate:.3f}")
    print(f"[{EXPERIMENT_ID}] KILLER 2 VERDICT: "
          f"{'PASSED' if result.killer_two_passed else 'FAILED'} "
          f"(both Honesty and Competence must exceed baseline)")


if __name__ == "__main__":
    out = Path(__file__).parent / "output"
    run(out)
