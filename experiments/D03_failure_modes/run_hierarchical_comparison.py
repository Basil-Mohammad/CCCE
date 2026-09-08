# Author: Basil M. Alzboun

"""
D03_FAILURE_MODES -- run_hierarchical_comparison.py

Compares the flat reference procedure (threshold_attribution_procedure,
Level C only) against the hierarchical Level C -> Level D cascade
(hierarchical_attribution_procedure) on Killer Experiment 2's Honesty
and Competence tests, per world. This directly quantifies Roadmap Item
#3's claimed improvement rather than leaving it as an informal code
comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ccce.attribution.hierarchical_procedure import hierarchical_attribution_procedure
from ccce.attribution.honesty_competence import (
    evaluate_competence,
    evaluate_honesty,
    threshold_attribution_procedure,
)

EXPERIMENT_ID = "D03_HIERARCHICAL_COMPARISON"
N_BATCHES = 100
BATCH_SIZE = 40
SEED = 2024


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, procedure in [("reference_flat", threshold_attribution_procedure),
                             ("hierarchical_cascade", hierarchical_attribution_procedure)]:
        rng = np.random.default_rng(SEED)
        honesty_rate, _ = evaluate_honesty(procedure, rng, n_batches=N_BATCHES, batch_size=BATCH_SIZE)
        competence_rate, _, per_world = evaluate_competence(
            procedure, rng, ("A", "B", "C"), n_batches=N_BATCHES, batch_size=BATCH_SIZE
        )
        results[name] = dict(
            honesty_rate=honesty_rate, competence_rate=competence_rate, competence_by_world=per_world,
        )
        print(f"[{EXPERIMENT_ID}] {name}: Honesty={honesty_rate:.3f} Competence={competence_rate:.3f}")
        for world_id, rate in per_world.items():
            print(f"    World {world_id}: {rate:.3f}")

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(dict(experiment_id=EXPERIMENT_ID, n_batches=N_BATCHES, batch_size=BATCH_SIZE,
                        seed=SEED, results=results), f, indent=2)

    print(f"\n[{EXPERIMENT_ID}] Improvement (hierarchical - reference):")
    for world_id in ("A", "B", "C"):
        delta = results["hierarchical_cascade"]["competence_by_world"][world_id] - \
                results["reference_flat"]["competence_by_world"][world_id]
        print(f"    World {world_id}: {delta:+.3f}")
    delta_honesty = results["hierarchical_cascade"]["honesty_rate"] - results["reference_flat"]["honesty_rate"]
    print(f"    Honesty (World D): {delta_honesty:+.3f}")


if __name__ == "__main__":
    out = Path(__file__).parent / "output_hierarchical"
    run(out)
