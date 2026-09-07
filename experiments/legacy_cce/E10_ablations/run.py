# Author: Basil M. Alzboun

"""
E10_ABLATIONS -- seed-count sensitivity (Master Prompt V3, Section 51,
factor 2 of 9). Uses the already-collected E01 discovery (N=10) and
confirmation (N=30) outputs directly -- no new training needed, since
this ablation only asks "does the aggregate metric change with N."

Only this one ablation factor is run in this session; the remaining 8
factors listed in Section 51 are NOT_RUN (see docs/DEVIATIONS.md).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ccce.statistics.inference import rmse, bias, coverage_rate

EXPERIMENT_ID = "E10_ABLATIONS"


def load_estimates(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    e01_dir = Path(__file__).parent.parent / "E01_ground_truth" / "output"

    rows = []
    for stage, n_seeds in (("discovery", 10), ("confirmation", 30)):
        est_path = e01_dir / stage / "ccce_estimates.csv"
        if not est_path.exists():
            continue
        data = load_estimates(est_path)
        certified = [r for r in data if r["identification_status"] == "CERTIFIED"]
        ests = [float(r["ccce"]) for r in certified]
        gts = [float(r["ccce_gt"]) for r in certified]
        los = [float(r["ci_low"]) for r in certified]
        his = [float(r["ci_high"]) for r in certified]
        rows.append(dict(
            ablation_factor="seed_count", value=n_seeds, stage=stage,
            n_certified=len(certified), rmse=rmse(ests, gts), bias=bias(ests, gts),
            coverage_95=coverage_rate(gts, los, his),
        ))

    csv_path = output_dir / "ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{EXPERIMENT_ID}] Seed-count ablation (derived from already-collected E01 data):")
    for row in rows:
        print(f"  N={row['value']:3d} ({row['stage']:12s}): RMSE={row['rmse']:.4f} "
              f"Bias={row['bias']:+.4f} Coverage95={row['coverage_95']:.2f}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(dict(
            factor="seed_count", rows=rows,
            caveat="Only 1 of 9 ablation factors listed in Section 51 is run in this "
                   "session (seed count, derived from already-collected E01 data at no "
                   "extra compute cost). The other 8 are NOT_RUN -- see docs/DEVIATIONS.md.",
        ), f, indent=2, default=str)


if __name__ == "__main__":
    out = Path(__file__).parent / "output"
    run(out)
