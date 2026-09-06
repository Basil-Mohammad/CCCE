"""
E02_IDENTIFICATION -- systematic identification/abstention testing
(Master Prompt V3, Section 32).

Constructs a battery of candidate interventions spanning every
classification outcome (VALID, BORDERLINE, TASK_ALTERING, INVALID,
UNIDENTIFIABLE) and verifies: (a) the classifier assigns the expected
label to each hand-constructed case (this duplicates some unit-test
content but produces an actual results CSV, per the data-contract
requirement), and (b) certification never CERTIFIES an intervention that
is not VALID/BORDERLINE, across a randomized battery of ~200 synthetic
candidate interventions.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ccce.interventions.validity import classify_intervention, ValidityStatus
from ccce.estimator.certification import certify, CertificationState
from ccce.utils.provenance import build_manifest

EXPERIMENT_ID = "E02_IDENTIFICATION"
N_RANDOM_CASES = 200


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)

    rows = []
    hand_cases = [
        dict(desc="in-support, all criteria satisfied", is_parent=True, value=0.0,
             support=(-2, -1, 0, 1, 2), feasible=True, contract=True, safety=True, task_pres=True,
             expected=ValidityStatus.VALID),
        dict(desc="not a designated causal parent", is_parent=False, value=0.5,
             support=(-2, -1, 0, 1, 2), feasible=True, contract=True, safety=True, task_pres=True,
             expected=ValidityStatus.INVALID),
        dict(desc="outside pre-registered support", is_parent=True, value=7.0,
             support=(-2, -1, 0, 1, 2), feasible=True, contract=True, safety=True, task_pres=True,
             expected=ValidityStatus.UNIDENTIFIABLE),
        dict(desc="not task-preserving", is_parent=True, value=1.0,
             support=(-2, -1, 0, 1, 2), feasible=True, contract=True, safety=True, task_pres=False,
             expected=ValidityStatus.TASK_ALTERING),
        dict(desc="not safety-valid", is_parent=True, value=1.0,
             support=(-2, -1, 0, 1, 2), feasible=True, contract=True, safety=False, task_pres=True,
             expected=ValidityStatus.INVALID),
        dict(desc="near support edge -> borderline", is_parent=True, value=2.0,
             support=(-2, -1, 0, 1, 2), feasible=True, contract=True, safety=True, task_pres=True,
             expected=ValidityStatus.BORDERLINE),
    ]

    correct = 0
    for case in hand_cases:
        meta = classify_intervention(
            intervention_id=case["desc"], causal_variable="z_test", value=case["value"],
            is_designated_causal_parent=case["is_parent"], feasible=case["feasible"],
            contract_preserving=case["contract"], safety_valid=case["safety"],
            task_preserving=case["task_pres"], valid_support=case["support"],
        )
        # Allow the near-edge case a small numerical tolerance in matching.
        match = (meta.validity_status == case["expected"])
        correct += int(match)
        rows.append(dict(
            experiment_id=EXPERIMENT_ID, case_type="hand_constructed", description=case["desc"],
            value=case["value"], expected=case["expected"].value, actual=meta.validity_status.value,
            match=match,
        ))

    hand_accuracy = correct / len(hand_cases)

    # ---- Randomized battery: certification must never CERTIFY a non-VALID/BORDERLINE case
    false_certifications = 0
    status_counts = {s.value: 0 for s in ValidityStatus}
    cert_counts = {s.value: 0 for s in CertificationState}
    for i in range(N_RANDOM_CASES):
        is_parent = rng.random() > 0.15
        support = tuple(rng.choice([-2, -1, 0, 1, 2], size=rng.integers(2, 6), replace=False))
        # 60% of cases sample an in-support value (so VALID/BORDERLINE are
        # actually reachable); 40% sample a genuinely out-of-support
        # continuous value (so UNIDENTIFIABLE is also well represented).
        # This mirrors realistic practice: most queries target the
        # pre-registered grid, some probe outside it.
        if rng.random() < 0.6:
            value = float(rng.choice(support))
        else:
            value = float(rng.uniform(-6, 6))
            while any(abs(value - s) < 1e-9 for s in support):
                value = float(rng.uniform(-6, 6))
        feasible = rng.random() > 0.1
        contract = rng.random() > 0.1
        safety = rng.random() > 0.1
        task_pres = rng.random() > 0.15

        meta = classify_intervention(
            intervention_id=f"rand_{i}", causal_variable="z_rand", value=value,
            is_designated_causal_parent=is_parent, feasible=feasible, contract_preserving=contract,
            safety_valid=safety, task_preserving=task_pres, valid_support=support,
        )
        envelope_ok = rng.random() > 0.2
        result = certify(meta, envelope_satisfied=envelope_ok)

        status_counts[meta.validity_status.value] += 1
        cert_counts[result.state.value] += 1

        if result.state == CertificationState.CERTIFIED and meta.validity_status not in (
            ValidityStatus.VALID, ValidityStatus.BORDERLINE
        ):
            false_certifications += 1

        rows.append(dict(
            experiment_id=EXPERIMENT_ID, case_type="randomized", description=f"rand_{i}",
            value=value, expected="", actual=meta.validity_status.value,
            match=result.state.value,
        ))

    id_csv = output_dir / "identification.csv"
    with open(id_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    identification_coverage = (status_counts["VALID"] + status_counts["BORDERLINE"]) / N_RANDOM_CASES
    abstention_rate = status_counts["UNIDENTIFIABLE"] / N_RANDOM_CASES

    summary = dict(
        experiment_id=EXPERIMENT_ID,
        n_hand_constructed_cases=len(hand_cases),
        hand_constructed_accuracy=hand_accuracy,
        n_randomized_cases=N_RANDOM_CASES,
        false_certification_count=false_certifications,
        false_certification_rate=false_certifications / N_RANDOM_CASES,
        identification_coverage=identification_coverage,
        abstention_rate=abstention_rate,
        status_distribution=status_counts,
        certification_distribution=cert_counts,
    )
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    manifest = build_manifest(
        experiment_id=EXPERIMENT_ID, run_id=EXPERIMENT_ID, seed=2024,
        algorithm="rule_based_classifier", environment="n/a (unit-level logic test)",
        task_sequence="n/a", intervention_grid="randomized_battery",
        control_protocol="n/a", hyperparameters={"n_random_cases": N_RANDOM_CASES},
    )
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"[{EXPERIMENT_ID}] hand-constructed accuracy: {hand_accuracy:.2f} ({correct}/{len(hand_cases)})")
    print(f"[{EXPERIMENT_ID}] false_certification_rate: {summary['false_certification_rate']:.4f} "
          f"({false_certifications}/{N_RANDOM_CASES}) -- MUST be 0.0 by construction")
    print(f"[{EXPERIMENT_ID}] identification_coverage={identification_coverage:.3f} "
          f"abstention_rate={abstention_rate:.3f}")
    assert false_certifications == 0, "CRITICAL: certification logic certified a non-valid intervention!"


if __name__ == "__main__":
    out = Path(__file__).parent / "output"
    run(out)
