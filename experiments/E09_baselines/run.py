"""
E09_CL_BASELINES -- Master Prompt V3 Section 14, reduced scale.

Compares naive sequential PPO, replay, and EWC (the three baselines with
the most mature implementations in this codebase) on the T1->T2->T3
sequence, reporting BWT on T1 after training through T3 for each method.
REDUCED SCALE: 10 seeds (meets Section 4's minimum), ~1200 steps/task (see docs/DEVIATIONS.md).
Distillation and UPGD are implemented in
src/ccce/baselines/continual_methods.py and unit-tested, but are not
included in this particular comparison run due to compute-time
constraints in this sandbox; this is stated explicitly rather than
silently omitted (Section 51: "prioritize ablations/baselines that test
scientific assumptions" under real time constraints).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ccce.environments.synthetic_env import SyntheticPointMassEnv, default_task_set
from ccce.learners.numpy_ppo import LinearGaussianPolicy, PPOHyperparameters, train_policy
from ccce.baselines.continual_methods import EWCState, ReplayBuffer, train_with_ewc, train_with_replay
from ccce.competence.contract import evaluate_competence
from ccce.utils.provenance import build_manifest

EXPERIMENT_ID = "E09_CL_BASELINES"
TASKS = default_task_set()
SEQUENCE = ["T1", "T2", "T3"]
SEEDS = list(range(10))  # N_dev = 10, matching Section 4's minimum
HP = PPOHyperparameters(total_env_steps=1200, rollout_length=60, n_epochs=3, learning_rate=0.04)


def run_naive(seed: int):
    rng = np.random.default_rng(seed)
    policy = LinearGaussianPolicy.initialize(4, 2, rng, log_std_init=-0.6)
    q_after_t1 = None
    for i, tid in enumerate(SEQUENCE):
        train_policy(lambda t=tid: SyntheticPointMassEnv(TASKS[t]), policy, rng, HP)
        if tid == "T1":
            q_after_t1 = evaluate_competence(TASKS["T1"], policy, 6, rng).scalar_competence()
    q_final_on_t1 = evaluate_competence(TASKS["T1"], policy, 6, rng).scalar_competence()
    return q_after_t1, q_final_on_t1


def run_replay(seed: int):
    rng = np.random.default_rng(seed)
    policy = LinearGaussianPolicy.initialize(4, 2, rng, log_std_init=-0.6)
    buf = ReplayBuffer(capacity=1000)
    q_after_t1 = None
    for tid in SEQUENCE:
        train_with_replay(lambda t=tid: SyntheticPointMassEnv(TASKS[t]), policy, rng, HP, buf)
        if tid == "T1":
            q_after_t1 = evaluate_competence(TASKS["T1"], policy, 6, rng).scalar_competence()
    q_final_on_t1 = evaluate_competence(TASKS["T1"], policy, 6, rng).scalar_competence()
    return q_after_t1, q_final_on_t1


def run_ewc(seed: int):
    rng = np.random.default_rng(seed)
    policy = LinearGaussianPolicy.initialize(4, 2, rng, log_std_init=-0.6)
    ewc = EWCState(lambda_ewc=300.0)
    q_after_t1 = None
    for tid in SEQUENCE:
        params_before = policy.flat_params().copy()
        probe_grads = []
        for _ in range(3):
            p0 = policy.flat_params().copy()
            train_policy(lambda t=tid: SyntheticPointMassEnv(TASKS[t]), policy, rng,
                         PPOHyperparameters(total_env_steps=60, rollout_length=60, n_epochs=1))
            probe_grads.append(policy.flat_params() - p0)
        policy.set_flat_params(params_before)
        train_with_ewc(lambda t=tid: SyntheticPointMassEnv(TASKS[t]), policy, rng, HP, ewc)
        if tid == "T1":
            q_after_t1 = evaluate_competence(TASKS["T1"], policy, 6, rng).scalar_competence()
    q_final_on_t1 = evaluate_competence(TASKS["T1"], policy, 6, rng).scalar_competence()
    return q_after_t1, q_final_on_t1


BASELINES = {"naive": run_naive, "replay": run_replay, "ewc": run_ewc}


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, fn in BASELINES.items():
        for seed in SEEDS:
            q_after_t1, q_final = fn(seed)
            bwt = q_final - q_after_t1
            rows.append(dict(experiment_id=EXPERIMENT_ID, baseline=name, seed=seed,
                              q_after_t1=q_after_t1, q_final_on_t1=q_final, bwt=bwt))
            print(f"[{EXPERIMENT_ID}] baseline={name:8s} seed={seed}  BWT(T1)={bwt:+.4f}")

    csv_path = output_dir / "diagnostics_by_baseline.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for name in BASELINES:
        bwts = [r["bwt"] for r in rows if r["baseline"] == name]
        summary[name] = dict(mean_bwt=float(np.mean(bwts)), std_bwt=float(np.std(bwts)), n_seeds=len(bwts))

    with open(output_dir / "summary.json", "w") as f:
        json.dump(dict(
            summary_by_baseline=summary,
            caveat="10 seeds (meets Section 4's minimum), ~1200 steps/task. Distillation and UPGD "
                   "baselines are implemented and unit-tested but not run in this "
                   "comparison due to sandbox compute-time constraints.",
        ), f, indent=2)

    manifest = build_manifest(
        experiment_id=EXPERIMENT_ID, run_id=EXPERIMENT_ID, seed=-1,
        algorithm="naive_ppo,replay_ppo,ewc_ppo (numpy)", environment="level2_synthetic_pointmass",
        task_sequence="T1->T2->T3", intervention_grid="n/a", control_protocol="n/a (baseline comparison)",
        hyperparameters={"total_env_steps_per_task": HP.total_env_steps, "seeds": SEEDS},
    )
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n[{EXPERIMENT_ID}] Summary (mean BWT on T1 after T1->T2->T3, n={len(SEEDS)} seeds):")
    for name, s in summary.items():
        print(f"  {name:8s}: {s['mean_bwt']:+.4f} +- {s['std_bwt']:.4f}")


if __name__ == "__main__":
    out = Path(__file__).parent / "output"
    run(out)
