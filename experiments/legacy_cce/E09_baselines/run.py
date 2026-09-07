# Author: Basil M. Alzboun

"""
E09_CL_BASELINES -- Master Prompt V3 Section 14.

--stage discovery:    10 seeds, ~1200 steps/task, 3 baselines (naive,
                       replay, EWC), PLUS a cheap order-confound check
                       (Section 11) comparing Sequence A (T1->T2->T3)
                       against Sequence B (T2->T1->T3) at discovery scale
                       only.
--stage confirmation: 30 FRESH seeds (1000-1029, disjoint from
                       discovery), 10x the discovery training budget
                       (~12,000 steps/task), ALL 5 baselines (naive,
                       replay, EWC, distillation, UPGD) on Sequence A.
                       Pre-registration lock file written before running
                       (Section 57).

Reports BWT on T1 after training through the full sequence, for each
baseline. See docs/DEVIATIONS.md for the from-scratch-NumPy-PPO
substitution this all sits on top of, and for why the training budget
here (even at confirmation scale) remains far below main_v2.tex's
2e6-step specification.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats as _stats

from ccce.environments.synthetic_env import SyntheticPointMassEnv, default_task_set
from ccce.learners.numpy_ppo import LinearGaussianPolicy, PPOHyperparameters, train_policy
from ccce.baselines.continual_methods import (
    EWCState, ReplayBuffer, train_with_ewc, train_with_replay, train_with_distillation, train_with_upgd,
)
from ccce.competence.contract import evaluate_competence
from ccce.statistics.inference import holm_correction
from ccce.utils.provenance import build_manifest
from ccce.utils.preregistration import write_confirmation_manifest

EXPERIMENT_ID = "E09_CL_BASELINES"
TASKS = default_task_set()
SEQUENCE_A = ["T1", "T2", "T3"]
SEQUENCE_B = ["T2", "T1", "T3"]  # order-confound check (Section 11): T1 learned second, not first


@dataclass
class StageConfig:
    seeds: List[int]
    hp: PPOHyperparameters
    baselines: List[str]


STAGES = {
    "discovery": StageConfig(
        seeds=list(range(10)),
        hp=PPOHyperparameters(total_env_steps=1200, rollout_length=60, n_epochs=3, learning_rate=0.04),
        baselines=["naive", "replay", "ewc"],
    ),
    "confirmation": StageConfig(
        seeds=list(range(1000, 1030)),
        hp=PPOHyperparameters(total_env_steps=12000, rollout_length=60, n_epochs=3, learning_rate=0.04),
        baselines=["naive", "replay", "ewc", "distillation", "upgd"],
    ),
}


def run_naive(seed: int, sequence: List[str], hp: PPOHyperparameters):
    rng = np.random.default_rng(seed)
    policy = LinearGaussianPolicy.initialize(4, 2, rng, log_std_init=-0.6)
    q_after_t1 = None
    for tid in sequence:
        train_policy(lambda t=tid: SyntheticPointMassEnv(TASKS[t]), policy, rng, hp)
        if tid == "T1":
            q_after_t1 = evaluate_competence(TASKS["T1"], policy, 6, rng).scalar_competence()
    q_final = evaluate_competence(TASKS["T1"], policy, 6, rng).scalar_competence()
    return q_after_t1, q_final


def run_replay(seed: int, sequence: List[str], hp: PPOHyperparameters):
    rng = np.random.default_rng(seed)
    policy = LinearGaussianPolicy.initialize(4, 2, rng, log_std_init=-0.6)
    buf = ReplayBuffer(capacity=1000)
    q_after_t1 = None
    for tid in sequence:
        train_with_replay(lambda t=tid: SyntheticPointMassEnv(TASKS[t]), policy, rng, hp, buf)
        if tid == "T1":
            q_after_t1 = evaluate_competence(TASKS["T1"], policy, 6, rng).scalar_competence()
    q_final = evaluate_competence(TASKS["T1"], policy, 6, rng).scalar_competence()
    return q_after_t1, q_final


def run_ewc(seed: int, sequence: List[str], hp: PPOHyperparameters):
    rng = np.random.default_rng(seed)
    policy = LinearGaussianPolicy.initialize(4, 2, rng, log_std_init=-0.6)
    ewc = EWCState(lambda_ewc=300.0)
    q_after_t1 = None
    for tid in sequence:
        params_before = policy.flat_params().copy()
        probe_grads = []
        for _ in range(3):
            p0 = policy.flat_params().copy()
            train_policy(lambda t=tid: SyntheticPointMassEnv(TASKS[t]), policy, rng,
                         PPOHyperparameters(total_env_steps=60, rollout_length=60, n_epochs=1))
            probe_grads.append(policy.flat_params() - p0)
        policy.set_flat_params(params_before)
        train_with_ewc(lambda t=tid: SyntheticPointMassEnv(TASKS[t]), policy, rng, hp, ewc)
        if tid == "T1":
            q_after_t1 = evaluate_competence(TASKS["T1"], policy, 6, rng).scalar_competence()
    q_final = evaluate_competence(TASKS["T1"], policy, 6, rng).scalar_competence()
    return q_after_t1, q_final


def run_distillation(seed: int, sequence: List[str], hp: PPOHyperparameters):
    rng = np.random.default_rng(seed)
    policy = LinearGaussianPolicy.initialize(4, 2, rng, log_std_init=-0.6)
    q_after_t1 = None
    for tid in sequence:
        teacher = policy.copy()
        train_with_distillation(lambda t=tid: SyntheticPointMassEnv(TASKS[t]), policy, teacher, rng, hp)
        if tid == "T1":
            q_after_t1 = evaluate_competence(TASKS["T1"], policy, 6, rng).scalar_competence()
    q_final = evaluate_competence(TASKS["T1"], policy, 6, rng).scalar_competence()
    return q_after_t1, q_final


def run_upgd(seed: int, sequence: List[str], hp: PPOHyperparameters):
    rng = np.random.default_rng(seed)
    policy = LinearGaussianPolicy.initialize(4, 2, rng, log_std_init=-0.6)
    utility_ema = None
    q_after_t1 = None
    for tid in sequence:
        result = train_with_upgd(lambda t=tid: SyntheticPointMassEnv(TASKS[t]), policy, rng, hp, utility_ema)
        utility_ema = result["utility_ema"]
        if tid == "T1":
            q_after_t1 = evaluate_competence(TASKS["T1"], policy, 6, rng).scalar_competence()
    q_final = evaluate_competence(TASKS["T1"], policy, 6, rng).scalar_competence()
    return q_after_t1, q_final


RUNNERS = {
    "naive": run_naive, "replay": run_replay, "ewc": run_ewc,
    "distillation": run_distillation, "upgd": run_upgd,
}


def run(stage: str, output_dir: Path) -> None:
    cfg = STAGES[stage]
    output_dir.mkdir(parents=True, exist_ok=True)

    lock_hash = None
    lock_path = output_dir / "confirmation_manifest.yaml"
    if stage == "confirmation":
        design = dict(
            experiment_id=EXPERIMENT_ID, stage=stage, seeds=cfg.seeds,
            total_env_steps_per_task=cfg.hp.total_env_steps, baselines=cfg.baselines,
            sequence=SEQUENCE_A, primary_outcome="mean_BWT_on_T1_per_baseline",
            multiple_testing_correction="holm_across_baseline_pairs_vs_naive",
            stopping_rule="fixed_n=30_no_early_stopping",
        )
        if lock_path.exists():
            # Resuming an interrupted confirmation run (Section 6): the
            # lock must NOT be rewritten (that would defeat its purpose).
            with open(lock_path) as f:
                existing = f.read()
            lock_hash = __import__("hashlib").sha256(existing.encode()).hexdigest()
            print(f"[{EXPERIMENT_ID}] RESUMING: found existing locked manifest "
                  f"(SHA-256: {lock_hash}); not rewriting it.")
        else:
            lock_hash = write_confirmation_manifest(lock_path, design)
            print(f"[{EXPERIMENT_ID}] LOCKED confirmation manifest SHA-256: {lock_hash}")

    # ---- Per-baseline checkpointing (Section 6): each baseline's full
    # seed sweep is written to its own CSV immediately on completion, so
    # an interrupted run (crash, timeout, kill) resumes from the last
    # completed baseline rather than losing all progress.
    rows = []
    for name in cfg.baselines:
        baseline_ckpt = output_dir / f"_checkpoint_{name}.csv"
        if baseline_ckpt.exists():
            with open(baseline_ckpt) as f:
                existing_rows = list(csv.DictReader(f))
            for r in existing_rows:
                r["seed"] = int(r["seed"])
                for k in ("q_after_t1", "q_final_on_t1", "bwt"):
                    r[k] = float(r[k])
            rows.extend(existing_rows)
            print(f"[{EXPERIMENT_ID}] baseline={name:12s} RESUMED from checkpoint "
                  f"({len(existing_rows)} seeds already done)")
            continue

        fn = RUNNERS[name]
        baseline_rows = []
        for seed in cfg.seeds:
            q_after_t1, q_final = fn(seed, SEQUENCE_A, cfg.hp)
            bwt = q_final - q_after_t1
            row = dict(experiment_id=EXPERIMENT_ID, baseline=name, sequence="A", seed=seed,
                       q_after_t1=q_after_t1, q_final_on_t1=q_final, bwt=bwt)
            baseline_rows.append(row)
            print(f"[{EXPERIMENT_ID}] baseline={name:12s} seed={seed}  BWT(T1)={bwt:+.4f}")

        # Checkpoint immediately after this baseline finishes.
        with open(baseline_ckpt, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(baseline_rows[0].keys()))
            writer.writeheader()
            writer.writerows(baseline_rows)
        rows.extend(baseline_rows)

    # ---- Order-confound check (Section 11), discovery stage only ----------
    order_rows = []
    if stage == "discovery":
        for name in ["naive", "ewc"]:  # cheap subset, both sequences
            fn = RUNNERS[name]
            for seed in cfg.seeds:
                qa, qfa = fn(seed, SEQUENCE_A, cfg.hp)
                qb, qfb = fn(seed, SEQUENCE_B, cfg.hp)
                order_rows.append(dict(
                    baseline=name, seed=seed,
                    bwt_sequence_A=qfa - qa, bwt_sequence_B=qfb - qb,
                ))
        order_csv = output_dir / "order_confound_check.csv"
        with open(order_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(order_rows[0].keys()))
            writer.writeheader()
            writer.writerows(order_rows)
        print(f"[{EXPERIMENT_ID}] Order-confound check (Sequence A: T1->T2->T3 vs "
              f"Sequence B: T2->T1->T3) written to {order_csv}")

    csv_path = output_dir / "diagnostics_by_baseline.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    naive_bwts = [r["bwt"] for r in rows if r["baseline"] == "naive"]
    p_values, names_tested = [], []
    for name in cfg.baselines:
        bwts = [r["bwt"] for r in rows if r["baseline"] == name]
        n_outliers = int(np.sum(np.abs(np.array(bwts) - np.median(bwts)) > 3 * (np.std(bwts) + 1e-9)))
        summary[name] = dict(
            mean_bwt=float(np.mean(bwts)), std_bwt=float(np.std(bwts)), n_seeds=len(bwts),
            n_outlier_seeds_gt_3std=n_outliers,
        )
        if name != "naive" and stage == "confirmation":
            diff = np.array(bwts) - np.array(naive_bwts)
            if diff.std(ddof=1) > 0:
                tstat = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
                p = float(2 * (1 - _stats.t.cdf(abs(tstat), df=len(diff) - 1)))
            else:
                p = 1.0
            p_values.append(p)
            names_tested.append(name)

    if p_values:
        reject = holm_correction(p_values, alpha=0.05)
        for name, r in zip(names_tested, reject):
            summary[name]["holm_significant_vs_naive_at_0.05"] = r

    with open(output_dir / "summary.json", "w") as f:
        json.dump(dict(
            summary_by_baseline=summary,
            confirmation_manifest_hash=lock_hash,
            caveat=(
                f"{'10' if stage == 'discovery' else '30'} seeds, "
                f"{cfg.hp.total_env_steps} steps/task ({stage} stage)."
            ),
        ), f, indent=2)

    manifest = build_manifest(
        experiment_id=EXPERIMENT_ID, run_id=f"{EXPERIMENT_ID}_{stage}", seed=-1,
        algorithm=",".join(cfg.baselines) + " (numpy)", environment="level2_synthetic_pointmass",
        task_sequence="->".join(SEQUENCE_A), intervention_grid="n/a",
        control_protocol="n/a (baseline comparison)",
        hyperparameters={"total_env_steps_per_task": cfg.hp.total_env_steps, "seeds": cfg.seeds},
    )
    manifest["confirmation_manifest_hash"] = lock_hash
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n[{EXPERIMENT_ID}] Summary (mean BWT on T1, {stage} stage, n={len(cfg.seeds)} seeds, "
          f"budget={cfg.hp.total_env_steps}):")
    for name, s in summary.items():
        extra = f" [Holm-sig vs naive: {s['holm_significant_vs_naive_at_0.05']}]" if "holm_significant_vs_naive_at_0.05" in s else ""
        print(f"  {name:12s}: {s['mean_bwt']:+.4f} +- {s['std_bwt']:.4f} "
              f"({s['n_outlier_seeds_gt_3std']} outlier seeds){extra}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["discovery", "confirmation"], default="discovery")
    args = parser.parse_args()
    out = Path(__file__).parent / "output" / args.stage
    run(args.stage, out)
