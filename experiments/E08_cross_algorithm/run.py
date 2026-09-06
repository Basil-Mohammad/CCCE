"""
E08_CROSS_ALGORITHM -- Master Prompt V3 Section 13.

FIRST GENUINE IMPLEMENTATION (previously NOT_RUN in this project; see
docs/DEVIATIONS.md for the full history: PyTorch/Stable-Baselines3 were
confirmed impossible to install in this sandbox in two earlier sessions,
then became installable in a third session for reasons not fully
understood -- possibly a change in the underlying sandbox image or
cache state. This experiment uses the now-functional PyTorch +
Stable-Baselines3 stack directly; no NumPy substitute is used here).

Design: naive sequential training on T1 -> T2 -> T3 (matching E09's
"naive" baseline exactly, for direct comparability), once with real SB3
PPO and once with real SB3 SAC, both evaluated identically. Reports BWT
on T1 after the full sequence for each algorithm, to test whether the
qualitative pattern found under PPO (near-zero mean BWT, with occasional
large-magnitude outlier seeds) is "cross-algorithm robust within the
evaluated settings" (Section 19) -- not algorithm-independence in
general.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from scipy import stats as _stats

from ccce.environments.gym_wrapper import GymSyntheticPointMassEnv
from ccce.environments.synthetic_env import default_task_set
from ccce.utils.provenance import build_manifest
from ccce.utils.preregistration import write_confirmation_manifest

EXPERIMENT_ID = "E08_CROSS_ALGORITHM"
TASKS = default_task_set()
SEQUENCE = ["T1", "T2", "T3"]

STEPS_PER_TASK = 1200
SEEDS = list(range(10))
N_EVAL_EPISODES = 6


def evaluate_on_t1(model, seed: int) -> float:
    env = GymSyntheticPointMassEnv(TASKS["T1"])
    total = 0.0
    for ep in range(N_EVAL_EPISODES):
        obs, _ = env.reset(seed=seed * 1000 + ep)
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated
        total += ep_return
    return total / N_EVAL_EPISODES


def run_algorithm(algo_name: str, seed: int):
    AlgoCls = {"PPO": PPO, "SAC": SAC}[algo_name]
    kwargs = {"n_steps": 64, "batch_size": 32} if algo_name == "PPO" else {"learning_starts": 100}

    env = GymSyntheticPointMassEnv(TASKS[SEQUENCE[0]])
    model = AlgoCls("MlpPolicy", env, verbose=0, seed=seed, **kwargs)

    q_after_t1 = None
    for tid in SEQUENCE:
        model.set_env(GymSyntheticPointMassEnv(TASKS[tid]))
        model.learn(total_timesteps=STEPS_PER_TASK, reset_num_timesteps=False)
        if tid == "T1":
            q_after_t1 = evaluate_on_t1(model, seed)

    q_final = evaluate_on_t1(model, seed)
    return q_after_t1, q_final


def run(output_dir: Path) -> None:
    import torch
    output_dir.mkdir(parents=True, exist_ok=True)

    design = dict(
        experiment_id=EXPERIMENT_ID, algorithms=["PPO", "SAC"], sequence=SEQUENCE,
        seeds=SEEDS, steps_per_task=STEPS_PER_TASK, n_eval_episodes=N_EVAL_EPISODES,
        primary_outcome="mean_BWT_on_T1_per_algorithm",
        backend=f"real_stable_baselines3 (torch {torch.__version__})",
    )
    lock_path = output_dir / "confirmation_manifest.yaml"
    lock_hash = write_confirmation_manifest(lock_path, design)
    print(f"[{EXPERIMENT_ID}] Design manifest SHA-256: {lock_hash}")
    print(f"[{EXPERIMENT_ID}] Backend: real Stable-Baselines3 (torch {torch.__version__})")

    rows = []
    for algo_name in ["PPO", "SAC"]:
        for seed in SEEDS:
            q_after, q_final = run_algorithm(algo_name, seed)
            bwt = q_final - q_after
            rows.append(dict(experiment_id=EXPERIMENT_ID, algorithm=algo_name, seed=seed,
                              q_after_t1=q_after, q_final_on_t1=q_final, bwt=bwt))
            print(f"[{EXPERIMENT_ID}] algorithm={algo_name:4s} seed={seed}  BWT(T1)={bwt:+.4f}")

    csv_path = output_dir / "diagnostics_by_algorithm.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ppo_bwts = [r["bwt"] for r in rows if r["algorithm"] == "PPO"]
    sac_bwts = [r["bwt"] for r in rows if r["algorithm"] == "SAC"]

    diff = np.array(sac_bwts) - np.array(ppo_bwts)
    if diff.std(ddof=1) > 0:
        tstat = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
        p_value = float(2 * (1 - _stats.t.cdf(abs(tstat), df=len(diff) - 1)))
    else:
        p_value = 1.0

    summary = dict(
        experiment_id=EXPERIMENT_ID,
        ppo=dict(mean_bwt=float(np.mean(ppo_bwts)), std_bwt=float(np.std(ppo_bwts)), n_seeds=len(ppo_bwts)),
        sac=dict(mean_bwt=float(np.mean(sac_bwts)), std_bwt=float(np.std(sac_bwts)), n_seeds=len(sac_bwts)),
        ppo_vs_sac_paired_diff_p_value=p_value,
        design_manifest_hash=lock_hash,
        backend="real_stable_baselines3",
        caveat=(
            f"{len(SEEDS)} seeds, {STEPS_PER_TASK} steps/task -- a modest but "
            "REAL deep-RL training budget (not a NumPy substitute). Still far "
            "below main_v2.tex's 2e6-step specification. See docs/DEVIATIONS.md."
        ),
    )
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    manifest = build_manifest(
        experiment_id=EXPERIMENT_ID, run_id=EXPERIMENT_ID, seed=-1,
        algorithm="PPO,SAC (real stable_baselines3)", environment="level2_synthetic_pointmass_gym_wrapped",
        task_sequence="->".join(SEQUENCE), intervention_grid="n/a",
        control_protocol="n/a (cross-algorithm comparison)",
        hyperparameters={"steps_per_task": STEPS_PER_TASK, "seeds": SEEDS},
    )
    manifest["design_manifest_hash"] = lock_hash
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n[{EXPERIMENT_ID}] Summary (mean BWT on T1, n={len(SEEDS)} seeds, "
          f"budget={STEPS_PER_TASK} steps/task, REAL SB3 backend):")
    print(f"  PPO: {summary['ppo']['mean_bwt']:+.4f} +- {summary['ppo']['std_bwt']:.4f}")
    print(f"  SAC: {summary['sac']['mean_bwt']:+.4f} +- {summary['sac']['std_bwt']:.4f}")
    print(f"  Paired difference p-value (uncorrected): {p_value:.4f}")


if __name__ == "__main__":
    out = Path(__file__).parent / "output"
    run(out)
