# Author: Basil M. Alzboun

"""
D06_STANDARD_BENCHMARKS -- main_v4.tex Section 8.6.

The FIRST real experiment run on a standard MuJoCo continuous-control
benchmark in this project (all prior experiments used the custom
synthetic point-mass environment). Prior work only smoke-tested MuJoCo
(HalfCheetah-v5, Hopper-v5 reset/step); this experiment closes that gap
with a genuine, executed diagnostic test.

Design
--------
Environment: HalfCheetah-v5 (real MuJoCo physics via Gymnasium).
Designated environmental causal factor: gravity magnitude
(`env.unwrapped.model.opt.gravity`), scaled by a pre-registered grid
{0.5, 0.75, 1.0, 1.5, 2.0} relative to Earth gravity (9.81 m/s^2), with
1.0 the nominal condition. This is a valid intervention under this
project's own criteria (main_v4.tex Section 5.1): it acts directly on a
designated causal parent of the physics simulation, not on the agent's
observation, and does not alter the task's goal or reward semantics.

For each of N independent seeds, a real Stable-Baselines3 PPO policy is
trained under NOMINAL gravity only (no intervention during training,
matching the standard "train once, evaluate under many conditions"
protocol used throughout this project's synthetic-environment
experiments), then evaluated -- using the SAME trained policy -- under
every point in the gravity intervention grid.

This directly instantiates a Level D (Interventional Diagnosis) test:
gravity is genuinely, deliberately manipulated across five conditions
with independent seeds as replicates, and
ccce.attribution.interventional.run_interventional_diagnosis is applied
exactly as it was for the synthetic-environment task-order experiment
(D04), giving Kruskal-Wallis omnibus test, Holm-corrected pairwise
contrasts, Cohen's d effect sizes, and Levene's variance-equality test,
on a real standard benchmark rather than a custom environment.

Scale
-------
10 seeds (this project's discovery-stage minimum), 8,000 training steps
per seed (a small budget relative to typical HalfCheetah training
regimes of 1-3 million steps; this is explicitly a diagnostic-
methodology validation, not a claim of near-optimal HalfCheetah
locomotion), 3 evaluation episodes x 300 steps per gravity condition.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from ccce.attribution.interventional import InterventionCondition, run_interventional_diagnosis
from ccce.utils.provenance import build_manifest
from ccce.utils.preregistration import write_confirmation_manifest

EXPERIMENT_ID = "D06_STANDARD_BENCHMARKS"
ENV_ID = "HalfCheetah-v5"
EARTH_GRAVITY = 9.81
GRAVITY_SCALE_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)
STEPS = 8000
SEEDS = list(range(10))
N_EVAL_EPISODES = 3
MAX_EVAL_STEPS = 300


def train_policy(seed: int) -> PPO:
    env = gym.make(ENV_ID)
    model = PPO("MlpPolicy", env, verbose=0, n_steps=256, batch_size=64, seed=seed)
    model.learn(total_timesteps=STEPS)
    return model


def evaluate_under_gravity(model: PPO, gravity_scale: float, seed: int) -> float:
    eval_env = gym.make(ENV_ID)
    returns = []
    for ep in range(N_EVAL_EPISODES):
        obs, _ = eval_env.reset(seed=seed * 1000 + ep)
        eval_env.unwrapped.model.opt.gravity[:] = np.array([0.0, 0.0, -EARTH_GRAVITY]) * gravity_scale
        total_r = 0.0
        for _ in range(MAX_EVAL_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(action)
            total_r += float(r)
            if term or trunc:
                break
        returns.append(total_r)
    eval_env.close()
    return float(np.mean(returns))


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    design = dict(
        experiment_id=EXPERIMENT_ID, environment=ENV_ID, factor="gravity_magnitude",
        gravity_scale_grid=list(GRAVITY_SCALE_GRID), seeds=SEEDS, training_steps=STEPS,
        n_eval_episodes=N_EVAL_EPISODES, max_eval_steps=MAX_EVAL_STEPS,
        backend="real_stable_baselines3_mujoco",
        primary_outcome="mean_episode_return_by_gravity_condition",
        statistical_test="kruskal_wallis_omnibus_plus_holm_corrected_pairwise_ttests_plus_levene",
    )
    lock_path = output_dir / "confirmation_manifest.yaml"
    if lock_path.exists():
        with open(lock_path) as f:
            existing = f.read()
        import hashlib
        lock_hash = hashlib.sha256(existing.encode()).hexdigest()
        print(f"[{EXPERIMENT_ID}] RESUMING with existing locked manifest SHA-256: {lock_hash}")
    else:
        lock_hash = write_confirmation_manifest(lock_path, design)
        print(f"[{EXPERIMENT_ID}] LOCKED manifest SHA-256: {lock_hash}")

    rows = []
    returns_by_gravity: Dict[float, List[float]] = {g: [] for g in GRAVITY_SCALE_GRID}

    for seed in SEEDS:
        ckpt_path = output_dir / f"_checkpoint_seed{seed}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                seed_result = json.load(f)
            print(f"[{EXPERIMENT_ID}] seed={seed} RESUMED from checkpoint")
        else:
            model = train_policy(seed)
            seed_result = {}
            for g in GRAVITY_SCALE_GRID:
                ret = evaluate_under_gravity(model, g, seed)
                seed_result[str(g)] = ret
            with open(ckpt_path, "w") as f:
                json.dump(seed_result, f)
            print(f"[{EXPERIMENT_ID}] seed={seed} " +
                  " ".join(f"g={g}:{seed_result[str(g)]:.1f}" for g in GRAVITY_SCALE_GRID))

        for g in GRAVITY_SCALE_GRID:
            val = seed_result[str(g)]
            returns_by_gravity[g].append(val)
            rows.append(dict(experiment_id=EXPERIMENT_ID, seed=seed, gravity_scale=g, mean_return=val))

    csv_path = output_dir / "evaluation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    conditions = [
        InterventionCondition(f"gravity_{g}", f"gravity_scale={g}", returns_by_gravity[g])
        for g in GRAVITY_SCALE_GRID
    ]
    result = run_interventional_diagnosis("gravity_magnitude", conditions)

    summary = dict(
        experiment_id=EXPERIMENT_ID, environment=ENV_ID,
        factor_name=result.factor_name,
        omnibus_test=result.omnibus_test, omnibus_p_value=result.omnibus_p_value,
        levene_p_value=result.levene_variance_test_p_value,
        variance_effect_detected=result.variance_effect_detected,
        conditions={
            cid: dict(mean=c.mean, std=c.std, n=c.n_replicates)
            for cid, c in result.conditions.items()
        },
        pairwise_contrasts=[
            dict(condition_a=c.condition_a, condition_b=c.condition_b,
                 mean_difference=c.mean_difference, cohens_d=c.cohens_d,
                 p_value_uncorrected=c.p_value_uncorrected,
                 holm_significant_at_0_05=c.holm_significant_at_0_05)
            for c in result.pairwise_contrasts
        ],
        any_holm_significant_contrast=result.any_holm_significant_contrast,
        level_d_causal_claim_supported=result.any_holm_significant_contrast,
        confirmation_manifest_hash=lock_hash,
    )
    with open(output_dir / "d06_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    manifest = build_manifest(
        experiment_id=EXPERIMENT_ID, run_id=EXPERIMENT_ID, seed=-1,
        algorithm="real_stable_baselines3_PPO", environment=ENV_ID,
        task_sequence="n/a (single-task, evaluated under intervention grid)",
        intervention_grid=list(GRAVITY_SCALE_GRID), control_protocol="n/a",
        hyperparameters={"training_steps": STEPS, "seeds": SEEDS},
    )
    manifest["confirmation_manifest_hash"] = lock_hash
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print()
    for line in result.summary_lines():
        print(f"[{EXPERIMENT_ID}] {line}")
    print(f"[{EXPERIMENT_ID}] LEVEL D CAUSAL CLAIM (gravity_magnitude, real MuJoCo): "
          f"{'SUPPORTED' if result.any_holm_significant_contrast else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    out = Path(__file__).parent / "output"
    run(out)
