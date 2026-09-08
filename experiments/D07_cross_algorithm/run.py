# Author: Basil M. Alzboun

"""
D07_CROSS_ALGORITHM -- main_v4.tex Section 8.7.

Tests whether D04's Level D task-order finding (NOT SUPPORTED for PPO,
at both discovery n=10 and confirmation n=40 scale) is at least
qualitatively robust under a structurally different algorithm (SAC),
per Failure F (main_v4.tex Section 6.6): "attribution conclusions do not
survive a change in learning algorithm."

Scope reduction, documented transparently
--------------------------------------------
A direct timing measurement showed real Stable-Baselines3 SAC costs
~110s per 3-task sequence at this project's standard 1200-steps/task
budget -- roughly 18x slower than PPO's ~6s for the same sequence
(SAC is off-policy and updates its Q-networks far more frequently per
environment step than PPO's on-policy rollout-then-update cycle). Running
D04's full 10-seed discovery scale under SAC would cost ~55 minutes of
additional wall-clock time on this sandbox's single CPU core, competing
with other concurrently-running experiments. This experiment therefore
uses 5 seeds per sequence (not 10), explicitly documented as a
REDUCED-SCALE CROSS-ALGORITHM SCREENING, not a full discovery-stage
replication -- consistent with this project's established practice of
transparently scoping down under real compute constraints rather than
silently reducing N or hiding the reduction.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from stable_baselines3 import SAC

from ccce.environments.gym_wrapper import GymSyntheticPointMassEnv
from ccce.environments.synthetic_env import SyntheticPointMassEnv, default_task_set
from ccce.attribution.interventional import InterventionCondition, run_interventional_diagnosis
from ccce.utils.provenance import build_manifest
from ccce.utils.preregistration import write_confirmation_manifest

EXPERIMENT_ID = "D07_CROSS_ALGORITHM"
TASKS = default_task_set()
SEQUENCES: Dict[str, List[str]] = {
    "seq1_T1_first": ["T1", "T2", "T3"],
    "seq2_T1_second": ["T2", "T1", "T3"],
    "seq3_T1_last": ["T3", "T2", "T1"],
}
COMPETENCE_OF_INTEREST = "T1"
STEPS_PER_TASK = 1200
SEEDS = list(range(5))  # REDUCED from D04's 10 -- see module docstring
N_EVAL_EPISODES = 8


def evaluate_t1_competence(model: SAC, seed: int) -> float:
    task = TASKS[COMPETENCE_OF_INTEREST]
    successes, safeties, effs, times = [], [], [], []
    for ep in range(N_EVAL_EPISODES):
        env = SyntheticPointMassEnv(task)
        rng = np.random.default_rng(seed * 1000 + ep)
        obs = env.reset(rng)
        done = False
        steps = 0
        total_cost = 0.0
        collided = False
        while not done:
            action, _ = model.predict(obs.astype(np.float32), deterministic=True)
            obs, reward, done, info = env.step(np.asarray(action, dtype=np.float64), rng)
            total_cost += float(np.sum(np.asarray(action) ** 2))
            collided = collided or info["collision"]
            steps += 1
        successes.append(float(info["success"]))
        safeties.append(float(1.0 - collided))
        effs.append(1.0 / (1.0 + total_cost / max(steps, 1)))
        times.append(1.0 - steps / task.max_steps + 1.0)
    return float(0.4 * np.mean(successes) + 0.3 * np.mean(safeties) + 0.1 * np.mean(effs) + 0.1 * np.mean(times) + 0.1)


def train_sequence(sequence: List[str], seed: int) -> float:
    env = GymSyntheticPointMassEnv(TASKS[sequence[0]])
    model = SAC("MlpPolicy", env, verbose=0, learning_starts=100, seed=seed)
    for task_id in sequence:
        model.set_env(GymSyntheticPointMassEnv(TASKS[task_id]))
        model.learn(total_timesteps=STEPS_PER_TASK, reset_num_timesteps=False)
    return evaluate_t1_competence(model, seed)


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    design = dict(
        experiment_id=EXPERIMENT_ID, factor="task_order", algorithm="SAC",
        conditions={cid: seq for cid, seq in SEQUENCES.items()},
        competence_of_interest=COMPETENCE_OF_INTEREST, seeds=SEEDS,
        steps_per_task=STEPS_PER_TASK, n_eval_episodes=N_EVAL_EPISODES,
        backend="real_stable_baselines3_SAC",
        primary_outcome="final_T1_competence_by_sequence_cross_algorithm_check",
        statistical_test="kruskal_wallis_omnibus_plus_holm_corrected_pairwise_ttests",
        scope_reduction_note="5 seeds/sequence (not D04's 10) due to SAC's ~18x higher "
                              "per-seed cost measured directly before this run; documented "
                              "screening-scale check, not full discovery-stage replication.",
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
    conditions = []
    for seq_id, sequence in SEQUENCES.items():
        ckpt_path = output_dir / f"_checkpoint_{seq_id}.json"
        seq_values: List[float] = []
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                seq_values = json.load(f)
            print(f"[{EXPERIMENT_ID}] {seq_id} RESUMING from checkpoint ({len(seq_values)}/{len(SEEDS)} done)")

        remaining_seeds = SEEDS[len(seq_values):]
        for seed in remaining_seeds:
            competence = train_sequence(sequence, seed)
            seq_values.append(competence)
            rows.append(dict(experiment_id=EXPERIMENT_ID, sequence_id=seq_id,
                              sequence="->".join(sequence), seed=seed, final_t1_competence=competence))
            print(f"[{EXPERIMENT_ID}] {seq_id} ({'->'.join(sequence)}) seed={seed}  "
                  f"final_T1_competence={competence:.4f}")
            with open(ckpt_path, "w") as f:
                json.dump(seq_values, f)

        conditions.append(InterventionCondition(
            condition_id=seq_id, description="->".join(sequence), replicate_values=seq_values,
        ))

    if rows:
        csv_path = output_dir / "evaluation.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    result = run_interventional_diagnosis("task_order", conditions)

    summary = dict(
        experiment_id=EXPERIMENT_ID, algorithm="SAC", factor_name=result.factor_name,
        omnibus_test=result.omnibus_test, omnibus_p_value=result.omnibus_p_value,
        levene_p_value=result.levene_variance_test_p_value,
        conditions={cid: dict(mean=c.mean, std=c.std, n=c.n_replicates) for cid, c in result.conditions.items()},
        pairwise_contrasts=[
            dict(condition_a=c.condition_a, condition_b=c.condition_b,
                 mean_difference=c.mean_difference, cohens_d=c.cohens_d,
                 p_value_uncorrected=c.p_value_uncorrected,
                 holm_significant_at_0_05=c.holm_significant_at_0_05)
            for c in result.pairwise_contrasts
        ],
        any_holm_significant_contrast=result.any_holm_significant_contrast,
        cross_algorithm_agrees_with_ppo=(not result.any_holm_significant_contrast),  # D04-PPO found NOT SUPPORTED
        confirmation_manifest_hash=lock_hash,
    )
    with open(output_dir / "d07_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    manifest = build_manifest(
        experiment_id=EXPERIMENT_ID, run_id=EXPERIMENT_ID, seed=-1,
        algorithm="real_stable_baselines3_SAC", environment="level2_synthetic_pointmass_gym_wrapped",
        task_sequence="3 pre-registered sequences over {T1,T2,T3}", intervention_grid="n/a",
        control_protocol="task_order (Category B intervenable factor)",
        hyperparameters={"steps_per_task": STEPS_PER_TASK, "seeds": SEEDS},
    )
    manifest["confirmation_manifest_hash"] = lock_hash
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print()
    for line in result.summary_lines():
        print(f"[{EXPERIMENT_ID}] {line}")
    print(f"[{EXPERIMENT_ID}] CROSS-ALGORITHM CHECK (SAC vs D04's PPO finding): "
          f"{'AGREES (both NOT SUPPORTED)' if summary['cross_algorithm_agrees_with_ppo'] else 'DISAGREES -- Failure F candidate'}")


if __name__ == "__main__":
    out = Path(__file__).parent / "output"
    run(out)
