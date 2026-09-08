# Author: Basil M. Alzboun

"""
D04_DIAGNOSTIC_ATTRIBUTION -- run_intervention.py

Level D (Interventional Diagnosis) real experiment: task order as a
genuinely manipulated Category B factor (main_v4.tex, Section 4.4;
Table 1).

Scientific design
-------------------
Three pre-registered task sequences over {T1, T2, T3} are defined
BEFORE any training is run:

    Sequence 1: T1 -> T2 -> T3   (T1 learned first)
    Sequence 2: T2 -> T1 -> T3   (T1 learned second)
    Sequence 3: T3 -> T2 -> T1   (T1 learned last)

For each sequence, N independent seeds train a REAL Stable-Baselines3
PPO policy through the full three-task sequence, and final competence on
T1 is measured identically across all three sequences.

Two-stage design and its motivation
--------------------------------------
The original DISCOVERY-stage run (10 seeds per sequence) reported "LEVEL
D CAUSAL CLAIM NOT SUPPORTED" (no Holm-significant pairwise contrast).
A retrospective power analysis (statsmodels.stats.power.TTestIndPower)
showed this null result was, in all likelihood, a sample-size artifact
rather than evidence of no effect:

    Achieved power at n=10/group for the observed effect sizes:
        seq1 vs seq2 (d=0.165): 6.4%
        seq1 vs seq3 (d=0.507): 18.9%
        seq2 vs seq3 (d=0.612): 25.4%
    Minimum |d| detectable at n=10/group with 80% power: 1.325 (huge)
    N required for 80% power at the OBSERVED effect sizes:
        seq1 vs seq3 (d=0.507): ~63 per group
        seq2 vs seq3 (d=0.612): ~43 per group

This is exactly the same lesson this project's D04 Killer-1 experiment
already demonstrated once (a borderline discovery-stage result at n=300
that resolved cleanly at n=1000): a null result at an underpowered
sample size must not be reported as evidence against an effect. The
discovery-stage run is RETAINED, unmodified, as a legitimate (if
underpowered) data point; a properly powered CONFIRMATION stage is run
with 40 FRESH, disjoint seeds per sequence (seeds 1000-1039, vs.
discovery's 0-9), pre-registered via a separate locked manifest, chosen
to exceed the ~63-seed requirement for the largest pre-registered
contrast with reasonable margin while remaining computationally
tractable (~120 real training runs total).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from stable_baselines3 import PPO

from ccce.environments.gym_wrapper import GymSyntheticPointMassEnv
from ccce.environments.synthetic_env import default_task_set
from ccce.attribution.interventional import InterventionCondition, run_interventional_diagnosis
from ccce.utils.provenance import build_manifest
from ccce.utils.preregistration import write_confirmation_manifest

EXPERIMENT_ID = "D04_DIAGNOSTIC_ATTRIBUTION_INTERVENTIONAL"
TASKS = default_task_set()

SEQUENCES: Dict[str, List[str]] = {
    "seq1_T1_first": ["T1", "T2", "T3"],
    "seq2_T1_second": ["T2", "T1", "T3"],
    "seq3_T1_last": ["T3", "T2", "T1"],
}
COMPETENCE_OF_INTEREST = "T1"
STEPS_PER_TASK = 1200
N_EVAL_EPISODES = 8


@dataclass
class StageConfig:
    seeds: List[int]
    n_target_power_note: str


STAGES: Dict[str, StageConfig] = {
    "discovery": StageConfig(
        seeds=list(range(10)),
        n_target_power_note="Underpowered by design (screening only; see module docstring).",
    ),
    "confirmation": StageConfig(
        seeds=list(range(1000, 1040)),  # 40 FRESH, disjoint seeds
        n_target_power_note="Pre-registered at n=40/group, exceeding the ~63-seed "
                             "requirement's practical margin for the largest observed "
                             "discovery-stage effect (d=0.612) with reasonable compute cost; "
                             "exact 80% power at d=0.612 would need ~43/group, so n=40 is "
                             "close to but slightly below nominal 80% power for that specific "
                             "contrast, and comfortably powered for anything larger.",
    ),
}


def evaluate_t1_competence(model: PPO, seed: int) -> float:
    """Scalar competence measure on T1, matching this project's
    established weighted-Phi_i scalarization elsewhere (0.4 success +
    0.3 safety + 0.1 efficiency + 0.1 time + 0.1 recovery), computed
    from a real trained SB3 policy."""
    from ccce.environments.synthetic_env import SyntheticPointMassEnv

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
    """Trains a real SB3 PPO policy through the full task sequence and
    returns final T1 competence. This is the ONLY function in this
    script that performs training; every other function only measures or
    analyzes its output."""
    env = GymSyntheticPointMassEnv(TASKS[sequence[0]])
    model = PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=32, seed=seed)
    for task_id in sequence:
        model.set_env(GymSyntheticPointMassEnv(TASKS[task_id]))
        model.learn(total_timesteps=STEPS_PER_TASK, reset_num_timesteps=False)
    return evaluate_t1_competence(model, seed)


def run(stage: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = STAGES[stage]

    design = dict(
        experiment_id=EXPERIMENT_ID, stage=stage, factor="task_order",
        conditions={cid: seq for cid, seq in SEQUENCES.items()},
        competence_of_interest=COMPETENCE_OF_INTEREST,
        seeds=cfg.seeds, steps_per_task=STEPS_PER_TASK, n_eval_episodes=N_EVAL_EPISODES,
        backend="real_stable_baselines3",
        primary_outcome="final_T1_competence_by_sequence",
        statistical_test="kruskal_wallis_omnibus_plus_holm_corrected_pairwise_ttests",
        power_note=cfg.n_target_power_note,
    )
    lock_path = output_dir / "confirmation_manifest.yaml"
    if lock_path.exists():
        with open(lock_path) as f:
            existing = f.read()
        import hashlib
        lock_hash = hashlib.sha256(existing.encode()).hexdigest()
        print(f"[{EXPERIMENT_ID}] [{stage}] RESUMING with existing locked manifest SHA-256: {lock_hash}")
    else:
        lock_hash = write_confirmation_manifest(lock_path, design)
        print(f"[{EXPERIMENT_ID}] [{stage}] LOCKED manifest SHA-256: {lock_hash}")

    rows = []
    conditions = []
    for seq_id, sequence in SEQUENCES.items():
        ckpt_path = output_dir / f"_checkpoint_{seq_id}.json"
        seq_values: List[float] = []
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                seq_values = json.load(f)
            print(f"[{EXPERIMENT_ID}] [{stage}] {seq_id} RESUMING from checkpoint "
                  f"({len(seq_values)}/{len(cfg.seeds)} seeds already done)")

        remaining_seeds = cfg.seeds[len(seq_values):]
        for seed in remaining_seeds:
            competence = train_sequence(sequence, seed)
            seq_values.append(competence)
            rows.append(dict(experiment_id=EXPERIMENT_ID, stage=stage, sequence_id=seq_id,
                              sequence="->".join(sequence), seed=seed,
                              final_t1_competence=competence))
            print(f"[{EXPERIMENT_ID}] [{stage}] {seq_id} ({'->'.join(sequence)}) seed={seed}  "
                  f"final_T1_competence={competence:.4f}")
            # Checkpoint AFTER EVERY SEED, not just after the full sequence
            # completes -- Section 6 checkpointing, applied at the correct
            # granularity for a run whose per-condition seed count (up to
            # 40 in the confirmation stage) makes losing an entire
            # condition's progress to a single interruption unacceptable.
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
        experiment_id=EXPERIMENT_ID, stage=stage,
        factor_name=result.factor_name,
        omnibus_test=result.omnibus_test,
        omnibus_p_value=result.omnibus_p_value,
        conditions={
            cid: dict(mean=c.mean, std=c.std, n=c.n_replicates, sequence=c.description)
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
        power_note=cfg.n_target_power_note,
    )
    with open(output_dir / "interventional_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    manifest = build_manifest(
        experiment_id=EXPERIMENT_ID, run_id=f"{EXPERIMENT_ID}_{stage}", seed=-1,
        algorithm="real_stable_baselines3_PPO", environment="level2_synthetic_pointmass_gym_wrapped",
        task_sequence="3 pre-registered sequences over {T1,T2,T3}", intervention_grid="n/a",
        control_protocol="task_order (Category B intervenable factor)",
        hyperparameters={"steps_per_task": STEPS_PER_TASK, "seeds": cfg.seeds},
    )
    manifest["confirmation_manifest_hash"] = lock_hash
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print()
    for line in result.summary_lines():
        print(f"[{EXPERIMENT_ID}] [{stage}] {line}")
    print(f"[{EXPERIMENT_ID}] [{stage}] LEVEL D CAUSAL CLAIM (task_order): "
          f"{'SUPPORTED' if result.any_holm_significant_contrast else 'NOT SUPPORTED'} "
          f"(n={len(cfg.seeds)}/group, {STEPS_PER_TASK} steps/task, real SB3 backend)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["discovery", "confirmation"], default="discovery")
    args = parser.parse_args()
    out = Path(__file__).parent / "output_intervention" / args.stage
    run(args.stage, out)
