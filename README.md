# CCCE: Counterfactual Cross-Competence Effects in Continual RL

Implementation of the experimental program specified for
"Counterfactual Cross-Competence Effects in Continual Reinforcement
Learning" (`main_v2.tex`).

## Read this first

- **`docs/DEVIATIONS.md`** -- every place this implementation deviates
  from the exact letter of the specification, why, and what the
  scientific consequence is. PyTorch/Stable-Baselines3/MuJoCo were
  confirmed **impossible** to install in two early sessions (disk space +
  network egress constraints), then became installable in a later
  session for reasons not fully understood -- see Section 10 for the
  full, transparent account. Only `E08_cross_algorithm` currently uses
  the real deep-RL backend; every other experiment still uses the
  from-scratch NumPy substitute for PPO described there.
- **`FINAL_EXPERIMENTAL_REPORT.md`** -- the honest, pre-specified
  scientific verdict: **UNIDENTIFIABLE** (Level 1 is fully compliant with
  spec; Level 2 now has genuine locked 30-seed confirmation-stage results
  for both the hidden-vulnerability experiment and the continual-learning
  baseline comparison, but SAC/MuJoCo remain out of reach and the full
  2e6-step training budget was not attempted; see the report for exactly
  what *was* run and what it found).
- **`scientific_audit_report.md`** -- itemized accounting of every
  experiment, every bug found and fixed during development, and every
  falsification condition's status.

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=src

# Run the full test suite (57 tests, <2 seconds)
pytest tests/ -v

# Run any experiment (each is independently executable and writes real,
# non-fabricated CSV/JSON outputs to experiments/<ID>/output/)
python experiments/E01_ground_truth/run.py --stage discovery
python experiments/E01_ground_truth/run.py --stage confirmation
python experiments/E02_identification/run.py
python experiments/E03_hidden_vulnerability/run.py --stage discovery
python experiments/E03_hidden_vulnerability/run.py --stage confirmation  # ~6 min
python experiments/E05_information_matched/run.py   # also produces E04's outputs
python experiments/E06_contrast_robustness/run.py
python experiments/E07_future_vulnerability/run.py
python experiments/E08_cross_algorithm/run.py         # ~8-9 min, real Stable-Baselines3 PPO+SAC
python experiments/E09_baselines/run.py --stage discovery
python experiments/E09_baselines/run.py --stage confirmation  # ~15-18 min, checkpointed/resumable
python experiments/E10_ablations/run.py              # run E01 first
python experiments/E11_negative_controls/run.py
```

## What actually works end-to-end

| Component | Status |
|---|---|
| Level 1 analytical SCM (exact ground truth) | Complete, tested, run at full spec'd scale (10 dev / 30 confirm seeds) |
| Level 2 synthetic RL environment | Complete, tested, run at **both discovery (10 seeds) and locked confirmation (30 fresh seeds) scale** |
| From-scratch NumPy PPO | Complete, tested |
| CL baselines (naive/replay/EWC/distillation/UPGD) | Complete, tested, **all 5 run in E09 confirmation stage** |
| Identification/certification logic | Complete, tested, run |
| Statistics (bootstrap, Holm/BH, effect sizes) | Complete, tested, **used in E03/E09 confirmation stages** |
| Pre-registration lock (Section 57) | **Implemented and used** for E03/E09 confirmation, SHA-256 hashed |
| Checkpoint/resume (Section 6) | **Implemented and exercised** for E09 after a real interruption |
| Task-order-confound check (Section 11) | **Implemented and run** (discovery scale) -- found task order changes outlier behavior by ~300x |
| SAC | **RUN (E08, real Stable-Baselines3 backend)** -- see docs/DEVIATIONS.md #10 |
| MuJoCo/standard benchmarks | Smoke-tested (HalfCheetah-v5, Hopper-v5 reset/step confirmed); no full experiment |
| Locked 50-seed confirmation tier | NOT run (30-seed tier is compliant and complete; 50-seed tier not attempted) |

## Directory structure

```
configs/            experiment registry, competence envelope, etc.
src/ccce/           the package: environments, scm, learners, protocols,
                     interventions, competence, oracle, estimator,
                     diagnostics, baselines, statistics, utils
experiments/         one directory per E01-E11, each with a real run.py
                     and an output/ directory of real results
tests/               57 unit + integration tests (pytest)
docs/DEVIATIONS.md   documented deviations from the master prompt
scientific_audit_report.md
FINAL_EXPERIMENTAL_REPORT.md
```

## Architectural guarantee worth knowing about

`src/ccce/oracle/analytical_oracle.py` is statically verified
(`tests/test_no_oracle_estimator_coupling.py`) to never import
`ccce.estimator`. The ground truth used to validate the estimator is
never computed by the estimator being validated.
