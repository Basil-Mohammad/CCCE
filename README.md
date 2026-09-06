# CCCE: Counterfactual Cross-Competence Effects in Continual RL

Implementation of the experimental program specified for
"Counterfactual Cross-Competence Effects in Continual Reinforcement
Learning" (`main_v2.tex`).

## Read this first

- **`docs/DEVIATIONS.md`** -- every place this implementation deviates
  from the exact letter of the specification, why, and what the
  scientific consequence is. This sandbox has 1 CPU core, no GPU, and
  cannot install PyTorch (confirmed: disk space is exhausted mid-install).
  As a direct consequence, **SAC and MuJoCo/standard-benchmark validation
  are `NOT_RUN`**, and PPO is a from-scratch NumPy substitute for
  Stable-Baselines3.
- **`FINAL_EXPERIMENTAL_REPORT.md`** -- the honest, pre-specified
  scientific verdict: **UNIDENTIFIABLE** (not enough was run at adequate
  scale to confirm or refute the manuscript's hypothesis; see the report
  for exactly what *was* run and what it found).
- **`scientific_audit_report.md`** -- itemized accounting of every
  experiment, every bug found and fixed during development, and every
  falsification condition's status.

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=src

# Run the full test suite (54 tests, <1 second)
pytest tests/ -v

# Run any experiment (each is independently executable and writes real,
# non-fabricated CSV/JSON outputs to experiments/<ID>/output/)
python experiments/E01_ground_truth/run.py --stage discovery
python experiments/E01_ground_truth/run.py --stage confirmation
python experiments/E02_identification/run.py
python experiments/E03_hidden_vulnerability/run.py --stage discovery
python experiments/E05_information_matched/run.py   # also produces E04's outputs
python experiments/E06_contrast_robustness/run.py
python experiments/E07_future_vulnerability/run.py
python experiments/E09_baselines/run.py
python experiments/E10_ablations/run.py              # run E01 first
python experiments/E11_negative_controls/run.py
```

## What actually works end-to-end

| Component | Status |
|---|---|
| Level 1 analytical SCM (exact ground truth) | Complete, tested, run at full spec'd scale |
| Level 2 synthetic RL environment | Complete, tested, run at REDUCED scale |
| From-scratch NumPy PPO | Complete, tested |
| CL baselines (naive/replay/EWC) | Complete, tested, run |
| CL baselines (distillation/UPGD) | Complete, tested, NOT run in a comparison |
| Identification/certification logic | Complete, tested, run |
| Statistics (bootstrap, Holm/BH, effect sizes) | Complete, tested |
| SAC | NOT implemented (no PyTorch) |
| MuJoCo/standard benchmarks | NOT implemented (no PyTorch/mujoco) |
| Locked 30/50-seed confirmation (Level 2) | NOT run (time) |

## Directory structure

```
configs/            experiment registry, competence envelope, etc.
src/ccce/           the package: environments, scm, learners, protocols,
                     interventions, competence, oracle, estimator,
                     diagnostics, baselines, statistics, utils
experiments/         one directory per E01-E11, each with a real run.py
                     and an output/ directory of real results
tests/               54 unit + integration tests (pytest)
docs/DEVIATIONS.md   documented deviations from the master prompt
scientific_audit_report.md
FINAL_EXPERIMENTAL_REPORT.md
```

## Architectural guarantee worth knowing about

`src/ccce/oracle/analytical_oracle.py` is statically verified
(`tests/test_no_oracle_estimator_coupling.py`) to never import
`ccce.estimator`. The ground truth used to validate the estimator is
never computed by the estimator being validated.
