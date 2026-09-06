# FINAL EXPERIMENTAL REPORT

## Executive Summary

This report covers the implementation and partial execution of the
experimental program specified in Master Prompt V3 for
"Counterfactual Cross-Competence Effects in Continual Reinforcement
Learning" (main_v2.tex). **The implementation is substantially complete
for Level 1 (analytical SCM) and partially complete for Level 2
(synthetic RL environment); Level 3 (standard/MuJoCo benchmarks) and SAC
cross-algorithm validation were not attempted**, due to a verified,
hard compute constraint of the execution sandbox (no GPU, 1 CPU core, and
insufficient disk space to install PyTorch -- see `docs/DEVIATIONS.md`).

No result in this report is fabricated. Every number below was produced
by running the code in this repository and is traceable to a CSV/JSON
file under `experiments/*/output/`. Where an experiment specified by the
master prompt was not run, it is labeled `NOT_RUN` and not represented by
an invented number.

## Experimental Setup

- **Language/stack**: Python 3.12, NumPy 2.4, SciPy, pandas, scikit-learn,
  pytest. No PyTorch, no Gymnasium/MuJoCo, no Stable-Baselines3 (see
  Deviations).
- **Hardware**: 1 logical CPU core, no GPU, ~3.9 GB RAM (auto-detected
  and logged in every run manifest via `src/ccce/utils/provenance.py`).

## Seed Structure

| Level | Discovery | Confirmation |
|---|---|---|
| Level 1 (analytical, E01) | 10 seeds | 30 fresh, disjoint seeds |
| Level 1 (other experiments) | 60-200 independent random scenarios each | n/a (single-stage design) |
| Level 2 (E03, E09) | 5 seeds | **not run** |

## Environments

1. **Level 1 -- Analytical SCM** (`src/ccce/scm/analytical.py`): linear-
   Gaussian parameter recursion with an exact closed-form mean, giving
   exact `CCCE^GT` with zero Monte Carlo error. Fully implemented, fully
   tested (14 unit tests), fully exercised across all required regimes.
2. **Level 2 -- Synthetic point-mass** (`src/ccce/environments/synthetic_env.py`):
   2D continuous-control environment with 4 causal factors (position
   dynamics, velocity damping, obstacle density, recovery perturbation)
   and 6 tasks with documented causal overlap. Fully implemented, fully
   tested (18 unit tests), exercised at reduced scale.
3. **Level 3 -- Standard benchmarks (MuJoCo/Gymnasium)**: **NOT_RUN.**

## Task Sequences

Six tasks (T1-T6) per `src/ccce/environments/synthetic_env.py::default_task_set`,
mirroring main_v2.tex Table 12's design intent (T1-T2 share position/
velocity dynamics; T2-T3 and T3-T5 share obstacle density; T1/T2-T4 and
T4-T5 share velocity dynamics/recovery perturbation; T5 combines high
obstacle density and high recovery-perturbation magnitude; T6 imposes a
conflicting near-zero-terminal-velocity requirement on the same shared
dynamics as T1). Only the single sequence T1->T2->T3->T4->T5(branch) was
tested (E03); the alternative pre-specified orderings required by
Section 11 for task-order-confound control were **not run**.

## Learning Algorithms

- **PPO**: from-scratch NumPy implementation (documented substitute for
  Stable-Baselines3; see Deviations #2). Real, gradient-trained, verified
  deterministic given a fixed seed.
- **SAC**: **NOT_RUN.**

## Baselines

Naive sequential, replay, and EWC were run in the E09 comparison (5
seeds, reduced budget). Distillation and UPGD are implemented and unit-
tested but not included in that comparison run (time constraints).

## Ground Truth

Level 1's `CCCE^GT` is available in exact closed form (not Monte Carlo)
via `src/ccce/oracle/analytical_oracle.py`, which is architecturally
prevented from importing the estimator it validates (enforced by
`tests/test_no_oracle_estimator_coupling.py`).

## CCCE Estimation

Paired Monte Carlo estimator with common evaluation randomness across the
target/control arms (`src/ccce/estimator/paired_estimator.py`). Recovery
against the exact Level-1 oracle:

| Regime | RMSE | Bias | 95% Coverage | Sign Accuracy | n (certified) |
|---|---|---|---|---|---|
| beneficial | 0.0043 | -0.0004 | 0.93 | 1.00 | 150 |
| neutral | 0.0037 | -0.0001 | 0.97 | 1.00 | 150 |
| harmful | 0.0039 | +0.0001 | 0.93 | 1.00 | 150 |
| hidden_vulnerability | 0.0161 | +0.0006 | 0.93 | 1.00 | 150 |
| negative_control | 0.0051 | -0.0005 | 0.93 | 1.00 | 150 |
| positive_control | 0.0085 | -0.0004 | 0.93 | 1.00 | 150 |

(confirmation stage, N=30 fresh seeds; source:
`experiments/E01_ground_truth/output/confirmation/summary_by_regime.csv`;
this run is now **verified bit-for-bit reproducible** across repeated
process invocations -- see `docs/DEVIATIONS.md` and
`scientific_audit_report.md` bug #6/#7 for the reproducibility fix that
made this guarantee possible.)

**The estimator recovers the exact analytical ground truth well across
every required regime**, including correctly reporting `UNIDENTIFIABLE`
for out-of-support queries (identification coverage 96.8% overall, with
the remaining 3.2% being the by-design-unidentifiable regime).

## Identification

E02's 200-case randomized battery: identification coverage 33.0%,
abstention rate 34.0%, and **critically, false-certification rate =
0.000** -- the certification logic never certified an intervention that
was not VALID/BORDERLINE, across 200 adversarially randomized cases.
Hand-constructed test cases: 100% classification accuracy (6/6).

## Hidden Vulnerability

**Not observed at the Level-2 scale actually run** (5 seeds, ~1800
training steps/task): nominal RC = -0.0125 (averaged across seeds; close
to the "RC ~ 0" half of the hypothesis but dominated by one outlier seed
-- seed 2 alone showed RC=-0.0625, an order of magnitude larger than the
other four seeds, which were all <0.0005 in absolute value). No
intervention value in the tested grid produced a *consistent* CCCE below
the pre-registered `-delta = -0.03` threshold in the hypothesized
direction across seeds; the strongest mean effect was +0.0124 (wrong
sign) at `c=2.0`, itself driven by a single outlier seed (seed 4, +0.0625
at c=2.0) rather than a consistent pattern. **This single-seed-driven
volatility is itself a finding**: per Master Prompt V3 Section 60 ("check
whether conclusions are driven by one seed"), they clearly are here, at
this reduced training scale -- another symptom of the ~1800-step budget
being too small for stable learning dynamics.

**However**, at Level 1, the same qualitative phenomenon (RC~0, CCCE(c*)
strongly negative under a valid intervention) is exactly constructible by
design and is correctly recovered by the estimator (see the
`hidden_vulnerability` row in the Ground Truth table above: sign accuracy
1.00). This means the pipeline's estimation and certification machinery
is capable of detecting this phenomenon when it is present with
sufficient effect size; **the Level-2 null result is most parsimoniously
explained by unstable, under-trained learning dynamics at this reduced
budget rather than the absence of any real phenomenon** -- E09's baseline
comparison (below) shows the same reduced-budget setup produces highly
seed-dependent, occasionally large (up to -0.08) BWT swings for replay
and EWC, i.e. the training dynamics at this scale are volatile enough
that a real effect could easily be swamped by seed-to-seed noise rather
than genuinely absent.

## Incremental Information

E04 (n=400 synthetic scenarios, 5-fold CV, logistic regression
predicting an independently-randomized future envelope-violation label):
M0 (conventional-diagnostic proxies) AUROC = 0.570 +/- 0.103; M1 (M0 +
CCCE) AUROC = 0.552 +/- 0.102. **Incremental gain = -0.018 -- NOT
SUPPORTED** under this specific dataset design.

**Important caveat** (see `docs/DEVIATIONS.md` #8): the prediction label
was constructed from a *fully independently randomized* future
continuation task, deliberately uncoupled from the current-step T/C
contrast that CCCE measures. This design may be structurally incapable of
showing a positive result for *any* current-step diagnostic, not just
CCCE. This result should be read as **inconclusive under this specific
design**, not as a confirmed general refutation of CCCE's incremental
value.

## Information-Matched Comparison

E05 (same dataset as E04): M1 (M0+CCCE) AUROC = 0.552; M2 (M0+raw
Q_T,Q_C) AUROC = 0.553. Gap = -0.0015. **SUPPORTED** in the narrow sense
that CCCE is not meaningfully worse than direct access to the raw
quantities it is computed from -- but since neither model is a strong
predictor of the (structurally decoupled) label in this run, this should
not be read as a strong positive result either.

## Contrast Robustness

E06 (1 scenario, exact Level-1 computation, 3 pre-specified controls x 5
intervention values): **CCCE sign was robust (consistent across all 3
controls) at every tested intervention value** for this scenario's
parameters. This is a single-scenario demonstration, not a general
robustness claim across the benchmark.

## Future Vulnerability

E07 (n=60 independently randomized scenarios): Pearson r=+0.020
(p=0.878), Spearman rho=-0.010 (p=0.938). **No detectable prospective
association** in this design. As with E04, the fully-independent-random-
future construction may be a conservative-to-the-point-of-uninformative
test; this is not interpreted as evidence that CCCE cannot be
prospectively predictive in a more realistically correlated task-sequence
setting, which was not tested.

## Cross-Algorithm Results

**NOT_RUN** (SAC requires PyTorch; see Deviations).

## Continual-Learning Baselines

E09 (5 seeds, ~1200 steps/task, T1->T2->T3 sequence, mean BWT on T1 after
the full sequence):

| Baseline | Mean BWT | Std BWT | Notes |
|---|---|---|---|
| naive | -0.0001 | 0.0001 | Consistently near zero across all 5 seeds |
| replay | -0.0162 | 0.0332 | Dominated by one outlier seed (seed 2: -0.0826); other 4 seeds all near zero |
| ewc | -0.0165 | 0.0329 | Dominated by one outlier seed (seed 3: -0.0823); other 4 seeds all near zero |

(source: `experiments/E09_baselines/output/summary.json`, now verified
deterministic -- see Reproducibility section)

**This result should not be read as "replay and EWC forget more than
naive training."** In both cases, 4 of 5 seeds show near-zero BWT
(consistent with naive), and the large mean/std is driven entirely by a
single outlier seed per method. This is exactly the seed-robustness check
Master Prompt V3 Section 60 requires ("check whether conclusions are
driven by one seed"), and the honest answer here is **yes, they are** --
at this reduced training scale, occasional unstable training runs (for
reasons not further diagnosed in this session -- possibly a poor
initialization interacting with the EWC penalty or replay buffer
composition) produce large one-off BWT swings that would disappear or
average out with more seeds. This instability is also the most likely
explanation for why E03's hidden-vulnerability pattern was not cleanly
observed (see above): at this scale, seed-to-seed training variance may
simply dominate any systematic cross-competence effect.

## Ablations

Only seed-count sensitivity was run (derived from already-collected E01
data at zero extra compute cost): RMSE was stable across N=10 (0.0071)
and N=30 (0.0080) seeds for the aggregate estimator. The other 8 ablation
factors specified in Section 51 (intervention-grid resolution,
competence-contract granularity, control protocol, task sequence,
evaluation budget, checkpoint frequency, paired vs. unpaired evaluation,
intervention coverage) are **NOT_RUN**.

## Negative Results

- E04's incremental-information test did not support the hypothesis
  under its specific design (see above, with caveats).
- E07's future-vulnerability test found no association (see above, with
  caveats).
- E03's hidden-vulnerability pattern was not observed at the tiny
  Level-2 training scale used.
- E09's baseline comparison revealed high seed-to-seed instability
  (replay and EWC each had one outlier seed with BWT around -0.08 versus
  near-zero for the other four seeds), suggesting the reduced training
  budget produces genuinely unstable learning dynamics at this scale.

These are reported here with the same weight as the positive Level-1
results (E01, E02, E06, E11), per Master Prompt V3 Section 62 ("negative
results are scientifically valuable") and Section 55 ("do not invent
results... if not supported, label NOT_SUPPORTED").

## Computational Cost

All experiments ran on 1 CPU core, no GPU, each completing in seconds.
Detailed wall-clock/GPU-hour instrumentation (Section 63) was not
implemented in this session.

## Reproducibility

- 54/54 unit and integration tests pass (`PYTHONPATH=src pytest tests/ -q`).
- Every experiment script is deterministic given its fixed seed(s) and
  produces its declared CSV/JSON outputs when re-run
  (`test_training_is_deterministic_given_same_seed`,
  `test_rng_bundle_reproducible_from_same_seed`, and the checkpoint
  round-trip test verify the underlying mechanisms directly).
- Every experiment's `manifest.json` records git commit, package
  versions (including explicit `NOT_INSTALLED_SANDBOX_CONSTRAINT` for
  torch/gymnasium/stable_baselines3/mujoco), hardware, and a
  hyperparameter hash.
- No locked `confirmation_manifest.yaml` pre-registration-lock file
  (Section 57) was generated in this session, because no Level-2
  confirmation stage was run to lock. This is a gap, not a completed
  requirement -- documented here rather than silently omitted.

## Scientific Verdict

Per Master Prompt V3 Section 83, classified from the pre-specified
criteria in `configs/experiments/experiments.yaml` and the falsification
conditions in `main_v2.tex` Section 8:

> ## **UNIDENTIFIABLE**
>
> Not "NOT SUPPORTED," not "FALSIFIED," and certainly not "SUPPORTED."
> The evidence actually collected in this session is insufficient, by
> design and by explicit scope limitation, to render any of the other
> five verdicts honestly:
>
> - The infrastructure, estimator, and identification logic are
>   validated and working correctly (Level 1, fully compliant scale).
> - The two experiments most directly relevant to the manuscript's
>   central "does CCCE add information" question (E04, E07) returned
>   null results, but under designs (independently-randomized future
>   labels) that may be incapable of detecting a real effect even if one
>   exists -- this is a limitation of this session's experimental design,
>   not strong evidence against the hypothesis.
> - The flagship hidden-vulnerability phenomenon (E03) was not observed,
>   but at a training scale (~1800 steps/task) that other evidence in
>   this same session (E09's highly seed-dependent, occasionally large
>   BWT swings) suggests produces unstable, high-variance learning
>   dynamics, confounding "CCCE doesn't detect an effect"
>   with "seed noise dominates any systematic effect at this scale."
> - The locked 30/50-seed confirmation stage required to make any
>   confirmatory claim about Level 2 was never run.
> - SAC cross-algorithm validation and standard-benchmark validation
>   were never run.
>
> **A properly resourced re-run of this exact codebase** -- with a real
> deep-RL backend (PyTorch + Stable-Baselines3), GPU compute, the full
> 2x10^6-step training budget, 30-50 seeds per Level-2 experiment, a
> locked confirmation stage, and a less conservative (more realistically
> correlated) future-risk label for E04/E07 -- **would be a fair test of
> the manuscript's hypothesis. This session's results should not be
> interpreted as that test's outcome.**
