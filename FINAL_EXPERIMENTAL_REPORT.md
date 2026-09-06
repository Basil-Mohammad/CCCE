# FINAL EXPERIMENTAL REPORT

## Executive Summary

This report covers the implementation and partial execution of the
experimental program specified in Master Prompt V3 for
"Counterfactual Cross-Competence Effects in Continual Reinforcement
Learning" (main_v2.tex). **The implementation is substantially complete
for Level 1 (analytical SCM, fully compliant with the specified 10/30
discovery/confirmation seed structure) and now includes genuine locked
30-seed confirmation-stage results for Level 2 (synthetic RL environment)
as well, added in a follow-up session. In a further session, PyTorch and
Stable-Baselines3 -- previously confirmed impossible to install in two
earlier sessions (see `docs/DEVIATIONS.md` #1-3) -- became installable
for reasons not fully understood, enabling a genuine SAC cross-algorithm
experiment (E08) with a real deep-RL backend.** Standard MuJoCo benchmark
validation (Section 9's "Level 3") is now technically possible (verified
via smoke tests on HalfCheetah-v5 and Hopper-v5) but was not run as a
full experiment this session due to time constraints.

No result in this report is fabricated. Every number below was produced
by running the code in this repository and is traceable to a CSV/JSON
file under `experiments/*/output/`. Where an experiment specified by the
master prompt was not run, it is labeled `NOT_RUN` and not represented by
an invented number. Both Level-2 confirmation-stage runs (E03, E09) use a
genuine pre-registration lock (Section 57): a design manifest is written
and SHA-256 hashed before any confirmation-stage result is computed.

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
| Level 2 (E03, E09) | 10 seeds | **30 fresh, disjoint seeds (1000-1029), locked pre-registration** |

## Environments

1. **Level 1 -- Analytical SCM** (`src/ccce/scm/analytical.py`): linear-
   Gaussian parameter recursion with an exact closed-form mean, giving
   exact `CCCE^GT` with zero Monte Carlo error. Fully implemented, fully
   tested (14 unit tests), fully exercised across all required regimes.
2. **Level 2 -- Synthetic point-mass** (`src/ccce/environments/synthetic_env.py`):
   2D continuous-control environment with 4 causal factors (position
   dynamics, velocity damping, obstacle density, recovery perturbation)
   and 6 tasks with documented causal overlap. Fully implemented, fully
   tested (18 unit tests), exercised at both discovery and confirmation
   scale.
3. **Level 3 -- Standard benchmarks (MuJoCo/Gymnasium)**: smoke-tested
   only (`HalfCheetah-v5`, `Hopper-v5` reset/step confirmed working after
   `mujoco`/`gymnasium[mujoco]` became installable -- see Learning
   Algorithms below and Deviations #10); **no full validation experiment
   run**.

## Task Sequences

Six tasks (T1-T6) per `src/ccce/environments/synthetic_env.py::default_task_set`,
mirroring main_v2.tex Table 12's design intent (T1-T2 share position/
velocity dynamics; T2-T3 and T3-T5 share obstacle density; T1/T2-T4 and
T4-T5 share velocity dynamics/recovery perturbation; T5 combines high
obstacle density and high recovery-perturbation magnitude; T6 imposes a
conflicting near-zero-terminal-velocity requirement on the same shared
dynamics as T1). The primary sequence T1->T2->T3->T4->T5(branch) was used
for E03's hidden-vulnerability test (both stages). A genuine order-
confound check (Section 11) was added this session to E09's discovery
stage: Sequence A (T1->T2->T3) vs. Sequence B (T2->T1->T3), for naive and
EWC. **Result: task order materially changes per-seed outlier behavior**
(e.g., naive seed 4: BWT = -0.0003 under Sequence A vs. +0.0838 under
Sequence B) -- a real, previously undetected order-sensitivity. This
check was run at discovery scale only; a full confirmation-scale,
multi-sequence design (Sequences A/B/C across all baselines) remains
NOT_RUN.

## Learning Algorithms

- **PPO**: primarily a from-scratch NumPy implementation (documented
  substitute for Stable-Baselines3; see Deviations #2), used for
  E01/E03/E04/E05/E06/E07/E09/E10/E11. **E08 uses real Stable-Baselines3
  PPO** (see below and Deviations #10). Both are verified deterministic
  given a fixed seed.
- **SAC**: **RUN (E08 only)**, using real Stable-Baselines3, after
  PyTorch became installable in this sandbox in a later session --
  reversing a constraint independently confirmed impossible twice earlier
  (standard PyPI install failing from disk exhaustion; the official
  CPU-only wheel index being outside this sandbox's network egress
  allowlist). See Deviations #10 for the complete, transparent account.

## Baselines

All 5 specified baselines (naive, replay, EWC, distillation, UPGD) were
run in the E09 confirmation stage (30 fresh seeds, 12,000 steps/task).
Discovery-stage screening used only naive/replay/EWC (3 of 5) for speed.

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

**Discovery stage** (10 seeds, ~1800 steps/task): mean nominal RC =
-0.0063 (close to the "RC~0" hypothesis). Strongest mean CCCE was -0.0069
at `c=1.5` -- correctly signed but below the pre-registered
`delta=0.03` threshold. `pattern_observed=False`.

**Confirmation stage** (30 fresh seeds 1000-1029, ~15,000 steps/task,
**locked pre-registration**, manifest SHA-256 `b9b3bae1...`): mean
nominal RC = +0.0021 (95% CI [-0.0027, 0.0068], consistent with "RC~0").
At `c=2.0`, mean CCCE = -0.0075 with a **bootstrap 95% CI of [-0.0164,
-0.0001] -- excluding zero** -- but this remains below the
pre-registered practical-significance threshold, and the Holm-corrected
significance flag (across the 4 non-nominal `c` values) is `False`.
`pattern_observed=False` by the pre-registered criterion (which requires
`|CCCE| > delta`, not merely a CI excluding zero).

**This is a materially more informative result than a simple null.** It
is a textbook instance of the distinction Master Prompt V3 Section 37
requires: "a statistically significant but practically negligible CCCE
should not be presented as an important discovery." Here we have exactly
that pattern, reported as such rather than rounded up to "detected" or
down to "no effect." Individual-seed variance in the confirmation run was
substantial: several seeds showed `|CCCE(c=2.0)|` exceeding 0.06 in
either direction (e.g., seed 1003: -0.0933, seed 1007: -0.0668, seed
1006: +0.0625), consistent with the seed-instability pattern documented
below (Continual-Learning Baselines).

**At Level 1**, the same qualitative phenomenon (RC~0, CCCE(c*) strongly
negative under a valid intervention) is exactly constructible by design
and correctly recovered by the estimator (sign accuracy 1.00, Ground
Truth table above). The Level-2 confirmation result -- a small,
correctly-signed, statistically-detectable-but-below-threshold effect --
sits between "phenomenon cleanly absent" and "phenomenon cleanly present
at the pre-registered scale," and is reported as exactly that rather than
forced into either category.

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

**RUN with a real deep-RL backend** (this is the only experiment in the
project using genuine PyTorch + Stable-Baselines3, not the NumPy
substitute -- see `docs/DEVIATIONS.md` #10 for the full, transparent
account of how torch/SB3/MuJoCo became installable in this sandbox
partway through this project, after being confirmed impossible in two
earlier sessions).

10 seeds, real SB3 PPO and SAC, T1->T2->T3 sequence, 1,200 steps/task
(matched to E09's discovery-stage budget for comparability):

| Algorithm | Mean BWT | Std BWT | Notes |
|---|---|---|---|
| PPO | -0.2388 | 0.7161 | Dominated by one catastrophic-collapse outlier (seed 0: BWT=-2.39); other 9 seeds all ~0.0000 |
| SAC | +0.0028 | 0.0134 | No comparable outlier; markedly more stable at this budget |

(source: `experiments/E08_cross_algorithm/output/summary.json`)

Paired-difference test: p=0.346 (not significant, but this is
uninformative given the extreme variance contributed by PPO's single
outlier seed). **The qualitative finding is that real neural-network PPO
at this very small training budget can occasionally collapse
catastrophically** (a ~2.4-unit swing in a system where per-episode
returns are of similar order of magnitude), while SAC's off-policy,
more sample-efficient learning did not exhibit a comparable failure mode
at the same budget. This is a genuinely different qualitative pattern
from the earlier NumPy-linear-policy experiments (E03, E09), where
outlier magnitudes were more modest (~0.08) and occurred at a similar
rate for multiple methods including naive. **This should be read as
"cross-algorithm robustness within the evaluated settings was not
observed at this specific tiny budget," not as a general claim about
PPO vs. SAC** -- 1,200 steps/task is an extremely small budget for a
neural-network policy, and the instability observed may be an artifact
of under-training rather than a property of PPO in general.

## Continual-Learning Baselines

**Discovery stage** (10 seeds, ~1200 steps/task, T1->T2->T3, 3 baselines):

| Baseline | Mean BWT | Std BWT | Notes |
|---|---|---|---|
| naive | +0.0000 | 0.0005 | Consistently near zero across all 10 seeds |
| replay | +0.0004 | 0.0371 | Two outlier seeds: seed 2 (-0.0826), seed 7 (+0.0832); other 8 near zero |
| ewc | -0.0165 | 0.0330 | Two outlier seeds: seed 3 (-0.0823), seed 6 (-0.0828); other 8 near zero |

**Confirmation stage** (30 fresh seeds 1000-1029, **locked
pre-registration**, manifest SHA-256 `6ef60363...`, ~12,000 steps/task,
all 5 baselines, Holm-corrected significance test vs. naive):

| Baseline | Mean BWT | Std BWT | Outlier seeds (of 30) | Holm-sig. vs. naive |
|---|---|---|---|---|
| naive | +0.0007 | 0.0332 | 1 | -- |
| replay | +0.0003 | 0.0013 | 0 | False |
| ewc | +0.0007 | 0.0216 | 2 | False |
| distillation | +0.0004 | 0.0211 | 2 | False |
| upgd | +0.0039 | 0.0437 | 1 | False |

(sources: `experiments/E09_baselines/output/{discovery,confirmation}/summary.json`,
both verified deterministic -- see Reproducibility section)

**At confirmation scale, no baseline differs significantly from naive**
(all Holm-corrected p-values non-significant at alpha=0.05). This is a
clean, statistically disciplined null result: with 5 methods tested
against a common baseline and correction for multiple comparisons,
none stands out. **This should not be read as "all methods are
equivalent to naive" in an unconditional sense** -- it should be read as
"at this training budget and this 3-task sequence, this test could not
distinguish any method from naive." Every method, including naive itself,
continued to show 0-2 outlier seeds out of 30 with large (~+/-0.08) BWT
swings, consistent with the discovery-stage finding: this is a recurring
seed-level instability affecting a small but consistent minority
(roughly 1-in-15 to 1-in-30 seeds) of training runs at this budget,
across every method tested, not a property that distinguishes one
continual-learning method from another. Master Prompt V3 Section 60's
seed-robustness check ("check whether conclusions are driven by one
seed") is directly answered here: conclusions are not driven by a single
seed at confirmation scale (the largest per-method outlier count is 2 of
30), but the *absence of a significant baseline difference* is itself a
finding that depends on this recurring background instability being
present in naive as much as in the other methods -- if naive's own
outlier (seed unspecified, BWT contributing to its 0.0332 std) were
removed, the comparison might look different. This residual sensitivity
is reported rather than resolved.

A genuine **order-confound check** (Section 11, discovery scale only,
naive and EWC on Sequence A [T1->T2->T3] vs. Sequence B [T2->T1->T3])
found that task order changes which seeds become outliers: naive seed 4
showed BWT=-0.0003 under Sequence A but +0.0838 under Sequence B --
a nearly 300x difference driven purely by task order. This confirms task
sequence is a genuine confound that a single-sequence design (as used for
the confirmation-stage comparison above) cannot rule out, and is an
explicit, acknowledged limitation of the confirmation-stage baseline
comparison.

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
- E03's hidden-vulnerability pattern was not confirmed at pre-registered
  significance at either discovery or confirmation scale: the
  confirmation-stage effect was statistically detectable (CI excludes
  zero at c=2.0) but below the practical-significance threshold, and not
  Holm-significant across the full intervention grid.
- E09's confirmation-stage baseline comparison found no method (replay,
  EWC, distillation, UPGD) significantly different from naive after
  multiple-comparison correction -- a clean null result for that specific
  comparison, at full 30-seed pre-registered scale.
- Both E03 and E09 confirmation-stage runs revealed recurring, low-rate
  (roughly 1-in-15 to 1-in-30 seeds) large-magnitude training instability
  across every method tested, including naive itself, suggesting this is
  a property of the training setup at this budget/architecture scale
  rather than a property distinguishing any one continual-learning method.
- A genuine order-confound was found (Section 11, discovery scale): task
  order changes per-seed outlier behavior by up to ~300x, an
  unaddressed threat to the single-sequence confirmation-stage design.

These are reported here with the same weight as the positive Level-1
results (E01, E02, E06, E11), per Master Prompt V3 Section 62 ("negative
results are scientifically valuable") and Section 55 ("do not invent
results... if not supported, label NOT_SUPPORTED").

## Computational Cost

Most experiments ran on 1 CPU core, no GPU, each completing in seconds.
The two confirmation-stage upgrades are the exception: E03's confirmation
stage (30 seeds x 15,000 steps/task) took ~6 minutes wall-clock; E09's
confirmation stage (5 baselines x 30 seeds x 12,000 steps/task) took
~15-18 minutes wall-clock and required background execution with
checkpoint/resume support because it exceeded this environment's
per-command execution window (an execution-sandbox constraint discovered
and worked around this session -- see `docs/DEVIATIONS.md` #3b). No
GPU-hours were used anywhere in this session. Detailed automated
wall-clock/GPU-hour instrumentation (a dedicated module per Section 63)
remains unimplemented; the timings above were measured manually.

## Reproducibility

- 57/57 unit and integration tests pass (`PYTHONPATH=src pytest tests/ -q`),
  including 3 new tests added this session for the pre-registration lock
  mechanism (`tests/test_preregistration.py`).
- Every experiment script is deterministic given its fixed seed(s) and
  produces its declared CSV/JSON outputs when re-run
  (`test_training_is_deterministic_given_same_seed`,
  `test_rng_bundle_reproducible_from_same_seed`, and the checkpoint
  round-trip test verify the underlying mechanisms directly). This was
  re-verified after this session's changes by diffing full stdout across
  repeated clean invocations of E01, E03 (discovery), and E09 (discovery).
- Every experiment's `manifest.json` records git commit, package
  versions (including explicit `NOT_INSTALLED_SANDBOX_CONSTRAINT` for
  torch/gymnasium/stable_baselines3/mujoco), hardware, and a
  hyperparameter hash.
- **Locked `confirmation_manifest.yaml` pre-registration files (Section
  57) are now generated and SHA-256 hashed for both E03 and E09 before
  their confirmation-stage results are computed**, closing the gap noted
  in the previous version of this report. The hash is carried through to
  each experiment's `verdict.json`/`summary.json`. This mechanism does
  not (and cannot, in a single-process script) cryptographically prevent
  a determined author from editing the file before "unlocking" it -- it
  provides an auditable paper trail, which is what Section 57 asks for.
- **Genuine checkpoint/resume support was added to `E09_baselines/run.py`**
  after an actual interruption during this session's confirmation-stage
  run (a background process was killed by the execution sandbox partway
  through, independent of any code bug). Each baseline's full 30-seed
  sweep is checkpointed to `_checkpoint_<baseline>.csv` immediately on
  completion; a restarted run detects and skips already-completed
  baselines. This is a real, exercised instance of Section 6's
  checkpointing requirement -- verified by the actual restart-and-resume
  that occurred during this session, not merely implemented and left
  untested. This checkpoint/resume pattern has not yet been generalized
  to E03 or the other experiment scripts.

## Scientific Verdict

Per Master Prompt V3 Section 83, classified from the pre-specified
criteria in `configs/experiments/experiments.yaml` and the falsification
conditions in `main_v2.tex` Section 8:

> ## **UNIDENTIFIABLE**
>
> Still not "NOT SUPPORTED," not "FALSIFIED," and not "SUPPORTED" --
> but the evidence base is now meaningfully stronger than in the
> discovery-only version of this report, and the reasons for
> "UNIDENTIFIABLE" are more specific:
>
> - The infrastructure, estimator, and identification logic are
>   validated and working correctly at fully compliant scale (Level 1:
>   10 discovery + 30 confirmation seeds, exactly as specified).
> - **Level 2 now has genuine locked confirmation-stage results** (30
>   fresh seeds, pre-registered, ~7-10x the discovery training budget)
>   for both E03 and E09, closing the single largest gap identified in
>   the previous version of this report. The confirmation-stage E03
>   result is a specific, nameable finding -- a statistically detectable
>   (CI excludes zero) but practically-below-threshold effect at the
>   pre-registered intervention point -- rather than a simple "not
>   observed." This is closer to a real answer than either "yes" or "no,"
>   and is reported as such rather than forced into a cleaner-sounding
>   category.
> - **E09's confirmation stage found no baseline (replay, EWC,
>   distillation, UPGD) significantly different from naive** after Holm
>   correction, across all 5 specified continual-learning methods at
>   locked, pre-registered, 30-seed scale -- a clean, well-powered null
>   result for that specific comparison.
> - **A genuine order-confound sensitivity was discovered** (Section 11):
>   task order changes per-seed outlier behavior by up to ~300x in the
>   discovery-stage check. This was not tested at confirmation scale for
>   the primary E03/E09 results, and remains an unaddressed threat to any
>   claim drawn from a single task-sequence design -- including the
>   confirmation-stage results reported above.
> - The two experiments most directly relevant to the manuscript's
>   central "does CCCE add information" question (E04, E07) still
>   returned null results under a design (independently-randomized future
>   labels) that may be structurally incapable of detecting a real effect
>   even if one exists -- this remains a limitation of the experimental
>   design, not evidence against the hypothesis, and was not revisited
>   this session.
> - **SAC cross-algorithm validation is now RUN** with a real
>   Stable-Baselines3 backend (E08), after PyTorch became installable in
>   this sandbox for reasons not fully understood (see
>   `docs/DEVIATIONS.md` #10). The result itself does not resolve
>   UNIDENTIFIABLE -- at the tiny 1,200-step/task budget tested, real PPO
>   showed a catastrophic single-seed collapse (BWT=-2.39) not mirrored by
>   SAC, which is a genuinely different and more extreme instability
>   pattern than anything seen in the NumPy-substitute experiments, and
>   raises rather than settles the question of what a fair budget for
>   cross-algorithm comparison would be. Standard MuJoCo benchmark
>   validation (Section 9's Level 3) is now also technically possible
>   (smoke-tested successfully on HalfCheetah-v5 and Hopper-v5) but no
>   full experiment was run this session.
> - The 50-seed tier specified for "the most critical claims" (Section 4)
>   was not attempted for Level 2, even after this session's upgrade to
>   30 seeds; nor was the full 2x10^6-step training budget, which remains
>   ~130-165x larger than what was actually run. Now that a real deep-RL
>   backend is available, closing this gap for E01/E03/E09 (not just E08)
>   is a concrete, achievable next step rather than a hypothetical one --
>   though SAC's ~85 steps/sec throughput on this CPU-only hardware means
>   a full 2e6-step run would still take on the order of hours per seed.
>
> **What would resolve UNIDENTIFIABLE from here**: (1) GPU compute, to
> make full-budget training (2x10^6 steps) practical in reasonable
> wall-clock time -- the software stack (PyTorch + Stable-Baselines3) is
> no longer the blocker it was; (2) migrating E01/E03/E09's confirmation
> stages from the NumPy substitute to the now-available real SB3 backend,
> so that the flagship hidden-vulnerability and baseline-comparison
> results are not qualified by "linear-policy substitute" the way they
> currently are; (3) further budget scaling to test whether the E03
> confirmation-stage effect's magnitude grows toward or away from the
> practical-significance threshold as training budget increases -- this
> session's data shows the effect became *more* detectable (CI excluding
> zero) as budget grew from 1,800 to 15,000 steps, which is suggestive
> but not conclusive; (4) a multi-sequence confirmation design addressing
> the order-confound found in Section 11; (5) a proper full-scale
> cross-algorithm comparison (E08 was run at only 1,200 steps/task,
> and its result -- a single catastrophic PPO outlier -- is itself
> plausibly a budget artifact rather than a stable finding); and (6) a
> redesigned
> E04/E07 label that is not structurally decoupled from the CCCE-
> generating contrast. None of these were completed this session, but
> items (2) and (3) are now partially informed by real evidence
> (the budget-scaling trend, and the order-confound's measured magnitude)
> rather than being purely speculative next steps.
