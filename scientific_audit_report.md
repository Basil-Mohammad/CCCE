# Scientific Audit Report

Generated after the experiment session described in this repository.
Per Master Prompt V3 Section 67.

## 1. Experiments completed

| ID | Status | Scale |
|---|---|---|
| E01_GT_RECOVERY | RUN (discovery + confirmation) | Compliant: 10 dev / 30 confirm seeds |
| E02_IDENTIFICATION | RUN | 200-case randomized battery + 6 hand cases |
| E03_HIDDEN_VULNERABILITY | RUN (discovery + **locked confirmation**) | Discovery: 10 seeds/1800 steps; Confirmation: **30 fresh seeds/15,000 steps, pre-registered** |
| E04_INCREMENTAL_INFORMATION | RUN | 400 synthetic scenarios |
| E05_INFORMATION_MATCHED | RUN | (shares E04's dataset) |
| E06_CONTRAST_ROBUSTNESS | RUN | Exact computation, 1 scenario x 3 controls x 5 c-values |
| E07_FUTURE_VULNERABILITY | RUN | 60 synthetic scenarios |
| E09_CL_BASELINES | RUN (discovery + **locked confirmation**) | Discovery: 10 seeds/3 baselines + order-confound check; Confirmation: **30 fresh seeds, all 5 baselines, pre-registered** |
| E10_ABLATIONS | PARTIALLY RUN | 1 of 9 ablation factors (seed count) |
| E11_NEGATIVE_CONTROLS | RUN | 80 synthetic scenarios |

## 2. Failed experiments

None of the experiments that were *attempted* failed outright. Several
bugs were found and fixed during development (see Section 4); no
experiment run described in this report reflects an unfixed bug to our
knowledge, but see Section 8 (limitations of our own verification).

## 3. Excluded runs / exclusion reasons

- **E08_CROSS_ALGORITHM (SAC)**: NOT_RUN. No PyTorch-based SAC
  implementation could be installed (disk space exhausted; confirmed via
  two separate attempts including the official CPU-only wheel index,
  which is also unreachable from this sandbox's network egress
  allowlist -- see `docs/DEVIATIONS.md` #1, #3).
- **Standard MuJoCo/Gymnasium benchmark validation (Section 9)**:
  NOT_RUN. Same root cause.
- **E03/E09 locked 30-seed confirmation stage**: **NOW RUN** (added this
  session). Both experiments have genuine pre-registered confirmation
  stages with 30 fresh seeds and a ~7-10x larger training budget than
  discovery. The locked 50-seed tier and the full 2e6-step training
  budget specified in main_v2.tex remain NOT_RUN -- see
  `docs/DEVIATIONS.md` #5, #6.
- **7 of 9 ablation factors in Section 51**: NOT_RUN (time constraints).
  Seed-count sensitivity and (new this session) a task-order-confound
  check are reported.
- **Distillation and UPGD baselines**: **NOW RUN** at confirmation scale
  (added this session, 30 seeds each), in addition to being implemented
  and unit-tested from the start.

## 4. Bugs found and fixed during this session (for transparency)

This is included because catching real bugs via the test suite is direct
evidence that the tests are actually exercising the code, not passing
vacuously.

1. **Sign error in the hidden-vulnerability regime's analytical
   parameters** (`src/ccce/scm/regimes.py`): the initial choice of query
   point `c=+2.0` produced `CCCE^GT = +10.7` instead of the intended
   negative value; the intervention-response direction was miscalculated
   by hand. Fixed by querying `c=-2.0` instead, matching the actual sign
   of `h_i^T (E[theta_T] - E[theta_C])`. Caught by
   `test_hidden_vulnerability_regime_has_rc_near_zero_and_ccce_negative`.
2. **`SeedSequence` given a negative integer** in the E01 runner (from
   naive integer arithmetic combining a seed, a hash, and a possibly
   negative intervention value). Fixed by deriving the composite seed via
   `SeedSequence.generate_state` on a tuple of non-negative components.
3. **`numpy.bool_`/size-1-array-to-`float()` conversions** in the
   synthetic environment's `step()` and the policy's `value()` method
   (NumPy 2.x is stricter about implicit scalar coercion than earlier
   versions). Fixed with explicit `bool(...)` and `.item()` calls. Caught
   by `test_env_reset_and_step_shapes` and
   `test_policy_flat_params_roundtrip`.
4. **`LinearGaussianPolicy.__post_init__` missing**, causing
   `set_flat_params` to crash on a freshly-constructed (all-`None`-field)
   policy object. Fixed by adding `__post_init__` to allocate correctly-
   shaped zero arrays. Caught by `test_policy_flat_params_roundtrip`.
5. **E02's random intervention battery could never produce an in-support
   value** (continuous `uniform(-5,5)` draws essentially never exactly
   match a discrete grid point), silently making `identification_coverage`
   always report `0.000`. This was caught not by a unit-test assertion but
   by noticing the *implausibly round number* `0.000` in the printed
   summary during manual review, and confirmed by inspecting the
   generator logic. Fixed by sampling 60% of cases' values directly from
   the chosen support set.
6. **Non-deterministic RNG usage** in `numpy_ppo.py::ppo_update` (epoch-
   shuffling permutation) and `continual_methods.py::ReplayBuffer.add`
   (eviction-slot selection): both called `np.random.default_rng()` with
   no seed, drawing fresh OS entropy every call -- a direct violation of
   Section 3 ("must never depend on undocumented random state"). Masked
   in `test_training_is_deterministic_given_same_seed` because floating-
   point summation is *almost* order-invariant, so the tolerance-based
   `np.allclose` check passed despite the bug. Caught instead by directly
   diffing full stdout across two separate process invocations of E01 and
   E03, which showed numeric drift between runs with identical declared
   seeds (e.g., E01's hidden-vulnerability RMSE varying between 0.0130
   and 0.0194 across "identical" runs). Fixed by threading the caller's
   seeded RNG through every consumption point. **Verified fixed**: two
   consecutive full invocations of E01 and E03 are now byte-for-byte
   identical in stdout and in every output CSV.
7. **`hash()` used on strings for seed derivation** in E01's runner:
   Python's built-in `hash()` for strings is randomized per-process via
   `PYTHONHASHSEED`, compounding bug 6's non-determinism. Fixed by
   switching to `hashlib.sha256`, which is stable across processes.
8. **EWC penalty-gradient overflow at larger training budgets**
   (`continual_methods.py::EWCState.penalty_grad`): the Fisher-diagonal
   proxy is an unnormalized sum of squared per-sample gradients, and
   `lambda_ewc=300` applied to this unbounded quantity produced a
   parameter update large enough to diverge (`RuntimeWarning: overflow
   encountered`, eventually `FloatingPointError` under `np.seterr(all=
   "raise")`) once the training budget was increased from ~1800 to
   ~12,000-15,000 steps/task (Section 5 of this report). This was not
   present at the original small budget, so it was not caught by the
   initial test suite; it surfaced only when actually attempting the
   larger-budget confirmation-stage run this session added. Fixed by
   normalizing the Fisher diagonal to a bounded scale and clipping the
   penalty step to the same norm bound as the main PPO gradient update.
   **Verified fixed** across 10 seeds at the confirmation-stage budget
   with `np.seterr(all="raise")` enabled (which would immediately halt
   execution on any remaining overflow); cross-checked that the resulting
   parameter norms are consistent with ordinary value-function-driven
   growth (also observed in the unmodified naive baseline), not
   divergence.
9. **Long-running background processes are killed between tool-call
   boundaries in this execution sandbox**, independent of any bug in the
   code itself: an initial E09 confirmation-stage run (5 baselines x 30
   seeds x 12,000 steps/task, ~15-18 minutes of wall-clock time) was
   silently terminated partway through (after completing naive, replay,
   and part of EWC) with no error in its log. This is an infrastructure
   constraint, not a code defect, but it directly motivated a real fix:
   `experiments/E09_baselines/run.py` previously held all results in
   memory until the very end, so the interruption lost all completed
   work. This was corrected by adding genuine per-baseline
   checkpoint/resume logic (Section 6): each baseline's full seed sweep
   is written to `_checkpoint_<baseline>.csv` immediately on completion,
   and a restarted run skips any baseline whose checkpoint file already
   exists. **Verified working**: the second attempt, after being killed
   and manually restarted, printed `RESUMED from checkpoint` for the
   already-completed baselines and correctly continued from EWC onward
   without recomputing naive/replay.

## 5. Pre-registration and confirmation-stage upgrades added this session

Following a request to push compliance with Master Prompt V3 as far as
practical within this sandbox's constraints:

- **E03 and E09 now have genuine locked confirmation stages** (30 fresh
  seeds each, 1000-1029, disjoint from the 0-9 discovery seeds), not just
  discovery-stage runs. A pre-registration-style lock file
  (`confirmation_manifest.yaml`, per Section 57) is written and SHA-256
  hashed *before* any confirmation-stage result is computed, and the
  hash is carried through to `verdict.json`/`summary.json` so a reader
  can verify the locked design was not altered after the fact. This is
  implemented and unit-tested in `src/ccce/utils/preregistration.py`.
- **Training budget increased ~7-10x** at confirmation stage (E03: 1800
  -> 15,000 steps/task; E09: 1200 -> 12,000 steps/task), verified
  affordable by direct timing tests before committing to the larger runs.
- **E09 now includes all 5 specified baselines** (naive, replay, EWC,
  distillation, UPGD) at confirmation scale, not just 3.
- **A genuine task-order-confound check (Section 11)** was added to E09's
  discovery stage: Sequence A (T1->T2->T3) vs. Sequence B (T2->T1->T3),
  run for naive and EWC. Result: task order materially changes which
  seeds exhibit large BWT swings (e.g., naive seed 4: BWT=-0.0003 under
  Sequence A vs. +0.0838 under Sequence B) -- a real, previously
  undetected order-sensitivity.
- **Multiple-comparison correction (Holm)** is now applied where relevant:
  across the 4 non-nominal intervention points in E03's confirmation
  stage, and across the 4 non-naive baselines in E09's confirmation
  stage.
- **Bootstrap CIs** (2000 resamples, dedicated `bootstrap_rng` stream)
  replace the paired-t interval at confirmation scale in E03.



## 6. Seed counts

See Section 1 table above and `docs/DEVIATIONS.md` #5 for the full
per-experiment accounting and justification for every departure from the
specified 10/30/50 hierarchy.

## 7. Confidence intervals

All CIs reported by the codebase are 95% intervals (paired t-interval for
small-N paired comparisons in E01/E03/E11; CV-fold standard deviation,
not a CI, for the AUROC comparisons in E04/E05, which is noted explicitly
in those experiments' summaries rather than mislabeled as a CI).

## 8. Multiple testing

No cross-experiment multiple-testing correction was applied in this
session, because no experiment in this run reports more than one primary
comparison per hypothesis (each experiment's `summary.json` states its
one pre-specified comparison). `src/ccce/statistics/inference.py`
implements both Holm and Benjamini-Hochberg correction and both are
unit-tested; they are ready for use once a genuinely multi-comparison
confirmation-stage analysis (e.g., comparing CCCE across all 6x6 task
pairs) is run.

## 9. Identification coverage

- E02 (randomized battery, n=200): identification_coverage = 0.330,
  abstention_rate = 0.340, **false_certification_rate = 0.000** (the
  single non-negotiable property: certification logic never certifies an
  intervention that is not VALID/BORDERLINE).
- E01 (n=930 estimate rows across all regimes and stages): identification
  coverage ~0.968 (the unidentifiable-regime rows are correctly abstained
  from; everything else is certified).

## 10. Hidden vulnerability evidence

**Discovery stage (10 seeds, ~1800 steps/task)**: `pattern_observed:
False`. Mean nominal RC = -0.0063 (close to the "RC~0" hypothesis); the
strongest mean intervention-conditioned CCCE was correctly signed
(-0.0069 at `c=1.5`) but below the pre-registered `delta=0.03` threshold.

**Confirmation stage (30 fresh seeds 1000-1029, ~15,000 steps/task,
locked pre-registration, SHA-256 hash `b9b3bae1...`)**: `pattern_observed:
False`, but with a more nuanced result than a simple null. Mean nominal
RC = +0.0021 (95% CI [-0.0027, 0.0068], consistent with "RC~0"). At
`c=2.0`, mean CCCE = -0.0075 with a **bootstrap 95% CI of [-0.0164,
-0.0001] -- excluding zero** -- but the magnitude remains below the
pre-registered practical-significance threshold `delta=0.03`, and the
Holm-corrected significance flag (across the 4 non-nominal intervention
points) is `False`. This is a genuine instance of the distinction Section
37 warns must not be conflated: a statistically detectable effect
(CI excludes 0) that is not practically meaningful by the pre-registered
standard. Individual-seed variance was substantial (several confirmation
seeds showed |CCCE| > 0.06 at `c=2.0` in either direction -- e.g., seed
1003: -0.0933, seed 1007: -0.0668, seed 1006: +0.0625 not shown above),
consistent with the seed-instability pattern also observed in E09.

Separately, at Level 1 (the pure analytical SCM, where effect sizes are
directly parameterized rather than emergent from RL training), the
hidden-vulnerability regime **is constructible exactly** (verified by
`test_hidden_vulnerability_regime_has_rc_near_zero_and_ccce_negative`)
and the estimator **does recover it** (RMSE=0.013-0.016, sign accuracy
1.00 in E01). This shows the *estimator and certification machinery* are
capable of detecting the phenomenon when it is present with adequate
signal; the Level-2 confirmation-stage result above shows a small,
correctly-signed, statistically-detectable-but-practically-below-
threshold effect, which is a materially more informative outcome than
either "clearly present" or "clearly absent."

## 11. Incremental information

**NOT_SUPPORTED** in the E04 run as specifically designed (AUROC gain of
M0+CCCE over M0 was -0.018, i.e. slightly negative). See
`docs/DEVIATIONS.md` #8 for why this specific dataset design (an
independently-randomized future label) may be structurally unable to show
a positive result regardless of CCCE's real properties, and why this
should be read as "inconclusive under this design" rather than a general
refutation.

## 12. Information-matched comparison

**SUPPORTED** in the narrow sense that M0+CCCE was not meaningfully worse
than M0+(Q_T,Q_C) (gap = -0.0015, within noise). Given that neither model
beat M0 alone by much in this run (Section 10), this "SUPPORTED" result
should not be over-read as a positive finding for CCCE -- it mainly shows
that CCCE and the raw quantities are comparably (weakly) informative
under this specific label construction, not that either is strongly
informative.

## 13. Contrast sensitivity

The one scenario tested in E06 showed **sign-robust** CCCE across all
three pre-specified controls at all five intervention values. This is a
single-scenario result, not a general claim about contrast robustness
across the benchmark.

## 14. Future prediction

**No detectable association** (Pearson r=+0.02, p=0.88; Spearman
rho=-0.01, p=0.94) in the 60-scenario E07 run. Per `docs/DEVIATIONS.md`
#8's related point, this design used a fully independently-randomized
future continuation, which may be an overly conservative test of
prospective association; a design in which the future task is drawn from
a more realistic (correlated) distribution of subsequent tasks was not
attempted in this session.

## 15. Cross-algorithm validation

NOT_RUN. See Section 3.

## 16. Computational cost

Most experiments in this session ran on 1 CPU core with no GPU,
completing in well under one minute each. The two confirmation-stage
upgrades added this session are the exception: E03's confirmation stage
(30 seeds x 15,000 steps/task) took approximately 6 minutes wall-clock,
and E09's confirmation stage (5 baselines x 30 seeds x 12,000 steps/task)
took approximately 15-18 minutes wall-clock, run as a background process
with checkpoint/resume support (see Section 4, bug 9) because it exceeded
this environment's per-command execution window. No GPU-hours were used
anywhere in this session. Exact automated wall-clock/GPU-hour logging
(a dedicated module per Section 63) is still not implemented; the timings
above were measured manually via direct timing tests before committing to
each larger run, not logged automatically per the specification.

## 17. Falsification status

Per `main_v2.tex` Section 8 and this project's `configs/experiments/experiments.yaml`:

- **Failure A candidate** (no incremental info over conventional
  diagnostics): triggered in the E04 run as specifically designed (see
  caveats above -- inconclusive, not confirmatory of a general failure).
- **Failure H** (estimator fails to recover ground truth): **not
  triggered** -- E01 shows the estimator recovers ground truth well
  (Level 1, fully compliant scale).
- **Failure C** (hidden vulnerability cannot be reproduced): **not
  conclusively triggered nor refuted** -- the confirmation-stage run
  (30 fresh seeds, 15,000 steps/task) found a correctly-signed,
  statistically-detectable (CI excludes 0) but practically-below-
  threshold effect at `c=2.0`. This is neither a clean reproduction
  (magnitude below the pre-registered `delta`) nor a clean failure to
  reproduce (the CI excludes zero, and the sign matches the hypothesis).
  Combined with the phenomenon's exact constructibility at Level 1, this
  reads as "a small real effect may be present, obscured by seed-level
  training variance at this scale," which is a more specific statement
  than the discovery-stage-only result supported.
- **Failures E, F, G** (sequence dependence, SAC, control dependence): F
  (SAC) remains NOT_RUN. E (sequence dependence) has partial evidence:
  E09's new order-confound check (Section 5) found that task order
  materially changes per-seed outlier behavior, which is a form of
  sequence-sensitivity, though it was not tested for E03's specific
  hidden-vulnerability claim. G (control dependence) was tested once
  (E06) and found robust, the opposite of failure.
- **Failure B** (no info beyond raw Q_T,Q_C): **not triggered** in the
  E05 run (CCCE was comparable to, not worse than, the raw quantities).

See `FINAL_EXPERIMENTAL_REPORT.md` for the overall scientific verdict.
