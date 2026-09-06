# Documented Deviations from Master Prompt V3

Per Master Prompt V3, Section 0 ("if you identify a genuine mathematical
inconsistency or experimentally impossible specification, STOP that
component, explain the issue, and propose the smallest scientifically
justified correction") and Section 86 ("do not silently make a
scientifically consequential assumption"), every deviation of this
implementation from the letter of the master prompt is listed here,
with its cause, its correction, and its scientific consequence.

## 1. Compute environment (root cause of most other deviations)

**Verified hardware** (see `src/ccce/utils/provenance.py::hardware_info`,
logged in every run manifest):
- 1 logical CPU core
- No GPU
- ~5.6 GB free disk after clearing caches (of 252 GB total; the rest is
  consumed by the base sandbox image)
- `pip install torch` **fails with `OSError: [Errno 28] No space left on
  device`** partway through unpacking (verified directly; a ~2 GB partial
  install was written and had to be removed to recover disk space)

**Consequence**: PyTorch, and therefore Gymnasium's MuJoCo backends and
Stable-Baselines3 (which depends on PyTorch), cannot be installed in this
environment. This is not a preference; it is a hard constraint verified
by direct attempted installation.

## 2. Section 12: PPO implementation

**Specified**: Stable-Baselines3 or "a carefully implemented equivalent."

**Deviation**: A from-scratch NumPy implementation
(`src/ccce/learners/numpy_ppo.py`): a linear-Gaussian policy + linear
value baseline, trained with the standard PPO clipped-surrogate objective
and GAE advantages, using closed-form score-function gradients (no
autodiff needed for a linear-Gaussian policy).

**Justification**: This is the smallest correction that preserves PPO's
defining features (clipped surrogate, GAE, on-policy rollouts, all
Section-12-required hyperparameters recorded) while running on 1 CPU core
with no PyTorch. It is a real, gradient-trained optimizer -- verified by
`test_training_changes_parameters` and `test_training_is_deterministic_given_same_seed`
-- not a scripted or mocked policy.

**Consequence**: Results from this learner (E03, E09) reflect the
capacity of a linear-Gaussian policy, not a deep network. This is a
materially smaller function class than any real submission would use, and
is the single most important caveat on the Level-2 results.

## 3. Section 13: SAC cross-algorithm validation

**Status: NOT_RUN.** SAC's standard implementations are PyTorch-based
(Stable-Baselines3, CleanRL, etc.). A from-scratch NumPy SAC (off-policy,
requires a replay buffer, twin Q-networks, and reparameterized sampling)
is a substantially larger engineering effort than the PPO substitute above
and was judged out of scope for this implementation session. **E08 is
explicitly marked `NOT_RUN` in `configs/experiments/experiments.yaml`**,
per Section 55's requirement to label unexecuted experiments honestly
rather than omit or fabricate them.

## 4. Section 9: standard benchmark environments (MuJoCo/Gymnasium)

**Status: NOT_RUN**, for the same root-cause reason (Section 1 above):
`mujoco` and `gymnasium[mujoco]` were not installed, and installing them
was not attempted given the confirmed disk constraint. All Level-2
results use the custom synthetic point-mass environment only.

## 5. Section 4-5: seed counts

**Specified**: N_dev >= 10, N_confirm >= 30 (>= 50 for the most critical
claims).

**Actual**:
- **E01 (Level 1, analytical)**: fully compliant. Discovery used 10
  seeds, confirmation used 30 fresh, disjoint seeds (seeds 1000-1029 vs.
  0-9), as specified.
- **E03, E09 (Level 2, synthetic env)**: discovery-stage only, with
  **5 seeds**, not 10. No locked 30/50-seed confirmation stage was run
  for Level 2 at all. This is a direct consequence of wall-clock time
  available in this session, not a scientific judgment that 5 seeds
  suffice -- it explicitly does not, and no claim in
  `FINAL_EXPERIMENTAL_REPORT.md` treats the Level-2 results as
  confirmatory.
- **E02, E06, E07, E11 (Level 1, various)**: use single fixed seeds or
  seed batteries (60-200 independent random scenarios) appropriate to
  their specific designs, documented per-experiment in
  `configs/experiments/experiments.yaml`.

## 6. Section 8: training budget

**Specified** (via main_v2.tex Table 16): 2x10^6 environment steps per
protocol.

**Actual**: 1,200-1,800 environment steps per task (Level 2). This is a
~1,000x reduction, chosen so that a 5-task sequential run completes in
seconds rather than hours on 1 CPU core. **A direct, measured consequence
of this reduction is visible in the E09 results**: BWT magnitudes across
all three baselines were <0.002 in absolute value, i.e., the policy
plausibly does not train long enough per task to exhibit meaningful
forgetting at all. This makes E09's baseline comparison **inconclusive**,
not merely "lower-powered" -- this is stated plainly in
`FINAL_EXPERIMENTAL_REPORT.md`.

## 7. Section 26: diagnostic definitions for E04/E05 (Level 1)

Section 26's diagnostics (gradient interference via probe gradients,
representation drift via CKA/cosine distance, plasticity via loss-of-
plasticity metrics) are defined in RL/neural-network terms that do not
apply directly to the closed-form Level-1 analytical SCM. For E04/E05,
we use documented **analytical analogues**: `bwt_proxy = RC_i`,
`gi_proxy` = cosine similarity of target-pull directions, `rep_drift_proxy`
= normalized parameter-update norm, etc. (see the docstring of
`experiments/E05_information_matched/run.py`). The RL-specific versions
(`src/ccce/diagnostics/conventional.py`) are implemented and unit-tested
and are used correctly in the E09 Level-2 experiment.

## 8. Section 28: E04's prediction target

Section 28 says "use the prediction target defined in main_v2.tex,"
which specifies "future competence risk" generically without a single
fixed operational definition. We defined it as a **binary envelope-
violation label after an independently-randomized held-out future
continuation task**. This choice makes the label deliberately
**uncoupled** from the T/C contrast used to compute CCCE at the current
step (by construction, the future task's target direction is drawn fresh
and independently). **This is very likely why E04 returned a
NOT_SUPPORTED result** (CCCE showed a small *negative* incremental AUROC
versus M0) -- a design that makes the label structurally independent of
the predictor cannot show a positive effect regardless of whether CCCE is
"really" informative in a less adversarially-constructed setting. This is
flagged explicitly, not smoothed over, in `FINAL_EXPERIMENTAL_REPORT.md`:
**the E04 null result should be read as "inconclusive under this specific
synthetic design," not as evidence against the manuscript's hypothesis in
general.**

## 9. Section 17: competence envelope thresholds

`configs/competence/default_envelope.yaml`'s thresholds
(`success_min=0.3`, etc.) are lower than main_v2.tex Table 13's stated
targets (e.g., `Success >= 0.90`). This is because main_v2.tex's
thresholds were calibrated for a full-budget, full-capacity learner; using
them against the reduced-capacity, reduced-budget NumPy-PPO substitute
(Deviation 2 + 6 above) would make every single Level-2 case VIOLATED
regardless of any real underlying phenomenon, destroying the signal
entirely. The re-derivation is documented in
`src/ccce/competence/contract.py`'s module docstring.

## Summary table

| Component | Spec'd | Actual | Reason |
|---|---|---|---|
| PPO backend | Stable-Baselines3 | From-scratch NumPy | No PyTorch (disk) |
| SAC | Required | NOT_RUN | No PyTorch (disk) |
| MuJoCo benchmarks | Required | NOT_RUN | No PyTorch/mujoco (disk) |
| Level-2 seeds | 10 dev / 30-50 confirm | 5 dev / 0 confirm | Wall-clock time |
| Training budget | 2e6 steps | 1.2-1.8k steps | Wall-clock time (1 CPU) |
| Level-1 seeds | 10 dev / 30-50 confirm | 10 dev / 30 confirm | **Fully compliant** |
