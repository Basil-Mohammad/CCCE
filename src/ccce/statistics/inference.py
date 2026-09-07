# Author: Basil M. Alzboun

"""
Statistical utilities (Master Prompt V3, Sections 36-38; main_v2.tex
Section 5.10). All confidence intervals are 95% unless stated otherwise.
Bootstrap resampling uses the dedicated `bootstrap_rng` stream (Section 59),
never a shared global generator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy import stats


@dataclass
class EstimateWithCI:
    mean: float
    standard_error: float
    ci_low: float
    ci_high: float
    n: int

    @property
    def sign(self) -> str:
        if self.ci_low > 0:
            return "+"
        if self.ci_high < 0:
            return "-"
        return "0"


def paired_mean_ci(
    deltas: Sequence[float], confidence: float = 0.95, method: str = "t"
) -> EstimateWithCI:
    """CI for the mean of paired differences (the CCCE estimator, Section 22).

    method="t": normal-theory t-interval (fast, used for quick checks).
    method="bootstrap": handled by `bootstrap_ci` instead; kept separate
    so callers are explicit about which method produced a given interval.
    """
    arr = np.asarray(deltas, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    if n > 1 and method == "t":
        tval = stats.t.ppf(0.5 + confidence / 2, df=n - 1)
        lo, hi = mean - tval * se, mean + tval * se
    else:
        lo, hi = mean, mean
    return EstimateWithCI(mean=mean, standard_error=se, ci_low=lo, ci_high=hi, n=n)


def bootstrap_ci(
    data: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> EstimateWithCI:
    """Percentile bootstrap CI, using the dedicated bootstrap RNG stream."""
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    if rng is None:
        rng = np.random.default_rng()  # caller should normally pass RNGBundle.bootstrap
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        boot_stats[b] = statistic(sample)
    alpha = 1 - confidence
    lo, hi = np.quantile(boot_stats, [alpha / 2, 1 - alpha / 2])
    return EstimateWithCI(
        mean=float(statistic(arr)),
        standard_error=float(np.std(boot_stats, ddof=1)),
        ci_low=float(lo),
        ci_high=float(hi),
        n=n,
    )


def cohens_d(x: Sequence[float], y: Sequence[float] | None = None) -> float:
    """Effect size. If y is None, treats x as paired differences (one-sample d)."""
    x = np.asarray(x, dtype=float)
    if y is None:
        sd = np.std(x, ddof=1)
        return float(np.mean(x) / sd) if sd > 0 else 0.0
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    pooled_sd = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2))
    return float((np.mean(x) - np.mean(y)) / pooled_sd) if pooled_sd > 0 else 0.0


def holm_correction(p_values: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Holm (1979) step-down procedure. Returns a boolean reject-list in
    the *original* order of `p_values`."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    reject_sorted = np.zeros(m, dtype=bool)
    for k, idx in enumerate(order):
        threshold = alpha / (m - k)
        if p[idx] <= threshold:
            reject_sorted[idx] = True
        else:
            break  # Holm stops rejecting once one test fails
    return reject_sorted.tolist()


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Benjamini-Hochberg (1995) FDR-controlling procedure."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    below = sorted_p <= thresholds
    reject_sorted = np.zeros(m, dtype=bool)
    if np.any(below):
        max_k = np.max(np.where(below)[0])
        reject_sorted[: max_k + 1] = True
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject.tolist()


def coverage_rate(
    ground_truth_values: Sequence[float], ci_low: Sequence[float], ci_high: Sequence[float]
) -> float:
    """Fraction of cases where the CI contains the ground-truth value."""
    gt = np.asarray(ground_truth_values)
    lo = np.asarray(ci_low)
    hi = np.asarray(ci_high)
    return float(np.mean((gt >= lo) & (gt <= hi)))


def rmse(estimates: Sequence[float], ground_truth: Sequence[float]) -> float:
    est = np.asarray(estimates, dtype=float)
    gt = np.asarray(ground_truth, dtype=float)
    return float(np.sqrt(np.mean((est - gt) ** 2)))


def bias(estimates: Sequence[float], ground_truth: Sequence[float]) -> float:
    est = np.asarray(estimates, dtype=float)
    gt = np.asarray(ground_truth, dtype=float)
    return float(np.mean(est - gt))


def sign_accuracy(estimates: Sequence[float], ground_truth: Sequence[float], delta: float) -> float:
    """Fraction of cases where sign(estimate) matches sign(ground truth),
    using `delta` as the practical-significance threshold for both."""

    def sgn(v: float) -> int:
        if v > delta:
            return 1
        if v < -delta:
            return -1
        return 0

    est = [sgn(e) for e in estimates]
    gt = [sgn(g) for g in ground_truth]
    matches = [int(a == b) for a, b in zip(est, gt)]
    return float(np.mean(matches))
