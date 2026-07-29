"""Statistical analysis of Monte Carlo ensembles (RMS 6.5, 6.8).

Implements the reporting contract the RMS imposes:

* **Distributions, not point estimates.** Every performance number carries an
  interval and upper percentiles, because Delta-v distributions are skewed and
  worst-case maintenance cost is tail-driven (RMS 6.6).
* **Paired, non-parametric comparison.** Trials use common random numbers, so
  policies are compared per-trial via the Wilcoxon signed-rank test rather than
  an unpaired or normality-assuming test (RMS 6.8).
* **Effect size with an interval, not a p-value alone.** "How much cheaper,
  with what uncertainty" is the scientific claim; significance is not
  (RMS 6.8).
* **Frontiers, not points.** The unit of comparison is the lower envelope of
  the (Delta-v, accuracy) cloud, and overlapping frontiers are reported as
  "comparable in this regime" rather than collapsed to a winner (RMS 6.5).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str

    def __str__(self) -> str:
        return f"{self.estimate:.4g} [{self.lower:.4g}, {self.upper:.4g}]"


@dataclass(frozen=True)
class Summary:
    """Distributional summary of one metric over one ensemble."""

    n: int
    mean: float
    median: float
    std: float
    percentiles: dict[str, float]
    ci: Interval

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ci"] = asdict(self.ci)
        return d


@dataclass(frozen=True)
class PairedComparison:
    """Paired per-trial comparison of two policies at matched conditions."""

    policy_a: str
    policy_b: str
    n_pairs: int
    median_difference: Interval
    wilcoxon_statistic: float
    p_value: float
    rank_biserial: float
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["median_difference"] = asdict(self.median_difference)
        return d


def bootstrap_ci(
    values: ArrayLike,
    statistic=np.mean,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap confidence interval.

    Preferred over a standard-error interval because Delta-v distributions are
    right-skewed, where the normal approximation understates the upper tail
    (RMS 6.8). Uses a fixed seed so intervals are themselves reproducible.

    Raises
    ------
    ValueError
        If fewer than two values are supplied.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError(f"need at least 2 values to bootstrap, got {values.size}")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(resamples, values.size))
    draws = statistic(values[idx], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lo, hi = np.percentile(draws, [100 * alpha, 100 * (1 - alpha)])
    return Interval(float(statistic(values)), float(lo), float(hi),
                    confidence_level, "percentile bootstrap")


def standard_error_ci(
    values: ArrayLike, confidence_level: float = 0.95
) -> Interval:
    """Normal-approximation interval on the mean.

    The sanctioned fallback under the RMS 3.5 cut policy when bootstrap
    resampling is too expensive.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError(f"need at least 2 values, got {values.size}")
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(values.size))
    z = float(stats.norm.ppf(0.5 + confidence_level / 2.0))
    return Interval(mean, mean - z * sem, mean + z * sem,
                    confidence_level, "standard error")


def summarise(
    values: ArrayLike,
    percentiles: Sequence[float] = (50.0, 95.0, 99.0),
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 10_000,
    use_bootstrap: bool = True,
) -> Summary:
    """Full distributional summary of one metric."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot summarise an empty ensemble")
    ci = (bootstrap_ci(values, resamples=bootstrap_resamples,
                       confidence_level=confidence_level)
          if use_bootstrap and values.size >= 2
          else standard_error_ci(values, confidence_level))
    return Summary(
        n=int(values.size), mean=float(values.mean()),
        median=float(np.median(values)), std=float(values.std(ddof=1)) if values.size > 1 else 0.0,
        percentiles={f"p{p:g}": float(np.percentile(values, p)) for p in percentiles},
        ci=ci,
    )


def paired_comparison(
    a: ArrayLike,
    b: ArrayLike,
    policy_a: str = "A",
    policy_b: str = "B",
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 10_000,
) -> PairedComparison:
    """Compare two policies trial-by-trial under common random numbers.

    Both arrays must be ordered by trial index, so that ``a[i]`` and ``b[i]``
    come from the identical noise realisation. Under that pairing the
    difference removes the between-trial variance that would otherwise swamp
    the between-policy signal.

    Uses Wilcoxon signed-rank rather than a paired *t*-test because Delta-v
    differences should not be assumed normal (RMS 6.8), and reports the median
    paired difference with a bootstrap interval as the effect size.

    The rank-biserial correlation is included as a scale-free effect size:
    it is the difference between the proportion of pairs favouring each policy,
    so +1 means A exceeded B on every trial and -1 the reverse.

    Raises
    ------
    ValueError
        If the arrays differ in length or are shorter than two elements.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"paired comparison requires matched arrays; got {a.shape} and {b.shape}. "
            "Both must be ordered by trial index for common random numbers to pair."
        )
    if a.size < 2:
        raise ValueError(f"need at least 2 pairs, got {a.size}")

    diff = a - b
    nonzero = diff[diff != 0.0]
    if nonzero.size == 0:
        stat, p = 0.0, 1.0
        rbc = 0.0
    else:
        stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        ranks = stats.rankdata(np.abs(nonzero))
        r_plus = ranks[nonzero > 0].sum()
        r_minus = ranks[nonzero < 0].sum()
        total = r_plus + r_minus
        rbc = float((r_plus - r_minus) / total) if total > 0 else 0.0

    med = bootstrap_ci(diff, statistic=np.median, resamples=bootstrap_resamples,
                       confidence_level=confidence_level, seed=1)

    if med.lower <= 0.0 <= med.upper:
        interp = (f"{policy_a} and {policy_b} are comparable: the interval on the "
                  "median paired difference spans zero")
    elif med.estimate < 0.0:
        interp = f"{policy_a} is lower than {policy_b} by {abs(med.estimate):.4g} (median paired)"
    else:
        interp = f"{policy_a} is higher than {policy_b} by {med.estimate:.4g} (median paired)"

    return PairedComparison(
        policy_a=policy_a, policy_b=policy_b, n_pairs=int(a.size),
        median_difference=med, wilcoxon_statistic=float(stat), p_value=float(p),
        rank_biserial=rbc, interpretation=interp,
    )


def pareto_frontier(
    cost: ArrayLike,
    error: ArrayLike,
) -> NDArray[np.intp]:
    """Indices of the non-dominated (lower-envelope) operating points.

    A point is dominated if another achieves both lower cost *and* lower error.
    The surviving set is the policy's achievable frontier in the Delta-v /
    accuracy plane (RMS 6.5) -- the unit of comparison, in place of a single
    hand-picked operating point.

    Returns
    -------
    Indices of the non-dominated points, sorted by increasing cost.
    """
    cost = np.asarray(cost, dtype=float)
    error = np.asarray(error, dtype=float)
    if cost.shape != error.shape:
        raise ValueError(f"cost and error must match; got {cost.shape} and {error.shape}")
    if cost.size == 0:
        return np.array([], dtype=np.intp)

    order = np.argsort(cost, kind="stable")
    keep: list[int] = []
    best_error = np.inf
    for i in order:
        # Strictly-better error is required, so ties do not both survive.
        if error[i] < best_error:
            keep.append(int(i))
            best_error = error[i]
    return np.array(keep, dtype=np.intp)


def frontier_dominates(
    cost_a: ArrayLike, error_a: ArrayLike,
    cost_b: ArrayLike, error_b: ArrayLike,
    tolerance: float = 0.0,
) -> str:
    """Classify the relationship between two frontiers.

    Returns one of ``"A dominates"``, ``"B dominates"``, ``"frontiers cross"``,
    or ``"insufficient overlap"``. RMS 6.5 is explicit that a crossing is a
    legitimate and often more honest result than a clean winner, so this
    reports the crossing rather than forcing a verdict.
    """
    ca, ea = np.asarray(cost_a, float), np.asarray(error_a, float)
    cb, eb = np.asarray(cost_b, float), np.asarray(error_b, float)
    if ca.size == 0 or cb.size == 0:
        return "insufficient overlap"

    lo = max(ea.min(), eb.min())
    hi = min(ea.max(), eb.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return "insufficient overlap"

    grid = np.linspace(lo, hi, 50)
    # Interpolate cost as a function of accuracy along each frontier.
    ia, ib = np.argsort(ea), np.argsort(eb)
    a_cost = np.interp(grid, ea[ia], ca[ia])
    b_cost = np.interp(grid, eb[ib], cb[ib])

    a_better = a_cost < b_cost * (1.0 - tolerance)
    b_better = b_cost < a_cost * (1.0 - tolerance)
    if a_better.all():
        return "A dominates"
    if b_better.all():
        return "B dominates"
    if a_better.any() and b_better.any():
        return "frontiers cross"
    return "comparable"


def summary_table(rows: Sequence[dict[str, Any]]) -> str:
    """Render summary rows as a fixed-width table for the README and logs."""
    if not rows:
        return "(no results)"
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), max(len(f"{r[c]}") for r in rows)) for c in cols}
    head = "  ".join(f"{c:<{widths[c]}}" for c in cols)
    rule = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join("  ".join(f"{r[c]!s:<{widths[c]}}" for c in cols) for r in rows)
    return f"{head}\n{rule}\n{body}"