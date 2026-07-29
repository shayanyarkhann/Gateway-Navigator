"""Statistical analysis: intervals, paired tests, frontiers (RMS 6.8)."""

from __future__ import annotations

import numpy as np
import pytest

from modules.m6_metrics import (
    bootstrap_ci, frontier_dominates, paired_comparison, pareto_frontier,
    standard_error_ci, summarise, summary_table,
)


def test_bootstrap_ci_covers_the_true_mean():
    rng = np.random.default_rng(0)
    hits = sum(
        bootstrap_ci(rng.normal(5.0, 1.0, 200), resamples=2000).lower <= 5.0
        <= bootstrap_ci(rng.normal(5.0, 1.0, 200), resamples=2000).upper
        for _ in range(20)
    )
    assert hits >= 15, f"nominal 95% interval covered only {hits}/20"


def test_bootstrap_is_reproducible():
    v = np.random.default_rng(3).lognormal(0, 0.5, 100)
    assert bootstrap_ci(v, resamples=2000) == bootstrap_ci(v, resamples=2000)


def test_bootstrap_handles_skew_asymmetrically():
    """On skewed data the interval should not be symmetric about the estimate."""
    v = np.random.default_rng(4).lognormal(0, 1.0, 500)
    ci = bootstrap_ci(v, resamples=4000)
    assert abs((ci.upper - ci.estimate) - (ci.estimate - ci.lower)) > 1e-3


def test_standard_error_ci_is_symmetric():
    ci = standard_error_ci(np.random.default_rng(5).normal(0, 1, 200))
    assert np.isclose(ci.upper - ci.estimate, ci.estimate - ci.lower)


@pytest.mark.parametrize("fn", [bootstrap_ci, standard_error_ci])
def test_intervals_reject_degenerate_input(fn):
    with pytest.raises(ValueError):
        fn([1.0])


def test_summary_reports_requested_percentiles():
    s = summarise(np.arange(1, 101.0), percentiles=(50.0, 95.0, 99.0),
                  bootstrap_resamples=2000)
    assert set(s.percentiles) == {"p50", "p95", "p99"}
    assert s.n == 100 and np.isclose(s.median, 50.5)


def test_paired_comparison_detects_a_real_difference():
    rng = np.random.default_rng(6)
    a = rng.lognormal(0, 0.3, 300)
    b = a * 0.75
    pc = paired_comparison(a, b, "PID", "LQR", bootstrap_resamples=2000)
    assert pc.p_value < 0.01
    assert pc.median_difference.estimate > 0
    assert "higher" in pc.interpretation


def test_paired_comparison_reports_comparable_when_equivalent():
    rng = np.random.default_rng(7)
    base = rng.lognormal(0, 0.3, 400)
    a = base + rng.normal(0, 0.01, 400)
    b = base + rng.normal(0, 0.01, 400)
    pc = paired_comparison(a, b, "PID", "LQR", bootstrap_resamples=2000)
    assert "comparable" in pc.interpretation


def test_paired_comparison_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="matched arrays"):
        paired_comparison(np.zeros(10), np.zeros(9))


def test_rank_biserial_saturates_for_uniform_dominance():
    a = np.arange(1.0, 51.0)
    pc = paired_comparison(a, a - 1.0, bootstrap_resamples=2000)
    assert np.isclose(pc.rank_biserial, 1.0)


def test_pareto_frontier_keeps_only_non_dominated_points():
    cost = np.array([1.0, 2.0, 3.0, 2.5])
    err = np.array([5.0, 3.0, 1.0, 4.0])
    keep = set(pareto_frontier(cost, err).tolist())
    assert keep == {0, 1, 2}, "point 3 is dominated by point 1 and must be dropped"


def test_pareto_frontier_handles_empty_and_single():
    assert pareto_frontier([], []).size == 0
    assert pareto_frontier([1.0], [1.0]).tolist() == [0]


def test_pareto_frontier_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        pareto_frontier([1.0, 2.0], [1.0])


def test_frontier_relations():
    assert frontier_dominates([1, 2, 3], [3, 2, 1], [2, 3, 4], [3, 2, 1]) == "A dominates"
    assert frontier_dominates([2, 3, 4], [3, 2, 1], [1, 2, 3], [3, 2, 1]) == "B dominates"
    assert frontier_dominates([1, 5], [1, 3], [4, 2], [1, 3]) == "frontiers cross"
    assert frontier_dominates([], [], [1], [1]) == "insufficient overlap"


def test_summary_table_renders():
    out = summary_table([{"policy": "PID", "dv": 0.5}, {"policy": "LQR", "dv": 0.2}])
    assert "policy" in out and "PID" in out and "LQR" in out
    assert summary_table([]) == "(no results)"