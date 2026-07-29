"""Figure generation: artifacts render and carry provenance."""

from __future__ import annotations

import numpy as np
import pytest

from modules.m6_figures import (
    plot_convergence, plot_distributions, plot_fault_recovery,
    plot_frontiers, plot_noise_robustness,
)
from modules.m6_montecarlo import convergence_trace


@pytest.fixture
def points():
    return {
        "PID": [(0.7, 5.0), (0.5, 6.5), (1.1, 4.1)],
        "LQR": [(0.4, 4.6), (0.3, 6.0), (0.8, 3.5)],
        "Targeting": [(0.25, 4.9), (0.2, 6.2), (0.5, 3.9)],
    }


def test_frontier_figure_is_written(config, points):
    p = plot_frontiers(points, config)
    assert p.exists() and p.stat().st_size > 1000


def test_noise_robustness_figure_is_written(config, points):
    p = plot_noise_robustness({"LOW": points, "MEDIUM": points, "HIGH": points}, config)
    assert p.exists()


def test_convergence_figure_is_written(config):
    rng = np.random.default_rng(0)
    traces = {k: convergence_trace(rng.normal(1, 0.2, 100)) for k in ("PID", "LQR")}
    assert plot_convergence(traces, config).exists()


def test_distribution_figure_is_written(config):
    rng = np.random.default_rng(1)
    assert plot_distributions({"PID": rng.lognormal(0, .3, 200),
                               "LQR": rng.lognormal(-.3, .3, 200)}, config).exists()


def test_fault_recovery_figure_is_written(config):
    rng = np.random.default_rng(2)
    assert plot_fault_recovery({"PID": rng.gamma(2, .3, 100),
                                "LQR": rng.gamma(2, .2, 100)}, config).exists()


def test_figures_handle_an_empty_policy(config):
    assert plot_frontiers({"PID": [], "LQR": [(0.4, 4.6), (0.3, 6.0)]}, config).exists()