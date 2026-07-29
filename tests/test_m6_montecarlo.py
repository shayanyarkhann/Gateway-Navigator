"""Monte Carlo engine: seed discipline and common random numbers (RMS 6.7)."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import load_config
from tests.conftest import CONFIG_DIR
from modules.m6_montecarlo import (
    convergence_trace, derive_trial_seed, is_converged,
)


def test_seed_derivation_is_deterministic():
    a = [derive_trial_seed(20260707, i) for i in range(50)]
    b = [derive_trial_seed(20260707, i) for i in range(50)]
    assert a == b


def test_seeds_are_distinct_across_trials():
    seeds = [derive_trial_seed(20260707, i) for i in range(500)]
    assert len(set(seeds)) == 500, "seed collisions would break trial independence"


def test_different_base_seeds_give_different_streams():
    a = [derive_trial_seed(1, i) for i in range(20)]
    b = [derive_trial_seed(2, i) for i in range(20)]
    assert not set(a) & set(b)


def test_seed_depends_only_on_trial_index():
    """The common-random-numbers guarantee (RMS 6.7).

    Nothing about the policy or tuning setting may enter the seed, or paired
    comparison silently degrades to an unpaired one.
    """
    import inspect
    sig = inspect.signature(derive_trial_seed)
    assert list(sig.parameters) == ["base_seed", "trial_index"]


def test_negative_trial_index_is_rejected():
    with pytest.raises(ValueError):
        derive_trial_seed(1, -1)


def test_convergence_trace_brackets_the_mean():
    rng = np.random.default_rng(0)
    vals = rng.normal(5.0, 1.0, 400)
    tr = convergence_trace(vals, n_points=8)
    assert tr["n"][-1] == 400
    assert np.all(tr["lower"] <= tr["mean"]) and np.all(tr["mean"] <= tr["upper"])
    assert abs(tr["mean"][-1] - vals.mean()) < 1e-12


def test_convergence_interval_narrows_with_n():
    rng = np.random.default_rng(1)
    tr = convergence_trace(rng.normal(0, 1, 800), n_points=10)
    width = tr["upper"] - tr["lower"]
    assert width[-1] < width[0]


def test_convergence_trace_needs_two_values():
    with pytest.raises(ValueError):
        convergence_trace([1.0])


def test_is_converged_detects_stability_and_drift():
    rng = np.random.default_rng(2)
    assert is_converged(rng.normal(10.0, 0.1, 500))
    assert not is_converged(np.linspace(1.0, 100.0, 500))


def test_smoke_config_is_small_enough_to_run():
    cfg = load_config(CONFIG_DIR / "v1-smoke.yaml")
    assert cfg.monte_carlo.trials <= 50 and cfg.scaffold.revolutions <= 10