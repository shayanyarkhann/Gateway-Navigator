"""Module 2 validation: noise statistics and RNG stream discipline."""

from __future__ import annotations

import numpy as np
import pytest

from modules.m2_noise import NOISE_LEVELS, inject_noise, inject_thruster_error, make_streams

X_TRUE = np.array([1.02, 0.0, -0.18, 0.0, -0.10, 0.0])


@pytest.mark.parametrize("level", ["LOW", "MEDIUM", "HIGH"])
def test_noise_is_zero_mean(level):
    """Blueprint M2 gate: mean of 10,000 samples within 3 sigma / sqrt(N)."""
    rng = make_streams(42)["sensor"]
    n = 10_000
    noise = np.array([inject_noise(X_TRUE, 0.0, rng, level) for _ in range(n)]) - X_TRUE
    sp, sv = NOISE_LEVELS[level]["sigma_pos"], NOISE_LEVELS[level]["sigma_vel"]
    assert abs(noise[:, 0:3].mean()) < 3 * sp / np.sqrt(n)
    assert abs(noise[:, 3:6].mean()) < 3 * sv / np.sqrt(n)


@pytest.mark.parametrize("level", ["LOW", "MEDIUM", "HIGH"])
def test_noise_standard_deviation_matches_spec(level):
    rng = make_streams(1)["sensor"]
    noise = np.array([inject_noise(X_TRUE, 0.0, rng, level) for _ in range(20_000)]) - X_TRUE
    assert np.isclose(noise[:, 0:3].std(), NOISE_LEVELS[level]["sigma_pos"], rtol=0.05)
    assert np.isclose(noise[:, 3:6].std(), NOISE_LEVELS[level]["sigma_vel"], rtol=0.05)


def test_streams_are_reproducible():
    a, b = make_streams(42), make_streams(42)
    assert np.array_equal(a["sensor"].normal(size=10), b["sensor"].normal(size=10))


def test_common_random_numbers_survive_divergent_thruster_draws():
    """The property Module 6's paired comparison depends on: arm B burns twice
    as often as arm A, yet both must see identical sensor noise."""
    arm_a, arm_b = make_streams(42), make_streams(42)
    for i in range(20):
        z_a = inject_noise(X_TRUE, i * 0.1, arm_a["sensor"])
        inject_thruster_error(np.ones(3), arm_a["thruster"])
        z_b = inject_noise(X_TRUE, i * 0.1, arm_b["sensor"])
        inject_thruster_error(np.ones(3), arm_b["thruster"])
        inject_thruster_error(np.ones(3), arm_b["thruster"])
        assert np.array_equal(z_a, z_b), f"CRN broken at step {i}"


def test_no_global_rng_mutation():
    """Exercising the noise model must not touch global NumPy state."""
    np.random.seed(123)
    before = np.random.random()
    np.random.seed(123)
    inject_noise(X_TRUE, 0.0, make_streams(7)["sensor"])
    assert np.random.random() == before


def test_drift_axes_are_not_perfectly_correlated():
    rng = make_streams(3)["sensor"]
    clean = inject_noise(X_TRUE, 2.5, rng, "LOW")
    drifted = inject_noise(X_TRUE, 2.5, make_streams(3)["sensor"], "HIGH")
    d = drifted[0:3] - clean[0:3]
    assert not np.allclose(d[0], d[1], rtol=1e-6)


def test_rejects_unknown_noise_level():
    with pytest.raises(KeyError):
        inject_noise(X_TRUE, 0.0, make_streams(1)["sensor"], "EXTREME")