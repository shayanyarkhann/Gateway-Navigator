

"""Sensor and actuator error models.

All stochastic draws take an explicit ``numpy.random.Generator``. Nothing in
this module touches global NumPy RNG state: reproducibility is a property of
the generator handed in, not of interpreter-wide state, which is what makes
common random numbers across controller arms exact rather than incidental.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, SeedSequence
from numpy.typing import ArrayLike, NDArray

from core.constants import L_STAR_KM, V_STAR_MS

NOISE_LEVELS: dict[str, dict[str, float | str | None]] = {
    "LOW":    {"sigma_pos": 0.05 / L_STAR_KM, "sigma_vel": 0.005 / V_STAR_MS, "drift": None},
    "MEDIUM": {"sigma_pos": 0.10 / L_STAR_KM, "sigma_vel": 0.010 / V_STAR_MS, "drift": "slow"},
    "HIGH":   {"sigma_pos": 0.50 / L_STAR_KM, "sigma_vel": 0.050 / V_STAR_MS, "drift": "fast"},
}

DRIFT_PARAMS: dict[str, dict[str, float]] = {
    "slow": {"alpha": 0.02 / L_STAR_KM, "T_drift": 10.0},
    "fast": {"alpha": 0.10 / L_STAR_KM, "T_drift": 3.0},
}


def make_streams(seed: int) -> dict[str, Generator]:
    """Build independent, named RNG streams from one master seed.

    Each noise source draws from its own stream, spawned from a shared
    ``SeedSequence``. Two consequences that matter for Module 6:

    * Common random numbers are exact. Two controller arms given the same
      ``seed`` see byte-identical sensor noise even if they consume different
      numbers of *thruster* draws, because the streams are independent.
    * Monte Carlo runs parallelise safely -- no shared mutable state.

    Adding a new noise source means adding a key here; it cannot perturb the
    realisations seen by existing sources.
    """
    names = ("sensor", "thruster", "initial_state")
    children = SeedSequence(seed).spawn(len(names))
    return {name: np.random.default_rng(child) for name, child in zip(names, children)}


def inject_noise(
    X_true: ArrayLike,
    t: float,
    rng: Generator,
    level: str = "MEDIUM",
) -> NDArray[np.float64]:
    """Corrupt a true state with Gaussian sensor noise and slow bias drift.

    Parameters
    ----------
    X_true
        Six-element true state.
    t
        Non-dimensional time (drives the deterministic drift term).
    rng
        Generator for this run's sensor stream -- typically
        ``make_streams(seed)["sensor"]``.
    level
        One of ``NOISE_LEVELS``.

    Raises
    ------
    KeyError
        If ``level`` is not a known noise level.
    ValueError
        If ``X_true`` is not six-element.
    """
    if level not in NOISE_LEVELS:
        raise KeyError(f"unknown noise level {level!r}; expected one of {sorted(NOISE_LEVELS)}")
    X_true = np.asarray(X_true, dtype=float)
    if X_true.shape != (6,):
        raise ValueError(f"expected a 6-element state, got shape {X_true.shape}")

    params = NOISE_LEVELS[level]
    X_meas = X_true.copy()
    X_meas[0:3] += rng.normal(0.0, params["sigma_pos"], 3)
    X_meas[3:6] += rng.normal(0.0, params["sigma_vel"], 3)

    drift = params["drift"]
    if drift is not None:
        d = DRIFT_PARAMS[drift]
        # Per-axis phase offsets: a single scalar bias applied identically to
        # x, y and z would make the three axes perfectly correlated, which no
        # physical sensor triad exhibits (see GN-014).
        phases = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
        X_meas[0:3] += d["alpha"] * np.sin(2.0 * np.pi * t / d["T_drift"] + phases)

    return X_meas


def inject_thruster_error(
    u: ArrayLike,
    rng: Generator,
    sigma_thr: float = 0.02,
) -> NDArray[np.float64]:
    """Apply multiplicative valve imprecision to a commanded burn."""
    u = np.asarray(u, dtype=float)
    return u * (1.0 + rng.normal(0.0, sigma_thr, u.shape))