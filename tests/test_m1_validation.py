"""Module 1 validation: propagator accuracy and reference-orbit identity."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from core.constants import L_STAR_KM, MOON_X, MU, SYNODIC_MONTH_DAYS, T_STAR_S
from core.nrho_ics import GATEWAY_NRHO_PERIOD, GATEWAY_NRHO_X0
from modules.m1_propagator import cr3bp_odes, jacobi_constant, propagate


@pytest.fixture(scope="module")
def one_period():
    """One full period of the reference NRHO, integrated once per module."""
    return propagate(GATEWAY_NRHO_X0, [0.0, GATEWAY_NRHO_PERIOD])


def test_periodicity_closure_under_one_metre(one_period):
    """Blueprint M1 gate (< 1 km), tightened: the corrector achieves < 1 m."""
    err_m = np.linalg.norm(one_period.y[:3, -1] - GATEWAY_NRHO_X0[:3]) * L_STAR_KM * 1e3
    assert err_m < 1.0, f"closure error {err_m:.4f} m exceeds 1 m"


def test_jacobi_constant_is_conserved(one_period):
    """Blueprint M1 gate: dC_J < 1e-10 over an uncontrolled orbit."""
    drift = abs(jacobi_constant(one_period.y[:, -1]) - jacobi_constant(GATEWAY_NRHO_X0))
    assert drift < 1e-10, f"Jacobi drift {drift:.3e} exceeds 1e-10"


def test_period_matches_92_synodic_resonance():
    """The reference must be Gateway's orbit, not merely *a* periodic NRHO."""
    expected = (2.0 * SYNODIC_MONTH_DAYS / 9.0) * 86400.0 / T_STAR_S
    assert abs(GATEWAY_NRHO_PERIOD - expected) < 1e-9


def test_geometry_matches_published_gateway_orbit(one_period):
    """Perilune/apolune radii must be consistent with NASA's published NRHO."""
    ts = np.linspace(0.0, GATEWAY_NRHO_PERIOD, 4000)
    r2 = np.linalg.norm(one_period.sol(ts)[:3].T - np.array([MOON_X, 0.0, 0.0]),
                        axis=1) * L_STAR_KM
    assert 3000.0 < r2.min() < 3600.0, f"perilune radius {r2.min():.0f} km off-family"
    assert 68000.0 < r2.max() < 74000.0, f"apolune radius {r2.max():.0f} km off-family"


def test_l2_lagrange_point_is_an_equilibrium():
    """Verifies the sign convention on the Coriolis terms."""
    x_l2 = brentq(lambda x: cr3bp_odes(0.0, [x, 0, 0, 0, 0, 0], MU)[3], 1.05, 1.30, xtol=1e-14)
    assert abs(x_l2 - 1.15568) < 1e-4, f"L2 at x={x_l2:.6f}, expected ~1.15568"
    accel = np.asarray(cr3bp_odes(0.0, [x_l2, 0, 0, 0, 0, 0], MU))[3:]
    assert np.allclose(accel, 0.0, atol=1e-12)


@pytest.mark.parametrize("method", ["DOP853", "RK45"])
def test_result_is_integrator_independent(method):
    """A result that depends on the integrator is a bug."""
    sol = propagate(GATEWAY_NRHO_X0, [0.0, GATEWAY_NRHO_PERIOD], method=method)
    err_km = np.linalg.norm(sol.y[:3, -1] - GATEWAY_NRHO_X0[:3]) * L_STAR_KM
    assert err_km < 1.0, f"{method} closure {err_km:.4f} km"


def test_uncontrolled_perturbation_diverges():
    """NRHOs are linearly unstable; if a perturbed orbit stays put, the
    reference orbit is wrong (blueprint M6 validation standard)."""
    t_end = 5.0 * GATEWAY_NRHO_PERIOD
    ts = np.linspace(0.0, t_end, 200)
    ref = propagate(GATEWAY_NRHO_X0, [0.0, t_end], t_eval=ts)
    per = propagate(GATEWAY_NRHO_X0 + np.array([0.5 / L_STAR_KM, 0, 0, 0, 0, 0]),
                    [0.0, t_end], t_eval=ts)
    sep = np.linalg.norm(ref.y[:3] - per.y[:3], axis=0) * L_STAR_KM
    growth = sep.max() / sep[0]
    assert growth > 10.0, (
        f"perturbation grew only {growth:.1f}x over 5 orbits "
        f"({sep[0]:.2f} -> {sep.max():.1f} km); reference may not be an NRHO"
    )


@pytest.mark.parametrize("bad", [np.zeros(5), np.zeros(7)])
def test_propagate_rejects_malformed_state(bad):
    with pytest.raises(ValueError):
        propagate(bad, [0.0, 1.0])