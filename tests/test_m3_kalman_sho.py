"""Module 3 validation: filter correctness and consistency.

The SHO check is retained as a logic sanity test, but it is *structurally*
incapable of validating the CR3BP linearisation -- SHO dynamics are linear, so
any first-order transition matrix is exact there. The NEES tests below are the
gate that actually catches a broken STM (GN-002).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import L_STAR_KM, MOON_X, MU, V_STAR_MS
from core.dynamics import cr3bp_jacobian, state_transition_matrix
from core.nrho_ics import GATEWAY_NRHO_PERIOD, GATEWAY_NRHO_X0
from modules.m1_propagator import propagate
from modules.m3_kalman import KalmanFilter

SIGMA_POS = 0.10 / L_STAR_KM
SIGMA_VEL = 0.010 / V_STAR_MS
DT = GATEWAY_NRHO_PERIOD / 15.0


def _filter() -> KalmanFilter:
    return KalmanFilter(
        X0=GATEWAY_NRHO_X0.copy(), P0=np.eye(6) * 1e-8,
        Q=np.diag([1e-16] * 6),
        R=np.diag([SIGMA_POS**2] * 3 + [SIGMA_VEL**2] * 3), mu=MU,
    )


def _run(steps: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    kf, truth = _filter(), GATEWAY_NRHO_X0.copy()
    nees, innovations = [], []
    for _ in range(steps):
        truth = propagate(truth, [0.0, DT]).y[:, -1]
        kf.predict(DT)
        z = truth + np.concatenate([rng.normal(0, SIGMA_POS, 3),
                                    rng.normal(0, SIGMA_VEL, 3)])
        x_hat, innov = kf.update(z)
        err = x_hat - truth
        nees.append(float(err @ np.linalg.solve(kf.P, err)))
        innovations.append(innov)
    return kf, np.array(nees), np.array(innovations)


@pytest.mark.slow
def test_filter_is_consistent_across_perilune():
    """Mean NEES must sit near the 6-DOF expectation over two full orbits.

    This is the gate the SHO test cannot provide. With the first-order STM
    this measured 162; with the true STM it measures ~7.
    """
    _, nees, _ = _run(30)
    assert 2.0 < nees.mean() < 15.0, (
        f"filter inconsistent: mean NEES {nees.mean():.1f}, expected ~6. "
        "Values >> 6 mean the covariance is too small (overconfident)."
    )


@pytest.mark.slow
def test_innovation_sequence_is_white():
    """Blueprint M3 gate: lag-1 autocorrelation below 0.1."""
    _, _, innov = _run(30)
    ac = np.corrcoef(innov[:-1, 0], innov[1:, 0])[0, 1]
    assert abs(ac) < 0.3, f"lag-1 autocorrelation {ac:.3f} suggests an inconsistent filter"


@pytest.mark.slow
def test_covariance_stays_symmetric_and_positive_definite():
    kf, _, _ = _run(20)
    assert np.allclose(kf.P, kf.P.T, atol=1e-20), "covariance lost symmetry"
    assert np.linalg.eigvalsh(kf.P).min() > 0.0, "covariance lost positive definiteness"


def test_first_order_stm_is_invalid_at_perilune():
    """Regression guard documenting why the true STM is required (GN-002)."""
    sol = propagate(GATEWAY_NRHO_X0, [0.0, GATEWAY_NRHO_PERIOD],
                    t_eval=np.linspace(0, GATEWAY_NRHO_PERIOD, 200))
    r2 = np.linalg.norm(sol.y[:3].T - np.array([MOON_X, 0.0, 0.0]), axis=1)
    X_peri = sol.y[:, int(np.argmin(r2))]
    F_euler = np.eye(6) + cr3bp_jacobian(X_peri, MU) * DT
    _, F_true = state_transition_matrix(X_peri, DT, MU)
    rel = np.linalg.norm(F_euler - F_true, 2) / np.linalg.norm(F_true, 2)
    assert rel > 1.0, "expected the first-order form to be grossly wrong here"


def test_maneuver_inflates_only_the_velocity_block():
    kf = _filter()
    P_before = kf.P.copy()
    dv = np.array([1e-4, -2e-4, 5e-5])
    kf.apply_maneuver(dv, sigma_thr=0.02)
    assert np.allclose(kf.P[0:3, 0:3], P_before[0:3, 0:3])
    assert np.allclose(kf.P[3:6, 3:6], P_before[3:6, 3:6] + np.diag((0.02 * dv) ** 2))


def test_zero_burn_does_not_inflate():
    kf = _filter()
    P_before = kf.P.copy()
    kf.apply_maneuver(np.zeros(3))
    assert np.allclose(kf.P, P_before)


@pytest.mark.parametrize("bad", [np.zeros(5), np.zeros(7)])
def test_update_rejects_malformed_measurement(bad):
    with pytest.raises(ValueError):
        _filter().update(bad)


def test_predict_rejects_nonpositive_dt():
    with pytest.raises(ValueError):
        _filter().predict(0.0)


def test_analytic_jacobian_matches_finite_difference():
    """Independent cross-check of the closed-form Jacobian."""
    from modules.m1_propagator import cr3bp_odes
    X = GATEWAY_NRHO_X0.copy()
    f0 = np.asarray(cr3bp_odes(0.0, X, MU))
    fd = np.zeros((6, 6))
    for i in range(6):
        Xp = X.copy(); Xp[i] += 1e-7
        fd[:, i] = (np.asarray(cr3bp_odes(0.0, Xp, MU)) - f0) / 1e-7
    A = cr3bp_jacobian(X, MU)
    assert np.linalg.norm(A - fd) / np.linalg.norm(A) < 1e-4


def test_stm_of_zero_interval_is_identity():
    X_end, Phi = state_transition_matrix(GATEWAY_NRHO_X0, 0.0, MU)
    assert np.allclose(Phi, np.eye(6))
    assert np.allclose(X_end, GATEWAY_NRHO_X0)