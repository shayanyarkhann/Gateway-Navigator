"""CR3BP linearisation: analytic Jacobian and state transition matrix.

Single source of truth for linearised dynamics. Both the Kalman filter
(covariance propagation) and the LQR controller (discrete-time plant model)
consume the STM from here, so the two can never drift out of agreement.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from core.constants import ATOL_STM, MU, RTOL_STM

_CORIOLIS = np.array([[0.0, 2.0, 0.0],
                      [-2.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])


def cr3bp_jacobian(X: ArrayLike, mu: float = MU) -> NDArray[np.float64]:
    """Analytic 6x6 Jacobian of the CR3BP vector field at ``X``.

    Block structure::

        A = [[    0      I    ]
             [ d2(Omega) Omega_c ]]

    where the lower-left block is the Hessian of the effective potential and
    the lower-right block holds the Coriolis terms. Computed in closed form
    rather than by finite differences: exact, and free of the step-size
    trade-off that makes differencing worst near perilune, where the second
    derivatives of the potential are largest (see GN-011).
    """
    X = np.asarray(X, dtype=float)
    r1_vec = np.array([X[0] + mu, X[1], X[2]])
    r2_vec = np.array([X[0] - 1.0 + mu, X[1], X[2]])
    r1 = float(np.linalg.norm(r1_vec))
    r2 = float(np.linalg.norm(r2_vec))
    if r1 < 1e-10 or r2 < 1e-10:
        raise ValueError(f"state is singular at a primary (r1={r1:.3e}, r2={r2:.3e})")

    eye3 = np.eye(3)
    hess = np.diag([1.0, 1.0, 0.0])
    hess -= (1.0 - mu) / r1**3 * (eye3 - 3.0 * np.outer(r1_vec, r1_vec) / r1**2)
    hess -= mu / r2**3 * (eye3 - 3.0 * np.outer(r2_vec, r2_vec) / r2**2)

    A = np.zeros((6, 6))
    A[0:3, 3:6] = eye3
    A[3:6, 0:3] = hess
    A[3:6, 3:6] = _CORIOLIS
    return A


def state_transition_matrix(
    X0: ArrayLike,
    dt: float,
    mu: float = MU,
    rtol: float = RTOL_STM,
    atol: float = ATOL_STM,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate the state and its STM over ``dt`` via the variational equations.

    Solves the augmented system::

        dX/dt   = f(X)
        dPhi/dt = A(X(t)) @ Phi,    Phi(0) = I_6

    This is exact to integrator tolerance for arbitrary ``dt``, unlike the
    first-order form ``I + A*dt``, which requires ``||A*dt|| << 1``. On this
    NRHO ``||A*dt||`` reaches ~6.9e3 at perilune at dt = T/15, where the
    first-order form carries >1300% error (GN-002).

    Returns
    -------
    X_end, Phi
        Nonlinear end state and the 6x6 transition matrix over ``dt``.

    Raises
    ------
    RuntimeError
        If the augmented integration fails.
    """
    X0 = np.asarray(X0, dtype=float)
    if X0.shape != (6,):
        raise ValueError(f"expected a 6-element state, got shape {X0.shape}")
    if dt == 0.0:
        return X0.copy(), np.eye(6)

    from modules.m1_propagator import cr3bp_odes

    def _augmented(t: float, y: NDArray[np.float64], mu: float) -> NDArray[np.float64]:
        X = y[:6]
        phi = y[6:].reshape(6, 6)
        return np.concatenate([
            np.asarray(cr3bp_odes(t, X, mu), dtype=float),
            (cr3bp_jacobian(X, mu) @ phi).ravel(),
        ])

    y0 = np.concatenate([X0, np.eye(6).ravel()])
    sol = solve_ivp(_augmented, [0.0, dt], y0, args=(mu,),
                    method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"STM integration failed: {sol.message}")

    return sol.y[:6, -1], sol.y[6:, -1].reshape(6, 6)