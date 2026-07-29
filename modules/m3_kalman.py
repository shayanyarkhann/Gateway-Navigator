"""Extended Kalman filter for CR3BP state estimation.

Consumes the linearisation from :mod:`core.dynamics` rather than maintaining
its own. The previous local ``numerical_jacobian`` / ``state_transition_matrix``
pair used the first-order form ``F = I + A*dt``, which requires ``||A*dt|| << 1``
-- on this NRHO that quantity reaches ~2.9e3 at perilune, where the first-order
form carries ~990% error and drove mean NEES to 162 against a target of 6
(GN-002). Both are removed; there is now one linearisation path in the
codebase, shared with the LQR controller (GN-022).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.dynamics import state_transition_matrix


class KalmanFilter:
    """Extended Kalman filter with full-state measurements.

    The mean is propagated through the true nonlinear dynamics and the
    covariance through the true state transition matrix, both obtained from a
    single integration of the augmented variational system -- so the mean and
    its linearisation are guaranteed consistent, at the cost of one integration
    per predict step rather than the two the previous implementation used.
    """

    def __init__(self, X0: ArrayLike, P0: ArrayLike, Q: ArrayLike,
                 R: ArrayLike, mu: float) -> None:
        """Initialise the filter.

        Parameters
        ----------
        X0
            Six-element initial state estimate.
        P0, Q, R
            6x6 initial state covariance, process noise, measurement noise.
        mu
            CR3BP mass parameter.

        Raises
        ------
        ValueError
            If any argument has the wrong shape.
        """
        self.x = np.array(X0, dtype=float)
        if self.x.shape != (6,):
            raise ValueError(f"expected a 6-element state, got shape {self.x.shape}")
        self.P = np.array(P0, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        for name, M in (("P0", self.P), ("Q", self.Q), ("R", self.R)):
            if M.shape != (6, 6):
                raise ValueError(f"{name} must be 6x6, got shape {M.shape}")
        self.mu = float(mu)
        self.H = np.eye(6)  # all six states directly measured

    def predict(self, dt: float) -> NDArray[np.float64]:
        """Propagate the estimate and its covariance forward by ``dt``.

        Raises
        ------
        ValueError
            If ``dt`` is not positive.
        RuntimeError
            If the augmented integration fails.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.x, F = state_transition_matrix(self.x, dt, self.mu)
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)
        return self.x

    def apply_maneuver(self, dv_cmd: ArrayLike, sigma_thr: float = 0.02) -> None:
        """Fold a commanded burn in, inflating covariance for execution error.

        The filter knows what it *commanded*, not what the thrusters delivered,
        so the mean shifts by ``dv_cmd`` while the velocity block grows by the
        execution-error covariance ``diag((sigma_thr * dv_cmd)^2)``. Omitting
        this leaves the filter overconfident after every burn (GN-004).
        """
        dv_cmd = np.asarray(dv_cmd, dtype=float)
        if dv_cmd.shape != (3,):
            raise ValueError(f"expected a 3-element dv, got shape {dv_cmd.shape}")
        if sigma_thr < 0.0:
            raise ValueError(f"sigma_thr must be non-negative, got {sigma_thr}")
        self.x[3:6] += dv_cmd
        self.P[3:6, 3:6] += np.diag((sigma_thr * dv_cmd) ** 2)
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Correct the estimate with measurement ``z``.

        Uses the Joseph-form covariance update, which stays symmetric and
        positive semi-definite for any gain rather than only the exactly
        optimal one, and solves the innovation system instead of forming
        ``inv(S)`` explicitly (GN-005).

        Returns
        -------
        x, innovation
            Updated estimate and measurement residual. The innovation is
            returned so callers can run whiteness and NEES diagnostics.
        """
        z = np.asarray(z, dtype=float)
        if z.shape != (6,):
            raise ValueError(f"expected a 6-element measurement, got shape {z.shape}")

        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T

        innovation = z - self.H @ self.x
        self.x = self.x + K @ innovation

        IKH = np.eye(6) - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return self.x, innovation
        


