import numpy as np
from scipy.integrate import solve_ivp
from modules.m1_propagator import cr3bp_odes
from numpy.typing import ArrayLike, NDArray

def numerical_jacobian(X, mu, eps=1e-7):
    # finite-difference approximation of the 6x6 system Jacobian
    # this tells us how small changes in state affect the dynamics,
    # which is what lets us linearize the CR3BP for the filter
    f0 = np.array(cr3bp_odes(0, X, mu))
    A = np.zeros((6, 6))
    for i in range(6):
        X_perturbed = X.copy()
        X_perturbed[i] += eps
        f1 = np.array(cr3bp_odes(0, X_perturbed, mu))
        A[:, i] = (f1 - f0) / eps
    return A


def state_transition_matrix(X, dt, mu):
    # first-order approximation: F ~ I + A*dt
    # good enough as long as dt stays small relative to orbital dynamics
    A = numerical_jacobian(X, mu)
    return np.eye(6) + A * dt


class KalmanFilter:
    def __init__(self, X0, P0, Q, R, mu):
        self.x = np.array(X0, dtype=float)
        self.P = P0
        self.Q = Q
        self.R = R
        self.mu = mu
        self.H = np.eye(6)  # we're assuming all 6 states are directly measurable

    def predict(self, dt):
        # Extended Kalman filter: the MEAN is propagated through the true
        # nonlinear dynamics (not the linearized F), while F is used ONLY to
        # propagate the covariance. F = I + A*dt is a first-order-accurate
        # approximation to the flow -- fine for covariance bookkeeping, but
        # applying it directly to the mean (F @ x) reintroduces uncorrected
        # linearization error every single step. Near the Moon, |A| can be
        # several units, so that error compounds fast across many predict
        # calls and blows up -- this was caught by testing the filter in
        # closed loop on the real NRHO, not by the SHO sanity check (where
        # the SHO dynamics are exactly linear, so F @ x was exact there).
        F = state_transition_matrix(self.x, dt, self.mu)

        # Note: deliberately NOT reusing m1_propagator.propagate() here -- that
        # function is fixed at rtol=1e-12/atol=1e-14 for M1's periodicity
        # validation, which is far tighter than a filter needs and is very
        # expensive when called every predict() step in a long closed-loop
        # run. rtol=1e-10 is still ~4 orders of magnitude better than the old
        # F@x approximation while running fast enough for real-time filtering.
        sol = solve_ivp(cr3bp_odes, [0, dt], self.x, args=(self.mu,),
                         method='RK45', rtol=1e-10, atol=1e-12)
        self.x = sol.y[:, -1]
        self.P = F @ self.P @ F.T + self.Q
        return self.x
    def apply_maneuver(
        self,
        dv_cmd: ArrayLike,
        sigma_thr: float = 0.02,
    ) -> None:
        """Fold a commanded burn into the estimate, inflating covariance for execution error.

        The filter knows what it *commanded*, not what the thrusters actually
        delivered, so the mean shifts by ``dv_cmd`` while the velocity block of
        the covariance grows by the execution-error covariance
        ``diag((sigma_thr * dv_cmd)^2)``. Omitting this inflation leaves the
        filter overconfident immediately after every maneuver (GN-004).

        Parameters
        ----------
        dv_cmd
            Three-element commanded velocity increment, non-dimensional.
        sigma_thr
            Multiplicative 1-sigma thruster execution error. Must match the
            value used by ``inject_thruster_error``.
        """
        dv_cmd = np.asarray(dv_cmd, dtype=float)
        if dv_cmd.shape != (3,):
            raise ValueError(f"expected a 3-element dv, got shape {dv_cmd.shape}")
        if sigma_thr < 0.0:
            raise ValueError(f"sigma_thr must be non-negative, got {sigma_thr}")

        self.x[3:6] += dv_cmd
        self.P[3:6, 3:6] += np.diag((sigma_thr * dv_cmd) ** 2)
        self.P = 0.5 * (self.P + self.P.T)
    def update(
        self,
        z: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Correct the estimate with measurement ``z``.

        Uses the Joseph-form covariance update, which stays symmetric and
        positive semi-definite for *any* gain rather than only the exactly
        optimal one, and solves the innovation system instead of forming
        ``inv(S)`` explicitly.

        Returns
        -------
        x, innovation
            Updated estimate and the measurement residual. The innovation is
            returned so callers can run whiteness and NEES diagnostics.

        Raises
        ------
        ValueError
            If ``z`` is not a 6-element measurement.
        numpy.linalg.LinAlgError
            If the innovation covariance is singular.
        """
        z = np.asarray(z, dtype=float)
        if z.shape != (6,):
            raise ValueError(f"expected a 6-element measurement, got shape {z.shape}")

        S = self.H @ self.P @ self.H.T + self.R
        # K = P H^T S^-1, obtained as the solution of S^T K^T = (P H^T)^T
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T

        innovation = z - self.H @ self.x
        self.x = self.x + K @ innovation

        # Joseph form: P = (I - KH) P (I - KH)^T + K R K^T
        IKH = np.eye(6) - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return self.x, innovation
        


