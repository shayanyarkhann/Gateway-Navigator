import numpy as np
from numpy.typing import ArrayLike
from core.constants import L_STAR_KM

_DEFAULT_INTEGRAL_LIMIT: float = 100.0 / L_STAR_KM


class PIDController:
    """
    PID station-keeping controller for impulsive NRHO maintenance burns.

    Operates on the POSITION error between the KF state estimate and the
    reference (nominal NRHO) trajectory, and outputs a velocity correction
    (an impulsive Delta-v) rather than a continuous force -- matching how
    real Gateway station-keeping burns are executed operationally.

    Design choice: the derivative term uses the ESTIMATED VELOCITY ERROR
    directly (x_ref[3:6] - x_hat[3:6]) instead of finite-differencing the
    position error frame-to-frame. Velocity is already the time-derivative
    of position, and we already have a Kalman-filtered velocity estimate --
    so this gives the "D" term for free without amplifying sensor noise
    through numerical differentiation.
    """

    def __init__(
        self,
        Kp: float | ArrayLike,
        Ki: float | ArrayLike,
        Kd: float | ArrayLike,
        dt_maneuver: float,
        integral_limit: float | None = _DEFAULT_INTEGRAL_LIMIT,
    ) -> None:
        """Construct a three-axis PID station-keeping controller.

        Parameters
        ----------
        Kp, Ki, Kd
            Scalar or three-element per-axis gains, synodic frame.
        dt_maneuver
            Non-dimensional interval between corrections.
        integral_limit
            Anti-windup clamp on the Euclidean norm of the accumulated
            integral term. Defaults to a finite value rather than ``None``:
            an unclamped integrator diverges in any scenario with sustained
            un-nullable error (thruster loss, axis loss, fuel cap), and
            opting *out* should be the deliberate choice (GN-007).
            Pass ``None`` explicitly to disable.

        Raises
        ------
        ValueError
            If ``dt_maneuver`` is non-positive or ``integral_limit`` negative.
        """
        if dt_maneuver <= 0.0:
            raise ValueError(f"dt_maneuver must be positive, got {dt_maneuver}")
        if integral_limit is not None and integral_limit < 0.0:
            raise ValueError(f"integral_limit must be non-negative, got {integral_limit}")

        self.Kp = np.asarray(Kp, dtype=float) * np.ones(3)
        self.Ki = np.asarray(Ki, dtype=float) * np.ones(3)
        self.Kd = np.asarray(Kd, dtype=float) * np.ones(3)
        self.dt = float(dt_maneuver)
        self.integral_limit = integral_limit
        self._integral = np.zeros(3)

    def reset(self):
        """Zero the accumulated integral term (call at the start of a new run)."""
        self._integral = np.zeros(3)

    def compute_dv(self, x_hat, x_ref, t=None):
        """
        Compute the commanded impulsive Delta-v for this maneuver epoch.

        Parameters:
            x_hat : array-like, shape (6,) -- current KF state estimate
            x_ref : array-like, shape (6,) -- nominal NRHO state at this epoch
            t     : unused, kept for interface parity with LQRController

        Returns:
            dv : np.array, shape (3,) -- commanded velocity correction
        """
        x_hat = np.asarray(x_hat, dtype=float)
        x_ref = np.asarray(x_ref, dtype=float)

        e_pos = x_ref[0:3] - x_hat[0:3]
        e_vel = x_ref[3:6] - x_hat[3:6]

        self._integral += e_pos * self.dt
        if self.integral_limit is not None:
            norm = np.linalg.norm(self._integral)
            if norm > self.integral_limit:
                self._integral *= self.integral_limit / norm

        return self.Kp * e_pos + self.Ki * self._integral + self.Kd * e_vel              
                    