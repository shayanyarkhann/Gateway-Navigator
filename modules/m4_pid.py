import numpy as np


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

    def __init__(self, Kp, Ki, Kd, dt_maneuver, integral_limit=None):
        """
        Parameters:
            Kp, Ki, Kd     : float or array-like(3,) -- gains, applied per-axis
                              (x,y,z) in the rotating synodic frame
            dt_maneuver    : float -- non-dimensional time between corrections;
                              needed to integrate the position error correctly
            integral_limit : float or None -- optional anti-windup clamp on the
                              norm of the accumulated integral term
        """
        self.Kp = np.asarray(Kp, dtype=float) * np.ones(3)
        self.Ki = np.asarray(Ki, dtype=float) * np.ones(3)
        self.Kd = np.asarray(Kd, dtype=float) * np.ones(3)
        self.dt = dt_maneuver
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
                    