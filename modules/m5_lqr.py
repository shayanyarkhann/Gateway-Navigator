import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_discrete_are
from modules.m1_propagator import cr3bp_odes
from modules.m3_kalman import numerical_jacobian


def compute_true_stm(X0, dt, mu):
    """
    Numerically integrate the CR3BP state AND its 6x6 state transition matrix
    (STM) together via the variational equations:

        dX/dt   = f(X)
        dPhi/dt = A(X(t)) @ Phi,      Phi(0) = I_6

    This is the RIGOROUS linearization needed for control design over a long
    interval (a full orbital period, in our case). The single-step Euler
    approximation F = I + A*dt used in m3_kalman.py's state_transition_matrix
    is only first-order accurate and is fine for the Kalman filter's SHORT
    sensor-sampling steps (dt ~ T_period/15) -- but over a FULL period, A*dt
    is order ~2-3 and that approximation is no longer trustworthy. Reusing it
    here for the LQR's much longer maneuver interval gave visibly worse
    control performance during tuning, which is what motivated writing this
    proper variational-equations version instead.

    Parameters:
        X0 : array-like, shape (6,) -- state to linearize about
        dt : float -- interval to propagate (here, one maneuver period)
        mu : float -- CR3BP mass parameter

    Returns:
        X_end   : np.array, shape (6,)   -- nonlinear end state
        Phi_end : np.array, shape (6,6)  -- true state transition matrix over dt
    """
    def augmented_odes(t, y, mu):
        X = y[:6]
        Phi = y[6:].reshape(6, 6)
        dX = np.array(cr3bp_odes(t, X, mu))
        A = numerical_jacobian(X, mu)
        dPhi = A @ Phi
        return np.concatenate([dX, dPhi.flatten()])

    y0 = np.concatenate([np.asarray(X0, dtype=float), np.eye(6).flatten()])
    sol = solve_ivp(augmented_odes, [0, dt], y0, args=(mu,),
                     method='DOP853', rtol=1e-10, atol=1e-12)

    X_end = sol.y[:6, -1]
    Phi_end = sol.y[6:, -1].reshape(6, 6)
    return X_end, Phi_end


class LQRController:
    """
    Discrete-time LQR station-keeping controller for impulsive NRHO maintenance.

    Formulation: over one maneuver interval dt_maneuver, the CR3BP dynamics
    are linearized about the reference trajectory using the TRUE state
    transition matrix (from compute_true_stm, i.e. the properly integrated
    variational equations -- not the short-step Euler approximation used in
    the Kalman filter, which is not valid over an interval this long).

    An impulsive Delta-v applied at the START of the interval is equivalent to
    adding it to the velocity states before propagating forward by Phi:

        x_{k+1} = Phi @ (x_k + [0,0,0,dv]) = Phi @ x_k + Phi[:, 3:6] @ dv

    so the discrete-time control input matrix is B = Phi[:, 3:6]. The optimal
    feedback gain K is then the solution of the discrete algebraic Riccati
    equation for (Phi, B, Q, R).
    """

    def __init__(self, Q, R, dt_maneuver, mu):
        """
        Parameters:
            Q           : (6,6) array -- state error penalty (position/velocity weights)
            R           : (3,3) array -- control effort penalty (Delta-v cost weight)
            dt_maneuver : float -- non-dimensional time between corrections
                          (intended for once-per-orbit corrections at apolune)
            mu          : float -- CR3BP mass parameter
        """
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.dt = dt_maneuver
        self.mu = mu
        self._K = None
        self._K_ref_state = None

    def _gain_for(self, x_ref):
        """
        Recompute the LQR gain only if the reference state has moved enough to
        meaningfully change the local linearization. Since maneuvers happen at
        the same orbital phase (apolune) every orbit, x_ref is nearly identical
        each time, so in practice this integrates the STM once and reuses it.
        """
        if self._K is not None and np.linalg.norm(x_ref - self._K_ref_state) < 1e-3:
            return self._K

        _, Phi = compute_true_stm(x_ref, self.dt, self.mu)
        B = Phi[:, 3:6]

        P = solve_discrete_are(Phi, B, self.Q, self.R)
        K = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ Phi)

        self._K = K
        self._K_ref_state = np.asarray(x_ref, dtype=float).copy()
        return K

    def compute_dv(self, x_hat, x_ref, t=None):
        """
        Compute the commanded impulsive Delta-v for this maneuver epoch.

        Parameters:
            x_hat : array-like, shape (6,) -- current KF state estimate
            x_ref : array-like, shape (6,) -- nominal NRHO state at this epoch
            t     : unused, kept for interface parity with PIDController

        Returns:
            dv : np.array, shape (3,) -- commanded velocity correction
        """
        x_hat = np.asarray(x_hat, dtype=float)
        x_ref = np.asarray(x_ref, dtype=float)

        K = self._gain_for(x_ref)
        error = x_hat - x_ref
        return -K @ error 