import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_discrete_are
from modules.m1_propagator import cr3bp_odes

from core.constants import L_STAR_KM, V_STAR_MS
from core.dynamics import state_transition_matrix

_POS_TOL: float = 1.0 / L_STAR_KM      # 1 km
_VEL_TOL: float = 0.1 / V_STAR_MS      # 0.1 m/s

_POS_TOL: float = 1.0 / L_STAR_KM      # 1 km
_VEL_TOL: float = 0.1 / V_STAR_MS      # 0.1 m/s




class LQRController:
    """
    Discrete-time LQR station-keeping controller for impulsive NRHO maintenance.

    Formulation: over one maneuver interval dt_maneuver, the CR3BP dynamics
    are linearized about the reference trajectory using the TRUE state
    transition matrix (from core.dynamics.state_transition_matrix, i.e. the properly integrated
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

    def _gain_is_stale(self, x_ref):
     if self._K is None:
        return True

     d = x_ref - self._K_ref_state

     return (
        np.linalg.norm(d[:3]) > _POS_TOL
        or np.linalg.norm(d[3:]) > _VEL_TOL
    )    

    def _gain_for(self, x_ref):
        """
        Recompute the LQR gain only if the reference state has moved enough to
        meaningfully change the local linearization. Since maneuvers happen at
        the same orbital phase (apolune) every orbit, x_ref is nearly identical
        each time, so in practice this integrates the STM once and reuses it.
        """
        if not self._gain_is_stale(x_ref):
         return self._K
        _, Phi = state_transition_matrix(x_ref, self.dt, self.mu)
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