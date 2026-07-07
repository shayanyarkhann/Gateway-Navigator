import numpy as np
from scipy.integrate import solve_ivp
from modules.m1_propagator import cr3bp_odes


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

    def update(self, z):
        # z is the noisy measurement coming in from m2_noise
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        innovation = z - self.H @ self.x
        self.x = self.x + K @ innovation
        self.P = (np.eye(6) - K @ self.H) @ self.P

        return self.x, innovation       

        


