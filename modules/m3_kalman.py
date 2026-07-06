import numpy as np
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
        F = state_transition_matrix(self.x, dt, self.mu)
        self.x = F @ self.x
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

        


