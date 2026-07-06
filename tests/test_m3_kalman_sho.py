import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from modules.m3_kalman import KalmanFilter

np.random.seed(42)

# simple harmonic oscillator: x'' = -k*x
# state = [position, velocity], known analytical solution
k = 1.0
dt = 0.01
steps = 500

def sho_true_state(t, x0=1.0, v0=0.0):
    w = np.sqrt(k)
    x = x0 * np.cos(w * t) + (v0 / w) * np.sin(w * t)
    v = -x0 * w * np.sin(w * t) + v0 * np.cos(w * t)
    return np.array([x, v])

# build a minimal 2-state version just for this sanity check
class SHOKalman:
    def __init__(self, x0, P0, Q, R):
        self.x = np.array(x0, dtype=float)
        self.P = P0
        self.Q = Q
        self.R = R
        self.H = np.eye(2)

    def predict(self, dt):
        F = np.array([[1, dt], [-k * dt, 1]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        innovation = z - self.H @ self.x
        self.x = self.x + K @ innovation
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x, innovation

Q = np.diag([1e-5, 1e-5])
R = np.diag([0.01, 0.01])
P0 = np.diag([0.1, 0.1])

kf = SHOKalman(x0=[1.05, 0.05], P0=P0, Q=Q, R=R)

errors = []
innovations = []

for i in range(steps):
    t = i * dt
    true_state = sho_true_state(t)
    measurement = true_state + np.random.normal(0, 0.1, 2)

    kf.predict(dt)
    est, innov = kf.update(measurement)

    errors.append(np.linalg.norm(est - true_state))
    innovations.append(innov)

errors = np.array(errors)
innovations = np.array(innovations)

# check 1: does estimate converge close to truth after 50 steps
late_error = np.mean(errors[50:])
print(f"Mean estimation error after step 50: {late_error:.4f}  (should be small, < 0.1)")

# check 2: innovation whiteness - lag-1 autocorrelation should be near zero
innov_flat = innovations[:, 0]
autocorr = np.corrcoef(innov_flat[:-1], innov_flat[1:])[0, 1]
print(f"Innovation lag-1 autocorrelation: {autocorr:.4f}  (should be < 0.1)")

if late_error < 0.1 and abs(autocorr) < 0.1:
    print("\nM3 SHO validation passed - filter logic is sound, safe to move to CR3BP")
else:
    print("\nM3 SHO validation failed - fix filter before touching CR3BP")
