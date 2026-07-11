import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from modules.m1_propagator import propagate
from modules.m5_lqr import LQRController
from core.closed_loop import run_closed_loop
from core.delta_v import delta_v_budget

MU = 0.012150584
L_STAR = 384400  # km

# Same NRHO IC/period validated in M1, same perturbation and cadence as M4
# (identical setup lets the PID and LQR results be compared apples-to-apples).
X0 = np.array([1.0170375034517611, 0.0, -0.1784174365278452, 0.0, -0.0921378916511874, 0.0])
T_PERIOD = 1.4451252712711455

np.random.seed(1)
pert = np.array([0.5 / L_STAR, 0.5 / L_STAR, 0.5 / L_STAR, 0, 0, 0])
X0_pert = X0 + pert

dt_sample = T_PERIOD / 15   # 15 KF updates per orbit
dt_maneuver = T_PERIOD      # 1 correction burn per orbit, at apolune
T_total = 15 * T_PERIOD     # ~15 orbits, ~94 days

# --- Baseline: identical perturbation, no control at all ---
sol_ref = propagate(X0, [0, T_total])
sol_pert = propagate(X0_pert, [0, T_total])
ts = np.linspace(0, T_total, 300)
uncontrolled_err_km = np.linalg.norm(
    sol_ref.sol(ts)[0:3] - sol_pert.sol(ts)[0:3], axis=0
) * L_STAR

# --- Controlled: LQR station-keeping, once per orbit at apolune ---
# Q = R = identity -- deliberately the simplest possible weighting, no hand
# tuning. Unlike the PID gains (which needed a real search to find a stable
# region), the LQR gain comes directly from solving the Riccati equation for
# the true linearized dynamics, so it doesn't need per-problem tuning to work
# well -- that's the whole point of using an optimal control formulation.
Q = np.eye(6)
R = np.eye(3)
lqr = LQRController(Q=Q, R=R, dt_maneuver=dt_maneuver, mu=MU)
res = run_closed_loop(lqr, X0_pert, T_PERIOD, T_total, dt_sample, dt_maneuver,
                       noise_level='MEDIUM', mu=MU, seed=1)

controlled_err_km = np.linalg.norm(res['X_true'][:, 0:3] - res['X_ref'][:, 0:3], axis=1) * L_STAR
dv_total_nondim = delta_v_budget(res['dv_history'])

rms_uncontrolled = np.sqrt(np.mean(uncontrolled_err_km ** 2))
rms_controlled = np.sqrt(np.mean(controlled_err_km ** 2))
max_controlled = controlled_err_km.max()

print(f"Uncontrolled RMS position error over 15 orbits : {rms_uncontrolled:10.3f} km")
print(f"Uncontrolled MAX position error over 15 orbits : {uncontrolled_err_km.max():10.3f} km")
print(f"LQR-controlled RMS position error              : {rms_controlled:10.3f} km")
print(f"LQR-controlled MAX position error              : {max_controlled:10.3f} km")
print(f"Total Delta-v budget (non-dim, ~x1.025 km/s)    : {dv_total_nondim:10.6f}")
print("  (must have controlled RMS < 5% of uncontrolled RMS, and controlled max < 100 km)")

if rms_controlled < 0.05 * rms_uncontrolled and max_controlled < 100.0:
    print("\nM5 LQR station-keeping PASSED -- controller measurably suppresses NRHO drift")
else:
    print("\nM5 LQR station-keeping FAILED")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            