import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from modules.m1_propagator import propagate
from modules.m4_pid import PIDController
from core.closed_loop import run_closed_loop
from core.delta_v import delta_v_budget

MU = 0.012150584
L_STAR = 384400  # km

# Same NRHO IC/period validated in M1
X0 = np.array([1.0170375034517611, 0.0, -0.1784174365278452, 0.0, -0.0921378916511874, 0.0])
T_PERIOD = 1.4451252712711455

# A small insertion/nav error (~0.87 km in position, zero velocity error).
# NRHOs are linearly UNSTABLE -- this tiny perturbation is enough to make the
# uncontrolled trajectory diverge by thousands of km within ~15 orbits, which
# is exactly why real Gateway station-keeping is needed even with no other
# perturbations modeled (solar radiation pressure, third-body, etc.).
np.random.seed(1)
pert = np.array([0.5 / L_STAR, 0.5 / L_STAR, 0.5 / L_STAR, 0, 0, 0])
X0_pert = X0 + pert

# Maneuver cadence: once per orbit, AT APOLUNE (t=0, T_PERIOD, 2*T_PERIOD, ...).
# Firing at a fixed absolute-time cadence instead of at a fixed orbital phase
# was tried first and was unstable over long horizons: it periodically fires
# near perilune (~2,500 km lunar altitude on this orbit), where the CR3BP
# dynamics are most sensitive and a single-snapshot correction can overshoot
# badly. Apolune (~69,500 km out) is the standard, calmer choice used in real
# NRHO maintenance design.
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

# --- Controlled: PID station-keeping, once per orbit at apolune ---
pid = PIDController(Kp=0.4, Ki=0.0, Kd=0.5, dt_maneuver=dt_maneuver)
res = run_closed_loop(pid, X0_pert, T_PERIOD, T_total, dt_sample, dt_maneuver,
                       noise_level='MEDIUM', mu=MU, seed=1)

controlled_err_km = np.linalg.norm(res['X_true'][:, 0:3] - res['X_ref'][:, 0:3], axis=1) * L_STAR
dv_total_nondim = delta_v_budget(res['dv_history'])

rms_uncontrolled = np.sqrt(np.mean(uncontrolled_err_km ** 2))
rms_controlled = np.sqrt(np.mean(controlled_err_km ** 2))
max_controlled = controlled_err_km.max()

print(f"Uncontrolled RMS position error over 15 orbits : {rms_uncontrolled:10.3f} km")
print(f"Uncontrolled MAX position error over 15 orbits : {uncontrolled_err_km.max():10.3f} km")
print(f"PID-controlled RMS position error              : {rms_controlled:10.3f} km")
print(f"PID-controlled MAX position error               : {max_controlled:10.3f} km")
print(f"Total Delta-v budget (non-dim, ~x1.025 km/s)    : {dv_total_nondim:10.6f}")
print("  (must have controlled RMS < 5% of uncontrolled RMS, and controlled max < 100 km)")

if rms_controlled < 0.05 * rms_uncontrolled and max_controlled < 100.0:
    print("\nM4 PID station-keeping PASSED -- controller measurably suppresses NRHO drift")
else:
    print("\nM4 PID station-keeping FAILED")
    