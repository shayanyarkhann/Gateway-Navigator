

import numpy as np
from modules.m1_propagator import propagate, jacobi_constant
from core.nrho_ics import GATEWAY_NRHO_PERIOD, GATEWAY_NRHO_X0

# NRHO Initial Conditions (non-dimensional)

X0 = GATEWAY_NRHO_X0
T_PERIOD = GATEWAY_NRHO_PERIOD
L_STAR = 384400    # km

sol = propagate(X0, [0, T_PERIOD])

X_end = sol.y[:, -1]

# Gate 1: Position return error
pos_error_nd = np.linalg.norm(X_end[:3] - np.array(X0[:3]))
pos_error_km = pos_error_nd * L_STAR

# Gate 2: Jacobi drift
CJ_start = jacobi_constant(X0)
CJ_end   = jacobi_constant(X_end)
jacobi_drift = abs(CJ_end - CJ_start)

print(f"Position return error : {pos_error_km:.4f} km  (must be < 1 km)")
print(f"Jacobi drift          : {jacobi_drift:.2e}     (must be < 1e-10)")

if pos_error_km < 1.0 and jacobi_drift < 1e-10:
    print("\n M1 VALIDATION PASSED — cleared to build M2")
else:
    print("\n M1 VALIDATION FAILED — do not proceed")