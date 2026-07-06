import numpy as np
from scipy.integrate import solve_ivp

MU = 0.012150584  # Earth-Moon mass parameter

def cr3bp_odes(t, X, mu=MU):
    """CR3BP equations of motion in the rotating synodic frame."""
    x, y, z, xd, yd, zd = X

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)       # Distance to Earth
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)   # Distance to Moon

    xdd = 2*yd + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    ydd = -2*xd + y - (1-mu)*y/r1**3 - mu*y/r2**3
    zdd = -(1-mu)*z/r1**3 - mu*z/r2**3

    return [xd, yd, zd, xdd, ydd, zdd]

def jacobi_constant(X, mu=MU):
    """Compute the Jacobi constant — conserved along uncontrolled trajectories."""
    x, y, z, xd, yd, zd = X

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    omega = 0.5*(x**2 + y**2) + (1-mu)/r1 + mu/r2
    return 2*omega - (xd**2 + yd**2 + zd**2)

def propagate(X0, t_span, t_eval=None, method='DOP853'):
    """Integrate CR3BP trajectory from initial conditions."""
    sol = solve_ivp(
        cr3bp_odes,
        t_span,
        X0,
        method=method,
        rtol=1e-12,
        atol=1e-14,
        t_eval=t_eval,
        dense_output=True
    )
    return sol