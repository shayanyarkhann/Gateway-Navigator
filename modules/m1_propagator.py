import numpy as np
from scipy.integrate import solve_ivp
from typing import Sequence
from numpy.typing import ArrayLike
from scipy.integrate._ivp.ivp import OdeResult

from core.constants import (
    MU,
    RTOL_TIGHT,
    ATOL_TIGHT,
)

def cr3bp_odes(t, X, mu=MU):
    """CR3BP equations of motion in the rotating synodic frame."""
    x, y, z, xd, yd, zd = X

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)       # Distance to Earth
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)   # Distance to Moon

    xdd = 2*yd + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    ydd = -2*xd + y - (1-mu)*y/r1**3 - mu*y/r2**3
    zdd = -(1-mu)*z/r1**3 - mu*z/r2**3

    return np.array(
    [xd, yd, zd, xdd, ydd, zdd],
    dtype=float,
)

def jacobi_constant(X, mu=MU):
    """Compute the Jacobi constant — conserved along uncontrolled trajectories."""
    x, y, z, xd, yd, zd = X

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    omega = 0.5*(x**2 + y**2) + (1-mu)/r1 + mu/r2
    return 2*omega - (xd**2 + yd**2 + zd**2)

def propagate(
    X0: ArrayLike,
    t_span: Sequence[float],
    t_eval: ArrayLike | None = None,
    method: str = "DOP853",
    rtol: float = RTOL_TIGHT,
    atol: float = ATOL_TIGHT,
    dense_output: bool = True,
    mu: float = MU,
) -> OdeResult:
    """Integrate the CR3BP equations of motion from ``X0`` over ``t_span``.

    Parameters
    ----------
    X0
        Six-element initial state ``[x, y, z, vx, vy, vz]``, non-dimensional.
    t_span
        ``(t0, tf)`` in non-dimensional time.
    t_eval
        Optional times at which to store the solution.
    method
        SciPy integrator. DOP853 is the default: at the tolerances used here an
        8th-order method takes far fewer steps than RK45 through perilune.
    dense_output
        Build an interpolant. Costs memory and time; pass ``False`` when only
        the end state is needed.
    mu
        CR3BP mass parameter.

    Returns
    -------
    OdeResult
        The SciPy solution object, guaranteed to have ``success = True``.

    Raises
    ------
    ValueError
        If ``X0`` is not six-element or ``t_span`` is malformed.
    RuntimeError
        If the integrator fails to reach ``tf``.
    """
    X0 = np.asarray(X0, dtype=float)
    if X0.shape != (6,):
        raise ValueError(f"expected a 6-element state, got shape {X0.shape}")
    if len(t_span) != 2:
        raise ValueError(f"t_span must be (t0, tf), got {t_span!r}")

    sol = solve_ivp(
        cr3bp_odes, t_span, X0, args=(mu,), method=method,
        rtol=rtol, atol=atol, t_eval=t_eval, dense_output=dense_output,
    )
    if not sol.success:
        raise RuntimeError(
            f"CR3BP integration failed over {tuple(t_span)}: {sol.message}. "
            f"Initial state: {X0}"
        )
    return sol 