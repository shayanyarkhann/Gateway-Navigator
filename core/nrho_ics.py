"""Gateway NRHO reference orbit: initial conditions and differential corrector.

The Gateway station occupies the 9:2 synodic resonant southern L2 NRHO --
nine revolutions per two synodic months. The initial conditions below were
obtained by symmetric single-shooting (see `correct_nrho`) with the period
constrained to that resonance, and close to sub-millimetre accuracy over a
full period.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from core.constants import ATOL_TIGHT, MU, RTOL_TIGHT, SYNODIC_MONTH_DAYS, T_STAR_S

# --- Gateway 9:2 synodic resonant NRHO -------------------------------------
# Corrected 28 Jul 2026. Supersedes the blueprint values, which fail the
# 1 km periodicity gate by 3415 km (see Findings Register GN-001).
GATEWAY_NRHO_X0: NDArray[np.float64] = np.array([
    1.0220282893062722,   # x
    0.0,                  # y   (xz-plane crossing)
    -0.1821014455365690,  # z   (southern family)
    0.0,                  # vx  (perpendicular crossing)
    -0.1032711113333490,  # vy
    0.0,                  # vz
])

GATEWAY_NRHO_PERIOD: float = 1.5112004307151059  # non-dim; 6.56235 days

#: Target period from the 9:2 synodic resonance, in non-dimensional time.
NRHO_92_PERIOD: float = (2.0 * SYNODIC_MONTH_DAYS / 9.0) * 86400.0 / T_STAR_S


def _next_plane_crossing(
    X0: ArrayLike,
    mu: float = MU,
    t_max: float = 4.0,
    t_min: float = 0.1,
) -> tuple[float, NDArray[np.float64]]:
    """Propagate to the next y = 0 crossing after ``t_min``.

    The initial state itself lies on the xz-plane, so crossings at t ~ 0 are
    the starting point re-detected and must be skipped -- hence ``t_min``.

    Raises
    ------
    RuntimeError
        If integration fails or no crossing is found before ``t_max``.
    """
    def _event(t: float, X: NDArray[np.float64], mu: float) -> float:
        return X[1]

    _event.terminal = False
    _event.direction = 0.0

    sol = solve_ivp(
        _cr3bp_rhs, [0.0, t_max], np.asarray(X0, dtype=float), args=(mu,),
        method="DOP853", rtol=RTOL_TIGHT, atol=ATOL_TIGHT, events=_event,
    )
    if not sol.success:
        raise RuntimeError(f"crossing search failed to integrate: {sol.message}")

    for t_e, y_e in zip(sol.t_events[0], sol.y_events[0]):
        if t_e > t_min:
            return float(t_e), np.asarray(y_e, dtype=float)
    raise RuntimeError(f"no y=0 crossing found in t <= {t_max}")


def correct_nrho(
    guess: ArrayLike,
    target_period: float = NRHO_92_PERIOD,
    mu: float = MU,
) -> tuple[NDArray[np.float64], float]:
    """Differentially correct onto the periodic NRHO of a *specified* period.

    Symmetric single-shooting. The state is parameterised as
    ``[x, 0, z, 0, vy, 0]`` -- a perpendicular crossing of the xz-plane -- and
    three residuals are driven to zero at the next crossing:

    ``vx = 0``, ``vz = 0`` (perpendicular return, i.e. periodicity) and
    ``2 * t_half - target_period = 0`` (family selection).

    The third constraint is what distinguishes this from a bare periodicity
    corrector. Without it the NRHO family parameter is unconstrained and the
    solver converges to whichever member is nearest the initial guess -- which
    is how the repository ended up on a 6.275-day orbit instead of Gateway's
    6.562-day one.

    Parameters
    ----------
    guess
        Three-element ``[x, z, vy]`` starting estimate.
    target_period
        Full period to target, non-dimensional. Defaults to the 9:2 resonance.
    mu
        CR3BP mass parameter.

    Returns
    -------
    X0, period
        Six-element corrected state and the achieved full period.

    Raises
    ------
    RuntimeError
        If the Newton solve does not converge.
    """
    def _residuals(p: NDArray[np.float64]) -> list[float]:
        X0 = np.array([p[0], 0.0, p[1], 0.0, p[2], 0.0])
        t_half, X_half = _next_plane_crossing(X0, mu)
        return [X_half[3], X_half[5], 2.0 * t_half - target_period]

    p_sol, _, ier, msg = fsolve(
        _residuals, np.asarray(guess, dtype=float),
        full_output=True, xtol=1e-13,
    )
    if ier != 1:
        raise RuntimeError(f"NRHO corrector did not converge: {msg.strip()}")

    X0 = np.array([p_sol[0], 0.0, p_sol[1], 0.0, p_sol[2], 0.0])
    t_half, _ = _next_plane_crossing(X0, mu)
    return X0, 2.0 * t_half


def _cr3bp_rhs(t, X, mu):  # local import shim -- see note in GN-022
    from modules.m1_propagator import cr3bp_odes
    return cr3bp_odes(t, X, mu)
    