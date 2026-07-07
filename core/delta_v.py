import numpy as np


def apply_delta_v(X, dv):
    """
    Apply an instantaneous velocity correction (impulsive maneuver) to a state.

    Position is unchanged -- an impulsive burn only changes velocity, on the
    assumption that the burn duration is short compared to orbital timescales
    (standard assumption for station-keeping maneuvers).

    Parameters:
        X  : array-like, shape (6,) -- state [x,y,z,xd,yd,zd] before the burn
        dv : array-like, shape (3,) -- velocity change [dvx,dvy,dvz], non-dim

    Returns:
        X_new : np.array, shape (6,) -- state immediately after the burn
    """
    X_new = np.array(X, dtype=float).copy()
    X_new[3:6] += np.asarray(dv, dtype=float)
    return X_new


def delta_v_budget(dv_history):
    """
    Total propellant cost of a maneuver sequence, non-dimensional velocity units.

    Each burn costs propellant regardless of direction, so magnitudes are
    summed (not vector-summed) -- this is the standard mission Delta-v budget.

    Parameters:
        dv_history : array-like, shape (N, 3) -- every commanded burn in a run
                     (zero-vectors for non-maneuver steps are fine, they add 0)

    Returns:
        total_dv : float
    """
    dv_history = np.atleast_2d(np.asarray(dv_history, dtype=float))
    if dv_history.size == 0:
        return 0.0
    return float(np.sum(np.linalg.norm(dv_history, axis=1))) 

