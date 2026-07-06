import numpy as np

# Fixed seed — non-negotiable for reproducible research
np.random.seed(42)

# Noise level presets (non-dimensional units)
NOISE_LEVELS = {
    'LOW':    {'sigma_pos': 0.05 / 384400, 'sigma_vel': 0.005 / 1025, 'drift': False},
    'MEDIUM': {'sigma_pos': 0.10 / 384400, 'sigma_vel': 0.010 / 1025, 'drift': 'slow'},
    'HIGH':   {'sigma_pos': 0.50 / 384400, 'sigma_vel': 0.050 / 1025, 'drift': 'fast'},
}

DRIFT_PARAMS = {
    'slow': {'alpha': 0.02 / 384400, 'T_drift': 10.0},
    'fast': {'alpha': 0.10 / 384400, 'T_drift': 3.0},
}

def inject_noise(X_true, t, level='MEDIUM'):
    """
    Inject Gaussian noise and sensor drift into a true state vector.

    Parameters:
        X_true : array-like, shape (6,) — true state [x,y,z,xd,yd,zd]
        t      : float — current non-dimensional time
        level  : str — 'LOW', 'MEDIUM', or 'HIGH'

    Returns:
        X_meas : np.array, shape (6,) — noisy measurement
    """
    params = NOISE_LEVELS[level]
    X_true = np.array(X_true)
    X_meas = X_true.copy()

    # Position noise — Gaussian
    X_meas[0:3] += np.random.normal(0, params['sigma_pos'], 3)

    # Velocity noise — Gaussian
    X_meas[3:6] += np.random.normal(0, params['sigma_vel'], 3)

    # Sensor drift — sinusoidal bias on position
    if params['drift']:
        d = DRIFT_PARAMS[params['drift']]
        bias = d['alpha'] * np.sin(2 * np.pi * t / d['T_drift'])
        X_meas[0:3] += bias

    return X_meas

def inject_thruster_error(u, sigma_thr=0.02):
    """
    Apply multiplicative thruster error to control input.

    Parameters:
        u         : array-like, shape (3,) — commanded thrust [ux, uy, uz]
        sigma_thr : float — thruster noise standard deviation (default 2%)

    Returns:
        u_actual : np.array, shape (3,) — actual thrust after valve imprecision
    """
    u = np.array(u)
    return u * (1 + np.random.normal(0, sigma_thr, u.shape))