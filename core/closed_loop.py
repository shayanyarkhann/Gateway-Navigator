import numpy as np
from modules.m1_propagator import propagate
from modules.m2_noise import inject_noise, inject_thruster_error, NOISE_LEVELS
from modules.m3_kalman import KalmanFilter
from core.delta_v import apply_delta_v


def run_closed_loop(controller, X0, T_period, T_total, dt_sample, dt_maneuver,
                     noise_level='MEDIUM', mu=0.012150584, P0_scale=1e-8,
                     seed=None):
    """
    Simulate closed-loop NRHO station-keeping: truth propagation -> noisy
    sensing -> Kalman estimation -> periodic controller correction.

    This harness is shared by the PID and LQR controllers (and will be reused
    by the Module 6 comparison) so both run through IDENTICAL dynamics, noise
    realizations, and estimator settings -- any performance difference between
    controllers is then attributable to the control law itself, not to
    incidental differences in how each was simulated.

    Parameters:
        controller  : object with .compute_dv(x_hat, x_ref, t) -> dv, shape (3,)
        X0          : array-like, shape (6,) -- true NRHO initial condition
        T_period    : float -- nominal NRHO orbital period (non-dim)
        T_total     : float -- total simulated time (non-dim)
        dt_sample   : float -- time between sensor/KF updates
        dt_maneuver : float -- time between controller corrections
                      (must be an integer multiple of dt_sample)
        noise_level : str -- 'LOW', 'MEDIUM', or 'HIGH' (see m2_noise)
        mu          : float -- CR3BP mass parameter
        P0_scale    : float -- initial KF covariance (diagonal), non-dim units
        seed        : int or None -- reseeds numpy for a reproducible run

    Returns:
        dict with keys 't', 'X_true', 'X_est', 'X_ref', 'dv_history'
        (np.arrays, time-indexed along axis 0)
    """
    if seed is not None:
        np.random.seed(seed)

    steps_per_maneuver = round(dt_maneuver / dt_sample)
    assert abs(steps_per_maneuver * dt_sample - dt_maneuver) < 1e-9, \
        "dt_maneuver must be an integer multiple of dt_sample"

    # Reference trajectory: propagate once with dense output, then sample it
    # modulo the period since the NRHO is (near-)periodic by construction.
    ref_sol = propagate(X0, [0, T_period], method='DOP853')

    def x_ref_at(t):
        return ref_sol.sol(t % T_period)

    X_true = np.array(X0, dtype=float)

    # Process noise: small and non-zero -- the dynamics model is very accurate
    # (validated to 1e-12 in M1) but never treated as exact.
    Q_kf = np.diag([1e-16] * 3 + [1e-16] * 3)
    R_kf = np.diag(
        [NOISE_LEVELS[noise_level]['sigma_pos'] ** 2] * 3 +
        [NOISE_LEVELS[noise_level]['sigma_vel'] ** 2] * 3
    )
    P0 = np.eye(6) * P0_scale
    kf = KalmanFilter(X0=X_true, P0=P0, Q=Q_kf, R=R_kf, mu=mu)

    n_steps = round(T_total / dt_sample)

    t_hist, X_true_hist, X_est_hist, X_ref_hist, dv_hist = [], [], [], [], []
    t = 0.0

    for i in range(n_steps):
        # --- truth propagates under natural dynamics between maneuvers ---
        sol = propagate(X_true, [t, t + dt_sample], method='DOP853')
        X_true = sol.y[:, -1]
        t += dt_sample

        # --- sense + estimate ---
        z = inject_noise(X_true, t, level=noise_level)
        kf.predict(dt_sample)
        x_hat, _ = kf.update(z)

        # --- maneuver epoch? ---
        if (i + 1) % steps_per_maneuver == 0:
            x_ref_now = x_ref_at(t)
            dv_cmd = controller.compute_dv(x_hat, x_ref_now, t)
            dv_actual = inject_thruster_error(dv_cmd)

            X_true = apply_delta_v(X_true, dv_actual)   # spacecraft feels the real burn
            kf.x = apply_delta_v(kf.x, dv_cmd)           # filter only knows what it commanded
            dv_hist.append(dv_cmd)
        else:
            dv_hist.append(np.zeros(3))

        t_hist.append(t)
        X_true_hist.append(X_true.copy())
        X_est_hist.append(kf.x.copy())
        X_ref_hist.append(x_ref_at(t))

    return {
        't': np.array(t_hist),
        'X_true': np.array(X_true_hist),
        'X_est': np.array(X_est_hist),
        'X_ref': np.array(X_ref_hist),
        'dv_history': np.array(dv_hist),
    } 
