"""Monte Carlo ensemble engine (RMS 6.6, 6.7).

Owns three things the comparison's validity rests on:

**Deterministic seed derivation.** Each trial's seed is a pure function of
``(base_seed, trial_index)`` and *nothing else*. It does not depend on the
policy, the tuning setting, or the noise level.

**Common random numbers.** Because the seed depends only on the trial index,
trial *i* presents byte-identical injection dispersion, sensor noise and
execution error to every policy at every tuning setting. Differences between
policies on a given trial are therefore attributable to the decision rule, not
sampling luck -- which is what makes the comparison *paired* (RMS 6.7) and
licenses the paired statistics in :mod:`modules.m6_metrics`.

**Convergence evidence.** RMS 6.6 requires showing that the running mean and
interval have stabilised as N grows, rather than assuming N = 500 suffices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from numpy.random import Generator, SeedSequence
from numpy.typing import NDArray

from core.closed_loop import run_closed_loop
from core.config import Config
from core.constants import L_STAR_KM, T_STAR_S, V_STAR_MS
from core.delta_v import delta_v_budget
from modules.m6_faults import NoFault, sample_fault

#: Streams drawn per trial, beyond those owned by the noise model. Named and
#: spawned separately so that adding a randomness source later cannot perturb
#: the realisations seen by existing ones.
_TRIAL_STREAMS = ("dispersion", "fault")


def derive_trial_seed(base_seed: int, trial_index: int) -> int:
    """Seed for one trial, from the base seed and trial index alone.

    Deliberately independent of policy and tuning: that independence *is* the
    common-random-numbers guarantee (RMS 6.7). Uses ``SeedSequence`` rather
    than arithmetic on the base seed, so that nearby trial indices produce
    well-separated, independent streams.
    """
    if trial_index < 0:
        raise ValueError(f"trial_index must be non-negative, got {trial_index}")
    child = SeedSequence(base_seed).spawn(trial_index + 1)[trial_index]
    return int(child.generate_state(1, dtype=np.uint32)[0])


def _trial_streams(seed: int) -> dict[str, Generator]:
    children = SeedSequence(seed).spawn(len(_TRIAL_STREAMS))
    return {n: np.random.default_rng(c) for n, c in zip(_TRIAL_STREAMS, children)}


@dataclass
class TrialResult:
    """Outcome of a single Monte Carlo trial."""

    trial_index: int
    seed: int
    policy: str
    setting: float
    noise_level: str
    delta_v_nondim: float
    delta_v_annual_ms: float
    rms_deviation_km: float
    max_deviation_km: float
    final_deviation_km: float
    estimation_rms_km: float
    fault_maneuver_index: int | None
    fault_severity: float | None
    diverged: bool

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnsembleResult:
    """All trials for one (policy, setting, noise level) operating point."""

    policy: str
    setting: float
    noise_level: str
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def delta_v(self) -> NDArray[np.float64]:
        return np.array([t.delta_v_annual_ms for t in self.trials])

    @property
    def deviation(self) -> NDArray[np.float64]:
        return np.array([t.rms_deviation_km for t in self.trials])

    @property
    def divergence_rate(self) -> float:
        return float(np.mean([t.diverged for t in self.trials])) if self.trials else 0.0


#: A policy factory maps a tuning setting to a controller object exposing
#: ``compute_dv(x_hat, x_ref, t)`` -- the only thing that varies across
#: policies in the common scaffold (RMS 6.2).
PolicyFactory = Callable[[float], Any]

#: Deviation beyond which a trial is recorded as diverged rather than
#: contributing a meaningless deviation number to the statistics.
DIVERGENCE_KM = 5_000.0


def run_trial(
    policy_name: str,
    factory: PolicyFactory,
    setting: float,
    trial_index: int,
    cfg: Config,
    X0: NDArray[np.float64],
    period: float,
    noise_level: str,
) -> TrialResult:
    """Execute one trial of one policy at one tuning setting.

    Every stochastic input is derived from ``derive_trial_seed``, so calling
    this with the same ``trial_index`` under a different policy reproduces the
    identical noise realisation.
    """
    seed = derive_trial_seed(cfg.monte_carlo.base_seed, trial_index)
    streams = _trial_streams(seed)

    disp = np.zeros(6)
    if cfg.dispersion.position_sigma_km > 0.0:
        disp[0:3] = streams["dispersion"].normal(
            0.0, cfg.dispersion.position_sigma_km / L_STAR_KM, 3)
    if cfg.dispersion.velocity_sigma_ms > 0.0:
        disp[3:6] = streams["dispersion"].normal(
            0.0, cfg.dispersion.velocity_sigma_ms / V_STAR_MS, 3)

    fault = sample_fault(cfg.fault, streams["fault"],
                         cfg.scaffold.maneuvers_per_revolution)

    controller = factory(setting)
    if hasattr(controller, "reset"):
        controller.reset()

    dt_sample = period / cfg.scaffold.samples_per_revolution
    dt_maneuver = period / cfg.scaffold.maneuvers_per_revolution
    t_total = cfg.scaffold.revolutions * period

    res = run_closed_loop(
        controller, X0, period, t_total, dt_sample, dt_maneuver,
        noise_level=noise_level, seed=seed, initial_dispersion=disp,
        fault=fault, q_accel=cfg.scaffold.q_accel,
        P0_scale=cfg.scaffold.p0_scale,
    )

    dev_km = np.linalg.norm(res["X_true"][:, 0:3] - res["X_ref"][:, 0:3], axis=1) * L_STAR_KM
    est_km = np.linalg.norm(res["X_true"][:, 0:3] - res["X_est"][:, 0:3], axis=1) * L_STAR_KM
    dv_nd = delta_v_budget(res["dv_history"])

    years = t_total * T_STAR_S / (86400.0 * 365.25)
    dv_annual = dv_nd * V_STAR_MS / years

    diverged = bool(dev_km.max() > DIVERGENCE_KM) or not np.all(np.isfinite(dev_km))

    return TrialResult(
        trial_index=trial_index, seed=seed, policy=policy_name, setting=setting,
        noise_level=noise_level,
        delta_v_nondim=float(dv_nd), delta_v_annual_ms=float(dv_annual),
        rms_deviation_km=float(np.sqrt(np.mean(dev_km**2))),
        max_deviation_km=float(dev_km.max()),
        final_deviation_km=float(dev_km[-1]),
        estimation_rms_km=float(np.sqrt(np.mean(est_km**2))),
        fault_maneuver_index=None if isinstance(fault, NoFault) else fault.maneuver_index,
        fault_severity=None if isinstance(fault, NoFault) else fault.severity,
        diverged=diverged,
    )


def run_ensemble(
    policy_name: str,
    factory: PolicyFactory,
    setting: float,
    cfg: Config,
    X0: NDArray[np.float64],
    period: float,
    noise_level: str,
    trials: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> EnsembleResult:
    """Run the full trial ensemble for one operating point."""
    n = trials if trials is not None else cfg.monte_carlo.trials
    out = EnsembleResult(policy=policy_name, setting=setting, noise_level=noise_level)
    for i in range(n):
        out.trials.append(
            run_trial(policy_name, factory, setting, i, cfg, X0, period, noise_level))
        if progress is not None:
            progress(i + 1, n)
    return out


def convergence_trace(
    values: Sequence[float],
    n_points: int = 10,
    confidence_level: float = 0.95,
) -> dict[str, NDArray[np.float64]]:
    """Running mean and standard-error band as the ensemble grows.

    RMS 6.6 requires *showing* that the estimate has stabilised rather than
    assuming it. Returns the sample sizes at which the running statistics were
    evaluated together with the mean and its interval, ready to plot.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 2:
        raise ValueError(f"need at least 2 values for a convergence trace, got {n}")

    from scipy.stats import norm
    z = float(norm.ppf(0.5 + confidence_level / 2.0))

    sizes = np.unique(np.linspace(max(2, n // n_points), n, n_points).astype(int))
    means = np.array([values[:k].mean() for k in sizes])
    sems = np.array([values[:k].std(ddof=1) / np.sqrt(k) for k in sizes])
    return {"n": sizes, "mean": means, "lower": means - z * sems, "upper": means + z * sems}


def is_converged(values: Sequence[float], tolerance: float = 0.02) -> bool:
    """Whether the running mean has stabilised within ``tolerance`` (relative).

    Compares the final mean against the mean over the first half of the
    ensemble. A crude but honest check: it detects a still-drifting estimate,
    which is the failure the RMS asks to guard against.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 4:
        return False
    half = values[: values.size // 2].mean()
    full = values.mean()
    if not np.isfinite(half) or not np.isfinite(full) or full == 0.0:
        return False
    return bool(abs(full - half) / abs(full) <= tolerance)