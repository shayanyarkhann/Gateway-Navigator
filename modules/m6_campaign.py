"""Campaign runner: one-command regeneration of every result and figure.

Implements the RMS Appendix C requirement that "a single entry point
regenerates every figure from scratch". Usage::

    python -m modules.m6_campaign configs/v1.yaml
    python -m modules.m6_campaign configs/v1-smoke.yaml --figures-only

The runner owns policy construction and sweep orchestration; it delegates trial
execution to :mod:`modules.m6_montecarlo`, statistics to
:mod:`modules.m6_metrics`, and artifacts to :mod:`modules.m6_export` and
:mod:`modules.m6_figures`, so no analysis logic lives here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from core.config import Config, load_config
from core.constants import MU
from core.nrho_ics import resolve_orbit
from modules.m4_pid import PIDController
from modules.m5_lqr import LQRController
from modules.m6_export import export_campaign, write_json
from modules.m6_metrics import (
    frontier_dominates, paired_comparison, pareto_frontier, summarise,
)
from modules.m6_montecarlo import (
    EnsembleResult, convergence_trace, is_converged, run_ensemble,
)
from modules.m6_targeting import TargetingController


def build_policies(cfg: Config, period: float) -> dict[str, tuple[Callable[[float], Any], tuple]]:
    """Map each policy name to (factory, sweep settings).

    The factory closes over everything held constant across the sweep, so the
    only thing that varies within a policy is its single tuning knob -- which
    is what makes the frontier a clean one-parameter family (RMS 6.5).
    """
    s = cfg.sweeps
    dt_maneuver = period / cfg.scaffold.maneuvers_per_revolution

    def pid(kp: float) -> PIDController:
        return PIDController(Kp=kp, Ki=s.pid_ki, Kd=kp * s.pid_kd_ratio,
                             dt_maneuver=dt_maneuver)

    def lqr(r_scale: float) -> LQRController:
        return LQRController(Q=np.eye(6), R=np.eye(3) * r_scale,
                             dt_maneuver=dt_maneuver, mu=MU)

    def targeting(coast: float) -> TargetingController:
        return TargetingController(coast_revolutions=int(coast), period=period, mu=MU)

    return {
        "PID": (pid, s.pid_kp),
        "LQR": (lqr, s.lqr_r_scale),
        "Targeting": (targeting, s.targeting_coast_revs),
    }


def run_campaign(cfg: Config, verbose: bool = True) -> list[EnsembleResult]:
    """Execute every (policy, setting, noise level) operating point."""
    X0, period = resolve_orbit(cfg.orbit.anchor)
    if verbose:
        print(f"reference orbit: anchor={cfg.orbit.anchor} period={period:.10f}")

    policies = build_policies(cfg, period)
    total = sum(len(settings) for _, settings in policies.values()) * len(cfg.monte_carlo.noise_levels)
    ensembles: list[EnsembleResult] = []
    done = 0

    for level in cfg.monte_carlo.noise_levels:
        for name, (factory, settings) in policies.items():
            for setting in settings:
                ens = run_ensemble(name, factory, float(setting), cfg, X0, period, level)
                ensembles.append(ens)
                done += 1
                if verbose:
                    print(f"  [{done}/{total}] {name:<10s} setting={setting:<8g} "
                          f"{level:<6s} dv={np.mean(ens.delta_v):8.3f} m/s/yr  "
                          f"dev={np.mean(ens.deviation):7.3f} km  "
                          f"div={ens.divergence_rate:.0%}")
    return ensembles


def analyse(ensembles: list[EnsembleResult], cfg: Config) -> dict[str, Any]:
    """Compute every statistic the RMS requires from the saved ensembles."""
    boot = cfg.stats.bootstrap_resamples
    pct = cfg.stats.percentiles
    conf = cfg.stats.confidence_level

    report: dict[str, Any] = {"operating_points": [], "paired": [], "frontiers": {}}

    for e in ensembles:
        if not e.trials:
            continue
        report["operating_points"].append({
            "policy": e.policy, "setting": e.setting, "noise_level": e.noise_level,
            "delta_v": summarise(e.delta_v, pct, conf, boot).to_dict(),
            "deviation": summarise(e.deviation, pct, conf, boot).to_dict(),
            "divergence_rate": e.divergence_rate,
            "converged": is_converged(e.delta_v),
        })

    # Frontier per (policy, noise level)
    for level in cfg.monte_carlo.noise_levels:
        per_policy: dict[str, list[tuple[float, float]]] = {}
        for e in ensembles:
            if e.noise_level != level or not e.trials:
                continue
            per_policy.setdefault(e.policy, []).append(
                (float(np.mean(e.delta_v)), float(np.mean(e.deviation))))
        report["frontiers"][level] = per_policy

        names = sorted(per_policy)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ca = [p[0] for p in per_policy[a]]; ea = [p[1] for p in per_policy[a]]
                cb = [p[0] for p in per_policy[b]]; eb = [p[1] for p in per_policy[b]]
                report["paired"].append({
                    "noise_level": level, "policy_a": a, "policy_b": b,
                    "frontier_relation": frontier_dominates(ca, ea, cb, eb),
                })

    # Paired per-trial comparison at each policy's best-accuracy setting
    for level in cfg.monte_carlo.noise_levels:
        best: dict[str, EnsembleResult] = {}
        for e in ensembles:
            if e.noise_level != level or not e.trials:
                continue
            cur = best.get(e.policy)
            if cur is None or np.mean(e.deviation) < np.mean(cur.deviation):
                best[e.policy] = e
        names = sorted(best)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ea, eb = best[a], best[b]
                if len(ea.trials) != len(eb.trials):
                    continue
                pc = paired_comparison(ea.delta_v, eb.delta_v, a, b, conf, boot)
                report["paired"].append({
                    "noise_level": level, "comparison": "tightest-accuracy setting",
                    **pc.to_dict(),
                })
    return report


def make_figures(ensembles: list[EnsembleResult], report: dict[str, Any], cfg: Config) -> list[str]:
    """Regenerate every figure from the saved ensembles."""
    from modules.m6_figures import (
        plot_convergence, plot_distributions, plot_frontiers, plot_noise_robustness,
    )
    paths: list[str] = []
    level = cfg.monte_carlo.noise_levels[len(cfg.monte_carlo.noise_levels) // 2]

    if level in report["frontiers"]:
        paths.append(str(plot_frontiers(report["frontiers"][level], cfg, level)))
    if len(report["frontiers"]) > 1:
        paths.append(str(plot_noise_robustness(report["frontiers"], cfg)))

    traces, samples = {}, {}
    for e in ensembles:
        if e.noise_level != level or not e.trials:
            continue
        cur = samples.get(e.policy)
        if cur is None or np.mean(e.deviation) < cur[1]:
            samples[e.policy] = (e.delta_v, float(np.mean(e.deviation)))
            if len(e.delta_v) >= 4:
                traces[e.policy] = convergence_trace(
                    e.delta_v, cfg.monte_carlo.convergence_check_points,
                    cfg.stats.confidence_level)
    if traces:
        paths.append(str(plot_convergence(traces, cfg)))
    if samples:
        paths.append(str(plot_distributions({k: v[0] for k, v in samples.items()}, cfg)))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path, help="path to the campaign YAML")
    parser.add_argument("--trials", type=int, default=None,
                        help="override the ensemble size (for smoke runs)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.trials is not None:
        object.__setattr__(cfg.monte_carlo, "trials", args.trials)

    verbose = not args.quiet
    if verbose:
        print(f"campaign {cfg.name}  fingerprint {cfg.fingerprint()}")

    ensembles = run_campaign(cfg, verbose)
    report = analyse(ensembles, cfg)
    written = export_campaign(ensembles, cfg)
    write_json(report, Path(cfg.outputs.results_dir) / f"{cfg.name}_report.json", cfg)
    figures = make_figures(ensembles, report, cfg)

    if verbose:
        print("\nartifacts:")
        for k, v in written.items():
            print(f"  {k:<12s} {v}")
        for f in figures:
            print(f"  figure       {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())