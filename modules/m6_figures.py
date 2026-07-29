"""Publication figure generation (RMS 6.5, Appendix C).

Every figure is built from *saved trial records*, never by re-running a
simulation. This is a hard architectural rule: it makes regeneration cheap,
guarantees that a figure and the table beside it were computed from identical
data, and means a reviewer can regenerate the plots without a compute budget.

Each figure carries its provenance stamp (config fingerprint, commit, seed) in
the footer, satisfying the RMS requirement that a figure record the settings
that produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")  # headless: figures are artifacts, not an interactive session
import matplotlib.pyplot as plt
import numpy as np

from core.config import Config
from modules.m6_export import provenance
from modules.m6_metrics import pareto_frontier

_POLICY_STYLE = {
    "PID": {"color": "#1f77b4", "marker": "o"},
    "LQR": {"color": "#d62728", "marker": "s"},
    "Targeting": {"color": "#2ca02c", "marker": "^"},
}


def _stamp(fig: plt.Figure, cfg: Config) -> None:
    """Footer stamp binding the figure to its (config, commit, seed) triple."""
    p = provenance(cfg)
    fig.text(
        0.005, 0.005,
        f"config {p['config_name']}·{p['config_fingerprint']}  "
        f"commit {p['git_commit'][:10]}  seed {p['base_seed']}",
        fontsize=5.5, color="#888888", ha="left", va="bottom",
    )


def _save(fig: plt.Figure, cfg: Config, name: str) -> Path:
    out = Path(cfg.outputs.figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.{cfg.outputs.figure_format}"
    _stamp(fig, cfg)
    fig.savefig(path, dpi=cfg.outputs.figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_frontiers(
    points: dict[str, list[tuple[float, float]]],
    cfg: Config,
    noise_level: str = "MEDIUM",
    name: str = "fig1_frontiers",
) -> Path:
    """Cost-accuracy frontiers, one curve per policy (RMS 6.5, Q1).

    ``points`` maps a policy name to its (annual Delta-v, RMS deviation)
    operating points. The non-dominated subset is drawn as a connected
    frontier; dominated points are shown faintly so the reader can see the
    tuning sweep behind the envelope rather than only its result.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for policy, pts in points.items():
        if not pts:
            continue
        cost = np.array([p[0] for p in pts])
        err = np.array([p[1] for p in pts])
        style = _POLICY_STYLE.get(policy, {"color": "#666666", "marker": "x"})
        ax.scatter(err, cost, s=18, alpha=0.30, **style)
        keep = pareto_frontier(cost, err)
        if keep.size:
            o = keep[np.argsort(err[keep])]
            ax.plot(err[o], cost[o], lw=1.8, label=policy, **style)

    ax.set_xlabel("RMS deviation from reference (km)")
    ax.set_ylabel("Annual maintenance cost (m/s/yr)")
    ax.set_title(f"Cost-accuracy frontiers — {noise_level} navigation error")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    return _save(fig, cfg, name)


def plot_noise_robustness(
    frontiers: dict[str, dict[str, list[tuple[float, float]]]],
    cfg: Config,
    name: str = "fig2_noise_robustness",
) -> Path:
    """How each policy's frontier shifts with navigation error (Q2, RMS 6.6)."""
    levels = list(frontiers.keys())
    fig, axes = plt.subplots(1, len(levels), figsize=(4.0 * len(levels), 3.8),
                             sharey=True, squeeze=False)
    for ax, level in zip(axes[0], levels):
        for policy, pts in frontiers[level].items():
            if not pts:
                continue
            cost = np.array([p[0] for p in pts]); err = np.array([p[1] for p in pts])
            keep = pareto_frontier(cost, err)
            if keep.size:
                o = keep[np.argsort(err[keep])]
                ax.plot(err[o], cost[o], lw=1.6, label=policy,
                        **_POLICY_STYLE.get(policy, {}))
        ax.set_title(level); ax.set_xlabel("RMS deviation (km)")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel("Annual Delta-v (m/s/yr)")
    axes[0][-1].legend(frameon=False)
    return _save(fig, cfg, name)


def plot_convergence(
    traces: dict[str, dict[str, np.ndarray]],
    cfg: Config,
    name: str = "fig3_convergence",
) -> Path:
    """Running mean with interval as N grows (RMS 6.6 convergence evidence)."""
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for policy, tr in traces.items():
        style = _POLICY_STYLE.get(policy, {})
        ax.plot(tr["n"], tr["mean"], lw=1.6, label=policy, **style)
        ax.fill_between(tr["n"], tr["lower"], tr["upper"], alpha=0.15,
                        color=style.get("color"))
    ax.set_xlabel("Monte Carlo trials")
    ax.set_ylabel("Running mean annual Delta-v (m/s/yr)")
    ax.set_title("Ensemble convergence")
    ax.legend(frameon=False); ax.grid(alpha=0.25, lw=0.5)
    return _save(fig, cfg, name)


def plot_distributions(
    samples: dict[str, np.ndarray],
    cfg: Config,
    xlabel: str = "Annual Delta-v (m/s/yr)",
    name: str = "fig4_distributions",
) -> Path:
    """Delta-v distributions with percentile markers.

    Plotted as distributions rather than bars because RMS 6.6 requires
    percentile reporting -- worst-case maintenance cost is tail-driven, and a
    mean alone hides exactly the behaviour that matters.
    """
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for policy, vals in samples.items():
        style = _POLICY_STYLE.get(policy, {})
        ax.hist(vals, bins=30, histtype="step", lw=1.6, density=True,
                label=policy, color=style.get("color"))
        for pct in cfg.stats.percentiles:
            ax.axvline(np.percentile(vals, pct), ls=":", lw=0.8,
                       color=style.get("color"), alpha=0.6)
    ax.set_xlabel(xlabel); ax.set_ylabel("density")
    ax.set_title("Cost distributions (dotted: p50 / p95 / p99)")
    ax.legend(frameon=False); ax.grid(alpha=0.25, lw=0.5)
    return _save(fig, cfg, name)


def plot_fault_recovery(
    excess: dict[str, np.ndarray],
    cfg: Config,
    name: str = "fig5_fault_recovery",
) -> Path:
    """Excess Delta-v attributable to fault recovery, per policy (Q3, RMS 7)."""
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    labels = list(excess.keys())
    ax.boxplot([excess[k] for k in labels], tick_labels=labels, showfliers=True, widths=0.55)
    ax.set_ylabel("Excess annual Delta-v vs fault-free (m/s/yr)")
    ax.set_title("Recovery cost after a missed / partial maneuver")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    return _save(fig, cfg, name)