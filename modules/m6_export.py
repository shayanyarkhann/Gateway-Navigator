"""Result export and provenance (RMS Appendix C).

The binding rule the RMS imposes is that *every figure and reported number is
regenerable from* ``(configuration file, git commit, base random seed)``. This
module writes that triple alongside every artifact, so a number found in the
paper can be traced back to the exact settings and code that produced it.

Trial-level records are written once; every downstream summary, table and
figure is derived from those saved records rather than by re-simulating, which
is what makes figure regeneration cheap and guarantees that the figure and the
table cannot disagree.
"""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from core.config import Config, git_commit
from modules.m6_montecarlo import EnsembleResult, TrialResult

SCHEMA_VERSION = "1.0"


def provenance(cfg: Config, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The provenance record attached to every artifact.

    Captures the RMS triple plus the environment needed to explain a numerical
    discrepancy between machines (library versions move; results should not).
    """
    import scipy

    rec = {
        "schema_version": SCHEMA_VERSION,
        "config_name": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "git_commit": git_commit(),
        "base_seed": cfg.monte_carlo.base_seed,
        "bit_generator": cfg.monte_carlo.bit_generator,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }
    if extra:
        rec.update(extra)
    return rec


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_trials_csv(trials: Sequence[TrialResult], path: str | Path) -> Path:
    """Write trial-level records: one row per trial, the rawest saved artifact.

    Raises
    ------
    ValueError
        If ``trials`` is empty -- an empty results file is worse than none,
        because it looks like a completed run.
    """
    if not trials:
        raise ValueError("refusing to write an empty trials file")
    path = _ensure(Path(path))
    rows = [t.to_row() for t in trials]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(payload: Any, path: str | Path, cfg: Config | None = None) -> Path:
    """Write a JSON artifact, embedding provenance when a config is supplied."""
    path = _ensure(Path(path))
    body = {"provenance": provenance(cfg), "data": payload} if cfg else payload
    path.write_text(json.dumps(body, indent=2, default=_json_default))
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"not JSON-serialisable: {type(obj).__name__}")


def write_summary_table(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write the headline results table as CSV for direct paper inclusion."""
    if not rows:
        raise ValueError("refusing to write an empty summary table")
    path = _ensure(Path(path))
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_campaign(
    ensembles: Iterable[EnsembleResult],
    cfg: Config,
    results_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write every artifact for a completed campaign.

    Returns
    -------
    Mapping from artifact name to the path written.
    """
    ensembles = list(ensembles)
    if not ensembles:
        raise ValueError("no ensembles to export")

    root = Path(results_dir) if results_dir is not None else Path(cfg.outputs.results_dir)
    trials_root = Path(cfg.outputs.trials_dir)

    all_trials = [t for e in ensembles for t in e.trials]
    written = {
        "trials": write_trials_csv(all_trials, trials_root / f"{cfg.name}_trials.csv"),
        "provenance": write_json(provenance(cfg), root / f"{cfg.name}_provenance.json"),
    }

    index = [
        {
            "policy": e.policy, "setting": e.setting, "noise_level": e.noise_level,
            "n_trials": len(e.trials), "divergence_rate": e.divergence_rate,
            "mean_delta_v_annual_ms": float(np.mean(e.delta_v)) if e.trials else None,
            "mean_rms_deviation_km": float(np.mean(e.deviation)) if e.trials else None,
        }
        for e in ensembles
    ]
    written["index"] = write_json(index, root / f"{cfg.name}_index.json", cfg)
    written["summary"] = write_summary_table(index, root / f"{cfg.name}_summary.csv")
    return written


def load_trials_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read back trial records, so figures regenerate without re-simulating."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"trial records not found: {path}")
    with path.open() as fh:
        return list(csv.DictReader(fh))