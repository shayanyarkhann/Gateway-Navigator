"""Shared pytest fixtures and path anchoring.

Two problems this file solves.

**Working-directory independence.** Tests previously resolved configuration
paths as bare relative strings (``"configs/v1.yaml"``), which resolves against
the *current working directory*, not the repository. That works only when
pytest happens to be invoked from the repository root and fails silently
everywhere else -- from an IDE runner, from a subdirectory, or from a CI job
with a different working directory. ``REPO_ROOT`` is derived from this file's
own location, so it is correct regardless of how pytest is started.

**Test isolation.** Unit tests for export, figures and statistics were
depending on a specific shipped YAML file merely to obtain a ``Config``
instance. A missing data file then surfaced as ten unrelated collection errors
rather than one clear failure. The ``config`` fixture below builds a valid
``Config`` in-process, so those tests exercise the code under test and nothing
else. Tests that genuinely assert something *about the shipped configs* still
read them from disk -- that is their purpose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import (
    Config, FaultConfig, InitialDispersionConfig, MonteCarloConfig,
    OrbitConfig, OutputConfig, ScaffoldConfig, StatsConfig, SweepConfig,
)

#: Repository root, anchored to this file rather than the working directory.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: Directory holding the shipped campaign configurations.
CONFIG_DIR: Path = REPO_ROOT / "configs"


def shipped_configs() -> list[Path]:
    """Every campaign configuration shipped with the repository."""
    return sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def config(tmp_path) -> Config:
    """A valid in-process ``Config`` with all outputs redirected to ``tmp_path``.

    Deliberately constructed rather than loaded from disk: a test of JSON
    serialisation or figure rendering should not fail because a YAML file is
    absent. Ensemble sizes are small so tests stay fast.
    """
    return Config(
        name="pytest-fixture",
        orbit=OrbitConfig(anchor="nine_two_period"),
        scaffold=ScaffoldConfig(revolutions=4, samples_per_revolution=8),
        dispersion=InitialDispersionConfig(position_sigma_km=0.5),
        fault=FaultConfig(enabled=False),
        monte_carlo=MonteCarloConfig(trials=6, base_seed=20260707,
                                     noise_levels=("MEDIUM",)),
        sweeps=SweepConfig(),
        stats=StatsConfig(bootstrap_resamples=2000),
        outputs=OutputConfig(
            results_dir=str(tmp_path / "results"),
            trials_dir=str(tmp_path / "trials"),
            figures_dir=str(tmp_path / "figures"),
            figure_format="png",
            figure_dpi=72,
        ),
    )