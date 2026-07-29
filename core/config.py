"""Configuration-as-code for the Gateway Navigator experiment campaign.

Implements RMS Appendix C: *every* scientific parameter -- orbit selection,
model constants, controller tuning ranges, noise levels, fault definition,
ensemble size, mission duration, seeds -- lives in a versioned YAML file, not
in the code body. Nothing downstream of this module may hardcode a scientific
parameter.

The dataclasses below are the schema. Loading validates eagerly and fails on
the first inconsistency, because a campaign that runs for hours before
discovering a malformed sweep range wastes more than it saves.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

# Orbit anchors. RMS Appendix A gives four targets (period band, Jacobi
# constant, perilune, apolune) that no single CR3BP member satisfies
# simultaneously -- see the campaign notes. The anchor is therefore an explicit,
# recorded choice rather than a silent one, and switching it is a config change.
OrbitAnchor = Literal["nine_two_period", "jacobi_3p0498", "period_6p50"]

ORBIT_ANCHORS: dict[str, dict[str, Any]] = {
    # 9:2 synodic resonance -- the orbit's *defining* property, named in the
    # RMS research question. Best match to the published perilune radius.
    "nine_two_period": {
        "kind": "period",
        "target": None,  # filled from the 9:2 resonance at load time
        "guess": [1.0220282893062722, -0.1821014455365690, -0.1032711113333490],
    },
    # Matches the tabulated Jacobi constant instead; period falls just below
    # the Appendix A band and perilune to ~2768 km.
    "jacobi_3p0498": {
        "kind": "jacobi",
        "target": 3.0498,
        "guess": [1.0187, -0.1797, -0.0962],
    },
    # Mid-band compromise: period inside the stated 6.4-6.55 d range.
    "period_6p50": {
        "kind": "period",
        "target": 6.50 * 86400.0 / 375190.0,
        "guess": [1.0187, -0.1797, -0.0962],
    },
}


@dataclass(frozen=True)
class OrbitConfig:
    """Reference-orbit selection (RMS A3, Section 5.3)."""

    anchor: OrbitAnchor = "nine_two_period"
    closure_tolerance_m: float = 1.0

    def __post_init__(self) -> None:
        if self.anchor not in ORBIT_ANCHORS:
            raise ValueError(
                f"unknown orbit anchor {self.anchor!r}; "
                f"expected one of {sorted(ORBIT_ANCHORS)}"
            )
        if self.closure_tolerance_m <= 0.0:
            raise ValueError("closure_tolerance_m must be positive")


@dataclass(frozen=True)
class ScaffoldConfig:
    """The common scaffold held identical across policies (RMS 6.2)."""

    revolutions: int = 56           # ~1 year at the ~6.5 d period (RMS 6.6)
    samples_per_revolution: int = 15
    maneuvers_per_revolution: int = 1
    maneuver_at_apolune: bool = True
    p0_scale: float = 1e-8
    q_accel: float = 1e-16

    def __post_init__(self) -> None:
        if self.revolutions < 1:
            raise ValueError("revolutions must be >= 1")
        if self.samples_per_revolution < 1:
            raise ValueError("samples_per_revolution must be >= 1")
        if self.maneuvers_per_revolution != 1:
            raise ValueError(
                "RMS 6.1 fixes one maneuver per revolution at apolune; "
                "changing this is a methodology amendment, not a config tweak"
            )
        if not self.maneuver_at_apolune:
            raise ValueError(
                "RMS 6.1(4) fixes the maneuver location at apolune -- perilune "
                "is where linear models are least valid"
            )


@dataclass(frozen=True)
class InitialDispersionConfig:
    """Injection dispersion, one of the Monte Carlo randomness sources (RMS 6.6)."""

    position_sigma_km: float = 0.5
    velocity_sigma_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.position_sigma_km < 0.0 or self.velocity_sigma_ms < 0.0:
            raise ValueError("dispersion sigmas must be non-negative")


@dataclass(frozen=True)
class FaultConfig:
    """The single canonical fault mode (RMS Section 7).

    RMS 7 specifies *one* primary mode -- a missed or partial maneuver at a
    randomized epoch with randomized severity -- and RMS 3.3 explicitly records
    "a general fault-injection framework" as an out-of-scope idea not to be
    pursued in V1. This schema is therefore deliberately narrow.
    """

    enabled: bool = False
    severity_range: tuple[float, float] = (0.0, 0.5)
    epoch_revolution_range: tuple[int, int] = (10, 46)

    def __post_init__(self) -> None:
        lo, hi = self.severity_range
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(
                f"severity_range must satisfy 0 <= lo <= hi <= 1, got {self.severity_range}. "
                "Severity is the *retained* thrust fraction: 0.0 is a fully "
                "missed maneuver, 1.0 is nominal."
            )
        r_lo, r_hi = self.epoch_revolution_range
        if not (0 <= r_lo <= r_hi):
            raise ValueError(f"epoch_revolution_range must be ordered and non-negative")


@dataclass(frozen=True)
class MonteCarloConfig:
    """Ensemble settings (RMS 6.6, 6.7)."""

    trials: int = 500
    base_seed: int = 20260707
    bit_generator: str = "PCG64"
    convergence_check_points: int = 10
    noise_levels: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")

    def __post_init__(self) -> None:
        if self.trials < 1:
            raise ValueError("trials must be >= 1")
        if self.trials < 500:
            # A warning, not an error: smoke configs legitimately run fewer.
            pass
        if self.bit_generator != "PCG64":
            raise ValueError(
                "RMS 6.7 requires the bit generator to be pinned and stated; "
                "PCG64 is the pinned choice for V1"
            )
        for level in self.noise_levels:
            if level not in ("LOW", "MEDIUM", "HIGH"):
                raise ValueError(f"unknown noise level {level!r}")


@dataclass(frozen=True)
class SweepConfig:
    """Tuning sweeps that generate each policy's frontier (RMS 6.5).

    Sweep effort must be symmetric across policies (RMS 6.9(4)), so the
    resolutions are validated to match.
    """

    pid_kp: tuple[float, ...] = (0.1, 0.2, 0.4, 0.8, 1.2)
    pid_ki: float = 0.02
    pid_kd_ratio: float = 1.25
    lqr_r_scale: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    targeting_coast_revs: tuple[int, ...] = (1, 2, 3, 4, 5)

    def __post_init__(self) -> None:
        n = {len(self.pid_kp), len(self.lqr_r_scale), len(self.targeting_coast_revs)}
        if len(n) != 1:
            raise ValueError(
                f"asymmetric sweep effort violates RMS 6.9(4): got "
                f"{len(self.pid_kp)} PID / {len(self.lqr_r_scale)} LQR / "
                f"{len(self.targeting_coast_revs)} targeting settings"
            )


@dataclass(frozen=True)
class StatsConfig:
    """Statistical analysis settings (RMS 6.8)."""

    bootstrap_resamples: int = 10_000
    confidence_level: float = 0.95
    percentiles: tuple[float, ...] = (50.0, 95.0, 99.0)

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.bootstrap_resamples < 1000:
            raise ValueError("bootstrap_resamples < 1000 gives unstable intervals")


@dataclass(frozen=True)
class OutputConfig:
    """Output locations and figure settings."""

    results_dir: str = "data/results"
    trials_dir: str = "data/trials"
    figures_dir: str = "figures"
    figure_format: str = "pdf"
    figure_dpi: int = 300


@dataclass(frozen=True)
class Config:
    """Top-level experiment configuration."""

    name: str
    orbit: OrbitConfig = field(default_factory=OrbitConfig)
    scaffold: ScaffoldConfig = field(default_factory=ScaffoldConfig)
    dispersion: InitialDispersionConfig = field(default_factory=InitialDispersionConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    sweeps: SweepConfig = field(default_factory=SweepConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form, for JSON export and provenance hashing."""
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable 16-hex-character digest of the whole configuration.

        Recorded alongside every figure and result file so that a number can be
        traced to the exact settings that produced it (RMS Appendix C).
        """
        blob = yaml.safe_dump(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _coerce_tuples(d: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """YAML gives lists; the frozen dataclasses want hashable tuples."""
    return {k: (tuple(v) if k in keys and isinstance(v, list) else v) for k, v in d.items()}


def load_config(path: str | Path) -> Config:
    """Load and validate a campaign configuration from YAML.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If any section fails validation, or an unknown section is present.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"configuration file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if "name" not in raw:
        raise ValueError(f"{path}: configuration must declare a 'name'")

    tuple_keys = (
        "severity_range", "epoch_revolution_range", "noise_levels", "pid_kp",
        "lqr_r_scale", "targeting_coast_revs", "percentiles",
    )
    sections = {
        "orbit": OrbitConfig, "scaffold": ScaffoldConfig,
        "dispersion": InitialDispersionConfig, "fault": FaultConfig,
        "monte_carlo": MonteCarloConfig, "sweeps": SweepConfig,
        "stats": StatsConfig, "outputs": OutputConfig,
    }
    unknown = set(raw) - set(sections) - {"name"}
    if unknown:
        raise ValueError(f"{path}: unknown configuration sections: {sorted(unknown)}")

    built = {
        key: cls(**_coerce_tuples(raw.get(key, {}) or {}, tuple_keys))
        for key, cls in sections.items()
    }
    return Config(name=raw["name"], **built)


def git_commit() -> str:
    """Current git commit, or ``"unknown"`` outside a repository.

    Part of the ``(configuration, commit, seed)`` provenance triple required by
    RMS Appendix C.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=5, check=False,
        )
        if out.returncode == 0:
            dirty = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True,
                timeout=5, check=False,
            ).stdout.strip()
            return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"