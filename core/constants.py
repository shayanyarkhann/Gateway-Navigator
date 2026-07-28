"""Physical and numerical constants for the Earth-Moon CR3BP.

Single source of truth. No physical constant may be written as a literal
anywhere else in the codebase; import from here (GN-008).
"""

from __future__ import annotations

from typing import Final

# --- CR3BP system parameters ----------------------------------------------
MU: Final[float] = 0.012150584
"""Earth-Moon mass parameter, M_moon / (M_earth + M_moon)."""

L_STAR_KM: Final[float] = 384_400.0
"""Characteristic length: mean Earth-Moon separation, km."""

T_STAR_S: Final[float] = 375_190.0
"""Characteristic time: 1/n, where n is the Earth-Moon mean motion, s."""

V_STAR_MS: Final[float] = L_STAR_KM * 1_000.0 / T_STAR_S
"""Characteristic velocity, m/s. Derived, never written as a literal --
the rounded value 1025 carries 0.044% error (GN-008). Equals 1024.548."""

MOON_X: Final[float] = 1.0 - MU
"""Moon's x-coordinate in the rotating frame, non-dimensional."""

R_MOON_KM: Final[float] = 1_737.4
"""Lunar mean radius, km. Used to convert perilune radius to altitude."""

SYNODIC_MONTH_DAYS: Final[float] = 29.530588
"""Mean synodic month, days. Defines the 9:2 resonance of Gateway's NRHO."""

# --- Integrator tolerances -------------------------------------------------
RTOL_TIGHT: Final[float] = 1e-12
ATOL_TIGHT: Final[float] = 1e-14
"""Validation-grade tolerances: periodicity gate and Jacobi conservation."""

RTOL_STM: Final[float] = 1e-11
ATOL_STM: Final[float] = 1e-13
"""Filter/controller-grade tolerances. Looser than validation grade because
the STM feeds covariance and gain computation, not the periodicity gate."""
    