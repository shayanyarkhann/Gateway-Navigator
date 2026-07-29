"""Fault injection: the single canonical NRHO failure mode (RMS Section 7).

RMS 7 specifies **one** primary fault for Version 1 -- a missed or partial
maneuver at a randomized epoch with randomized severity -- and RMS 3.3 records
"a general fault-injection framework" as an out-of-scope idea deliberately not
pursued. This module is therefore intentionally narrow: it models that one mode
well rather than modelling many modes shallowly.

Severity is expressed as the *retained* thrust fraction: 0.0 is a fully missed
maneuver (the canonical missed-thrust event), 1.0 is nominal execution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from core.config import FaultConfig


@dataclass(frozen=True)
class FaultEvent:
    """A single scheduled maneuver fault.

    Attributes
    ----------
    maneuver_index
        Zero-based index of the affected maneuver epoch.
    severity
        Retained thrust fraction in [0, 1]; 0.0 is a fully missed maneuver.
    """

    maneuver_index: int
    severity: float

    def applies_to(self, maneuver_index: int) -> bool:
        return maneuver_index == self.maneuver_index

    def apply(self, dv: ArrayLike) -> NDArray[np.float64]:
        """Scale a commanded correction by the retained thrust fraction."""
        return np.asarray(dv, dtype=float) * self.severity


#: Sentinel for a fault-free trial. Kept as a distinct type rather than ``None``
#: so callers need no null checks in the maneuver loop.
@dataclass(frozen=True)
class NoFault:
    """Null fault: every maneuver executes nominally."""

    def applies_to(self, maneuver_index: int) -> bool:  # noqa: D102
        return False

    def apply(self, dv: ArrayLike) -> NDArray[np.float64]:  # noqa: D102
        return np.asarray(dv, dtype=float)


def sample_fault(
    cfg: FaultConfig,
    rng: Generator,
    maneuvers_per_revolution: int = 1,
) -> FaultEvent | NoFault:
    """Draw a fault event for one Monte Carlo trial.

    Both *when* the fault occurs and *how severe* it is are randomized across
    the ensemble (RMS 7), so the comparison measures each policy's recovery
    across the fault space rather than at one hand-picked epoch.

    The draw consumes randomness from the caller's dedicated fault stream even
    when faults are disabled would break common random numbers -- so when
    ``cfg.enabled`` is false this returns immediately **without** drawing,
    keeping the fault-free and faulted campaigns on identical sensor and
    thruster streams for a paired comparison.

    Parameters
    ----------
    cfg
        Fault configuration section.
    rng
        Generator for this trial's fault stream.
    maneuvers_per_revolution
        Maneuver cadence, used to convert the configured revolution range to a
        maneuver index.

    Returns
    -------
    FaultEvent or NoFault
    """
    if not cfg.enabled:
        return NoFault()

    r_lo, r_hi = cfg.epoch_revolution_range
    idx_lo = r_lo * maneuvers_per_revolution
    idx_hi = r_hi * maneuvers_per_revolution
    maneuver_index = int(rng.integers(idx_lo, idx_hi + 1))

    s_lo, s_hi = cfg.severity_range
    severity = float(rng.uniform(s_lo, s_hi))

    return FaultEvent(maneuver_index=maneuver_index, severity=severity)