"""Reference x-axis-crossing targeting policy (RMS 6.3, benchmark).

The domain-standard NRHO maintenance method (Guzzetti et al. 2017; Davis et al.
2017): at apolune, solve for the impulsive correction that drives the
trajectory back onto the reference a stated number of revolutions downstream,
by differential correction on the state-transition matrix.

Why this exists: RMS 1.1 identifies the framing threat that PID and LQR are not
how NRHOs are actually station-kept. Without this benchmark the study compares
two classical controllers to each other and cannot say where either sits
relative to operational practice. RMS 3.5 makes it the *last* discretionary
item to cut for exactly that reason.

It reuses :func:`core.dynamics.state_transition_matrix` -- the same STM that
drives the differential corrector and the LQR plant model -- rather than
duplicating any linearisation logic (GN-022).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.constants import L_STAR_KM, MU
from core.dynamics import state_transition_matrix


class TargetingController:
    """Impulsive x-axis-crossing targeting policy.

    Over a coast arc of ``coast_revolutions`` periods the linearised map from a
    deviation at the maneuver epoch to a deviation downstream is::

        dx(t_k + N*T) = Phi @ (dx_k + [0; dv])
                      = Phi @ dx_k + Phi[:, 3:6] @ dv

    Nulling the *position* components downstream gives a square 3x3 system::

        Phi[0:3, :] @ dx_k + Phi[0:3, 3:6] @ dv = 0

    Position-only targeting (rather than full-state) is the deliberate choice:
    the system is then exactly determined, and it mirrors the operational
    method, which targets a downstream plane-crossing condition rather than a
    complete state. Targeting all six components would be overdetermined and
    would require a least-squares compromise that has no operational analogue.

    **Known issue — horizon/cadence mismatch.** RMS 6.2 fixes the maneuver
    cadence at once per revolution for every policy, while RMS 6.3 names the
    targeting policy's knob as the coast duration. This implementation targets
    ``coast_revolutions`` downstream but still fires every revolution, so a
    correction sized for an N-revolution arc is re-applied N times before it
    matures. The smoke campaign shows this is benign at N = 1, 4, 5 and
    divergent at N = 2, 3 -- non-monotonically, so it is not simple
    over-correction. The targeting map itself is well conditioned throughout
    (cond 21-310 over N = 1..6), which rules out a singular solve. Resolve
    before the production campaign; until then, treat N > 1 results as
    provisional.

    The gain depends only on the reference state at the maneuver epoch and the
    coast duration. Because maneuvers occur at a fixed orbital phase, the STM is
    integrated once and cached -- an important saving, since a Monte Carlo
    campaign evaluates this policy hundreds of thousands of times.
    """

    #: Reference-state movement beyond which the cached STM is recomputed.
    _POS_TOL: float = 1.0 / L_STAR_KM   # 1 km

    def __init__(
        self,
        coast_revolutions: int,
        period: float,
        mu: float = MU,
        max_dv: float | None = None,
    ) -> None:
        """Construct the targeting policy.

        Parameters
        ----------
        coast_revolutions
            Number of reference periods downstream at which the deviation is
            nulled. The policy's tuning knob (RMS 6.3): a longer arc buys a
            cheaper correction at the cost of larger intermediate excursion.
        period
            Reference orbit period, non-dimensional.
        mu
            CR3BP mass parameter.
        max_dv
            Optional saturation on the commanded correction magnitude,
            non-dimensional. Guards against the ill-conditioned solve that can
            occur if the targeting arc lands near a singular geometry.

        Raises
        ------
        ValueError
            If ``coast_revolutions`` or ``period`` is non-positive.
        """
        if coast_revolutions < 1:
            raise ValueError(f"coast_revolutions must be >= 1, got {coast_revolutions}")
        if period <= 0.0:
            raise ValueError(f"period must be positive, got {period}")
        if max_dv is not None and max_dv <= 0.0:
            raise ValueError(f"max_dv must be positive, got {max_dv}")

        self.coast_revolutions = int(coast_revolutions)
        self.period = float(period)
        self.mu = float(mu)
        self.max_dv = max_dv
        self.horizon = self.coast_revolutions * self.period
        self._gain: NDArray[np.float64] | None = None
        self._gain_ref: NDArray[np.float64] | None = None
        self.last_condition_number: float = float("nan")

    def reset(self) -> None:
        """Clear the cached gain (call between independent runs)."""
        self._gain = None
        self._gain_ref = None

    def _gain_for(self, x_ref: NDArray[np.float64]) -> NDArray[np.float64]:
        """Targeting gain at ``x_ref``, recomputing only when the phase moves."""
        if (self._gain is not None
                and np.linalg.norm(x_ref[:3] - self._gain_ref[:3]) <= self._POS_TOL):
            return self._gain

        _, Phi = state_transition_matrix(x_ref, self.horizon, self.mu)
        M = Phi[0:3, 3:6]          # position response to an impulsive dv
        cond = float(np.linalg.cond(M))
        self.last_condition_number = cond
        if cond > 1e12:
            raise RuntimeError(
                f"targeting map is numerically singular (cond {cond:.2e}) over a "
                f"{self.coast_revolutions}-revolution arc; choose a different "
                "coast duration"
            )
        # dv = -M^-1 @ Phi[0:3, :] @ dx  ->  gain applied to the full deviation
        self._gain = -np.linalg.solve(M, Phi[0:3, :])
        self._gain_ref = np.asarray(x_ref, dtype=float).copy()
        return self._gain

    def compute_dv(
        self,
        x_hat: ArrayLike,
        x_ref: ArrayLike,
        t: float | None = None,
    ) -> NDArray[np.float64]:
        """Commanded impulsive correction at this maneuver epoch.

        Parameters
        ----------
        x_hat
            Six-element estimated state.
        x_ref
            Six-element reference state at the same epoch.
        t
            Unused; present for interface parity with the PID and LQR policies
            so all three drop into the common scaffold unchanged (RMS 6.2).

        Returns
        -------
        dv
            Three-element velocity correction, non-dimensional.
        """
        x_hat = np.asarray(x_hat, dtype=float)
        x_ref = np.asarray(x_ref, dtype=float)
        if x_hat.shape != (6,) or x_ref.shape != (6,):
            raise ValueError("x_hat and x_ref must both be 6-element states")

        dv = self._gain_for(x_ref) @ (x_hat - x_ref)

        if self.max_dv is not None:
            norm = float(np.linalg.norm(dv))
            if norm > self.max_dv:
                dv = dv * (self.max_dv / norm)
        return dv