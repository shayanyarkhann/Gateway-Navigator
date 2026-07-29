"""Targeting policy: the RMS 6.3 benchmark controller."""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import L_STAR_KM, MU
from core.dynamics import state_transition_matrix
from core.nrho_ics import GATEWAY_NRHO_PERIOD as T, GATEWAY_NRHO_X0 as X0
from modules.m6_targeting import TargetingController


def _tc(coast: int = 2, **kw) -> TargetingController:
    return TargetingController(coast_revolutions=coast, period=T, mu=MU, **kw)


def test_zero_deviation_commands_zero_correction():
    assert np.allclose(_tc().compute_dv(X0, X0), 0.0, atol=1e-14)


def test_correction_nulls_downstream_position_deviation():
    """The defining property: after the burn, the linearised downstream
    position deviation must vanish."""
    tc = _tc(coast=2)
    dx = np.array([1e-5, -2e-5, 5e-6, 0.0, 0.0, 0.0])
    dv = tc.compute_dv(X0 + dx, X0)
    _, Phi = state_transition_matrix(X0, 2 * T, MU)
    downstream = Phi @ (dx + np.concatenate([np.zeros(3), dv]))
    assert np.linalg.norm(downstream[:3]) * L_STAR_KM < 1e-6


def test_correction_scales_linearly_with_deviation():
    tc = _tc()
    dx = np.array([1e-6, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(tc.compute_dv(X0 + 3 * dx, X0), 3 * tc.compute_dv(X0 + dx, X0), rtol=1e-6)


@pytest.mark.parametrize("coast", [1, 2, 3])
def test_targeting_map_is_well_conditioned(coast):
    tc = _tc(coast)
    tc.compute_dv(X0 + np.array([1e-6, 0, 0, 0, 0, 0]), X0)
    assert tc.last_condition_number < 1e8


def test_gain_is_cached_across_calls():
    tc = _tc()
    tc.compute_dv(X0 + np.array([1e-6, 0, 0, 0, 0, 0]), X0)
    gain = tc._gain
    tc.compute_dv(X0 + np.array([2e-6, 0, 0, 0, 0, 0]), X0)
    assert tc._gain is gain, "STM should not be re-integrated for the same phase"


def test_reset_clears_the_cache():
    tc = _tc()
    tc.compute_dv(X0 + np.array([1e-6, 0, 0, 0, 0, 0]), X0)
    tc.reset()
    assert tc._gain is None


def test_max_dv_saturates():
    tc = _tc(max_dv=1e-9)
    dv = tc.compute_dv(X0 + np.array([1e-3, 0, 0, 0, 0, 0]), X0)
    assert np.linalg.norm(dv) <= 1e-9 * (1 + 1e-12)


@pytest.mark.parametrize("kw", [{"coast_revolutions": 0}, {"coast_revolutions": -1}])
def test_invalid_coast_is_rejected(kw):
    with pytest.raises(ValueError):
        TargetingController(period=T, **kw)


def test_invalid_period_is_rejected():
    with pytest.raises(ValueError):
        TargetingController(coast_revolutions=1, period=-1.0)


def test_malformed_state_is_rejected():
    with pytest.raises(ValueError):
        _tc().compute_dv(np.zeros(5), X0)


def test_interface_matches_the_other_policies():
    """All three policies must drop into the common scaffold unchanged (RMS 6.2)."""
    import inspect
    from modules.m4_pid import PIDController
    from modules.m5_lqr import LQRController
    sig = lambda c: list(inspect.signature(c.compute_dv).parameters)
    assert sig(TargetingController) == sig(PIDController) == sig(LQRController)