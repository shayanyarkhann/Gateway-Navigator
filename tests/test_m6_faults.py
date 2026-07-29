"""Fault injection: the single canonical failure mode (RMS Section 7)."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import FaultConfig
from modules.m6_faults import FaultEvent, NoFault, sample_fault


def test_disabled_config_yields_no_fault():
    assert isinstance(sample_fault(FaultConfig(enabled=False), np.random.default_rng(0)), NoFault)


def test_disabled_config_draws_no_randomness():
    """Drawing when faults are off would desynchronise the fault-free and
    faulted campaigns and break the paired comparison."""
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    sample_fault(FaultConfig(enabled=False), rng_a)
    assert rng_a.random() == rng_b.random()


def test_enabled_config_yields_an_event_in_range():
    cfg = FaultConfig(enabled=True, severity_range=(0.0, 0.5),
                      epoch_revolution_range=(10, 46))
    rng = np.random.default_rng(1)
    for _ in range(200):
        f = sample_fault(cfg, rng)
        assert isinstance(f, FaultEvent)
        assert 10 <= f.maneuver_index <= 46
        assert 0.0 <= f.severity <= 0.5


def test_fault_sampling_is_reproducible():
    cfg = FaultConfig(enabled=True)
    a = sample_fault(cfg, np.random.default_rng(9))
    b = sample_fault(cfg, np.random.default_rng(9))
    assert a == b


def test_fault_applies_only_at_its_epoch():
    f = FaultEvent(maneuver_index=7, severity=0.25)
    assert f.applies_to(7) and not f.applies_to(6)
    dv = np.array([1.0, 2.0, 3.0])
    assert np.allclose(f.apply(dv), dv * 0.25)


def test_missed_thrust_event_zeroes_the_burn():
    assert np.allclose(FaultEvent(3, 0.0).apply(np.ones(3)), 0.0)


def test_no_fault_is_a_pass_through():
    dv = np.array([1.0, -2.0, 0.5])
    assert np.allclose(NoFault().apply(dv), dv)


def test_severity_spans_the_configured_range():
    cfg = FaultConfig(enabled=True, severity_range=(0.0, 1.0))
    rng = np.random.default_rng(2)
    sev = np.array([sample_fault(cfg, rng).severity for _ in range(500)])
    assert sev.min() < 0.1 and sev.max() > 0.9