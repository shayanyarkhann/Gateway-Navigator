"""Configuration layer: schema validation and provenance (RMS Appendix C)."""

from __future__ import annotations

import pytest
import yaml

from tests.conftest import CONFIG_DIR, shipped_configs

from core.config import (
    Config, FaultConfig, MonteCarloConfig, ORBIT_ANCHORS, OrbitConfig,
    ScaffoldConfig, StatsConfig, SweepConfig, git_commit, load_config,
)

CONFIGS = shipped_configs()


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_load(path):
    cfg = load_config(path)
    assert cfg.name
    assert cfg.orbit.anchor in ORBIT_ANCHORS


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_fingerprint_is_stable_and_specific(path):
    a, b = load_config(path), load_config(path)
    assert a.fingerprint() == b.fingerprint()
    assert len(a.fingerprint()) == 16


def test_fingerprints_differ_between_configs():
    prints = {load_config(p).fingerprint() for p in CONFIGS}
    assert len(prints) == len(CONFIGS), "distinct configs must fingerprint distinctly"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(CONFIG_DIR / "does-not-exist.yaml")


def test_expected_configs_are_shipped():
    """The campaign, smoke and fault configs must all be present.

    Guards the integration failure this suite previously surfaced as ten
    unrelated collection errors: a config silently absent from the repository.
    """
    names = {p.name for p in CONFIGS}
    missing = {"v1.yaml", "v1-smoke.yaml", "v1-fault.yaml"} - names
    assert not missing, f"missing shipped configs: {sorted(missing)}"


def test_unknown_section_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"name": "x", "nonsense": {"a": 1}}))
    with pytest.raises(ValueError, match="unknown configuration sections"):
        load_config(p)


def test_missing_name_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"orbit": {"anchor": "nine_two_period"}}))
    with pytest.raises(ValueError, match="must declare a 'name'"):
        load_config(p)


def test_unknown_orbit_anchor_is_rejected():
    with pytest.raises(ValueError, match="unknown orbit anchor"):
        OrbitConfig(anchor="not_an_orbit")


def test_maneuver_location_is_locked_by_methodology():
    """RMS 6.1(4) fixes apolune; the schema must refuse to move it silently."""
    with pytest.raises(ValueError, match="apolune"):
        ScaffoldConfig(maneuver_at_apolune=False)


def test_maneuver_cadence_is_locked_by_methodology():
    with pytest.raises(ValueError, match="methodology amendment"):
        ScaffoldConfig(maneuvers_per_revolution=3)


def test_bit_generator_must_be_pinned():
    """RMS 6.7 requires the bit generator pinned and stated."""
    with pytest.raises(ValueError, match="pinned"):
        MonteCarloConfig(bit_generator="MT19937")


@pytest.mark.parametrize("bad", [(-0.1, 0.5), (0.6, 0.2), (0.0, 1.5)])
def test_invalid_fault_severity_range_is_rejected(bad):
    with pytest.raises(ValueError, match="severity_range"):
        FaultConfig(severity_range=bad)


def test_asymmetric_sweep_effort_is_rejected():
    """RMS 6.9(4): unequal search effort could bias the comparison."""
    with pytest.raises(ValueError, match="asymmetric sweep effort"):
        SweepConfig(pid_kp=(0.1, 0.2), lqr_r_scale=(1.0, 2.0, 3.0),
                    targeting_coast_revs=(1, 2, 3))


def test_unstable_bootstrap_size_is_rejected():
    with pytest.raises(ValueError, match="unstable intervals"):
        StatsConfig(bootstrap_resamples=100)


def test_git_commit_is_a_string():
    assert isinstance(git_commit(), str)