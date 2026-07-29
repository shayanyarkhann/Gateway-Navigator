"""Result export and provenance (RMS Appendix C)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from modules.m6_export import (
    SCHEMA_VERSION, export_campaign, load_trials_csv, provenance,
    write_json, write_summary_table, write_trials_csv,
)
from modules.m6_montecarlo import EnsembleResult, TrialResult


def _trial(i: int = 0, policy: str = "PID") -> TrialResult:
    return TrialResult(
        trial_index=i, seed=1000 + i, policy=policy, setting=0.4,
        noise_level="MEDIUM", delta_v_nondim=1e-4, delta_v_annual_ms=0.5,
        rms_deviation_km=4.2, max_deviation_km=9.1, final_deviation_km=3.3,
        estimation_rms_km=0.12, fault_maneuver_index=None, fault_severity=None,
        diverged=False,
    )


def test_provenance_carries_the_rms_triple(config):
    p = provenance(config)
    for key in ("config_fingerprint", "git_commit", "base_seed"):
        assert key in p and p[key] is not None
    assert p["schema_version"] == SCHEMA_VERSION
    assert p["bit_generator"] == "PCG64"


def test_trials_round_trip(tmp_path):
    trials = [_trial(i) for i in range(5)]
    path = write_trials_csv(trials, tmp_path / "t.csv")
    back = load_trials_csv(path)
    assert len(back) == 5
    assert int(back[0]["trial_index"]) == 0
    assert back[0]["policy"] == "PID"


def test_empty_exports_are_refused(tmp_path):
    with pytest.raises(ValueError):
        write_trials_csv([], tmp_path / "t.csv")
    with pytest.raises(ValueError):
        write_summary_table([], tmp_path / "s.csv")


def test_missing_trials_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_trials_csv(tmp_path / "absent.csv")


def test_json_embeds_provenance_when_config_given(tmp_path, config):
    p = write_json({"x": 1}, tmp_path / "r.json", config)
    body = json.loads(p.read_text())
    assert body["data"]["x"] == 1
    assert body["provenance"]["config_fingerprint"] == config.fingerprint()


def test_json_serialises_numpy(tmp_path):
    p = write_json({"a": np.float64(1.5), "b": np.arange(3)}, tmp_path / "n.json")
    body = json.loads(p.read_text())
    assert body["a"] == 1.5 and body["b"] == [0, 1, 2]


def test_export_campaign_writes_every_artifact(tmp_path, config):
    
    ens = EnsembleResult(policy="PID", setting=0.4, noise_level="MEDIUM",
                         trials=[_trial(i) for i in range(4)])
    written = export_campaign([ens], config, results_dir=tmp_path / "results")
    assert set(written) == {"trials", "provenance", "index", "summary"}
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0


def test_export_refuses_empty_campaign(config, tmp_path):
    with pytest.raises(ValueError):
        export_campaign([], config, results_dir=tmp_path)