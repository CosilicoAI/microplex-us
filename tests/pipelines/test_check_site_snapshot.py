"""Tests for the US site snapshot consistency checker."""

from __future__ import annotations

import json

import pytest

from microplex_us.pipelines.check_site_snapshot import (
    check_us_microplex_site_snapshot,
)
from microplex_us.pipelines.site_snapshot import write_us_microplex_site_snapshot


def test_check_us_microplex_site_snapshot_accepts_matching_snapshot(tmp_path) -> None:
    artifact_dir = tmp_path / "run-1"
    artifact_dir.mkdir()
    _write_us_artifact_bundle(artifact_dir)
    snapshot_path = tmp_path / "snapshots" / "site_snapshot_us.json"
    write_us_microplex_site_snapshot(artifact_dir, snapshot_path)

    snapshot = json.loads(snapshot_path.read_text())
    assert snapshot["sourceArtifact"]["artifactPath"] == "../run-1"

    assert check_us_microplex_site_snapshot(snapshot_path) == snapshot_path


def test_check_us_microplex_site_snapshot_rejects_stale_snapshot(tmp_path) -> None:
    artifact_dir = tmp_path / "run-1"
    artifact_dir.mkdir()
    _write_us_artifact_bundle(artifact_dir)
    snapshot_path = tmp_path / "site_snapshot_us.json"
    write_us_microplex_site_snapshot(artifact_dir, snapshot_path)

    snapshot = json.loads(snapshot_path.read_text())
    snapshot["currentRun"]["candidateMeanAbsRelativeError"] = 9.9
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))

    with pytest.raises(SystemExit, match="stale or inconsistent"):
        check_us_microplex_site_snapshot(snapshot_path)


def test_check_us_microplex_site_snapshot_rejects_stale_data_flow_sidecar(
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "run-1"
    artifact_dir.mkdir()
    _write_us_artifact_bundle(artifact_dir)
    snapshot_path = tmp_path / "site_snapshot_us.json"
    write_us_microplex_site_snapshot(artifact_dir, snapshot_path)

    data_flow_path = artifact_dir / "data_flow_snapshot.json"
    data_flow = json.loads(data_flow_path.read_text())
    data_flow["runtime"]["scaffoldSource"] = "stale_source"
    data_flow_path.write_text(json.dumps(data_flow, indent=2, sort_keys=True))

    with pytest.raises(SystemExit, match="data-flow snapshot is stale or inconsistent"):
        check_us_microplex_site_snapshot(snapshot_path)


def _write_us_artifact_bundle(artifact_dir) -> None:
    (artifact_dir / "seed_data.parquet").write_text("")
    (artifact_dir / "synthetic_data.parquet").write_text("")
    (artifact_dir / "calibrated_data.parquet").write_text("")
    (artifact_dir / "targets.json").write_text("{}")
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-03-29T00:00:00+00:00",
                "config": {"n_synthetic": 2000},
                "artifacts": {
                    "seed_data": "seed_data.parquet",
                    "synthetic_data": "synthetic_data.parquet",
                    "calibrated_data": "calibrated_data.parquet",
                    "targets": "targets.json",
                    "policyengine_harness": "policyengine_harness.json",
                },
                "synthesis": {
                    "scaffold_source": "cps_asec_2023",
                    "state_program_support_proxies": {
                        "available": ["ssi"],
                        "missing": ["snap"],
                    },
                },
                "calibration": {
                    "n_loaded_targets": 100,
                    "n_supported_targets": 90,
                    "converged": False,
                    "weight_collapse_suspected": False,
                    "household_weight_diagnostics": {
                        "effective_sample_size": 40.0,
                        "tiny_share": 0.01,
                    },
                    "person_weight_diagnostics": {
                        "effective_sample_size": 80.0,
                        "tiny_share": 0.02,
                    },
                },
                "policyengine_harness": {
                    "candidate_mean_abs_relative_error": 0.9,
                    "baseline_mean_abs_relative_error": 1.1,
                    "mean_abs_relative_error_delta": -0.2,
                },
            }
        )
    )
    (artifact_dir / "policyengine_harness.json").write_text(
        json.dumps(
            {
                "summary": {
                    "candidate_mean_abs_relative_error": 0.9,
                    "baseline_mean_abs_relative_error": 1.1,
                    "mean_abs_relative_error_delta": -0.2,
                    "candidate_composite_parity_loss": 0.8,
                    "baseline_composite_parity_loss": 1.2,
                    "target_win_rate": 0.2,
                    "slice_win_rate": 0.5,
                    "supported_target_rate": 0.9,
                    "tag_summaries": {
                        "state": {
                            "candidate_mean_abs_relative_error": 0.7,
                            "baseline_mean_abs_relative_error": 0.8,
                            "mean_abs_relative_error_delta": -0.1,
                            "candidate_composite_parity_loss": 0.6,
                            "baseline_composite_parity_loss": 0.9,
                            "target_win_rate": 0.3,
                            "slice_win_rate": 1.0,
                            "supported_target_rate": 0.85,
                        }
                    },
                    "parity_scorecard": {"overall": {"candidate_beats_baseline": True}},
                    "attribute_cell_summaries": {
                        "geo=state|feature=snap": {"candidate_target_count": 10}
                    },
                }
            }
        )
    )
