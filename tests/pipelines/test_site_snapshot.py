"""Tests for canonical US site snapshot generation."""

import json

from microplex_us.pipelines.site_snapshot import build_us_microplex_site_snapshot


def test_build_us_microplex_site_snapshot_reads_manifest_and_harness(tmp_path):
    artifact_dir = tmp_path / "run-1"
    artifact_dir.mkdir()
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
    (artifact_dir / "seed_data.parquet").write_text("")
    (artifact_dir / "synthetic_data.parquet").write_text("")
    (artifact_dir / "calibrated_data.parquet").write_text("")
    (artifact_dir / "targets.json").write_text("{}")
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

    snapshot = build_us_microplex_site_snapshot(artifact_dir)

    assert snapshot["currentRun"]["benchmarkTag"] == "state"
    assert snapshot["currentRun"]["candidateMeanAbsRelativeError"] == 0.7
    assert snapshot["currentRun"]["nSynthetic"] == 2000
    assert snapshot["currentRun"]["supportProxies"]["available"] == ["ssi"]
    assert snapshot["summary"]["supported_target_rate"] == 0.9
