"""Tests for the persistent US microplex run registry."""

import json

from microplex_us.pipelines.registry import (
    append_us_microplex_run_registry_entry,
    build_us_microplex_run_registry_entry,
    load_us_microplex_run_registry,
)


def _manifest(
    *,
    created_at: str,
    synthesis_backend: str,
    calibration_backend: str,
    candidate_error: float,
    baseline_error: float,
    delta: float,
    candidate_composite_loss: float | None = None,
    baseline_composite_loss: float | None = None,
    composite_delta: float | None = None,
) -> dict:
    resolved_candidate_composite_loss = (
        candidate_error if candidate_composite_loss is None else candidate_composite_loss
    )
    resolved_baseline_composite_loss = (
        baseline_error if baseline_composite_loss is None else baseline_composite_loss
    )
    resolved_composite_delta = (
        resolved_candidate_composite_loss - resolved_baseline_composite_loss
        if composite_delta is None
        else composite_delta
    )
    return {
        "created_at": created_at,
        "config": {
            "synthesis_backend": synthesis_backend,
            "calibration_backend": calibration_backend,
            "random_seed": 42,
        },
        "rows": {"seed": 10, "synthetic": 20, "calibrated": 20},
        "weights": {"nonzero": 20, "total": 1000.0},
        "synthesis": {"source_names": ["cps", "puf"]},
        "policyengine_harness": {
            "candidate_mean_abs_relative_error": candidate_error,
            "baseline_mean_abs_relative_error": baseline_error,
            "mean_abs_relative_error_delta": delta,
            "candidate_composite_parity_loss": resolved_candidate_composite_loss,
            "baseline_composite_parity_loss": resolved_baseline_composite_loss,
            "composite_parity_loss_delta": resolved_composite_delta,
            "slice_win_rate": 1.0 if delta < 0 else 0.0,
            "target_win_rate": 1.0 if delta < 0 else 0.0,
            "supported_target_rate": 1.0,
            "tag_summaries": {
                "national": {
                    "candidate_mean_abs_relative_error": candidate_error,
                    "baseline_mean_abs_relative_error": baseline_error,
                    "mean_abs_relative_error_delta": delta,
                    "candidate_composite_parity_loss": resolved_candidate_composite_loss,
                    "baseline_composite_parity_loss": resolved_baseline_composite_loss,
                    "composite_parity_loss_delta": resolved_composite_delta,
                    "slice_win_rate": 1.0 if delta < 0 else 0.0,
                    "target_win_rate": 1.0 if delta < 0 else 0.0,
                    "supported_target_rate": 1.0,
                }
            },
            "parity_scorecard": {
                "overall": {
                    "candidate_mean_abs_relative_error": candidate_error,
                    "baseline_mean_abs_relative_error": baseline_error,
                    "mean_abs_relative_error_delta": delta,
                    "candidate_composite_parity_loss": resolved_candidate_composite_loss,
                    "baseline_composite_parity_loss": resolved_baseline_composite_loss,
                    "composite_parity_loss_delta": resolved_composite_delta,
                    "slice_win_rate": 1.0 if delta < 0 else 0.0,
                    "target_win_rate": 1.0 if delta < 0 else 0.0,
                    "supported_target_rate": 1.0,
                    "candidate_beats_baseline": delta < 0,
                },
                "national": {
                    "candidate_mean_abs_relative_error": candidate_error,
                    "baseline_mean_abs_relative_error": baseline_error,
                    "mean_abs_relative_error_delta": delta,
                    "candidate_composite_parity_loss": resolved_candidate_composite_loss,
                    "baseline_composite_parity_loss": resolved_baseline_composite_loss,
                    "composite_parity_loss_delta": resolved_composite_delta,
                    "slice_win_rate": 1.0 if delta < 0 else 0.0,
                    "target_win_rate": 1.0 if delta < 0 else 0.0,
                    "supported_target_rate": 1.0,
                    "candidate_beats_baseline": delta < 0,
                },
            },
        },
    }


def _harness_payload() -> dict:
    return {
        "metadata": {
            "baseline_dataset": "enhanced_cps_2024.h5",
            "targets_db": "policy_data.db",
            "target_period": 2024,
            "target_variables": ["snap", "household_count"],
            "target_domains": ["snap"],
            "target_geo_levels": ["state"],
            "target_reform_id": 0,
            "policyengine_us_runtime_version": "1.587.0",
        }
    }


def test_append_and_load_us_microplex_run_registry(tmp_path):
    registry_path = tmp_path / "runs.jsonl"

    first_entry = build_us_microplex_run_registry_entry(
        artifact_dir=tmp_path / "v1",
        manifest_path=tmp_path / "v1" / "manifest.json",
        manifest=_manifest(
            created_at="2026-03-25T12:00:00+00:00",
            synthesis_backend="bootstrap",
            calibration_backend="entropy",
            candidate_error=0.20,
            baseline_error=0.30,
            delta=-0.10,
        ),
        policyengine_harness_path=tmp_path / "v1" / "policyengine_harness.json",
        policyengine_harness_payload=_harness_payload(),
        metadata={"git_commit": "abc123"},
    )
    recorded_first = append_us_microplex_run_registry_entry(registry_path, first_entry)

    second_entry = build_us_microplex_run_registry_entry(
        artifact_dir=tmp_path / "v2",
        manifest_path=tmp_path / "v2" / "manifest.json",
        manifest=_manifest(
            created_at="2026-03-25T13:00:00+00:00",
            synthesis_backend="synthesizer",
            calibration_backend="entropy",
            candidate_error=0.25,
            baseline_error=0.30,
            delta=-0.05,
        ),
        policyengine_harness_path=tmp_path / "v2" / "policyengine_harness.json",
        policyengine_harness_payload=_harness_payload(),
        metadata={"git_commit": "def456"},
    )
    recorded_second = append_us_microplex_run_registry_entry(registry_path, second_entry)

    entries = load_us_microplex_run_registry(registry_path)

    assert len(entries) == 2
    assert entries[0].artifact_id == "v1"
    assert entries[0].source_names == ("cps", "puf")
    assert entries[0].target_variables == ("snap", "household_count")
    assert entries[0].policyengine_us_runtime_version == "1.587.0"
    assert entries[0].supported_target_rate == 1.0
    assert entries[0].tag_summaries["national"]["supported_target_rate"] == 1.0
    assert entries[0].candidate_composite_parity_loss == 0.20
    assert entries[0].parity_scorecard["overall"]["candidate_beats_baseline"] is True
    assert entries[0].metadata["git_commit"] == "abc123"
    assert entries[0].improved_candidate_frontier is True
    assert entries[0].improved_delta_frontier is True
    assert entries[0].improved_composite_frontier is True
    assert entries[1].artifact_id == "v2"
    assert entries[1].metadata["git_commit"] == "def456"
    assert entries[1].improved_candidate_frontier is False
    assert entries[1].improved_delta_frontier is False
    assert entries[1].improved_composite_frontier is False
    assert recorded_first.config_hash is not None
    assert recorded_second.config_hash is not None
    assert recorded_second.config_hash != recorded_first.config_hash

    raw_lines = registry_path.read_text().splitlines()
    assert len(raw_lines) == 2
    assert json.loads(raw_lines[0])["artifact_id"] == "v1"


def test_default_frontier_metric_prefers_composite_parity_loss(tmp_path):
    from microplex_us.pipelines.registry import select_us_microplex_frontier_entry

    registry_path = tmp_path / "runs.jsonl"

    append_us_microplex_run_registry_entry(
        registry_path,
        build_us_microplex_run_registry_entry(
            artifact_dir=tmp_path / "run-1",
            manifest_path=tmp_path / "run-1" / "manifest.json",
            manifest=_manifest(
                created_at="2026-03-25T12:00:00+00:00",
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                candidate_error=0.18,
                baseline_error=0.30,
                delta=-0.12,
                candidate_composite_loss=0.45,
                baseline_composite_loss=0.50,
                composite_delta=-0.05,
            ),
            policyengine_harness_payload=_harness_payload(),
        ),
    )
    append_us_microplex_run_registry_entry(
        registry_path,
        build_us_microplex_run_registry_entry(
            artifact_dir=tmp_path / "run-2",
            manifest_path=tmp_path / "run-2" / "manifest.json",
            manifest=_manifest(
                created_at="2026-03-25T13:00:00+00:00",
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                candidate_error=0.20,
                baseline_error=0.30,
                delta=-0.10,
                candidate_composite_loss=0.35,
                baseline_composite_loss=0.50,
                composite_delta=-0.15,
            ),
            policyengine_harness_payload=_harness_payload(),
        ),
    )

    assert select_us_microplex_frontier_entry(registry_path).artifact_id == "run-2"
    assert (
        select_us_microplex_frontier_entry(
            registry_path,
            metric="candidate_mean_abs_relative_error",
        ).artifact_id
        == "run-1"
    )
