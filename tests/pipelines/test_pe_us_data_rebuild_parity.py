"""Tests for the PE-US-data rebuild parity artifact helpers."""

from __future__ import annotations

import json

from microplex_us.pipelines.pe_us_data_rebuild import (
    default_policyengine_us_data_rebuild_config,
)
from microplex_us.pipelines.pe_us_data_rebuild_parity import (
    build_policyengine_us_data_rebuild_parity_artifact,
    write_policyengine_us_data_rebuild_parity_artifact,
)


def test_build_policyengine_us_data_rebuild_parity_artifact_summarizes_comparison(
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    config = default_policyengine_us_data_rebuild_config(
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
    ).to_dict()
    manifest = {
        "config": config,
        "artifacts": {
            "policyengine_harness": "policyengine_harness.json",
            "policyengine_native_scores": "policyengine_native_scores.json",
        },
    }
    harness_payload = {
        "candidate_label": "microplex",
        "baseline_label": "policyengine_us_data",
        "period": 2024,
        "metadata": {"slice_profile": "pe_native_broad"},
        "summary": {
            "candidate_mean_abs_relative_error": 0.12,
            "baseline_mean_abs_relative_error": 0.10,
            "mean_abs_relative_error_delta": 0.02,
            "candidate_composite_parity_loss": 0.20,
            "baseline_composite_parity_loss": 0.15,
            "composite_parity_loss_delta": 0.05,
            "slice_win_rate": 0.40,
            "target_win_rate": 0.35,
            "supported_target_rate": 0.97,
            "baseline_supported_target_rate": 0.99,
            "tag_summaries": {"national": {"target_win_rate": 0.5}},
        },
    }
    native_scores_payload = {
        "metric": "enhanced_cps_native_loss",
        "period": 2024,
        "summary": {
            "candidate_enhanced_cps_native_loss": 0.30,
            "baseline_enhanced_cps_native_loss": 0.20,
            "enhanced_cps_native_loss_delta": 0.10,
            "candidate_beats_baseline": False,
            "candidate_unweighted_msre": 0.31,
            "baseline_unweighted_msre": 0.21,
            "unweighted_msre_delta": 0.10,
            "n_targets_total": 100,
            "n_targets_kept": 90,
            "n_targets_zero_dropped": 5,
            "n_targets_bad_dropped": 5,
            "n_national_targets": 20,
            "n_state_targets": 70,
        },
    }

    payload = build_policyengine_us_data_rebuild_parity_artifact(
        artifact_dir,
        manifest_payload=manifest,
        harness_payload=harness_payload,
        native_scores_payload=native_scores_payload,
    )

    assert payload["schemaVersion"] == 1
    assert payload["program"]["programId"] == "pe-us-data-rebuild-v1"
    assert payload["profileConformance"]["exactMatch"] is True
    assert payload["evidence"]["manifest"]["source"] == "in_memory_override"
    assert payload["evidence"]["policyengineHarness"]["source"] == "in_memory_override"
    assert payload["evidence"]["policyengineNativeScores"]["source"] == "in_memory_override"
    assert payload["baselineSlice"]["baselineDatasetPath"] == "/tmp/enhanced_cps_2024.h5"
    assert payload["baselineSlice"]["baselineLabel"] == "policyengine_us_data"
    assert payload["comparison"]["policyengineHarness"]["isPolicyEngineComparison"] is True
    assert (
        payload["comparison"]["policyengineNativeScores"]["isPolicyEngineComparison"]
        is True
    )
    assert (
        payload["comparison"]["policyengineHarness"]["mean_abs_relative_error_delta"]
        == 0.02
    )
    assert (
        payload["comparison"]["policyengineNativeScores"]["enhanced_cps_native_loss_delta"]
        == 0.10
    )
    assert payload["verdict"]["candidateBeatsHarnessMeanAbsRelativeError"] is False
    assert payload["verdict"]["candidateBeatsNativeBroadLoss"] is False
    assert payload["verdict"]["hasRealPolicyEngineComparison"] is True


def test_write_policyengine_us_data_rebuild_parity_artifact_records_config_drift(
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    config = {
        **default_policyengine_us_data_rebuild_config(
            donor_imputer_condition_selection="top_correlated",
            policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        ).to_dict(),
        "experimental_override": 7,
    }
    manifest = {"config": config}
    harness_payload = {
        "candidate_label": "microplex",
        "baseline_label": "policyengine_us_data",
        "period": 2024,
        "metadata": {},
        "summary": {
            "candidate_mean_abs_relative_error": 0.08,
            "baseline_mean_abs_relative_error": 0.10,
            "mean_abs_relative_error_delta": -0.02,
            "candidate_composite_parity_loss": 0.14,
            "baseline_composite_parity_loss": 0.15,
            "composite_parity_loss_delta": -0.01,
            "slice_win_rate": 0.55,
            "target_win_rate": 0.58,
            "supported_target_rate": 0.98,
            "baseline_supported_target_rate": 0.99,
            "tag_summaries": {},
        },
    }

    output_path = write_policyengine_us_data_rebuild_parity_artifact(
        artifact_dir,
        manifest_payload=manifest,
        harness_payload=harness_payload,
    )

    written = json.loads(output_path.read_text())
    drift = {
        item["key"]: item
        for item in written["profileConformance"]["differingKeys"]
    }

    assert output_path == artifact_dir / "pe_us_data_rebuild_parity.json"
    assert written["profileConformance"]["exactMatch"] is False
    assert drift["donor_imputer_condition_selection"]["expected"] == "pe_prespecified"
    assert drift["donor_imputer_condition_selection"]["observed"] == "top_correlated"
    assert drift["experimental_override"]["expected"] is None
    assert drift["experimental_override"]["observed"] == 7
    assert written["verdict"]["candidateBeatsHarnessMeanAbsRelativeError"] is True


def test_write_policyengine_us_data_rebuild_parity_artifact_uses_bundle_files_and_validates_identity(
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    manifest = {
        "config": default_policyengine_us_data_rebuild_config(
            policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
            policyengine_targets_db="/tmp/policy_data.db",
        ).to_dict(),
    }
    harness_payload = {
        "candidate_label": "microplex",
        "baseline_label": "not_policyengine",
        "period": 2024,
        "metadata": {},
        "summary": {
            "candidate_mean_abs_relative_error": 0.08,
            "baseline_mean_abs_relative_error": 0.10,
            "mean_abs_relative_error_delta": -0.02,
            "candidate_composite_parity_loss": 0.14,
            "baseline_composite_parity_loss": 0.15,
            "composite_parity_loss_delta": -0.01,
            "slice_win_rate": 0.55,
            "target_win_rate": 0.58,
            "supported_target_rate": 0.98,
            "baseline_supported_target_rate": 0.99,
            "tag_summaries": {},
        },
    }
    native_scores_payload = {
        "metric": "not_enhanced_cps_native_loss",
        "period": 2024,
        "summary": {
            "candidate_beats_baseline": True,
        },
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
    (artifact_dir / "policyengine_harness.json").write_text(json.dumps(harness_payload))
    (artifact_dir / "policyengine_native_scores.json").write_text(
        json.dumps(native_scores_payload)
    )

    output_path = write_policyengine_us_data_rebuild_parity_artifact(artifact_dir)
    written = json.loads(output_path.read_text())

    assert written["evidence"]["manifest"]["source"] == "artifact_bundle"
    assert written["evidence"]["manifest"]["exists"] is True
    assert written["evidence"]["policyengineHarness"]["exists"] is True
    assert written["evidence"]["policyengineNativeScores"]["exists"] is True
    assert written["comparison"]["policyengineHarness"]["isPolicyEngineComparison"] is False
    assert (
        written["comparison"]["policyengineNativeScores"]["isPolicyEngineComparison"]
        is False
    )
    assert written["verdict"]["candidateBeatsHarnessMeanAbsRelativeError"] is None
    assert written["verdict"]["candidateBeatsNativeBroadLoss"] is None
    assert written["verdict"]["hasRealPolicyEngineComparison"] is False
