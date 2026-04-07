"""Tests for PE-US-data rebuild native audit sidecars."""

import json

from microplex_us.pipelines.pe_us_data_rebuild_audit import (
    build_policyengine_us_data_rebuild_native_audit,
    write_policyengine_us_data_rebuild_native_audit,
)


def test_build_policyengine_us_data_rebuild_native_audit_summarizes_saved_artifact(
    tmp_path,
    monkeypatch,
):
    artifact_dir = tmp_path / "run-1"
    artifact_dir.mkdir()
    candidate_dataset = artifact_dir / "policyengine_us.h5"
    candidate_dataset.write_text("dataset")
    baseline_dataset = tmp_path / "enhanced_cps_2024.h5"
    baseline_dataset.write_text("baseline")
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "policyengine_baseline_dataset": str(baseline_dataset),
                    "policyengine_dataset_year": 2024,
                    "policyengine_target_period": 2024,
                },
                "artifacts": {
                    "policyengine_dataset": candidate_dataset.name,
                },
            }
        )
    )
    (artifact_dir / "policyengine_native_scores.json").write_text(
        json.dumps(
            {
                "metric": "enhanced_cps_native_loss",
                "period": 2024,
                "summary": {
                    "candidate_enhanced_cps_native_loss": 0.98,
                    "baseline_enhanced_cps_native_loss": 0.02,
                    "enhanced_cps_native_loss_delta": 0.96,
                    "candidate_beats_baseline": False,
                },
                "family_breakdown": [
                    {
                        "family": "national_irs_other",
                        "loss_contribution_delta": 0.30,
                    },
                    {
                        "family": "state_agi_distribution",
                        "loss_contribution_delta": 0.20,
                    },
                    {
                        "family": "state_snap_cost",
                        "loss_contribution_delta": -0.01,
                    },
                ],
            }
        )
    )

    def fake_target_delta(**kwargs):
        assert kwargs["from_dataset_path"] == baseline_dataset
        assert kwargs["to_dataset_path"] == candidate_dataset
        return {
            "top_regressions": [
                {
                    "target_name": "state/CA/adjusted_gross_income/amount/100k_200k",
                    "weighted_term_delta": 0.12,
                }
            ],
            "top_improvements": [
                {
                    "target_name": "state/NY/snap-cost",
                    "weighted_term_delta": -0.03,
                }
            ],
        }

    def fake_support_audit(**kwargs):
        assert kwargs["candidate_dataset_path"] == candidate_dataset
        assert kwargs["baseline_dataset_path"] == baseline_dataset
        return {
            "comparisons": {
                "critical_input_support": [
                    {
                        "variable": "has_esi",
                        "candidate_stored": False,
                        "baseline_stored": True,
                        "weighted_nonzero_delta": -1250.0,
                    },
                    {
                        "variable": "rental_income",
                        "candidate_stored": True,
                        "baseline_stored": True,
                        "weighted_nonzero_delta": -250.0,
                    },
                ],
                "filing_status_weighted_delta": [
                    {
                        "filing_status": "SEPARATE",
                        "weighted_count_delta": -400.0,
                    }
                ],
                "mfs_high_agi_delta": [
                    {
                        "agi_bin": "200k_to_500k",
                        "weighted_count_delta": -75.0,
                    }
                ],
                "state_marketplace_enrollment_top_gaps": [
                    {
                        "state": "CA",
                        "weighted_marketplace_enrollment_delta": -210.0,
                    }
                ],
                "state_age_bucket_top_gaps": [
                    {
                        "state": "TX",
                        "age_bucket": "18_to_29",
                        "weight_delta": -180.0,
                    }
                ],
            }
        }

    monkeypatch.setattr(
        "microplex_us.pipelines.pe_us_data_rebuild_audit.compare_us_pe_native_target_deltas",
        fake_target_delta,
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_us_data_rebuild_audit.compute_us_pe_native_support_audit",
        fake_support_audit,
    )

    audit = build_policyengine_us_data_rebuild_native_audit(artifact_dir, top_k=2)

    assert audit["artifactId"] == "run-1"
    assert audit["nativeBroadLossSummary"]["enhanced_cps_native_loss_delta"] == 0.96
    assert audit["topFamilyRegressions"][0]["family"] == "national_irs_other"
    assert audit["topFamilyImprovements"][0]["family"] == "state_snap_cost"
    assert audit["topTargetRegressions"][0]["target_name"].startswith("state/CA/")
    assert audit["supportAuditSummary"]["missingStoredCriticalInputs"] == ["has_esi"]
    assert audit["supportAuditSummary"]["topCriticalInputSupportGaps"][0]["variable"] == "has_esi"
    assert audit["verdictHints"]["largestRegressingFamily"] == "national_irs_other"
    assert audit["verdictHints"]["largestRegressingTarget"].startswith("state/CA/")


def test_write_policyengine_us_data_rebuild_native_audit_writes_default_sidecar(
    tmp_path,
    monkeypatch,
):
    artifact_dir = tmp_path / "run-2"
    artifact_dir.mkdir()
    candidate_dataset = artifact_dir / "policyengine_us.h5"
    candidate_dataset.write_text("dataset")
    baseline_dataset = tmp_path / "enhanced_cps_2024.h5"
    baseline_dataset.write_text("baseline")
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config": {"policyengine_baseline_dataset": str(baseline_dataset)},
                "artifacts": {"policyengine_dataset": candidate_dataset.name},
            }
        )
    )
    (artifact_dir / "policyengine_native_scores.json").write_text(
        json.dumps(
            {
                "summary": {
                    "candidate_enhanced_cps_native_loss": 1.0,
                    "baseline_enhanced_cps_native_loss": 0.1,
                    "enhanced_cps_native_loss_delta": 0.9,
                },
                "family_breakdown": [],
            }
        )
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_us_data_rebuild_audit.compare_us_pe_native_target_deltas",
        lambda **_kwargs: {"top_regressions": [], "top_improvements": []},
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_us_data_rebuild_audit.compute_us_pe_native_support_audit",
        lambda **_kwargs: {"comparisons": {}},
    )

    output_path = write_policyengine_us_data_rebuild_native_audit(artifact_dir)

    assert output_path == artifact_dir / "pe_us_data_rebuild_native_audit.json"
    payload = json.loads(output_path.read_text())
    assert payload["candidateDatasetPath"] == str(candidate_dataset)
    assert payload["baselineDatasetPath"] == str(baseline_dataset.resolve())
