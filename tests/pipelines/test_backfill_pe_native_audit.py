"""Tests for historical PE rebuild native-audit backfill."""

from __future__ import annotations

import json

from microplex_us.pipelines.backfill_pe_native_audit import (
    backfill_us_pe_native_audit_bundle,
    backfill_us_pe_native_audit_root,
)


def test_backfill_us_pe_native_audit_root_updates_manifest_and_snapshot(
    monkeypatch,
    tmp_path,
) -> None:
    artifact_root = tmp_path / "live_runs"
    bundle_dir = artifact_root / "run-1"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "policyengine_us.h5").write_text("candidate")
    (bundle_dir / "policyengine_native_scores.json").write_text(
        json.dumps(
            {
                "metric": "enhanced_cps_native_loss",
                "summary": {
                    "enhanced_cps_native_loss_delta": 0.25,
                },
            }
        )
    )
    (bundle_dir / "data_flow_snapshot.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "stages": [
                    {
                        "id": "benchmark",
                        "outputs": ["policyengine_native_scores.json"],
                        "metrics": [],
                        "status": "ready",
                    }
                ],
            }
        )
    )

    manifest = {
        "created_at": "2026-03-29T12:00:00+00:00",
        "config": {
            "policyengine_baseline_dataset": str((tmp_path / "baseline.h5").resolve()),
            "policyengine_dataset_year": 2024,
        },
        "rows": {"seed": 10, "synthetic": 20, "calibrated": 20},
        "weights": {"nonzero": 20, "total": 1000.0},
        "synthesis": {"source_names": ["cps", "puf"]},
        "calibration": {"converged": True},
        "artifacts": {
            "policyengine_dataset": "policyengine_us.h5",
            "policyengine_native_scores": "policyengine_native_scores.json",
        },
        "policyengine_native_scores": {
            "enhanced_cps_native_loss_delta": 0.25,
        },
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    monkeypatch.setattr(
        "microplex_us.pipelines.backfill_pe_native_audit.build_policyengine_us_data_rebuild_native_audit",
        lambda *args, **kwargs: {
            "artifactId": "run-1",
            "verdictHints": {
                "largestRegressingFamily": "national_irs_other",
                "productionImputationVariant": "structured_pe_conditioning",
                "productionImputationVariantIsMaeWinner": False,
                "productionImputationVariantIsSupportWinner": True,
            },
        },
    )

    manifest_paths = backfill_us_pe_native_audit_root(artifact_root)

    assert manifest_paths == [manifest_path]
    updated_manifest = json.loads(manifest_path.read_text())
    assert (
        updated_manifest["artifacts"]["policyengine_native_audit"]
        == "pe_us_data_rebuild_native_audit.json"
    )
    assert (
        updated_manifest["policyengine_native_audit"][
            "productionImputationVariantIsSupportWinner"
        ]
        is True
    )
    snapshot = json.loads((bundle_dir / "data_flow_snapshot.json").read_text())
    benchmark = next(stage for stage in snapshot["stages"] if stage["id"] == "benchmark")
    assert benchmark["outputs"] == [
        "policyengine_native_scores.json",
        "pe_us_data_rebuild_native_audit.json",
    ]


def test_backfill_us_pe_native_audit_bundle_reuses_existing_sidecar_without_recomputing(
    monkeypatch,
    tmp_path,
) -> None:
    bundle_dir = tmp_path / "run-1"
    bundle_dir.mkdir()
    (bundle_dir / "policyengine_us.h5").write_text("candidate")
    (bundle_dir / "policyengine_native_scores.json").write_text(
        json.dumps({"metric": "enhanced_cps_native_loss", "summary": {}})
    )
    (bundle_dir / "pe_us_data_rebuild_native_audit.json").write_text(
        json.dumps(
            {
                "artifactId": "run-1",
                "verdictHints": {
                    "largestRegressingFamily": "national_irs_other",
                    "productionImputationVariantIsMaeWinner": False,
                },
            }
        )
    )
    manifest = {
        "created_at": "2026-03-29T12:00:00+00:00",
        "config": {
            "policyengine_baseline_dataset": str((tmp_path / "baseline.h5").resolve()),
            "policyengine_dataset_year": 2024,
        },
        "rows": {"seed": 10, "synthetic": 20, "calibrated": 20},
        "weights": {"nonzero": 20, "total": 1000.0},
        "synthesis": {"source_names": ["cps", "puf"]},
        "calibration": {"converged": True},
        "artifacts": {
            "policyengine_dataset": "policyengine_us.h5",
            "policyengine_native_scores": "policyengine_native_scores.json",
        },
        "policyengine_native_scores": {},
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    monkeypatch.setattr(
        "microplex_us.pipelines.backfill_pe_native_audit.build_policyengine_us_data_rebuild_native_audit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("should not recompute when audit sidecar already exists")
        ),
    )

    returned_manifest_path = backfill_us_pe_native_audit_bundle(bundle_dir)

    assert returned_manifest_path == manifest_path
    updated_manifest = json.loads(manifest_path.read_text())
    assert (
        updated_manifest["artifacts"]["policyengine_native_audit"]
        == "pe_us_data_rebuild_native_audit.json"
    )
    assert (
        updated_manifest["policyengine_native_audit"]["largestRegressingFamily"]
        == "national_irs_other"
    )
