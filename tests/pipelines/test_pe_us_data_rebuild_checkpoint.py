"""Tests for the PE-US-data rebuild checkpoint runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
from microplex.core import SourceQuery

from microplex_us.pipelines.artifacts import (
    USMicroplexArtifactPaths,
    USMicroplexVersionedBuildArtifacts,
)
from microplex_us.pipelines.pe_us_data_rebuild import (
    default_policyengine_us_data_rebuild_source_providers,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint import (
    attach_policyengine_us_data_rebuild_checkpoint_evidence,
    default_policyengine_us_data_rebuild_checkpoint_config,
    default_policyengine_us_data_rebuild_queries,
    run_policyengine_us_data_rebuild_checkpoint,
)
from microplex_us.pipelines.registry import load_us_microplex_run_registry


def test_default_policyengine_us_data_rebuild_checkpoint_config_sets_pe_context() -> None:
    config = default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
        target_period=2024,
        target_profile="pe_native_broad",
        n_synthetic=500,
        random_seed=123,
    )

    assert config.synthesis_backend == "seed"
    assert config.calibration_backend == "entropy"
    assert config.donor_imputer_backend == "qrf"
    assert config.donor_imputer_condition_selection == "pe_prespecified"
    assert config.policyengine_baseline_dataset == "/tmp/enhanced_cps_2024.h5"
    assert config.policyengine_targets_db == "/tmp/policy_data.db"
    assert config.policyengine_dataset_year == 2024
    assert config.policyengine_target_period == 2024
    assert config.policyengine_target_profile == "pe_native_broad"
    assert config.policyengine_calibration_target_profile == "pe_native_broad"
    assert config.policyengine_direct_override_variables == (
        "filing_status",
        "non_sch_d_capital_gains",
        "pre_tax_contributions",
    )
    assert config.policyengine_prefer_existing_tax_unit_ids is True
    assert config.n_synthetic == 500
    assert config.random_seed == 123


def test_default_policyengine_us_data_rebuild_checkpoint_config_infers_total_weight_targets(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_us_data_rebuild_checkpoint._infer_policyengine_baseline_household_weight_sum",
        lambda dataset, *, target_period: 150_000_000.0,
    )

    config = default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
        target_period=2024,
    )

    assert config.policyengine_calibration_target_total_weight == 150_000_000.0
    assert config.policyengine_calibration_rescale_to_target_total_weight is True
    assert config.policyengine_selection_target_total_weight == 150_000_000.0


def test_default_policyengine_us_data_rebuild_checkpoint_config_respects_explicit_total_weight_overrides(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_us_data_rebuild_checkpoint._infer_policyengine_baseline_household_weight_sum",
        lambda dataset, *, target_period: 150_000_000.0,
    )

    config = default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
        target_period=2024,
        policyengine_calibration_target_total_weight=123.0,
        policyengine_selection_target_total_weight=456.0,
    )

    assert config.policyengine_calibration_target_total_weight == 123.0
    assert config.policyengine_selection_target_total_weight == 456.0


def test_default_policyengine_us_data_rebuild_checkpoint_config_skips_calibration_total_weight_when_rescaling_to_input_sum(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_us_data_rebuild_checkpoint._infer_policyengine_baseline_household_weight_sum",
        lambda dataset, *, target_period: 150_000_000.0,
    )

    config = default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
        target_period=2024,
        policyengine_calibration_rescale_to_input_weight_sum=True,
    )

    assert config.policyengine_calibration_target_total_weight is None
    assert config.policyengine_calibration_rescale_to_target_total_weight is False
    assert config.policyengine_selection_target_total_weight == 150_000_000.0


def test_default_policyengine_us_data_rebuild_checkpoint_config_skips_inferred_total_weight_targets_for_no_calibration(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_us_data_rebuild_checkpoint._infer_policyengine_baseline_household_weight_sum",
        lambda dataset, *, target_period: 150_000_000.0,
    )

    config = default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
        target_period=2024,
        calibration_backend="none",
    )

    assert config.calibration_backend == "none"
    assert config.policyengine_calibration_target_total_weight is None
    assert config.policyengine_calibration_rescale_to_target_total_weight is False
    assert config.policyengine_selection_target_total_weight is None


def test_infer_policyengine_baseline_household_weight_sum_returns_none_when_weight_array_missing(
    tmp_path,
) -> None:
    from microplex_us.pipelines.pe_us_data_rebuild_checkpoint import (
        _infer_policyengine_baseline_household_weight_sum,
    )

    dataset_path = tmp_path / "baseline.h5"
    with h5py.File(dataset_path, "w") as handle:
        household_id = handle.create_group("household_id")
        household_id.create_dataset("2024", data=[1, 2, 3])

    inferred = _infer_policyengine_baseline_household_weight_sum(
        dataset_path,
        target_period=2024,
    )

    assert inferred is None


def test_default_policyengine_us_data_rebuild_queries_assign_sample_sizes_by_provider_type() -> None:
    providers = default_policyengine_us_data_rebuild_source_providers(
        include_donor_surveys=True,
        cps_download=False,
    )

    queries = default_policyengine_us_data_rebuild_queries(
        providers,
        cps_sample_n=11,
        puf_sample_n=22,
        donor_sample_n=33,
        random_seed=7,
    )

    assert queries[providers[0].descriptor.name].provider_filters == {
        "sample_n": 11,
        "random_seed": 7,
    }
    assert queries[providers[1].descriptor.name].provider_filters == {
        "sample_n": 22,
        "random_seed": 7,
    }
    for provider in providers[2:]:
        assert queries[provider.descriptor.name].provider_filters == {
            "sample_n": 33,
            "random_seed": 7,
        }


@dataclass(frozen=True)
class _FakeProvider:
    descriptor: Any


def test_run_policyengine_us_data_rebuild_checkpoint_builds_bundle_and_parity(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_us_data_rebuild_checkpoint._infer_policyengine_baseline_household_weight_sum",
        lambda dataset, *, target_period: 150_000_000.0,
    )
    artifact_dir = tmp_path / "artifacts" / "run-1"
    artifact_dir.mkdir(parents=True)
    provider = _FakeProvider(descriptor=SimpleNamespace(name="fake_source"))
    query = SourceQuery(provider_filters={"sample_n": 5})
    captured: dict[str, Any] = {}

    def fake_build_and_save_versioned_us_microplex_from_source_providers(
        *,
        providers,
        output_root,
        config,
        queries,
        version_id,
        frontier_metric,
        policyengine_comparison_cache,
        policyengine_target_provider,
        policyengine_baseline_dataset,
        policyengine_harness_slices,
        policyengine_harness_metadata,
        policyengine_us_data_repo,
        defer_policyengine_harness,
        require_policyengine_native_score,
        defer_policyengine_native_score,
        precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores,
        run_registry_path,
        run_index_path,
        run_registry_metadata,
    ):
        captured.update(
            {
                "providers": providers,
                "output_root": output_root,
                "config": config,
                "queries": queries,
                "version_id": version_id,
                "frontier_metric": frontier_metric,
                "policyengine_baseline_dataset": policyengine_baseline_dataset,
                "policyengine_harness_metadata": policyengine_harness_metadata,
                "run_registry_metadata": run_registry_metadata,
                "defer_policyengine_harness": defer_policyengine_harness,
                "defer_policyengine_native_score": defer_policyengine_native_score,
            }
        )
        manifest = {
            "created_at": "2026-04-06T00:00:00+00:00",
            "config": config.to_dict(),
            "rows": {"seed": 10, "synthetic": 20, "calibrated": 20},
            "weights": {"nonzero": 20, "total": 20.0},
            "targets": {"n_marginal_groups": 1, "n_continuous": 0},
            "synthesis": {
                "scaffold_source": "fake_source",
                "source_names": ["fake_source"],
                "backend": "seed",
                "condition_vars": [],
                "target_vars": [],
                "donor_integrated_variables": [],
                "state_program_support_proxies": {"available": [], "missing": []},
            },
            "calibration": {"converged": True, "n_loaded_targets": 1, "n_supported_targets": 1},
            "artifacts": {
                "seed_data": "seed_data.parquet",
                "synthetic_data": "synthetic_data.parquet",
                "calibrated_data": "calibrated_data.parquet",
                "targets": "targets.json",
                "policyengine_dataset": "policyengine_us.h5",
            },
        }
        (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
        (artifact_dir / "policyengine_us.h5").write_text("dataset")
        return USMicroplexVersionedBuildArtifacts(
            build_result=SimpleNamespace(config=config),
            artifact_paths=USMicroplexArtifactPaths(
                output_dir=artifact_dir,
                version_id="run-1",
                seed_data=artifact_dir / "seed_data.parquet",
                synthetic_data=artifact_dir / "synthetic_data.parquet",
                calibrated_data=artifact_dir / "calibrated_data.parquet",
                targets=artifact_dir / "targets.json",
                manifest=artifact_dir / "manifest.json",
            ),
        )

    def fake_write_policyengine_us_data_rebuild_parity_artifact(
        artifact_dir_arg,
        output_path=None,
        *,
        program=None,
        manifest_payload=None,
        harness_payload=None,
        native_scores_payload=None,
    ) -> Path:
        assert manifest_payload is None
        assert harness_payload is None
        assert native_scores_payload is None
        path = (
            Path(output_path)
            if output_path is not None
            else Path(artifact_dir_arg) / "pe_us_data_rebuild_parity.json"
        )
        path.write_text(
            json.dumps(
                {
                    "program": {"programId": program.program_id},
                    "verdict": {"hasRealPolicyEngineComparison": False},
                }
            )
        )
        return path

    def fake_build_policyengine_us_data_rebuild_parity_artifact(
        artifact_dir_arg,
        *,
        program=None,
        manifest_payload=None,
        harness_payload=None,
        native_scores_payload=None,
    ) -> dict[str, Any]:
        assert manifest_payload is None
        assert harness_payload is None
        assert native_scores_payload is None
        return {
            "artifactId": Path(artifact_dir_arg).name,
            "program": {"programId": program.program_id},
            "verdict": {"hasRealPolicyEngineComparison": False},
        }

    module_name = "microplex_us.pipelines.pe_us_data_rebuild_checkpoint"
    monkeypatch.setattr(
        f"{module_name}.build_and_save_versioned_us_microplex_from_source_providers",
        fake_build_and_save_versioned_us_microplex_from_source_providers,
    )
    def fake_attach_policyengine_us_data_rebuild_checkpoint_evidence(
        artifact_dir_arg,
        **kwargs,
    ):
        artifact_root = Path(artifact_dir_arg)
        registry_path = tmp_path / "artifacts" / "run_registry.jsonl"
        run_index_path = tmp_path / "artifacts" / "run_index.duckdb"
        manifest_path = artifact_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"]["policyengine_harness"] = "policyengine_harness.json"
        (artifact_root / "policyengine_harness.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "candidate_mean_abs_relative_error": 0.08,
                        "baseline_mean_abs_relative_error": 0.10,
                        "mean_abs_relative_error_delta": -0.02,
                    }
                }
            )
        )
        manifest["policyengine_harness"] = {
            "candidate_mean_abs_relative_error": 0.08,
            "baseline_mean_abs_relative_error": 0.10,
            "mean_abs_relative_error_delta": -0.02,
        }
        registry_path.write_text(
            json.dumps(
                {
                    "created_at": "2026-04-06T00:00:00+00:00",
                    "artifact_id": "run-1",
                    "artifact_dir": str(artifact_root.resolve()),
                    "manifest_path": str(manifest_path.resolve()),
                    "policyengine_harness_path": str(
                        (artifact_root / "policyengine_harness.json").resolve()
                    ),
                    "enhanced_cps_native_loss_delta": 0.5,
                }
            )
            + "\n"
        )
        run_index_path.write_text("")
        manifest["run_registry"] = {
            "path": "artifacts/run_registry.jsonl",
            "artifact_id": "run-1",
        }
        manifest["run_index"] = {
            "path": "artifacts/run_index.duckdb",
            "artifact_id": "run-1",
        }
        manifest_path.write_text(json.dumps(manifest))
        return SimpleNamespace(
            artifact_dir=artifact_root,
            manifest_path=manifest_path,
            harness_path=artifact_root / "policyengine_harness.json",
            native_scores_path=None,
            parity_path=fake_write_policyengine_us_data_rebuild_parity_artifact(
                artifact_dir_arg,
                program=kwargs.get("program"),
            ),
            parity_payload=fake_build_policyengine_us_data_rebuild_parity_artifact(
                artifact_dir_arg,
                program=kwargs.get("program"),
            ),
        )

    monkeypatch.setattr(
        f"{module_name}.attach_policyengine_us_data_rebuild_checkpoint_evidence",
        fake_attach_policyengine_us_data_rebuild_checkpoint_evidence,
    )

    result = run_policyengine_us_data_rebuild_checkpoint(
        output_root=tmp_path / "artifacts",
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
        providers=(provider,),
        queries={"fake_source": query},
        version_id="run-1",
    )

    assert result.provider_names == ("fake_source",)
    assert result.queries == {"fake_source": query}
    assert result.parity_path == artifact_dir / "pe_us_data_rebuild_parity.json"
    assert result.parity_payload["program"]["programId"] == "pe-us-data-rebuild-v1"
    assert captured["providers"] == [provider]
    assert captured["queries"] == {"fake_source": query}
    assert captured["version_id"] == "run-1"
    assert captured["frontier_metric"] == "enhanced_cps_native_loss_delta"
    assert (
        captured["policyengine_baseline_dataset"] == "/tmp/enhanced_cps_2024.h5"
    )
    assert captured["config"].policyengine_targets_db == "/tmp/policy_data.db"
    assert captured["config"].policyengine_calibration_target_total_weight == 150_000_000.0
    assert captured["config"].policyengine_calibration_rescale_to_target_total_weight is True
    assert captured["config"].policyengine_selection_target_total_weight == 150_000_000.0
    assert captured["defer_policyengine_harness"] is True
    assert captured["defer_policyengine_native_score"] is True
    assert captured["policyengine_harness_metadata"]["rebuild_checkpoint"] is True
    assert captured["policyengine_harness_metadata"]["rebuild_program_id"] == (
        "pe-us-data-rebuild-v1"
    )
    assert captured["policyengine_harness_metadata"]["rebuild_provider_names"] == [
        "fake_source"
    ]
    assert captured["run_registry_metadata"]["rebuild_profile_expected"] is True
    assert (
        result.artifacts.artifact_paths.policyengine_harness
        == artifact_dir / "policyengine_harness.json"
    )
    assert result.artifacts.artifact_paths.run_registry == tmp_path / "artifacts" / "run_registry.jsonl"
    assert result.artifacts.artifact_paths.run_index_db == tmp_path / "artifacts" / "run_index.duckdb"
    assert result.artifacts.current_entry is not None
    assert result.artifacts.current_entry.artifact_id == "run-1"
    assert result.artifacts.frontier_entry is not None
    assert result.artifacts.frontier_entry.artifact_id == "run-1"
    assert result.artifacts.frontier_delta == 0.0


def test_run_policyengine_us_data_rebuild_checkpoint_rejects_empty_provider_sequence(
    tmp_path,
) -> None:
    try:
        run_policyengine_us_data_rebuild_checkpoint(
            output_root=tmp_path / "artifacts",
            policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
            policyengine_targets_db="/tmp/policy_data.db",
            providers=(),
        )
    except ValueError as exc:
        assert "non-empty provider sequence" in str(exc)
    else:
        raise AssertionError("Expected empty providers to fail closed")


def test_run_policyengine_us_data_rebuild_checkpoint_rejects_unknown_query_keys(
    tmp_path,
) -> None:
    provider = _FakeProvider(descriptor=SimpleNamespace(name="fake_source"))
    try:
        run_policyengine_us_data_rebuild_checkpoint(
            output_root=tmp_path / "artifacts",
            policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
            policyengine_targets_db="/tmp/policy_data.db",
            providers=(provider,),
            queries={"typo_source": SourceQuery(provider_filters={"sample_n": 5})},
        )
    except ValueError as exc:
        assert "unknown provider keys" in str(exc)
        assert "fake_source" in str(exc)
    else:
        raise AssertionError("Expected unknown query keys to fail")


def test_run_policyengine_us_data_rebuild_checkpoint_rejects_mismatched_explicit_config(
    tmp_path,
) -> None:
    config = default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
        target_period=2024,
    )
    provider = _FakeProvider(descriptor=SimpleNamespace(name="fake_source"))

    try:
        run_policyengine_us_data_rebuild_checkpoint(
            output_root=tmp_path / "artifacts",
            policyengine_baseline_dataset="/tmp/other_baseline.h5",
            policyengine_targets_db="/tmp/policy_data.db",
            config=config,
            providers=(provider,),
            queries={"fake_source": SourceQuery(provider_filters={"sample_n": 5})},
        )
    except ValueError as exc:
        assert "does not match the requested PE rebuild context" in str(exc)
        assert "policyengine_baseline_dataset" in str(exc)
    else:
        raise AssertionError("Expected mismatched explicit config to fail")


def test_run_policyengine_us_data_rebuild_checkpoint_rejects_custom_python_without_native_defer(
    tmp_path,
) -> None:
    provider = _FakeProvider(descriptor=SimpleNamespace(name="fake_source"))
    try:
        run_policyengine_us_data_rebuild_checkpoint(
            output_root=tmp_path / "artifacts",
            policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
            policyengine_targets_db="/tmp/policy_data.db",
            providers=(provider,),
            queries={"fake_source": SourceQuery(provider_filters={"sample_n": 5})},
            policyengine_us_data_python="/tmp/venv/bin/python",
        )
    except ValueError as exc:
        assert "defer_policyengine_native_score=True" in str(exc)
    else:
        raise AssertionError("Expected unsupported custom PE Python path to fail")


def test_attach_policyengine_us_data_rebuild_checkpoint_evidence_updates_manifest(
    monkeypatch,
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    manifest = {
        "created_at": "2026-04-06T00:00:00+00:00",
        "config": default_policyengine_us_data_rebuild_checkpoint_config(
            policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
            policyengine_targets_db="/tmp/policy_data.db",
            target_period=2024,
        ).to_dict(),
        "rows": {"seed": 10, "synthetic": 20, "calibrated": 20},
        "weights": {"nonzero": 20, "total": 20.0},
        "targets": {"n_marginal_groups": 1, "n_continuous": 0},
        "synthesis": {
            "scaffold_source": "cps_asec_2023",
            "source_names": ["cps_asec_2023", "irs_soi_puf"],
            "backend": "seed",
            "condition_vars": [],
            "target_vars": [],
            "donor_integrated_variables": [],
            "state_program_support_proxies": {"available": [], "missing": []},
        },
        "calibration": {"converged": True, "n_loaded_targets": 1, "n_supported_targets": 1},
        "artifacts": {
            "seed_data": "seed_data.parquet",
            "synthetic_data": "synthetic_data.parquet",
            "calibrated_data": "calibrated_data.parquet",
            "targets": "targets.json",
            "policyengine_dataset": "policyengine_us.h5",
        },
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
    (artifact_dir / "data_flow_snapshot.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "stages": [
                    {
                        "id": "benchmark",
                        "status": "missing",
                        "metrics": [],
                        "outputs": [],
                    }
                ],
            }
        )
    )
    for name in (
        "seed_data.parquet",
        "synthetic_data.parquet",
        "calibrated_data.parquet",
        "targets.json",
        "policyengine_us.h5",
    ):
        (artifact_dir / name).write_text("{}")

    harness_payload = {
        "candidate_label": "microplex",
        "baseline_label": "policyengine_us_data",
        "period": 2024,
        "metadata": {"slice_profile": "pe_native_broad"},
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
            "parity_scorecard": {},
            "attribute_cell_summaries": {},
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
        },
    }

    module_name = "microplex_us.pipelines.pe_us_data_rebuild_checkpoint"
    monkeypatch.setattr(
        f"{module_name}.write_policyengine_us_data_rebuild_parity_artifact",
        lambda artifact_dir_arg, **kwargs: (Path(artifact_dir_arg) / "pe_us_data_rebuild_parity.json"),
    )
    monkeypatch.setattr(
        f"{module_name}.build_policyengine_us_data_rebuild_parity_artifact",
        lambda artifact_dir_arg, **kwargs: {
            "artifactId": Path(artifact_dir_arg).name,
            "verdict": {"hasRealPolicyEngineComparison": True},
        },
    )

    result = attach_policyengine_us_data_rebuild_checkpoint_evidence(
        artifact_dir,
        compute_harness=False,
        compute_native_scores=False,
        precomputed_policyengine_harness_payload=harness_payload,
        precomputed_policyengine_native_scores=native_scores_payload,
        run_registry_path=tmp_path / "run_registry.jsonl",
        run_index_path=tmp_path,
        run_registry_metadata={"checkpoint_test": True},
    )

    written_manifest = json.loads((artifact_dir / "manifest.json").read_text())
    refreshed_snapshot = json.loads((artifact_dir / "data_flow_snapshot.json").read_text())
    benchmark_stage = next(
        stage for stage in refreshed_snapshot["stages"] if stage["id"] == "benchmark"
    )
    registry_entries = load_us_microplex_run_registry(tmp_path / "run_registry.jsonl")
    assert result.harness_path == artifact_dir / "policyengine_harness.json"
    assert result.native_scores_path == artifact_dir / "policyengine_native_scores.json"
    assert written_manifest["artifacts"]["policyengine_harness"] == "policyengine_harness.json"
    assert (
        written_manifest["artifacts"]["policyengine_native_scores"]
        == "policyengine_native_scores.json"
    )
    assert written_manifest["policyengine_harness"]["mean_abs_relative_error_delta"] == -0.02
    assert (
        written_manifest["policyengine_native_scores"]["enhanced_cps_native_loss_delta"]
        == 0.10
    )
    assert written_manifest["run_registry"]["artifact_id"] == "artifact"
    assert written_manifest["run_index"]["artifact_id"] == "artifact"
    assert (tmp_path / "run_index.duckdb").exists()
    assert len(registry_entries) == 1
    assert registry_entries[0].artifact_id == "artifact"
    assert registry_entries[0].metadata["checkpoint_test"] is True
    assert benchmark_stage["status"] == "ready"
    assert benchmark_stage["outputs"] == [
        "policyengine_harness.json",
        "policyengine_native_scores.json",
    ]
