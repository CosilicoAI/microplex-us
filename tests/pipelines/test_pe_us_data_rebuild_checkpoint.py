"""Tests for the PE-US-data rebuild checkpoint runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from microplex.core import SourceQuery

from microplex_us.pipelines.artifacts import (
    USMicroplexArtifactPaths,
    USMicroplexVersionedBuildArtifacts,
)
from microplex_us.pipelines.pe_us_data_rebuild import (
    default_policyengine_us_data_rebuild_source_providers,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint import (
    default_policyengine_us_data_rebuild_checkpoint_config,
    default_policyengine_us_data_rebuild_queries,
    run_policyengine_us_data_rebuild_checkpoint,
)


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
    assert config.n_synthetic == 500
    assert config.random_seed == 123


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
            }
        )
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
    monkeypatch.setattr(
        f"{module_name}.write_policyengine_us_data_rebuild_parity_artifact",
        fake_write_policyengine_us_data_rebuild_parity_artifact,
    )
    monkeypatch.setattr(
        f"{module_name}.build_policyengine_us_data_rebuild_parity_artifact",
        fake_build_policyengine_us_data_rebuild_parity_artifact,
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
    assert captured["policyengine_harness_metadata"]["rebuild_checkpoint"] is True
    assert captured["policyengine_harness_metadata"]["rebuild_program_id"] == (
        "pe-us-data-rebuild-v1"
    )
    assert captured["policyengine_harness_metadata"]["rebuild_provider_names"] == [
        "fake_source"
    ]
    assert captured["run_registry_metadata"]["rebuild_profile_expected"] is True


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
