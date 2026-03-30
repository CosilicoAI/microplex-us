"""Artifact persistence for production pipeline outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from microplex.core import SourceProvider, SourceQuery
from microplex.targets import (
    TargetProvider,
    assert_valid_benchmark_artifact_manifest,
)

from microplex_us.pipelines.index_db import (
    append_us_microplex_run_index_entry,
)
from microplex_us.pipelines.pe_native_scores import (
    compute_us_pe_native_scores,
)
from microplex_us.pipelines.registry import (
    FrontierMetric,
    append_us_microplex_run_registry_entry,
    build_us_microplex_run_registry_entry,
    load_us_microplex_run_registry,
    select_us_microplex_frontier_entry,
)
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexBuildResult,
    USMicroplexPipeline,
    build_us_microplex,
)
from microplex_us.policyengine.harness import (
    PolicyEngineUSComparisonCache,
    PolicyEngineUSHarnessSlice,
    default_policyengine_us_db_all_target_slices,
    default_policyengine_us_harness_slices,
    evaluate_policyengine_us_harness,
    filter_nonempty_policyengine_us_harness_slices,
)
from microplex_us.policyengine.us import (
    PolicyEngineUSDBTargetProvider,
)


@dataclass(frozen=True)
class USMicroplexArtifactPaths:
    """Filesystem locations for persisted pipeline artifacts."""

    output_dir: Path
    seed_data: Path
    synthetic_data: Path
    calibrated_data: Path
    targets: Path
    manifest: Path
    version_id: str | None = None
    synthesizer: Path | None = None
    policyengine_dataset: Path | None = None
    policyengine_harness: Path | None = None
    policyengine_native_scores: Path | None = None
    run_registry: Path | None = None
    run_index_db: Path | None = None


@dataclass(frozen=True)
class USMicroplexVersionedBuildArtifacts:
    """End-to-end build, save, and frontier-tracking result."""

    build_result: USMicroplexBuildResult
    artifact_paths: USMicroplexArtifactPaths
    current_entry: Any | None = None
    frontier_entry: Any | None = None
    frontier_delta: float | None = None


def save_us_microplex_artifacts(
    result: USMicroplexBuildResult,
    output_dir: str | Path,
    *,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
) -> USMicroplexArtifactPaths:
    """Persist a build result as a reproducible artifact bundle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_data_path = output_dir / "seed_data.parquet"
    synthetic_data_path = output_dir / "synthetic_data.parquet"
    calibrated_data_path = output_dir / "calibrated_data.parquet"
    targets_path = output_dir / "targets.json"
    manifest_path = output_dir / "manifest.json"
    synthesizer_path = output_dir / "synthesizer.pt" if result.synthesizer else None
    policyengine_dataset_path = (
        output_dir / "policyengine_us.h5" if result.policyengine_tables is not None else None
    )
    policyengine_harness_path = None
    policyengine_native_scores_path = None
    resolved_run_registry_path = None
    resolved_run_index_path = None
    harness_payload = None

    result.seed_data.to_parquet(seed_data_path, index=False)
    result.synthetic_data.to_parquet(synthetic_data_path, index=False)
    result.calibrated_data.to_parquet(calibrated_data_path, index=False)
    targets_path.write_text(
        json.dumps(
            {
                "marginal": result.targets.marginal,
                "continuous": result.targets.continuous,
            },
            indent=2,
            sort_keys=True,
        )
    )

    if result.synthesizer is not None and synthesizer_path is not None:
        result.synthesizer.save(synthesizer_path)

    if result.policyengine_tables is not None and policyengine_dataset_path is not None:
        period = result.config.policyengine_dataset_year or 2024
        USMicroplexPipeline(result.config).export_policyengine_dataset(
            result,
            policyengine_dataset_path,
            period=period,
        )

    (
        resolved_target_provider,
        resolved_baseline_dataset,
        resolved_harness_slices,
        resolved_harness_metadata,
    ) = _resolve_policyengine_harness_context(
        result,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
    )

    harness_summary = None
    native_scores_payload = (
        dict(precomputed_policyengine_native_scores)
        if precomputed_policyengine_native_scores is not None
        else None
    )
    if precomputed_policyengine_harness_payload is not None:
        harness_payload = dict(precomputed_policyengine_harness_payload)
        policyengine_harness_path = output_dir / "policyengine_harness.json"
        policyengine_harness_path.write_text(
            json.dumps(harness_payload, indent=2, sort_keys=True)
        )
        harness_summary = harness_payload.get("summary")
    elif (
        not defer_policyengine_harness
        and result.policyengine_tables is not None
        and resolved_target_provider is not None
        and resolved_baseline_dataset is not None
        and resolved_harness_slices
    ):
        harness_period = result.config.policyengine_dataset_year or 2024
        harness_run = evaluate_policyengine_us_harness(
            result.policyengine_tables,
            resolved_target_provider,
            resolved_harness_slices,
            baseline_dataset=str(resolved_baseline_dataset),
            dataset_year=harness_period,
            simulation_cls=result.config.policyengine_simulation_cls,
            candidate_label="microplex",
            baseline_label="policyengine_us_data",
            metadata=resolved_harness_metadata,
            cache=policyengine_comparison_cache,
        )
        policyengine_harness_path = output_dir / "policyengine_harness.json"
        harness_run.save(policyengine_harness_path)
        harness_payload = harness_run.to_dict()
        harness_summary = harness_payload["summary"]

    if native_scores_payload is not None:
        policyengine_native_scores_path = output_dir / "policyengine_native_scores.json"
        policyengine_native_scores_path.write_text(
            json.dumps(native_scores_payload, indent=2, sort_keys=True)
        )
    elif (
        not defer_policyengine_native_score
        and policyengine_dataset_path is not None
        and resolved_baseline_dataset is not None
    ):
        try:
            native_scores_payload = compute_us_pe_native_scores(
                candidate_dataset_path=policyengine_dataset_path,
                baseline_dataset_path=resolved_baseline_dataset,
                period=result.config.policyengine_dataset_year or 2024,
                policyengine_us_data_repo=policyengine_us_data_repo,
            )
            policyengine_native_scores_path = (
                output_dir / "policyengine_native_scores.json"
            )
            policyengine_native_scores_path.write_text(
                json.dumps(native_scores_payload, indent=2, sort_keys=True)
            )
        except Exception:
            if require_policyengine_native_score:
                raise

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": result.config.to_dict(),
        "rows": {
            "seed": int(len(result.seed_data)),
            "synthetic": int(len(result.synthetic_data)),
            "calibrated": int(len(result.calibrated_data)),
        },
        "weights": {
            "nonzero": result.n_nonzero_weights,
            "total": result.total_weighted_population,
        },
        "targets": {
            "n_marginal_groups": len(result.targets.marginal),
            "n_continuous": len(result.targets.continuous),
        },
        "synthesis": result.synthesis_metadata,
        "calibration": result.calibration_summary,
        "artifacts": {
            "seed_data": seed_data_path.name,
            "synthetic_data": synthetic_data_path.name,
            "calibrated_data": calibrated_data_path.name,
            "targets": targets_path.name,
            "synthesizer": synthesizer_path.name if synthesizer_path else None,
            "policyengine_dataset": (
                policyengine_dataset_path.name if policyengine_dataset_path else None
            ),
            "policyengine_harness": (
                policyengine_harness_path.name if policyengine_harness_path else None
            ),
            "policyengine_native_scores": (
                policyengine_native_scores_path.name
                if policyengine_native_scores_path is not None
                else None
            ),
        },
    }
    if harness_summary is not None:
        manifest["policyengine_harness"] = harness_summary
    if native_scores_payload is not None:
        manifest["policyengine_native_scores"] = dict(
            native_scores_payload.get("summary", {})
        )
    assert_valid_benchmark_artifact_manifest(
        manifest,
        artifact_dir=output_dir,
        manifest_path=manifest_path,
        summary_section=(
            "policyengine_harness" if harness_summary is not None else None
        ),
        required_artifact_keys=(
            "seed_data",
            "synthetic_data",
            "calibrated_data",
            "targets",
            *(
                ("policyengine_native_scores",)
                if native_scores_payload is not None
                else ()
            ),
        ),
        required_summary_keys=(
            (
                "candidate_mean_abs_relative_error",
                "baseline_mean_abs_relative_error",
                "mean_abs_relative_error_delta",
            )
            if harness_summary is not None
            else ()
        ),
    )
    if harness_summary is not None or native_scores_payload is not None:
        resolved_run_registry_path = Path(run_registry_path or output_dir.parent / "run_registry.jsonl")
        run_entry = build_us_microplex_run_registry_entry(
            artifact_dir=output_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            policyengine_harness_path=policyengine_harness_path,
            policyengine_harness_payload=harness_payload,
            metadata=dict(run_registry_metadata or {}),
        )
        recorded_entry = append_us_microplex_run_registry_entry(
            resolved_run_registry_path,
            run_entry,
        )
        resolved_run_index_path = append_us_microplex_run_index_entry(
            run_index_path or output_dir.parent,
            recorded_entry,
            policyengine_harness_payload=harness_payload,
        )
        manifest["run_registry"] = {
            "path": str(resolved_run_registry_path),
            "artifact_id": recorded_entry.artifact_id,
            "improved_candidate_frontier": recorded_entry.improved_candidate_frontier,
            "improved_delta_frontier": recorded_entry.improved_delta_frontier,
            "improved_composite_frontier": recorded_entry.improved_composite_frontier,
            "improved_native_frontier": recorded_entry.improved_native_frontier,
            "default_frontier_metric": (
                "enhanced_cps_native_loss_delta"
                if native_scores_payload is not None
                else "candidate_composite_parity_loss"
            ),
        }
        manifest["run_index"] = {
            "path": str(resolved_run_index_path),
            "artifact_id": recorded_entry.artifact_id,
        }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    return USMicroplexArtifactPaths(
        output_dir=output_dir,
        version_id=output_dir.name,
        seed_data=seed_data_path,
        synthetic_data=synthetic_data_path,
        calibrated_data=calibrated_data_path,
        targets=targets_path,
        manifest=manifest_path,
        synthesizer=synthesizer_path,
        policyengine_dataset=policyengine_dataset_path,
        policyengine_harness=policyengine_harness_path,
        policyengine_native_scores=policyengine_native_scores_path,
        run_registry=resolved_run_registry_path,
        run_index_db=resolved_run_index_path,
    )


def save_versioned_us_microplex_artifacts(
    result: USMicroplexBuildResult,
    output_root: str | Path,
    *,
    version_id: str | None = None,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
) -> USMicroplexArtifactPaths:
    """Persist a build under a stable versioned directory beneath one output root."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_version_id, output_dir = _allocate_versioned_output_dir(
        output_root,
        version_id=version_id,
        result=result,
    )
    paths = save_us_microplex_artifacts(
        result,
        output_dir,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path or output_root / "run_registry.jsonl",
        run_index_path=run_index_path or output_root,
        run_registry_metadata=run_registry_metadata,
    )
    return USMicroplexArtifactPaths(
        output_dir=paths.output_dir,
        version_id=resolved_version_id,
        seed_data=paths.seed_data,
        synthetic_data=paths.synthetic_data,
        calibrated_data=paths.calibrated_data,
        targets=paths.targets,
        manifest=paths.manifest,
        synthesizer=paths.synthesizer,
        policyengine_dataset=paths.policyengine_dataset,
        policyengine_harness=paths.policyengine_harness,
        policyengine_native_scores=paths.policyengine_native_scores,
        run_registry=paths.run_registry,
        run_index_db=paths.run_index_db,
    )


def build_and_save_versioned_us_microplex(
    persons: Any,
    households: Any,
    output_root: str | Path,
    *,
    config: USMicroplexBuildConfig | None = None,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
) -> USMicroplexVersionedBuildArtifacts:
    """Build a US microplex dataset, save a versioned bundle, and report frontier gap."""
    build_result = build_us_microplex(persons, households, config=config)
    return save_versioned_us_microplex_build_result(
        build_result,
        output_root,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
    )


def save_versioned_us_microplex_build_result(
    build_result: USMicroplexBuildResult,
    output_root: str | Path,
    *,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
) -> USMicroplexVersionedBuildArtifacts:
    """Save an already-built result as a versioned bundle and report frontier gap."""
    return _finalize_versioned_build_artifacts(
        build_result,
        output_root=output_root,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
    )


def build_and_save_versioned_us_microplex_from_source_provider(
    provider: SourceProvider,
    output_root: str | Path,
    *,
    config: USMicroplexBuildConfig | None = None,
    query: SourceQuery | None = None,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
) -> USMicroplexVersionedBuildArtifacts:
    """Build from one source provider, save a versioned bundle, and report frontier gap."""
    pipeline = USMicroplexPipeline(config)
    build_result = pipeline.build_from_source_provider(provider, query=query)
    return _finalize_versioned_build_artifacts(
        build_result,
        output_root=output_root,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
    )


def build_and_save_versioned_us_microplex_from_source_providers(
    providers: list[SourceProvider],
    output_root: str | Path,
    *,
    config: USMicroplexBuildConfig | None = None,
    queries: dict[str, SourceQuery] | None = None,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
) -> USMicroplexVersionedBuildArtifacts:
    """Build from multiple source providers, save a versioned bundle, and report frontier gap."""
    pipeline = USMicroplexPipeline(config)
    build_result = pipeline.build_from_source_providers(providers, queries=queries)
    return _finalize_versioned_build_artifacts(
        build_result,
        output_root=output_root,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
    )


def build_and_save_versioned_us_microplex_from_data_dir(
    data_dir: str | Path,
    output_root: str | Path,
    *,
    config: USMicroplexBuildConfig | None = None,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
) -> USMicroplexVersionedBuildArtifacts:
    """Build from a CPS-style parquet directory, save a versioned bundle, and report frontier gap."""
    pipeline = USMicroplexPipeline(config)
    build_result = pipeline.build_from_data_dir(data_dir)
    return _finalize_versioned_build_artifacts(
        build_result,
        output_root=output_root,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
    )


def _finalize_versioned_build_artifacts(
    build_result: USMicroplexBuildResult,
    *,
    output_root: str | Path,
    version_id: str | None,
    frontier_metric: FrontierMetric,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None,
    policyengine_target_provider: TargetProvider | None,
    policyengine_baseline_dataset: str | Path | None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ),
    policyengine_harness_metadata: dict[str, Any] | None,
    policyengine_us_data_repo: str | Path | None,
    defer_policyengine_harness: bool,
    require_policyengine_native_score: bool,
    defer_policyengine_native_score: bool,
    precomputed_policyengine_harness_payload: dict[str, Any] | None,
    precomputed_policyengine_native_scores: dict[str, Any] | None,
    run_registry_path: str | Path | None,
    run_index_path: str | Path | None,
    run_registry_metadata: dict[str, Any] | None,
) -> USMicroplexVersionedBuildArtifacts:
    artifact_paths = save_versioned_us_microplex_artifacts(
        build_result,
        output_root,
        version_id=version_id,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
    )
    current_entry = None
    frontier_entry = None
    frontier_delta = None
    if artifact_paths.run_registry is not None and artifact_paths.version_id is not None:
        registry_entries = load_us_microplex_run_registry(artifact_paths.run_registry)
        current_entry = next(
            (
                entry
                for entry in reversed(registry_entries)
                if entry.artifact_id == artifact_paths.version_id
            ),
            None,
        )
        frontier_entry = select_us_microplex_frontier_entry(
            artifact_paths.run_registry,
            metric=frontier_metric,
        )
        current_value = _registry_metric_value(current_entry, frontier_metric)
        frontier_value = _registry_metric_value(frontier_entry, frontier_metric)
        if current_value is not None and frontier_value is not None:
            frontier_delta = current_value - frontier_value
    return USMicroplexVersionedBuildArtifacts(
        build_result=build_result,
        artifact_paths=artifact_paths,
        current_entry=current_entry,
        frontier_entry=frontier_entry,
        frontier_delta=frontier_delta,
    )


def _resolve_policyengine_harness_context(
    result: USMicroplexBuildResult,
    *,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None,
    policyengine_target_provider: TargetProvider | None,
    policyengine_baseline_dataset: str | Path | None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ),
    policyengine_harness_metadata: dict[str, Any] | None,
) -> tuple[
    TargetProvider | None,
    str | Path | None,
    tuple[PolicyEngineUSHarnessSlice, ...],
    dict[str, Any],
]:
    resolved_target_provider = policyengine_target_provider
    if resolved_target_provider is None and result.config.policyengine_targets_db is not None:
        resolved_target_provider = PolicyEngineUSDBTargetProvider(
            result.config.policyengine_targets_db
        )

    resolved_baseline_dataset = (
        policyengine_baseline_dataset or result.config.policyengine_baseline_dataset
    )

    harness_period = result.config.policyengine_dataset_year or 2024
    if policyengine_harness_slices is not None:
        resolved_harness_slices = tuple(policyengine_harness_slices)
    elif result.config.policyengine_targets_db is not None:
        resolved_harness_slices = default_policyengine_us_db_all_target_slices(
            period=harness_period,
            reform_id=result.config.policyengine_target_reform_id,
        )
    else:
        resolved_harness_slices = default_policyengine_us_harness_slices(
            period=harness_period
        )
    if resolved_target_provider is not None and resolved_harness_slices:
        resolved_harness_slices = filter_nonempty_policyengine_us_harness_slices(
            resolved_target_provider,
            resolved_harness_slices,
            cache=policyengine_comparison_cache,
        )

    resolved_harness_metadata = {
        "baseline_dataset": (
            Path(resolved_baseline_dataset).name
            if resolved_baseline_dataset is not None
            else None
        ),
        "targets_db": (
            Path(result.config.policyengine_targets_db).name
            if result.config.policyengine_targets_db is not None
            else None
        ),
        "target_period": result.config.policyengine_target_period,
        "target_variables": list(result.config.policyengine_target_variables),
        "target_domains": list(result.config.policyengine_target_domains),
        "target_geo_levels": list(result.config.policyengine_target_geo_levels),
        "target_profile": result.config.policyengine_target_profile,
        "calibration_target_profile": (
            result.config.policyengine_calibration_target_profile
        ),
        "target_reform_id": result.config.policyengine_target_reform_id,
        "harness_slice_names": [slice_spec.name for slice_spec in resolved_harness_slices],
        "policyengine_us_runtime_version": _resolve_policyengine_us_runtime_version(),
        "harness_suite": (
            "policyengine_us_all_targets"
            if result.config.policyengine_targets_db is not None
            and policyengine_harness_slices is None
            else None
        ),
        **dict(policyengine_harness_metadata or {}),
    }
    return (
        resolved_target_provider,
        resolved_baseline_dataset,
        resolved_harness_slices,
        resolved_harness_metadata,
    )


def _resolve_policyengine_us_runtime_version() -> str | None:
    try:
        return version("policyengine-us")
    except PackageNotFoundError:
        return None


def _allocate_versioned_output_dir(
    output_root: Path,
    *,
    version_id: str | None,
    result: USMicroplexBuildResult,
) -> tuple[str, Path]:
    if version_id is not None:
        output_dir = output_root / version_id
        if output_dir.exists():
            raise FileExistsError(f"Versioned artifact directory already exists: {output_dir}")
        return version_id, output_dir

    config_hash = _short_config_hash(result.config.to_dict())
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base_version_id = f"{timestamp}-{config_hash}"
    candidate_version_id = base_version_id
    suffix = 2
    output_dir = output_root / candidate_version_id
    while output_dir.exists():
        candidate_version_id = f"{base_version_id}-{suffix}"
        output_dir = output_root / candidate_version_id
        suffix += 1
    return candidate_version_id, output_dir


def _short_config_hash(config: dict[str, Any]) -> str:
    import hashlib
    import json

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _registry_metric_value(entry: Any | None, metric: FrontierMetric) -> float | None:
    if entry is None:
        return None
    return getattr(entry, metric, None)
