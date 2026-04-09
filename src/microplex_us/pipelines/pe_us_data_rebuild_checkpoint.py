"""Concrete checkpoint runner for the PE-US-data rebuild profile."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
from microplex.core import SourceQuery
from microplex.targets import assert_valid_benchmark_artifact_manifest

from microplex_us.pipelines.artifacts import (
    USMicroplexArtifactPaths,
    USMicroplexVersionedBuildArtifacts,
    build_and_save_versioned_us_microplex_from_source_providers,
)
from microplex_us.pipelines.index_db import append_us_microplex_run_index_entry
from microplex_us.pipelines.pe_us_data_rebuild import (
    PEUSDataRebuildProgram,
    default_policyengine_us_data_rebuild_config,
    default_policyengine_us_data_rebuild_program,
    default_policyengine_us_data_rebuild_source_providers,
)
from microplex_us.pipelines.pe_us_data_rebuild_parity import (
    build_policyengine_us_data_rebuild_parity_artifact,
    write_policyengine_us_data_rebuild_parity_artifact,
)
from microplex_us.pipelines.registry import (
    append_us_microplex_run_registry_entry,
    build_us_microplex_run_registry_entry,
    load_us_microplex_run_registry,
    select_us_microplex_frontier_entry,
)

if TYPE_CHECKING:
    from microplex.core import SourceProvider
    from microplex.targets import TargetProvider

    from microplex_us.pipelines.registry import FrontierMetric
    from microplex_us.pipelines.us import USMicroplexBuildConfig
    from microplex_us.policyengine.harness import (
        PolicyEngineUSComparisonCache,
        PolicyEngineUSHarnessSlice,
    )


@dataclass(frozen=True)
class PEUSDataRebuildCheckpointResult:
    """Saved artifact bundle plus parity sidecar for one rebuild checkpoint."""

    build_config: USMicroplexBuildConfig
    provider_names: tuple[str, ...]
    queries: dict[str, SourceQuery]
    artifacts: USMicroplexVersionedBuildArtifacts
    parity_path: Path
    parity_payload: dict[str, Any]


@dataclass(frozen=True)
class PEUSDataRebuildCheckpointEvidenceResult:
    """Comparison evidence attached to one saved rebuild artifact."""

    artifact_dir: Path
    manifest_path: Path
    harness_path: Path | None
    native_scores_path: Path | None
    parity_path: Path
    parity_payload: dict[str, Any]


def _normalize_path_value(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser())


def _validate_checkpoint_config_context(
    config: USMicroplexBuildConfig,
    *,
    policyengine_baseline_dataset: str | Path,
    policyengine_targets_db: str | Path,
    target_period: int,
    target_profile: str,
    calibration_target_profile: str | None,
    target_variables: tuple[str, ...],
    target_domains: tuple[str, ...],
    target_geo_levels: tuple[str, ...],
    calibration_target_variables: tuple[str, ...],
    calibration_target_domains: tuple[str, ...],
    calibration_target_geo_levels: tuple[str, ...],
) -> None:
    expected_pairs = {
        "policyengine_baseline_dataset": _normalize_path_value(
            policyengine_baseline_dataset
        ),
        "policyengine_targets_db": _normalize_path_value(policyengine_targets_db),
        "policyengine_dataset_year": int(target_period),
        "policyengine_target_period": int(target_period),
        "policyengine_target_profile": target_profile,
        "policyengine_calibration_target_profile": (
            calibration_target_profile or target_profile
        ),
        "policyengine_target_variables": tuple(target_variables),
        "policyengine_target_domains": tuple(target_domains),
        "policyengine_target_geo_levels": tuple(target_geo_levels),
        "policyengine_calibration_target_variables": tuple(
            calibration_target_variables
        ),
        "policyengine_calibration_target_domains": tuple(
            calibration_target_domains
        ),
        "policyengine_calibration_target_geo_levels": tuple(
            calibration_target_geo_levels
        ),
    }
    for key, expected in expected_pairs.items():
        observed = getattr(config, key)
        if observed != expected:
            raise ValueError(
                "Explicit config does not match the requested PE rebuild context for "
                f"{key}: expected {expected!r}, observed {observed!r}"
            )


def _validate_query_keys(
    provider_names: tuple[str, ...],
    queries: dict[str, SourceQuery],
) -> None:
    unexpected = sorted(set(queries) - set(provider_names))
    if unexpected:
        allowed = ", ".join(provider_names)
        unexpected_text = ", ".join(unexpected)
        raise ValueError(
            "Checkpoint queries include unknown provider keys: "
            f"{unexpected_text}. Expected one of: {allowed}"
        )


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp_path.replace(path)


def _resolve_policyengine_us_runtime_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("policyengine-us")
    except PackageNotFoundError:
        return None


def _registry_metric_value(entry: Any | None, metric: FrontierMetric) -> float | None:
    if entry is None:
        return None
    return getattr(entry, metric, None)


def _resolve_saved_artifact_path(
    artifact_root: Path,
    relative_or_absolute: str | Path | None,
) -> Path | None:
    if relative_or_absolute is None:
        return None
    candidate = Path(relative_or_absolute)
    if not candidate.is_absolute():
        artifact_relative = artifact_root / candidate
        if artifact_relative.exists():
            return artifact_relative
        cwd_relative = candidate.resolve()
        if cwd_relative.exists():
            return cwd_relative
        candidate = artifact_relative
    return candidate


def _infer_policyengine_baseline_household_weight_sum(
    baseline_dataset: str | Path,
    *,
    target_period: int,
) -> float | None:
    """Best-effort household-weight target inferred from the PE baseline dataset."""

    dataset_path = Path(baseline_dataset).expanduser()
    if not dataset_path.exists():
        return None
    try:
        with h5py.File(dataset_path, "r") as handle:
            weights = handle.get("household_weight")
            if weights is None:
                return None
            period_key = str(int(target_period))
            if period_key not in weights:
                return None
            weight_sum = float(weights[period_key][...].sum())
    except (FileNotFoundError, OSError, ValueError):
        return None
    return weight_sum if weight_sum > 0.0 else None


def _build_checkpoint_benchmark_stage(manifest: dict[str, Any]) -> dict[str, Any]:
    artifacts = dict(manifest.get("artifacts", {}))
    harness_summary = dict(manifest.get("policyengine_harness", {}))
    native_scores_summary = dict(manifest.get("policyengine_native_scores", {}))
    return {
        "id": "benchmark",
        "step": "06",
        "title": "PolicyEngine benchmark",
        "summary": "Harness and native-loss diagnostics stay attached to the same artifact bundle.",
        "status": "ready" if harness_summary or native_scores_summary else "missing",
        "metrics": [
            {
                "label": "Harness delta",
                "value": harness_summary.get("mean_abs_relative_error_delta"),
            },
            {
                "label": "Native delta",
                "value": native_scores_summary.get("enhanced_cps_native_loss_delta"),
            },
            {
                "label": "Win rate",
                "value": harness_summary.get("target_win_rate"),
            },
        ],
        "outputs": [
            value
            for value in (
                artifacts.get("policyengine_harness"),
                artifacts.get("policyengine_native_scores"),
            )
            if value
        ],
    }


def _refresh_checkpoint_data_flow_snapshot(
    artifact_root: Path,
    manifest: dict[str, Any],
) -> Path | None:
    snapshot_path = artifact_root / "data_flow_snapshot.json"
    if not snapshot_path.exists():
        return None
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get("schemaVersion") != 1:
        return snapshot_path
    stages = list(snapshot.get("stages", []))
    benchmark_stage = _build_checkpoint_benchmark_stage(manifest)
    replaced = False
    for index, stage in enumerate(stages):
        if isinstance(stage, dict) and stage.get("id") == "benchmark":
            stages[index] = benchmark_stage
            replaced = True
            break
    if not replaced:
        stages.append(benchmark_stage)
    snapshot["stages"] = stages
    _write_json_atomically(snapshot_path, snapshot)
    return snapshot_path


def _attach_checkpoint_registry_and_index(
    artifact_root: Path,
    manifest: dict[str, Any],
    *,
    harness_path: Path | None,
    harness_payload: dict[str, Any] | None,
    run_registry_path: str | Path | None,
    run_index_path: str | Path | None,
    run_registry_metadata: dict[str, Any] | None,
) -> tuple[Path | None, Path | None]:
    if (
        "policyengine_harness" not in manifest
        and "policyengine_native_scores" not in manifest
    ):
        return None, None

    resolved_harness_payload = (
        dict(harness_payload)
        if harness_payload is not None
        else (
            json.loads(harness_path.read_text())
            if harness_path is not None and harness_path.exists()
            else None
        )
    )
    resolved_run_registry_path = Path(
        run_registry_path or artifact_root.parent / "run_registry.jsonl"
    )
    existing_entry = next(
        (
            entry
            for entry in reversed(load_us_microplex_run_registry(resolved_run_registry_path))
            if entry.artifact_id == artifact_root.name
        ),
        None,
    )
    if existing_entry is None:
        run_entry = build_us_microplex_run_registry_entry(
            artifact_dir=artifact_root,
            manifest_path=artifact_root / "manifest.json",
            manifest=manifest,
            policyengine_harness_path=harness_path,
            policyengine_harness_payload=resolved_harness_payload,
            metadata=dict(run_registry_metadata or {}),
        )
        recorded_entry = append_us_microplex_run_registry_entry(
            resolved_run_registry_path,
            run_entry,
        )
    else:
        recorded_entry = existing_entry
    resolved_run_index_path = append_us_microplex_run_index_entry(
        run_index_path or artifact_root.parent,
        recorded_entry,
        policyengine_harness_payload=resolved_harness_payload,
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
            if "policyengine_native_scores" in manifest
            else "candidate_composite_parity_loss"
        ),
    }
    manifest["run_index"] = {
        "path": str(resolved_run_index_path),
        "artifact_id": recorded_entry.artifact_id,
    }
    return resolved_run_registry_path, resolved_run_index_path


def _load_checkpoint_versioned_artifacts(
    *,
    build_result: Any,
    artifact_root: Path,
    frontier_metric: FrontierMetric,
) -> USMicroplexVersionedBuildArtifacts:
    manifest_path = artifact_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = dict(manifest.get("artifacts", {}))
    artifact_paths = USMicroplexArtifactPaths(
        output_dir=artifact_root,
        version_id=artifact_root.name,
        seed_data=artifact_root / str(artifacts["seed_data"]),
        synthetic_data=artifact_root / str(artifacts["synthetic_data"]),
        calibrated_data=artifact_root / str(artifacts["calibrated_data"]),
        targets=artifact_root / str(artifacts["targets"]),
        manifest=manifest_path,
        synthesizer=_resolve_saved_artifact_path(
            artifact_root,
            artifacts.get("synthesizer"),
        ),
        policyengine_dataset=_resolve_saved_artifact_path(
            artifact_root,
            artifacts.get("policyengine_dataset"),
        ),
        data_flow_snapshot=(
            artifact_root / "data_flow_snapshot.json"
            if (artifact_root / "data_flow_snapshot.json").exists()
            else None
        ),
        policyengine_harness=_resolve_saved_artifact_path(
            artifact_root,
            artifacts.get("policyengine_harness"),
        ),
        policyengine_native_scores=_resolve_saved_artifact_path(
            artifact_root,
            artifacts.get("policyengine_native_scores"),
        ),
        run_registry=_resolve_saved_artifact_path(
            artifact_root,
            dict(manifest.get("run_registry", {})).get("path"),
        ),
        run_index_db=_resolve_saved_artifact_path(
            artifact_root,
            dict(manifest.get("run_index", {})).get("path"),
        ),
    )
    current_entry = None
    frontier_entry = None
    frontier_delta = None
    if artifact_paths.run_registry is not None:
        registry_entries = load_us_microplex_run_registry(artifact_paths.run_registry)
        current_entry = next(
            (
                entry
                for entry in reversed(registry_entries)
                if entry.artifact_id == artifact_root.name
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


def _build_checkpoint_harness_context(
    *,
    manifest: dict[str, Any],
    policyengine_target_provider: TargetProvider | None,
    policyengine_baseline_dataset: str | Path | None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...]
        | list[PolicyEngineUSHarnessSlice]
        | None
    ),
    policyengine_harness_metadata: dict[str, Any] | None,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None,
) -> tuple[
    TargetProvider | None,
    str | Path | None,
    tuple[PolicyEngineUSHarnessSlice, ...],
    dict[str, Any],
]:
    from microplex_us.policyengine.harness import (
        default_policyengine_us_db_all_target_slices,
        default_policyengine_us_harness_slices,
        filter_nonempty_policyengine_us_harness_slices,
    )
    from microplex_us.policyengine.us import PolicyEngineUSDBTargetProvider

    config = dict(manifest.get("config", {}))
    resolved_target_provider = policyengine_target_provider
    if resolved_target_provider is None and config.get("policyengine_targets_db") is not None:
        resolved_target_provider = PolicyEngineUSDBTargetProvider(
            config["policyengine_targets_db"]
        )
    resolved_baseline_dataset = (
        policyengine_baseline_dataset or config.get("policyengine_baseline_dataset")
    )
    harness_period = (
        config.get("policyengine_dataset_year")
        or config.get("policyengine_target_period")
        or 2024
    )
    if policyengine_harness_slices is not None:
        resolved_harness_slices = tuple(policyengine_harness_slices)
    elif config.get("policyengine_targets_db") is not None:
        resolved_harness_slices = default_policyengine_us_db_all_target_slices(
            period=int(harness_period),
            reform_id=int(config.get("policyengine_target_reform_id", 0) or 0),
        )
    else:
        resolved_harness_slices = default_policyengine_us_harness_slices(
            period=int(harness_period)
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
            Path(config["policyengine_targets_db"]).name
            if config.get("policyengine_targets_db") is not None
            else None
        ),
        "target_period": config.get("policyengine_target_period"),
        "target_variables": list(config.get("policyengine_target_variables", ())),
        "target_domains": list(config.get("policyengine_target_domains", ())),
        "target_geo_levels": list(config.get("policyengine_target_geo_levels", ())),
        "target_profile": config.get("policyengine_target_profile"),
        "calibration_target_profile": config.get(
            "policyengine_calibration_target_profile"
        ),
        "target_reform_id": config.get("policyengine_target_reform_id"),
        "harness_slice_names": [slice_spec.name for slice_spec in resolved_harness_slices],
        "policyengine_us_runtime_version": _resolve_policyengine_us_runtime_version(),
        "harness_suite": (
            "policyengine_us_all_targets"
            if config.get("policyengine_targets_db") is not None
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


def attach_policyengine_us_data_rebuild_checkpoint_evidence(
    artifact_dir: str | Path,
    *,
    program: PEUSDataRebuildProgram | None = None,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...]
        | list[PolicyEngineUSHarnessSlice]
        | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
    compute_harness: bool = True,
    compute_native_scores: bool = True,
    require_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
) -> PEUSDataRebuildCheckpointEvidenceResult:
    """Attach PE comparison evidence to an already-saved rebuild artifact."""

    from microplex_us.pipelines.pe_native_scores import compute_us_pe_native_scores
    from microplex_us.policyengine.harness import evaluate_policyengine_us_harness
    from microplex_us.policyengine.us import load_policyengine_us_entity_tables

    artifact_root = Path(artifact_dir)
    manifest_path = artifact_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = dict(manifest.get("config", {}))
    artifacts = dict(manifest.get("artifacts", {}))
    dataset_name = artifacts.get("policyengine_dataset")
    dataset_path = artifact_root / dataset_name if isinstance(dataset_name, str) else None
    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError(
            "Saved rebuild artifact is missing policyengine_dataset output"
        )

    harness_path: Path | None = None
    harness_payload = (
        dict(precomputed_policyengine_harness_payload)
        if precomputed_policyengine_harness_payload is not None
        else None
    )
    if harness_payload is None and compute_harness:
        (
            resolved_target_provider,
            resolved_baseline_dataset,
            resolved_harness_slices,
            resolved_harness_metadata,
        ) = _build_checkpoint_harness_context(
            manifest=manifest,
            policyengine_target_provider=policyengine_target_provider,
            policyengine_baseline_dataset=policyengine_baseline_dataset,
            policyengine_harness_slices=policyengine_harness_slices,
            policyengine_harness_metadata=policyengine_harness_metadata,
            policyengine_comparison_cache=policyengine_comparison_cache,
        )
        if resolved_target_provider is None:
            raise ValueError(
                "Cannot compute rebuild checkpoint harness without a target provider"
            )
        if resolved_baseline_dataset is None:
            raise ValueError(
                "Cannot compute rebuild checkpoint harness without a baseline dataset"
            )
        if not resolved_harness_slices:
            raise ValueError(
                "Cannot compute rebuild checkpoint harness because no nonempty slices resolved"
            )
        candidate_tables = load_policyengine_us_entity_tables(
            dataset_path,
            period=(
                config.get("policyengine_dataset_year")
                or config.get("policyengine_target_period")
                or 2024
            ),
        )
        harness_run = evaluate_policyengine_us_harness(
            candidate_tables,
            resolved_target_provider,
            resolved_harness_slices,
            baseline_dataset=str(resolved_baseline_dataset),
            dataset_year=config.get("policyengine_dataset_year"),
            simulation_cls=None,
            candidate_label="microplex",
            baseline_label="policyengine_us_data",
            metadata=resolved_harness_metadata,
            cache=policyengine_comparison_cache,
        )
        harness_payload = harness_run.to_dict()
    if harness_payload is not None:
        harness_path = artifact_root / "policyengine_harness.json"
        _write_json_atomically(harness_path, harness_payload)
        artifacts["policyengine_harness"] = harness_path.name
        manifest["policyengine_harness"] = dict(harness_payload.get("summary", {}))

    native_scores_path: Path | None = None
    native_scores_payload = (
        dict(precomputed_policyengine_native_scores)
        if precomputed_policyengine_native_scores is not None
        else None
    )
    if native_scores_payload is None and compute_native_scores:
        resolved_baseline_dataset = (
            policyengine_baseline_dataset or config.get("policyengine_baseline_dataset")
        )
        if resolved_baseline_dataset is None:
            raise ValueError(
                "Cannot compute PE-native scores without a baseline dataset"
            )
        native_scores_payload = compute_us_pe_native_scores(
            candidate_dataset_path=dataset_path,
            baseline_dataset_path=resolved_baseline_dataset,
            period=(
                config.get("policyengine_dataset_year")
                or config.get("policyengine_target_period")
                or 2024
            ),
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
        )
    if native_scores_payload is not None:
        native_scores_path = artifact_root / "policyengine_native_scores.json"
        _write_json_atomically(native_scores_path, native_scores_payload)
        artifacts["policyengine_native_scores"] = native_scores_path.name
        manifest["policyengine_native_scores"] = dict(
            native_scores_payload.get("summary", {})
        )
    elif require_policyengine_native_score:
        raise ValueError(
            "require_policyengine_native_score=True but no PE-native scores were computed"
        )

    manifest["artifacts"] = artifacts
    _attach_checkpoint_registry_and_index(
        artifact_root,
        manifest,
        harness_path=harness_path,
        harness_payload=harness_payload,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=run_registry_metadata,
    )
    assert_valid_benchmark_artifact_manifest(
        manifest,
        artifact_dir=artifact_root,
        manifest_path=manifest_path,
        summary_section=(
            "policyengine_harness" if "policyengine_harness" in manifest else None
        ),
        required_artifact_keys=(
            "seed_data",
            "synthetic_data",
            "calibrated_data",
            "targets",
            *(
                ("policyengine_harness",)
                if artifacts.get("policyengine_harness") is not None
                else ()
            ),
            *(
                ("policyengine_native_scores",)
                if artifacts.get("policyengine_native_scores") is not None
                else ()
            ),
        ),
        required_summary_keys=(
            (
                "candidate_mean_abs_relative_error",
                "baseline_mean_abs_relative_error",
                "mean_abs_relative_error_delta",
            )
            if "policyengine_harness" in manifest
            else ()
        ),
    )
    _refresh_checkpoint_data_flow_snapshot(artifact_root, manifest)
    _write_json_atomically(manifest_path, manifest)

    resolved_program = program or default_policyengine_us_data_rebuild_program()
    parity_path = write_policyengine_us_data_rebuild_parity_artifact(
        artifact_root,
        program=resolved_program,
    )
    parity_payload = build_policyengine_us_data_rebuild_parity_artifact(
        artifact_root,
        program=resolved_program,
    )
    return PEUSDataRebuildCheckpointEvidenceResult(
        artifact_dir=artifact_root,
        manifest_path=manifest_path,
        harness_path=harness_path,
        native_scores_path=native_scores_path,
        parity_path=parity_path,
        parity_payload=parity_payload,
    )


def default_policyengine_us_data_rebuild_checkpoint_config(
    *,
    policyengine_baseline_dataset: str | Path,
    policyengine_targets_db: str | Path,
    target_period: int = 2024,
    target_profile: str = "pe_native_broad",
    calibration_target_profile: str | None = None,
    target_variables: tuple[str, ...] = (),
    target_domains: tuple[str, ...] = (),
    target_geo_levels: tuple[str, ...] = (),
    calibration_target_variables: tuple[str, ...] = (),
    calibration_target_domains: tuple[str, ...] = (),
    calibration_target_geo_levels: tuple[str, ...] = (),
    **overrides: Any,
) -> USMicroplexBuildConfig:
    """Return the canonical rebuild config with required PE comparison context."""

    resolved_target_period = int(target_period)
    resolved_baseline_weight_sum = _infer_policyengine_baseline_household_weight_sum(
        policyengine_baseline_dataset,
        target_period=resolved_target_period,
    )
    resolved_overrides = dict(overrides)
    infer_total_weight_targets = (
        resolved_baseline_weight_sum is not None
        and resolved_overrides.get("calibration_backend") != "none"
    )
    if infer_total_weight_targets:
        resolved_overrides.setdefault(
            "policyengine_selection_target_total_weight",
            resolved_baseline_weight_sum,
        )
        if not resolved_overrides.get(
            "policyengine_calibration_rescale_to_input_weight_sum",
            False,
        ):
            resolved_overrides.setdefault(
                "policyengine_calibration_target_total_weight",
                resolved_baseline_weight_sum,
            )
            resolved_overrides.setdefault(
                "policyengine_calibration_rescale_to_target_total_weight",
                True,
            )
    return default_policyengine_us_data_rebuild_config(
        policyengine_baseline_dataset=str(policyengine_baseline_dataset),
        policyengine_targets_db=str(policyengine_targets_db),
        policyengine_dataset_year=resolved_target_period,
        policyengine_target_period=resolved_target_period,
        policyengine_target_profile=target_profile,
        policyengine_calibration_target_profile=(
            calibration_target_profile or target_profile
        ),
        policyengine_target_variables=tuple(target_variables),
        policyengine_target_domains=tuple(target_domains),
        policyengine_target_geo_levels=tuple(target_geo_levels),
        policyengine_calibration_target_variables=tuple(calibration_target_variables),
        policyengine_calibration_target_domains=tuple(calibration_target_domains),
        policyengine_calibration_target_geo_levels=tuple(
            calibration_target_geo_levels
        ),
        **resolved_overrides,
    )


def default_policyengine_us_data_rebuild_queries(
    providers: tuple[SourceProvider, ...] | list[SourceProvider],
    *,
    cps_sample_n: int | None = None,
    puf_sample_n: int | None = None,
    donor_sample_n: int | None = None,
    random_seed: int = 0,
) -> dict[str, SourceQuery]:
    """Return default provider queries for a rebuild checkpoint smoke run."""

    from microplex_us.data_sources.cps import CPSASECSourceProvider
    from microplex_us.data_sources.donor_surveys import DonorSurveySourceProvider
    from microplex_us.data_sources.puf import PUFSourceProvider

    resolved_donor_sample_n = donor_sample_n
    if resolved_donor_sample_n is None:
        source_sample_sizes = tuple(
            int(sample_n)
            for sample_n in (cps_sample_n, puf_sample_n)
            if sample_n is not None
        )
        if source_sample_sizes:
            resolved_donor_sample_n = max(source_sample_sizes)

    queries: dict[str, SourceQuery] = {}
    for provider in providers:
        sample_n: int | None = None
        if isinstance(provider, CPSASECSourceProvider):
            sample_n = cps_sample_n
        elif isinstance(provider, PUFSourceProvider):
            sample_n = puf_sample_n
        elif isinstance(provider, DonorSurveySourceProvider):
            sample_n = resolved_donor_sample_n
        if sample_n is None:
            continue
        queries[provider.descriptor.name] = SourceQuery(
            provider_filters={
                "sample_n": int(sample_n),
                "random_seed": int(random_seed),
            }
        )
    return queries


def run_policyengine_us_data_rebuild_checkpoint(
    output_root: str | Path,
    *,
    policyengine_baseline_dataset: str | Path,
    policyengine_targets_db: str | Path,
    target_period: int = 2024,
    target_profile: str = "pe_native_broad",
    calibration_target_profile: str | None = None,
    target_variables: tuple[str, ...] = (),
    target_domains: tuple[str, ...] = (),
    target_geo_levels: tuple[str, ...] = (),
    calibration_target_variables: tuple[str, ...] = (),
    calibration_target_domains: tuple[str, ...] = (),
    calibration_target_geo_levels: tuple[str, ...] = (),
    config: USMicroplexBuildConfig | None = None,
    config_overrides: dict[str, Any] | None = None,
    providers: tuple[SourceProvider, ...] | list[SourceProvider] | None = None,
    queries: dict[str, SourceQuery] | None = None,
    cps_source_year: int = 2023,
    cps_cache_dir: str | Path | None = None,
    cps_download: bool = True,
    puf_target_year: int | None = None,
    puf_cps_reference_year: int | None = None,
    puf_cache_dir: str | Path | None = None,
    puf_path: str | Path | None = None,
    puf_demographics_path: str | Path | None = None,
    puf_expand_persons: bool = True,
    include_donor_surveys: bool = True,
    acs_year: int = 2022,
    sipp_year: int = 2023,
    scf_year: int = 2022,
    donor_cache_dir: str | Path | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
    cps_sample_n: int | None = None,
    puf_sample_n: int | None = None,
    donor_sample_n: int | None = None,
    query_random_seed: int = 0,
    version_id: str | None = None,
    frontier_metric: FrontierMetric = "enhanced_cps_native_loss_delta",
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_target_provider: TargetProvider | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...]
        | list[PolicyEngineUSHarnessSlice]
        | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    defer_policyengine_harness: bool = False,
    require_policyengine_native_score: bool = False,
    defer_policyengine_native_score: bool = False,
    precomputed_policyengine_harness_payload: dict[str, Any] | None = None,
    precomputed_policyengine_native_scores: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    run_index_path: str | Path | None = None,
    run_registry_metadata: dict[str, Any] | None = None,
) -> PEUSDataRebuildCheckpointResult:
    """Run one saved rebuild checkpoint and write its parity sidecar."""

    if config is not None and config_overrides:
        raise ValueError(
            "config_overrides cannot be used when an explicit config is supplied"
        )
    resolved_config = config or default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_targets_db=policyengine_targets_db,
        target_period=target_period,
        target_profile=target_profile,
        calibration_target_profile=calibration_target_profile,
        target_variables=target_variables,
        target_domains=target_domains,
        target_geo_levels=target_geo_levels,
        calibration_target_variables=calibration_target_variables,
        calibration_target_domains=calibration_target_domains,
        calibration_target_geo_levels=calibration_target_geo_levels,
        **dict(config_overrides or {}),
    )
    if config is not None:
        _validate_checkpoint_config_context(
            resolved_config,
            policyengine_baseline_dataset=policyengine_baseline_dataset,
            policyengine_targets_db=policyengine_targets_db,
            target_period=target_period,
            target_profile=target_profile,
            calibration_target_profile=calibration_target_profile,
            target_variables=target_variables,
            target_domains=target_domains,
            target_geo_levels=target_geo_levels,
            calibration_target_variables=calibration_target_variables,
            calibration_target_domains=calibration_target_domains,
            calibration_target_geo_levels=calibration_target_geo_levels,
        )
    if providers is None:
        resolved_providers = tuple(
            default_policyengine_us_data_rebuild_source_providers(
            cps_source_year=cps_source_year,
            cps_cache_dir=cps_cache_dir,
            cps_download=cps_download,
            puf_target_year=(
                int(puf_target_year)
                if puf_target_year is not None
                else int(target_period)
            ),
            puf_cps_reference_year=puf_cps_reference_year,
            puf_cache_dir=puf_cache_dir,
            puf_path=puf_path,
            puf_demographics_path=puf_demographics_path,
            puf_expand_persons=puf_expand_persons,
            include_donor_surveys=include_donor_surveys,
            acs_year=acs_year,
            sipp_year=sipp_year,
            scf_year=scf_year,
            donor_cache_dir=donor_cache_dir,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
        )
        )
    else:
        resolved_providers = tuple(providers)
        if not resolved_providers:
            raise ValueError(
                "providers must be None or a non-empty provider sequence for a rebuild checkpoint"
            )
    resolved_queries = (
        dict(queries)
        if queries is not None
        else default_policyengine_us_data_rebuild_queries(
            resolved_providers,
            cps_sample_n=cps_sample_n,
            puf_sample_n=puf_sample_n,
            donor_sample_n=donor_sample_n,
            random_seed=query_random_seed,
        )
    )
    program = default_policyengine_us_data_rebuild_program()
    provider_names = tuple(provider.descriptor.name for provider in resolved_providers)
    _validate_query_keys(provider_names, resolved_queries)
    if (
        policyengine_us_data_python is not None
        and not defer_policyengine_native_score
        and precomputed_policyengine_native_scores is None
    ):
        raise ValueError(
            "policyengine_us_data_python requires defer_policyengine_native_score=True "
            "or precomputed_policyengine_native_scores because the automatic native-score "
            "save path cannot yet honor a custom PE-US-data interpreter"
        )
    resolved_harness_metadata = {
        "rebuild_checkpoint": True,
        "rebuild_program_id": program.program_id,
        "rebuild_provider_names": list(provider_names),
        **dict(policyengine_harness_metadata or {}),
    }
    resolved_registry_metadata = {
        "rebuild_checkpoint": True,
        "rebuild_program_id": program.program_id,
        "rebuild_provider_names": list(provider_names),
        "rebuild_profile_expected": True,
        **dict(run_registry_metadata or {}),
    }

    artifacts = build_and_save_versioned_us_microplex_from_source_providers(
        providers=list(resolved_providers),
        output_root=output_root,
        config=resolved_config,
        queries=resolved_queries or None,
        version_id=version_id,
        frontier_metric=frontier_metric,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=resolved_config.policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=resolved_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        defer_policyengine_harness=True,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=True,
        precomputed_policyengine_harness_payload=None,
        precomputed_policyengine_native_scores=None,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=resolved_registry_metadata,
    )
    evidence = attach_policyengine_us_data_rebuild_checkpoint_evidence(
        artifacts.artifact_paths.output_dir,
        program=program,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=resolved_config.policyengine_baseline_dataset,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=resolved_harness_metadata,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
        compute_harness=not defer_policyengine_harness,
        compute_native_scores=not defer_policyengine_native_score,
        require_policyengine_native_score=require_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=resolved_registry_metadata,
    )
    refreshed_artifacts = _load_checkpoint_versioned_artifacts(
        build_result=artifacts.build_result,
        artifact_root=artifacts.artifact_paths.output_dir,
        frontier_metric=frontier_metric,
    )
    return PEUSDataRebuildCheckpointResult(
        build_config=resolved_config,
        provider_names=provider_names,
        queries=resolved_queries,
        artifacts=refreshed_artifacts,
        parity_path=evidence.parity_path,
        parity_payload=evidence.parity_payload,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for one PE-US-data rebuild checkpoint."""

    parser = argparse.ArgumentParser(
        description="Run a versioned PE-US-data rebuild checkpoint in microplex-us."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-dataset", required=True)
    parser.add_argument("--targets-db", required=True)
    parser.add_argument("--policyengine-us-data-repo")
    parser.add_argument("--policyengine-us-data-python")
    parser.add_argument("--version-id")
    parser.add_argument("--target-period", type=int, default=2024)
    parser.add_argument("--target-profile", default="pe_native_broad")
    parser.add_argument("--calibration-target-profile")
    parser.add_argument("--n-synthetic", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cps-source-year", type=int, default=2023)
    parser.add_argument("--puf-target-year", type=int)
    parser.add_argument("--puf-cps-reference-year", type=int)
    parser.add_argument("--acs-year", type=int, default=2022)
    parser.add_argument("--sipp-year", type=int, default=2023)
    parser.add_argument("--scf-year", type=int, default=2022)
    parser.add_argument("--cps-cache-dir")
    parser.add_argument("--puf-cache-dir")
    parser.add_argument("--donor-cache-dir")
    parser.add_argument("--puf-path")
    parser.add_argument("--puf-demographics-path")
    parser.add_argument("--cps-sample-n", type=int)
    parser.add_argument("--puf-sample-n", type=int)
    parser.add_argument("--donor-sample-n", type=int)
    parser.add_argument("--query-random-seed", type=int, default=0)
    parser.add_argument("--target-variable", action="append", default=[])
    parser.add_argument("--target-domain", action="append", default=[])
    parser.add_argument("--target-geo-level", action="append", default=[])
    parser.add_argument("--calibration-target-variable", action="append", default=[])
    parser.add_argument("--calibration-target-domain", action="append", default=[])
    parser.add_argument("--calibration-target-geo-level", action="append", default=[])
    parser.add_argument(
        "--include-donor-surveys",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-cps-download", action="store_true")
    parser.add_argument("--no-puf-expand-persons", action="store_true")
    parser.add_argument("--defer-policyengine-harness", action="store_true")
    parser.add_argument("--defer-policyengine-native-score", action="store_true")
    parser.add_argument("--require-policyengine-native-score", action="store_true")
    args = parser.parse_args(argv)

    result = run_policyengine_us_data_rebuild_checkpoint(
        output_root=args.output_root,
        policyengine_baseline_dataset=args.baseline_dataset,
        policyengine_targets_db=args.targets_db,
        target_period=args.target_period,
        target_profile=args.target_profile,
        calibration_target_profile=args.calibration_target_profile,
        target_variables=tuple(args.target_variable),
        target_domains=tuple(args.target_domain),
        target_geo_levels=tuple(args.target_geo_level),
        calibration_target_variables=tuple(args.calibration_target_variable),
        calibration_target_domains=tuple(args.calibration_target_domain),
        calibration_target_geo_levels=tuple(args.calibration_target_geo_level),
        config_overrides={
            "n_synthetic": int(args.n_synthetic),
            "random_seed": int(args.random_seed),
        },
        cps_source_year=args.cps_source_year,
        cps_cache_dir=args.cps_cache_dir,
        cps_download=not args.no_cps_download,
        puf_target_year=args.puf_target_year,
        puf_cps_reference_year=args.puf_cps_reference_year,
        puf_cache_dir=args.puf_cache_dir,
        puf_path=args.puf_path,
        puf_demographics_path=args.puf_demographics_path,
        puf_expand_persons=not args.no_puf_expand_persons,
        include_donor_surveys=args.include_donor_surveys,
        acs_year=args.acs_year,
        sipp_year=args.sipp_year,
        scf_year=args.scf_year,
        donor_cache_dir=args.donor_cache_dir,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        policyengine_us_data_python=args.policyengine_us_data_python,
        cps_sample_n=args.cps_sample_n,
        puf_sample_n=args.puf_sample_n,
        donor_sample_n=args.donor_sample_n,
        query_random_seed=args.query_random_seed,
        version_id=args.version_id,
        defer_policyengine_harness=args.defer_policyengine_harness,
        require_policyengine_native_score=args.require_policyengine_native_score,
        defer_policyengine_native_score=args.defer_policyengine_native_score,
    )

    print(result.artifacts.artifact_paths.output_dir)
    print(result.parity_path)
    print(json.dumps(result.parity_payload["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
