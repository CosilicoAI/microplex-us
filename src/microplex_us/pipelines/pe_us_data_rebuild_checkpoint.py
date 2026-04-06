"""Concrete checkpoint runner for the PE-US-data rebuild profile."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from microplex.core import SourceQuery

from microplex_us.pipelines.artifacts import (
    USMicroplexVersionedBuildArtifacts,
    build_and_save_versioned_us_microplex_from_source_providers,
)
from microplex_us.pipelines.pe_us_data_rebuild import (
    default_policyengine_us_data_rebuild_config,
    default_policyengine_us_data_rebuild_program,
    default_policyengine_us_data_rebuild_source_providers,
)
from microplex_us.pipelines.pe_us_data_rebuild_parity import (
    build_policyengine_us_data_rebuild_parity_artifact,
    write_policyengine_us_data_rebuild_parity_artifact,
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
        **overrides,
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

    queries: dict[str, SourceQuery] = {}
    for provider in providers:
        sample_n: int | None = None
        if isinstance(provider, CPSASECSourceProvider):
            sample_n = cps_sample_n
        elif isinstance(provider, PUFSourceProvider):
            sample_n = puf_sample_n
        elif isinstance(provider, DonorSurveySourceProvider):
            sample_n = donor_sample_n
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
    include_donor_surveys: bool = False,
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
        defer_policyengine_harness=defer_policyengine_harness,
        require_policyengine_native_score=require_policyengine_native_score,
        defer_policyengine_native_score=defer_policyengine_native_score,
        precomputed_policyengine_harness_payload=precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores=precomputed_policyengine_native_scores,
        run_registry_path=run_registry_path,
        run_index_path=run_index_path,
        run_registry_metadata=resolved_registry_metadata,
    )
    parity_path = write_policyengine_us_data_rebuild_parity_artifact(
        artifacts.artifact_paths.output_dir,
        program=program,
    )
    parity_payload = build_policyengine_us_data_rebuild_parity_artifact(
        artifacts.artifact_paths.output_dir,
        program=program,
    )
    return PEUSDataRebuildCheckpointResult(
        build_config=resolved_config,
        provider_names=provider_names,
        queries=resolved_queries,
        artifacts=artifacts,
        parity_path=parity_path,
        parity_payload=parity_payload,
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
    parser.add_argument("--include-donor-surveys", action="store_true")
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
