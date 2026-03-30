"""Performance harness for iterative US microplex optimization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import pandas as pd
from microplex.core import ObservationFrame, SourceProvider, SourceQuery
from microplex.fusion import FusionPlan
from microplex.targets import TargetSet

from microplex_us.pipelines.pe_native_scores import compute_us_pe_native_scores
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexBuildResult,
    USMicroplexPipeline,
    USMicroplexTargets,
)
from microplex_us.policyengine.harness import (
    PolicyEngineUSComparisonCache,
    PolicyEngineUSHarnessRun,
    default_policyengine_us_db_harness_slices,
    evaluate_policyengine_us_harness,
    filter_nonempty_policyengine_us_harness_slices,
)
from microplex_us.policyengine.us import (
    PolicyEngineUSDBTargetProvider,
    PolicyEngineUSEntityTableBundle,
)

CacheValue = object
BuildConfigCacheKey = tuple[tuple[str, CacheValue], ...]
SourceQueryCacheKey = tuple[
    str,
    tuple[tuple[str, CacheValue], ...],
    int | str | None,
    tuple[tuple[str, CacheValue], ...],
]
PreCalibrationCacheKey = tuple[tuple[SourceQueryCacheKey, ...], BuildConfigCacheKey]
CalibrationCacheKey = tuple[PreCalibrationCacheKey, BuildConfigCacheKey]

PRECALIBRATION_STAGE_NAMES = (
    "prepare_source_inputs",
    "prepare_seed_data",
    "integrate_donor_sources",
    "build_targets",
    "resolve_synthesis_variables",
    "synthesize",
    "ensure_target_support",
    "build_policyengine_tables",
)


def default_fast_calibration_target_variables(
    variables: tuple[str, ...],
) -> tuple[str, ...]:
    """Drop redundant targets when a downstream PE tax aggregate already subsumes them."""
    filtered = list(variables)
    if "income_tax" in filtered and "adjusted_gross_income" in filtered:
        filtered = [
            variable for variable in filtered if variable != "adjusted_gross_income"
        ]
    return tuple(filtered) or variables


@dataclass(frozen=True)
class USMicroplexPerformanceHarnessConfig:
    """Configuration for a repeatable local optimization harness."""

    sample_n: int = 100
    n_synthetic: int = 100
    random_seed: int = 42
    targets_db: str | Path | None = None
    baseline_dataset: str | Path | None = None
    target_period: int = 2024
    target_variables: tuple[str, ...] | None = None
    target_domains: tuple[str, ...] | None = None
    target_geo_levels: tuple[str, ...] | None = None
    target_profile: str | None = None
    calibration_target_variables: tuple[str, ...] | None = None
    calibration_target_domains: tuple[str, ...] | None = None
    calibration_target_geo_levels: tuple[str, ...] | None = None
    calibration_target_profile: str | None = None
    build_config: USMicroplexBuildConfig | None = None
    evaluate_parity: bool = True
    evaluate_pe_native_loss: bool = False
    policyengine_us_data_repo: str | Path | None = None
    strict_materialization: bool = True
    fast_inner_loop_calibration: bool = False


@dataclass(frozen=True)
class USMicroplexPerformanceHarnessResult:
    """Stage timings plus parity metrics for one performance-harness run."""

    config: USMicroplexPerformanceHarnessConfig
    build_config: USMicroplexBuildConfig
    build_result: USMicroplexBuildResult
    source_names: tuple[str, ...]
    stage_timings: dict[str, float]
    total_seconds: float
    parity_run: PolicyEngineUSHarnessRun | None = None
    pe_native_scores: dict[str, object] | None = None

    def _parity_run_attr(self, name: str) -> float | None:
        if self.parity_run is None:
            return None
        return getattr(self.parity_run, name)

    def _pe_native_score_attr(self, name: str) -> float | bool | int | None:
        if self.pe_native_scores is None:
            return None
        summary = self.pe_native_scores.get("summary")
        if not isinstance(summary, dict):
            return None
        return summary.get(name)

    @property
    def candidate_composite_parity_loss(self) -> float | None:
        return self._parity_run_attr("candidate_composite_parity_loss")

    @property
    def baseline_composite_parity_loss(self) -> float | None:
        return self._parity_run_attr("baseline_composite_parity_loss")

    @property
    def target_win_rate(self) -> float | None:
        return self._parity_run_attr("target_win_rate")

    @property
    def slice_win_rate(self) -> float | None:
        return self._parity_run_attr("slice_win_rate")

    @property
    def candidate_enhanced_cps_native_loss(self) -> float | None:
        value = self._pe_native_score_attr("candidate_enhanced_cps_native_loss")
        return float(value) if value is not None else None

    @property
    def baseline_enhanced_cps_native_loss(self) -> float | None:
        value = self._pe_native_score_attr("baseline_enhanced_cps_native_loss")
        return float(value) if value is not None else None

    @property
    def enhanced_cps_native_loss_delta(self) -> float | None:
        value = self._pe_native_score_attr("enhanced_cps_native_loss_delta")
        return float(value) if value is not None else None


@dataclass
class USMicroplexPreCalibrationCacheEntry:
    """Reusable upstream build state prior to PE-US calibration."""

    seed_data: pd.DataFrame
    synthetic_data: pd.DataFrame
    synthetic_tables: PolicyEngineUSEntityTableBundle
    targets: USMicroplexTargets
    synthesis_metadata: dict[str, object]
    synthesizer: object | None
    source_frame: ObservationFrame
    source_frames: tuple[ObservationFrame, ...]
    fusion_plan: FusionPlan


@dataclass
class USMicroplexCalibrationCacheEntry:
    """Reusable PE-US calibration result for a fixed synthetic table bundle."""

    policyengine_tables: PolicyEngineUSEntityTableBundle
    calibrated_data: pd.DataFrame
    calibration_summary: dict[str, object]


def _stage(stage_timings: dict[str, float], name: str) -> float:
    start = perf_counter()
    stage_timings.setdefault(name, 0.0)
    return start


def _finish_stage(stage_timings: dict[str, float], name: str, start: float) -> None:
    stage_timings[name] += perf_counter() - start


def _normalize_source_query_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return tuple(
            sorted(
                (str(key), _normalize_source_query_value(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_source_query_value(item) for item in value)
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return str(value)


def _sorted_normalized_items(
    items: Iterable[tuple[str, object]],
) -> tuple[tuple[str, object], ...]:
    return tuple(
        sorted(
            (str(key), _normalize_source_query_value(value))
            for key, value in items
        )
    )


PRECALIBRATION_EXCLUDED_BUILD_CONFIG_FIELDS = frozenset(
    {
        "calibration_backend",
        "target_sparsity",
        "device",
        "policyengine_baseline_dataset",
        "policyengine_targets_db",
        "policyengine_target_period",
        "policyengine_target_variables",
        "policyengine_target_domains",
        "policyengine_target_geo_levels",
        "policyengine_calibration_target_variables",
        "policyengine_calibration_target_domains",
        "policyengine_calibration_target_geo_levels",
        "policyengine_target_reform_id",
        "policyengine_simulation_cls",
    }
)

CALIBRATION_INCLUDED_BUILD_CONFIG_FIELDS = frozenset(
    {
        "calibration_backend",
        "target_sparsity",
        "device",
        "policyengine_targets_db",
        "policyengine_target_period",
        "policyengine_target_variables",
        "policyengine_target_domains",
        "policyengine_target_geo_levels",
        "policyengine_calibration_target_variables",
        "policyengine_calibration_target_domains",
        "policyengine_calibration_target_geo_levels",
        "policyengine_target_reform_id",
        "policyengine_simulation_cls",
        "policyengine_dataset_year",
    }
)


def _provider_cache_identity(provider: SourceProvider) -> tuple[str, tuple[tuple[str, object], ...]]:
    provider_type = f"{provider.__class__.__module__}.{provider.__class__.__qualname__}"
    public_items = _sorted_normalized_items(
        (key, value)
        for key, value in vars(provider).items()
        if not key.startswith("_") and not callable(value)
    )
    return provider_type, public_items


def _build_config_key(
    build_config: USMicroplexBuildConfig,
    *,
    included_fields: frozenset[str] | None = None,
    excluded_fields: frozenset[str] = frozenset(),
) -> tuple[tuple[str, object], ...]:
    return _sorted_normalized_items(
        (key, value)
        for key, value in build_config.to_dict().items()
        if (included_fields is None or key in included_fields)
        and key not in excluded_fields
    )


def _precalibration_build_config_key(
    build_config: USMicroplexBuildConfig,
) -> BuildConfigCacheKey:
    return _build_config_key(
        build_config,
        excluded_fields=PRECALIBRATION_EXCLUDED_BUILD_CONFIG_FIELDS,
    )


def _calibration_build_config_key(
    build_config: USMicroplexBuildConfig,
) -> BuildConfigCacheKey:
    return _build_config_key(
        build_config,
        included_fields=CALIBRATION_INCLUDED_BUILD_CONFIG_FIELDS,
    )


def _source_query_cache_key(
    provider: SourceProvider,
    query: SourceQuery | None,
) -> SourceQueryCacheKey:
    query = query or SourceQuery()
    provider_type, provider_items = _provider_cache_identity(provider)
    return (
        provider_type,
        provider_items,
        query.period,
        _sorted_normalized_items(query.provider_filters.items()),
    )


def _copy_optional_table(table: pd.DataFrame | None) -> pd.DataFrame | None:
    if table is None:
        return None
    return table.copy()


def _clone_policyengine_us_tables(
    tables: PolicyEngineUSEntityTableBundle,
) -> PolicyEngineUSEntityTableBundle:
    if not isinstance(tables, PolicyEngineUSEntityTableBundle):
        return tables
    return PolicyEngineUSEntityTableBundle(
        households=tables.households.copy(),
        persons=_copy_optional_table(tables.persons),
        tax_units=_copy_optional_table(tables.tax_units),
        spm_units=_copy_optional_table(tables.spm_units),
        families=_copy_optional_table(tables.families),
        marital_units=_copy_optional_table(tables.marital_units),
    )


def _clone_calibration_cache_entry(
    entry: USMicroplexCalibrationCacheEntry,
) -> USMicroplexCalibrationCacheEntry:
    return USMicroplexCalibrationCacheEntry(
        policyengine_tables=_clone_policyengine_us_tables(entry.policyengine_tables),
        calibrated_data=entry.calibrated_data.copy(deep=True),
        calibration_summary=dict(entry.calibration_summary),
    )


def _default_provider_query(
    config: USMicroplexPerformanceHarnessConfig,
) -> SourceQuery:
    return SourceQuery(
        provider_filters={
            "sample_n": config.sample_n,
            "random_seed": config.random_seed,
        }
    )


def _load_frames(
    providers: list[SourceProvider],
    provider_queries: dict[str, SourceQuery],
    *,
    frame_cache: dict[SourceQueryCacheKey, ObservationFrame] | None,
) -> tuple[list[ObservationFrame], list[SourceQueryCacheKey]]:
    frames: list[ObservationFrame] = []
    frame_keys: list[SourceQueryCacheKey] = []
    for provider in providers:
        provider_query = provider_queries.get(provider.descriptor.name)
        cache_key = _source_query_cache_key(provider, provider_query)
        frame_keys.append(cache_key)
        if frame_cache is not None and cache_key in frame_cache:
            frames.append(frame_cache[cache_key])
            continue
        frame = provider.load_frame(provider_query)
        if frame_cache is not None:
            frame_cache[cache_key] = frame
        frames.append(frame)
    return frames, frame_keys


def _mark_skipped_stages(stage_timings: dict[str, float], stage_names: tuple[str, ...]) -> None:
    for stage_name in stage_names:
        stage_timings.setdefault(stage_name, 0.0)


def _resolve_build_config(config: USMicroplexPerformanceHarnessConfig) -> USMicroplexBuildConfig:
    default_base_kwargs: dict[str, object] = {
        "synthesis_backend": "bootstrap",
        "calibration_backend": "entropy",
    }
    if config.target_profile is None and config.calibration_target_profile is None:
        default_base_kwargs.update(
            {
                "policyengine_target_variables": (
                    "adjusted_gross_income",
                    "income_tax",
                    "dividend_income",
                    "taxable_interest_income",
                    "self_employment_income",
                ),
                "policyengine_target_geo_levels": ("national",),
            }
        )
    base = config.build_config or USMicroplexBuildConfig(**default_base_kwargs)
    target_variables = (
        base.policyengine_target_variables
        if config.target_variables is None
        else config.target_variables
    )
    target_domains = (
        base.policyengine_target_domains
        if config.target_domains is None
        else config.target_domains
    )
    target_geo_levels = (
        base.policyengine_target_geo_levels
        if config.target_geo_levels is None
        else config.target_geo_levels
    )
    calibration_target_variables = (
        config.calibration_target_variables
        if config.calibration_target_variables is not None
        else base.policyengine_calibration_target_variables
        or (
            default_fast_calibration_target_variables(target_variables)
            if config.fast_inner_loop_calibration and target_variables
            else target_variables
        )
    )
    calibration_target_domains = (
        config.calibration_target_domains
        if config.calibration_target_domains is not None
        else base.policyengine_calibration_target_domains or target_domains
    )
    calibration_target_geo_levels = (
        config.calibration_target_geo_levels
        if config.calibration_target_geo_levels is not None
        else base.policyengine_calibration_target_geo_levels or target_geo_levels
    )
    return replace(
        base,
        n_synthetic=config.n_synthetic,
        random_seed=config.random_seed,
        policyengine_targets_db=(
            str(config.targets_db)
            if config.targets_db is not None
            else base.policyengine_targets_db
        ),
        policyengine_baseline_dataset=(
            str(config.baseline_dataset)
            if config.baseline_dataset is not None
            else base.policyengine_baseline_dataset
        ),
        policyengine_target_period=config.target_period,
        policyengine_target_variables=target_variables,
        policyengine_target_domains=target_domains,
        policyengine_target_geo_levels=target_geo_levels,
        policyengine_target_profile=config.target_profile or base.policyengine_target_profile,
        policyengine_calibration_target_variables=calibration_target_variables,
        policyengine_calibration_target_domains=calibration_target_domains,
        policyengine_calibration_target_geo_levels=calibration_target_geo_levels,
        policyengine_calibration_target_profile=(
            config.calibration_target_profile
            or base.policyengine_calibration_target_profile
        ),
        policyengine_dataset_year=base.policyengine_dataset_year or config.target_period,
    )


@dataclass
class USMicroplexPerformanceSession:
    """Reusable local optimization session with a persistent PE-US comparison cache."""

    comparison_cache: PolicyEngineUSComparisonCache = field(
        default_factory=PolicyEngineUSComparisonCache
    )
    frame_cache: dict[SourceQueryCacheKey, ObservationFrame] = field(default_factory=dict)
    precalibration_cache: dict[PreCalibrationCacheKey, USMicroplexPreCalibrationCacheEntry] = field(
        default_factory=dict
    )
    calibration_cache: dict[CalibrationCacheKey, USMicroplexCalibrationCacheEntry] = field(
        default_factory=dict
    )

    def warm_parity_cache(
        self,
        *,
        config: USMicroplexPerformanceHarnessConfig,
    ) -> PolicyEngineUSComparisonCache:
        return warm_us_microplex_parity_cache(
            config=config,
            comparison_cache=self.comparison_cache,
        )

    def run(
        self,
        providers: list[SourceProvider],
        *,
        config: USMicroplexPerformanceHarnessConfig,
        queries: dict[str, SourceQuery] | None = None,
    ) -> USMicroplexPerformanceHarnessResult:
        return run_us_microplex_performance_harness(
            providers,
            config=config,
            queries=queries,
            comparison_cache=self.comparison_cache,
            frame_cache=self.frame_cache,
            precalibration_cache=self.precalibration_cache,
            calibration_cache=self.calibration_cache,
        )


def _union_target_set(target_sets: dict[str, TargetSet]) -> TargetSet:
    union = TargetSet()
    seen_names: set[str] = set()
    for target_set in target_sets.values():
        for target in target_set.targets:
            if target.name in seen_names:
                continue
            seen_names.add(target.name)
            union.add(target)
    return union


def warm_us_microplex_parity_cache(
    *,
    config: USMicroplexPerformanceHarnessConfig,
    comparison_cache: PolicyEngineUSComparisonCache | None = None,
) -> PolicyEngineUSComparisonCache:
    """Preload target slices and the baseline PE-US report into a reusable cache."""
    if config.targets_db is None or config.baseline_dataset is None:
        raise ValueError(
            "warm_us_microplex_parity_cache requires both targets_db and baseline_dataset"
        )

    cache = comparison_cache or PolicyEngineUSComparisonCache()
    build_config = _resolve_build_config(config)
    target_provider = PolicyEngineUSDBTargetProvider(str(config.targets_db))
    slices = default_policyengine_us_db_harness_slices(
        period=config.target_period,
        variables=build_config.policyengine_target_variables,
        domain_variables=build_config.policyengine_target_domains,
        geo_levels=build_config.policyengine_target_geo_levels,
        reform_id=build_config.policyengine_target_reform_id,
    )
    slices = filter_nonempty_policyengine_us_harness_slices(
        target_provider,
        slices,
        cache=cache,
    )
    slice_target_sets = {
        slice_spec.name: cache.load_target_set(target_provider, slice_spec.query)
        for slice_spec in slices
    }
    union_target_set = _union_target_set(slice_target_sets)
    cache.load_baseline_report(
        target_set=union_target_set,
        baseline_dataset=str(config.baseline_dataset),
        period=config.target_period,
        dataset_year=build_config.policyengine_dataset_year,
        simulation_cls=build_config.policyengine_simulation_cls,
        baseline_label="policyengine_baseline",
        strict_materialization=config.strict_materialization,
    )
    return cache


def run_us_microplex_performance_harness(
    providers: list[SourceProvider],
    *,
    config: USMicroplexPerformanceHarnessConfig,
    queries: dict[str, SourceQuery] | None = None,
    comparison_cache: PolicyEngineUSComparisonCache | None = None,
    frame_cache: dict[SourceQueryCacheKey, ObservationFrame] | None = None,
    precalibration_cache: dict[PreCalibrationCacheKey, USMicroplexPreCalibrationCacheEntry]
    | None = None,
    calibration_cache: dict[CalibrationCacheKey, USMicroplexCalibrationCacheEntry]
    | None = None,
) -> USMicroplexPerformanceHarnessResult:
    """Run a repeatable build+parity loop with stage-level timings."""
    if not providers:
        raise ValueError("USMicroplex performance harness requires at least one provider")
    if config.evaluate_parity and (
        config.targets_db is None or config.baseline_dataset is None
    ):
        raise ValueError(
            "USMicroplex performance harness requires both targets_db and baseline_dataset"
        )
    if config.evaluate_pe_native_loss and config.baseline_dataset is None:
        raise ValueError(
            "USMicroplex performance harness requires baseline_dataset for PE-native loss scoring"
        )

    build_config = _resolve_build_config(config)
    pipeline = USMicroplexPipeline(build_config)
    provider_queries = dict(queries or {})
    for provider in providers:
        provider_queries.setdefault(
            provider.descriptor.name,
            _default_provider_query(config),
        )

    stage_timings: dict[str, float] = {}
    total_start = perf_counter()

    start = _stage(stage_timings, "load_frames")
    frames, frame_keys = _load_frames(
        providers,
        provider_queries,
        frame_cache=frame_cache,
    )
    _finish_stage(stage_timings, "load_frames", start)
    precalibration_key = (
        tuple(frame_keys),
        _precalibration_build_config_key(build_config),
    )
    precalibration = (
        precalibration_cache.get(precalibration_key)
        if precalibration_cache is not None
        else None
    )

    if precalibration is None:
        start = _stage(stage_timings, "prepare_source_inputs")
        source_inputs = [pipeline.prepare_source_input(frame) for frame in frames]
        fusion_plan = FusionPlan.from_sources([frame.source for frame in frames])
        scaffold_input = pipeline._select_scaffold_source(source_inputs)
        _finish_stage(stage_timings, "prepare_source_inputs", start)

        start = _stage(stage_timings, "prepare_seed_data")
        seed_data = pipeline.prepare_seed_data_from_source(scaffold_input)
        _finish_stage(stage_timings, "prepare_seed_data", start)

        start = _stage(stage_timings, "integrate_donor_sources")
        donor_integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=scaffold_input,
            donor_inputs=[
                source for source in source_inputs if source is not scaffold_input
            ],
        )
        seed_data = donor_integration["seed_data"]
        _finish_stage(stage_timings, "integrate_donor_sources", start)

        start = _stage(stage_timings, "build_targets")
        targets = pipeline.build_targets(seed_data)
        _finish_stage(stage_timings, "build_targets", start)

        start = _stage(stage_timings, "resolve_synthesis_variables")
        synthesis_variables = pipeline._resolve_synthesis_variables(
            scaffold_input,
            fusion_plan=fusion_plan,
            include_all_observed_targets=len(source_inputs) > 1,
            available_columns=set(seed_data.columns),
        )
        _finish_stage(stage_timings, "resolve_synthesis_variables", start)

        start = _stage(stage_timings, "synthesize")
        synthetic_data, synthesizer, synthesis_metadata = pipeline.synthesize(
            seed_data,
            synthesis_variables=synthesis_variables,
        )
        _finish_stage(stage_timings, "synthesize", start)

        start = _stage(stage_timings, "ensure_target_support")
        synthetic_data = pipeline.ensure_target_support(
            synthetic_data,
            seed_data,
            targets,
        )
        _finish_stage(stage_timings, "ensure_target_support", start)

        start = _stage(stage_timings, "build_policyengine_tables")
        synthetic_tables = pipeline.build_policyengine_entity_tables(synthetic_data)
        _finish_stage(stage_timings, "build_policyengine_tables", start)

        synthesis_metadata = {
            **synthesis_metadata,
            "source_names": fusion_plan.source_names,
            "condition_vars": list(synthesis_variables.condition_vars),
            "target_vars": list(synthesis_variables.target_vars),
            "scaffold_source": scaffold_input.frame.source.name,
            "donor_integrated_variables": donor_integration["integrated_variables"],
        }
        precalibration = USMicroplexPreCalibrationCacheEntry(
            seed_data=seed_data,
            synthetic_data=synthetic_data,
            synthetic_tables=synthetic_tables,
            targets=targets,
            synthesis_metadata=synthesis_metadata,
            synthesizer=synthesizer,
            source_frame=scaffold_input.frame,
            source_frames=tuple(frames),
            fusion_plan=fusion_plan,
        )
        if precalibration_cache is not None:
            precalibration_cache[precalibration_key] = precalibration
    else:
        _mark_skipped_stages(stage_timings, PRECALIBRATION_STAGE_NAMES)

    calibration_key = (
        precalibration_key,
        _calibration_build_config_key(build_config),
    )
    calibration = (
        calibration_cache.get(calibration_key)
        if calibration_cache is not None
        else None
    )
    if calibration is None:
        start = _stage(stage_timings, "calibrate_policyengine_tables")
        policyengine_tables, calibrated_data, calibration_summary = (
            pipeline.calibrate_policyengine_tables(
                _clone_policyengine_us_tables(precalibration.synthetic_tables)
            )
        )
        _finish_stage(stage_timings, "calibrate_policyengine_tables", start)
        calibration = USMicroplexCalibrationCacheEntry(
            policyengine_tables=_clone_policyengine_us_tables(policyengine_tables),
            calibrated_data=calibrated_data.copy(deep=True),
            calibration_summary=dict(calibration_summary),
        )
        if calibration_cache is not None:
            calibration_cache[calibration_key] = calibration
    else:
        stage_timings.setdefault("calibrate_policyengine_tables", 0.0)

    calibration = _clone_calibration_cache_entry(calibration)

    build_result = USMicroplexBuildResult(
        config=build_config,
        seed_data=precalibration.seed_data,
        synthetic_data=precalibration.synthetic_data,
        calibrated_data=calibration.calibrated_data,
        targets=precalibration.targets,
        calibration_summary=calibration.calibration_summary,
        synthesis_metadata=dict(precalibration.synthesis_metadata),
        synthesizer=precalibration.synthesizer,
        policyengine_tables=calibration.policyengine_tables,
        source_frame=precalibration.source_frame,
        source_frames=precalibration.source_frames,
        fusion_plan=precalibration.fusion_plan,
    )

    parity_run = None
    if config.evaluate_parity:
        start = _stage(stage_timings, "evaluate_parity_harness")
        target_provider = PolicyEngineUSDBTargetProvider(str(config.targets_db))
        slices = default_policyengine_us_db_harness_slices(
            period=config.target_period,
            variables=build_config.policyengine_target_variables,
            domain_variables=build_config.policyengine_target_domains,
            geo_levels=build_config.policyengine_target_geo_levels,
            reform_id=build_config.policyengine_target_reform_id,
        )
        slices = filter_nonempty_policyengine_us_harness_slices(
            target_provider,
            slices,
            cache=comparison_cache,
        )
        parity_run = evaluate_policyengine_us_harness(
            build_result.policyengine_tables,
            target_provider,
            slices,
            baseline_dataset=str(config.baseline_dataset),
            dataset_year=build_config.policyengine_dataset_year,
            simulation_cls=build_config.policyengine_simulation_cls,
            metadata={
                "sample_n": config.sample_n,
                "n_synthetic": config.n_synthetic,
                "source_names": precalibration.fusion_plan.source_names,
                "target_variables": list(build_config.policyengine_target_variables),
                "calibration_target_variables": list(
                    build_config.policyengine_calibration_target_variables
                ),
            },
            strict_materialization=config.strict_materialization,
            cache=comparison_cache,
        )
        _finish_stage(stage_timings, "evaluate_parity_harness", start)

    pe_native_scores = None
    if config.evaluate_pe_native_loss:
        start = _stage(stage_timings, "evaluate_pe_native_loss")
        with TemporaryDirectory(prefix="microplex-us-native-score-") as temp_dir:
            candidate_dataset_path = pipeline.export_policyengine_dataset(
                build_result,
                Path(temp_dir) / "candidate_policyengine_us.h5",
                direct_override_variables=build_config.policyengine_direct_override_variables,
            )
            pe_native_scores = compute_us_pe_native_scores(
                candidate_dataset_path=candidate_dataset_path,
                baseline_dataset_path=str(config.baseline_dataset),
                period=build_config.policyengine_dataset_year or config.target_period,
                policyengine_us_data_repo=config.policyengine_us_data_repo,
            )
        _finish_stage(stage_timings, "evaluate_pe_native_loss", start)

    total_seconds = perf_counter() - total_start
    return USMicroplexPerformanceHarnessResult(
        config=config,
        build_config=build_config,
        build_result=build_result,
        source_names=precalibration.fusion_plan.source_names,
        stage_timings=stage_timings,
        total_seconds=total_seconds,
        parity_run=parity_run,
        pe_native_scores=pe_native_scores,
    )


__all__ = [
    "USMicroplexPerformanceHarnessConfig",
    "USMicroplexPerformanceHarnessResult",
    "USMicroplexCalibrationCacheEntry",
    "USMicroplexPreCalibrationCacheEntry",
    "USMicroplexPerformanceSession",
    "default_fast_calibration_target_variables",
    "run_us_microplex_performance_harness",
    "warm_us_microplex_parity_cache",
]
