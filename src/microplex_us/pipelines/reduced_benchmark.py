"""Reduced benchmark harness for staged Microplex-US debugging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from microplex.core import EntityType, SourceProvider, SourceQuery
from microplex.targets import (
    FilterOperator,
    TargetAggregation,
    TargetFilter,
    TargetSpec,
)

from microplex_us.pipelines.local_reweighting import reweight_us_household_targets
from microplex_us.pipelines.performance import (
    USMicroplexPerformanceHarnessConfig,
    USMicroplexPerformanceHarnessResult,
    run_us_microplex_performance_harness,
)
from microplex_us.policyengine.us import (
    PolicyEngineUSEntityTableBundle,
    build_policyengine_us_time_period_arrays,
    load_policyengine_us_entity_tables,
    write_policyengine_us_time_period_dataset,
)

USReducedBenchmarkEntity = Literal[
    "household", "person", "tax_unit", "spm_unit", "family", "marital_unit"
]
USReducedBenchmarkAggregation = Literal[
    "weighted_count", "weighted_sum", "weighted_mean"
]

DEFAULT_ATOMIC_AGE_BINS: tuple[float, ...] = (0.0, 18.0, 30.0, 45.0, 65.0, 200.0)
DEFAULT_ATOMIC_AGE_LABELS: tuple[str, ...] = (
    "0_to_17",
    "18_to_29",
    "30_to_44",
    "45_to_64",
    "65_plus",
)


@dataclass(frozen=True)
class USMicroplexReducedDimensionSpec:
    """One grouped dimension for a reduced benchmark rung."""

    variable: str
    label: str | None = None
    bins: tuple[float, ...] | None = None
    bin_labels: tuple[str, ...] | None = None
    right: bool = False
    include_lowest: bool = True
    missing_label: str | None = "__missing__"
    zero_pad: int | None = None

    @property
    def output_name(self) -> str:
        return self.label or self.variable


@dataclass(frozen=True)
class USMicroplexReducedMeasureSpec:
    """One weighted measure to compare on a reduced rung."""

    name: str
    aggregation: USReducedBenchmarkAggregation = "weighted_count"
    variable: str | None = None


@dataclass(frozen=True)
class USMicroplexReducedBenchmarkSpec:
    """A reduced benchmark rung defined on one entity table and a small target surface."""

    name: str
    entity: USReducedBenchmarkEntity
    dimensions: tuple[USMicroplexReducedDimensionSpec, ...]
    measures: tuple[USMicroplexReducedMeasureSpec, ...] = field(
        default_factory=lambda: (
            USMicroplexReducedMeasureSpec(name="weighted_count"),
        )
    )
    top_k: int = 10


@dataclass(frozen=True)
class USMicroplexReducedBenchmarkReport:
    """Candidate-vs-baseline comparison for one reduced benchmark rung."""

    spec: USMicroplexReducedBenchmarkSpec
    candidate_dataset: str
    baseline_dataset: str
    period: int
    summary: dict[str, Any]
    measure_summaries: dict[str, dict[str, Any]]
    top_cell_gaps: dict[str, list[dict[str, Any]]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": _json_compatible_value(asdict(self.spec)),
            "candidate_dataset": self.candidate_dataset,
            "baseline_dataset": self.baseline_dataset,
            "period": self.period,
            "summary": _json_compatible_value(self.summary),
            "measure_summaries": _json_compatible_value(self.measure_summaries),
            "top_cell_gaps": _json_compatible_value(self.top_cell_gaps),
        }


@dataclass(frozen=True)
class USMicroplexReducedBenchmarkHarnessConfig:
    """Config for building one candidate and evaluating reduced benchmark rungs."""

    performance_config: USMicroplexPerformanceHarnessConfig = field(
        default_factory=USMicroplexPerformanceHarnessConfig
    )
    benchmark_specs: tuple[USMicroplexReducedBenchmarkSpec, ...] = field(
        default_factory=lambda: default_us_atomic_rung0_benchmarks()
    )
    baseline_dataset: str | Path | None = None
    period: int | None = None
    output_json_path: str | Path | None = None
    output_policyengine_dataset_path: str | Path | None = None


@dataclass(frozen=True)
class USMicroplexReducedBenchmarkHarnessResult:
    """One reduced benchmark harness run plus the inner build/export result."""

    config: USMicroplexReducedBenchmarkHarnessConfig
    performance_result: USMicroplexPerformanceHarnessResult
    benchmark_reports: dict[str, USMicroplexReducedBenchmarkReport]
    candidate_dataset_path: str
    baseline_dataset_path: str
    period: int
    stage_timings: dict[str, float]
    total_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": _json_compatible_value(asdict(self.config)),
            "performance_result": self.performance_result.to_dict(),
            "candidate_dataset_path": self.candidate_dataset_path,
            "baseline_dataset_path": self.baseline_dataset_path,
            "period": self.period,
            "stage_timings": dict(self.stage_timings),
            "total_seconds": float(self.total_seconds),
            "benchmark_reports": {
                name: report.to_dict() for name, report in self.benchmark_reports.items()
            },
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return destination


@dataclass(frozen=True)
class USMicroplexReducedCalibrationReport:
    """Pre/post reduced benchmark comparison around a small household reweight step."""

    calibration_spec: USMicroplexReducedBenchmarkSpec
    evaluation_specs: tuple[USMicroplexReducedBenchmarkSpec, ...]
    candidate_dataset: str
    baseline_dataset: str
    period: int
    target_count: int
    reweighting_summary: dict[str, Any]
    pre_reports: dict[str, USMicroplexReducedBenchmarkReport]
    post_reports: dict[str, USMicroplexReducedBenchmarkReport]
    benchmark_deltas: dict[str, dict[str, Any]]
    reweighted_dataset_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_spec": _json_compatible_value(asdict(self.calibration_spec)),
            "evaluation_specs": _json_compatible_value(
                [asdict(spec) for spec in self.evaluation_specs]
            ),
            "candidate_dataset": self.candidate_dataset,
            "baseline_dataset": self.baseline_dataset,
            "period": self.period,
            "target_count": self.target_count,
            "reweighting_summary": _json_compatible_value(self.reweighting_summary),
            "benchmark_deltas": _json_compatible_value(self.benchmark_deltas),
            "reweighted_dataset_path": self.reweighted_dataset_path,
            "pre_reports": {
                name: report.to_dict() for name, report in self.pre_reports.items()
            },
            "post_reports": {
                name: report.to_dict() for name, report in self.post_reports.items()
            },
        }


def default_us_atomic_rung0_benchmarks() -> tuple[USMicroplexReducedBenchmarkSpec, ...]:
    """A minimal first rung: counts by state, then by state x age."""

    return (
        USMicroplexReducedBenchmarkSpec(
            name="household_count_by_state",
            entity="household",
            dimensions=(
                USMicroplexReducedDimensionSpec(
                    variable="state_fips",
                    zero_pad=2,
                ),
            ),
            measures=(
                USMicroplexReducedMeasureSpec(name="weighted_household_count"),
            ),
        ),
        USMicroplexReducedBenchmarkSpec(
            name="person_count_by_state_age",
            entity="person",
            dimensions=(
                USMicroplexReducedDimensionSpec(
                    variable="state_fips",
                    zero_pad=2,
                ),
                USMicroplexReducedDimensionSpec(
                    variable="age",
                    label="age_bucket",
                    bins=DEFAULT_ATOMIC_AGE_BINS,
                    bin_labels=DEFAULT_ATOMIC_AGE_LABELS,
                ),
            ),
            measures=(
                USMicroplexReducedMeasureSpec(name="weighted_person_count"),
            ),
        ),
    )


def default_us_atomic_rung1_benchmarks() -> tuple[USMicroplexReducedBenchmarkSpec, ...]:
    """A slightly richer CPS-like rung on person demographics and earnings."""

    return (
        USMicroplexReducedBenchmarkSpec(
            name="person_count_by_state_sex",
            entity="person",
            dimensions=(
                USMicroplexReducedDimensionSpec(
                    variable="state_fips",
                    zero_pad=2,
                ),
                USMicroplexReducedDimensionSpec(
                    variable="is_female",
                ),
            ),
            measures=(
                USMicroplexReducedMeasureSpec(name="weighted_person_count"),
            ),
        ),
        USMicroplexReducedBenchmarkSpec(
            name="employment_income_sum_by_state",
            entity="person",
            dimensions=(
                USMicroplexReducedDimensionSpec(
                    variable="state_fips",
                    zero_pad=2,
                ),
            ),
            measures=(
                USMicroplexReducedMeasureSpec(
                    name="weighted_employment_income_sum",
                    aggregation="weighted_sum",
                    variable="employment_income_before_lsr",
                ),
            ),
        ),
        USMicroplexReducedBenchmarkSpec(
            name="employment_income_mean_by_state_sex",
            entity="person",
            dimensions=(
                USMicroplexReducedDimensionSpec(
                    variable="state_fips",
                    zero_pad=2,
                ),
                USMicroplexReducedDimensionSpec(
                    variable="is_female",
                ),
            ),
            measures=(
                USMicroplexReducedMeasureSpec(
                    name="weighted_employment_income_mean",
                    aggregation="weighted_mean",
                    variable="employment_income_before_lsr",
                ),
            ),
        ),
    )


def default_us_atomic_rung2_calibration() -> tuple[
    USMicroplexReducedBenchmarkSpec,
    tuple[USMicroplexReducedBenchmarkSpec, ...],
]:
    """Default reduced calibration comparison: fit state household counts, then evaluate rung 0+1."""

    rung0 = default_us_atomic_rung0_benchmarks()
    return (
        rung0[0],
        rung0 + default_us_atomic_rung1_benchmarks(),
    )


def evaluate_us_reduced_benchmark(
    candidate_dataset: str | Path,
    baseline_dataset: str | Path,
    spec: USMicroplexReducedBenchmarkSpec,
    *,
    period: int = 2024,
) -> USMicroplexReducedBenchmarkReport:
    """Compare one candidate H5 to one baseline H5 on a small grouped target surface."""

    _validate_reduced_benchmark_spec(spec)
    requested_variables = _required_reduced_benchmark_variables(spec)
    candidate_bundle = load_policyengine_us_entity_tables(
        candidate_dataset,
        period=period,
        variables=tuple(sorted(requested_variables)),
    )
    baseline_bundle = load_policyengine_us_entity_tables(
        baseline_dataset,
        period=period,
        variables=tuple(sorted(requested_variables)),
    )
    return _evaluate_us_reduced_benchmark_bundles(
        candidate_bundle,
        baseline_bundle,
        spec,
        candidate_dataset=candidate_dataset,
        baseline_dataset=baseline_dataset,
        period=period,
    )


def _evaluate_us_reduced_benchmark_bundles(
    candidate_bundle: PolicyEngineUSEntityTableBundle,
    baseline_bundle: PolicyEngineUSEntityTableBundle,
    spec: USMicroplexReducedBenchmarkSpec,
    *,
    candidate_dataset: str | Path,
    baseline_dataset: str | Path,
    period: int,
) -> USMicroplexReducedBenchmarkReport:
    """Compare one candidate bundle to one baseline bundle on a small grouped target surface."""

    _validate_reduced_benchmark_spec(spec)
    candidate_grouped, candidate_row_count = _group_reduced_benchmark_bundle(
        candidate_bundle,
        spec,
    )
    baseline_grouped, baseline_row_count = _group_reduced_benchmark_bundle(
        baseline_bundle,
        spec,
    )

    dimension_names = [dimension.output_name for dimension in spec.dimensions]
    merged = candidate_grouped.merge(
        baseline_grouped,
        on=dimension_names,
        how="outer",
        suffixes=("_candidate", "_baseline"),
    )
    # Fill NaN with 0.0 only for count/sum measures (missing cell = zero).
    # Leave weighted_mean NaN so the error metric correctly reflects missing cells.
    fill_zero_columns = []
    for measure in spec.measures:
        if measure.aggregation in {"weighted_count", "weighted_sum"}:
            fill_zero_columns.append(f"{measure.name}_candidate")
            fill_zero_columns.append(f"{measure.name}_baseline")
    if fill_zero_columns:
        merged[fill_zero_columns] = merged[fill_zero_columns].fillna(0.0)

    measure_summaries: dict[str, dict[str, Any]] = {}
    top_cell_gaps: dict[str, list[dict[str, Any]]] = {}
    for measure in spec.measures:
        candidate_column = f"{measure.name}_candidate"
        baseline_column = f"{measure.name}_baseline"
        candidate_values = merged[candidate_column].to_numpy(dtype=float)
        baseline_values = merged[baseline_column].to_numpy(dtype=float)
        delta = candidate_values - baseline_values
        abs_relative_error = np.abs(delta) / (np.abs(baseline_values) + 1.0)
        baseline_nonzero = np.abs(baseline_values) > 1e-9
        candidate_nonzero = np.abs(candidate_values) > 1e-9
        shared_nonzero = baseline_nonzero & candidate_nonzero
        measure_summaries[measure.name] = {
            "candidate_total": float(candidate_values.sum()),
            "baseline_total": float(baseline_values.sum()),
            "total_delta": float(delta.sum()),
            "n_cells": int(len(candidate_values)),
            "baseline_nonzero_cell_count": int(baseline_nonzero.sum()),
            "candidate_nonzero_cell_count": int(candidate_nonzero.sum()),
            "shared_nonzero_cell_count": int(shared_nonzero.sum()),
            "support_recall": _safe_ratio(
                int(shared_nonzero.sum()),
                int(baseline_nonzero.sum()),
            ),
            "mean_abs_relative_error": float(np.nanmean(abs_relative_error)),
            "max_abs_relative_error": float(np.nanmax(abs_relative_error)),
        }
        sort_frame = merged[dimension_names + [candidate_column, baseline_column]].copy()
        sort_frame["delta"] = delta
        sort_frame["abs_relative_error"] = abs_relative_error
        sort_frame["abs_delta"] = np.abs(delta)
        sort_frame = sort_frame.sort_values(
            by=["abs_relative_error", "abs_delta"],
            ascending=False,
            kind="mergesort",
        )
        rows: list[dict[str, Any]] = []
        for _, row in sort_frame.head(spec.top_k).iterrows():
            payload = {
                dimension_name: _json_compatible_value(row[dimension_name])
                for dimension_name in dimension_names
            }
            payload.update(
                {
                    "candidate_value": float(row[candidate_column]),
                    "baseline_value": float(row[baseline_column]),
                    "delta": float(row["delta"]),
                    "abs_relative_error": float(row["abs_relative_error"]),
                }
            )
            rows.append(payload)
        top_cell_gaps[measure.name] = rows

    summary = {
        "entity": spec.entity,
        "n_candidate_rows": candidate_row_count,
        "n_baseline_rows": baseline_row_count,
        "n_dimensions": len(spec.dimensions),
        "n_measures": len(spec.measures),
        "n_cells": int(len(merged)),
        "mean_measure_mare": float(
            np.nanmean(
                [
                    measure_summaries[measure.name]["mean_abs_relative_error"]
                    for measure in spec.measures
                ]
            )
        ),
    }
    return USMicroplexReducedBenchmarkReport(
        spec=spec,
        candidate_dataset=str(Path(candidate_dataset).expanduser().resolve()),
        baseline_dataset=str(Path(baseline_dataset).expanduser().resolve()),
        period=int(period),
        summary=summary,
        measure_summaries=measure_summaries,
        top_cell_gaps=top_cell_gaps,
    )


def run_us_microplex_reduced_benchmark_harness(
    providers: list[SourceProvider],
    *,
    config: USMicroplexReducedBenchmarkHarnessConfig,
    queries: dict[str, SourceQuery] | None = None,
    **performance_kwargs: Any,
) -> USMicroplexReducedBenchmarkHarnessResult:
    """Build one candidate dataset, then score reduced benchmark rungs against a baseline H5."""

    if not providers:
        raise ValueError("US reduced benchmark harness requires at least one provider")
    if not config.benchmark_specs:
        raise ValueError("US reduced benchmark harness requires at least one benchmark spec")

    baseline_dataset = (
        Path(config.baseline_dataset).expanduser().resolve()
        if config.baseline_dataset is not None
        else (
            Path(config.performance_config.baseline_dataset).expanduser().resolve()
            if config.performance_config.baseline_dataset is not None
            else None
        )
    )
    if baseline_dataset is None:
        raise ValueError(
            "US reduced benchmark harness requires baseline_dataset either directly or via performance_config"
        )
    period = int(config.period or config.performance_config.target_period)

    stage_timings: dict[str, float] = {}
    total_start = perf_counter()
    with TemporaryDirectory(prefix="microplex-us-reduced-benchmark-") as temp_dir:
        candidate_output = (
            Path(config.output_policyengine_dataset_path).expanduser().resolve()
            if config.output_policyengine_dataset_path is not None
            else (Path(temp_dir) / "candidate_policyengine_us.h5")
        )
        inner_config = replace(
            config.performance_config,
            evaluate_parity=False,
            evaluate_pe_native_loss=False,
            evaluate_matched_pe_native_loss=False,
            reweight_matched_pe_native_loss=False,
            optimize_pe_native_loss=False,
            output_json_path=None,
            output_pe_native_target_delta_path=None,
            output_pe_native_support_audit_path=None,
            output_matched_baseline_dataset_path=None,
            output_policyengine_dataset_path=str(candidate_output),
        )
        build_start = perf_counter()
        performance_result = run_us_microplex_performance_harness(
            providers,
            config=inner_config,
            queries=queries,
            **performance_kwargs,
        )
        stage_timings["build_candidate_dataset"] = perf_counter() - build_start
        candidate_dataset_path = (
            performance_result.policyengine_dataset_path or str(candidate_output.resolve())
        )

        benchmark_reports: dict[str, USMicroplexReducedBenchmarkReport] = {}
        reduced_start = perf_counter()
        for spec in config.benchmark_specs:
            benchmark_reports[spec.name] = evaluate_us_reduced_benchmark(
                candidate_dataset_path,
                baseline_dataset,
                spec,
                period=period,
            )
        stage_timings["evaluate_reduced_benchmarks"] = perf_counter() - reduced_start

    result = USMicroplexReducedBenchmarkHarnessResult(
        config=config,
        performance_result=performance_result,
        benchmark_reports=benchmark_reports,
        candidate_dataset_path=candidate_dataset_path,
        baseline_dataset_path=str(baseline_dataset),
        period=period,
        stage_timings=stage_timings,
        total_seconds=perf_counter() - total_start,
    )
    if config.output_json_path is not None:
        result.save(config.output_json_path)
    return result


def reduced_benchmark_to_calibration_targets(
    spec: USMicroplexReducedBenchmarkSpec,
    baseline_dataset: str | Path,
    *,
    period: int = 2024,
) -> list[TargetSpec]:
    """Convert a reduced weighted-count spec into household reweighting targets from baseline cells."""

    _validate_reduced_benchmark_spec(spec)
    unsupported = [
        measure.name
        for measure in spec.measures
        if measure.aggregation != "weighted_count"
    ]
    if unsupported:
        raise ValueError(
            "Reduced calibration currently supports weighted_count measures only: "
            + ", ".join(unsupported)
        )

    requested_variables = _required_reduced_benchmark_variables(spec)
    baseline_bundle = load_policyengine_us_entity_tables(
        baseline_dataset,
        period=period,
        variables=tuple(sorted(requested_variables)),
    )
    grouped, _ = _group_reduced_benchmark_bundle(baseline_bundle, spec)
    entity = _entity_type_for_reduced_entity(spec.entity)
    targets: list[TargetSpec] = []
    for _, row in grouped.iterrows():
        filters: list[TargetFilter] = []
        name_parts: list[str] = []
        for dimension in spec.dimensions:
            value = row[dimension.output_name]
            filters.extend(_dimension_filters_for_value(dimension, value))
            name_parts.append(f"{dimension.output_name}={_json_compatible_value(value)}")
        measure = spec.measures[0]
        target_value = float(row[measure.name])
        targets.append(
            TargetSpec(
                name=f"{spec.name}::{'|'.join(name_parts)}",
                entity=entity,
                value=target_value,
                period=period,
                aggregation=TargetAggregation.COUNT,
                filters=tuple(filters),
            )
        )
    return targets


def calibrate_and_evaluate_us_reduced_benchmarks(
    candidate_dataset: str | Path,
    baseline_dataset: str | Path,
    calibration_spec: USMicroplexReducedBenchmarkSpec,
    evaluation_specs: tuple[USMicroplexReducedBenchmarkSpec, ...],
    *,
    period: int = 2024,
    max_iter: int = 8,
    tol: float = 1e-4,
    factor_bounds: tuple[float, float] = (0.5, 2.0),
    output_reweighted_dataset_path: str | Path | None = None,
) -> USMicroplexReducedCalibrationReport:
    """Reweight candidate household weights to a reduced surface, then compare pre/post rung errors."""

    if not evaluation_specs:
        raise ValueError("Reduced calibration requires at least one evaluation spec")

    candidate_path = Path(candidate_dataset).expanduser().resolve()
    baseline_path = Path(baseline_dataset).expanduser().resolve()
    calibration_targets = reduced_benchmark_to_calibration_targets(
        calibration_spec,
        baseline_path,
        period=period,
    )
    requested_variables = set().union(
        _required_reduced_benchmark_variables(calibration_spec),
        *(_required_reduced_benchmark_variables(spec) for spec in evaluation_specs),
    )
    candidate_bundle = load_policyengine_us_entity_tables(
        candidate_path,
        period=period,
        variables=tuple(sorted(requested_variables)),
    )
    baseline_bundle = load_policyengine_us_entity_tables(
        baseline_path,
        period=period,
        variables=tuple(sorted(requested_variables)),
    )

    pre_reports = {
        spec.name: _evaluate_us_reduced_benchmark_bundles(
            candidate_bundle,
            baseline_bundle,
            spec,
            candidate_dataset=candidate_path,
            baseline_dataset=baseline_path,
            period=period,
        )
        for spec in evaluation_specs
    }

    reweight_result = reweight_us_household_targets(
        candidate_bundle,
        targets=calibration_targets,
        max_iter=max_iter,
        tol=tol,
        factor_bounds=factor_bounds,
    )
    reweighted_dataset_path: str | None = None
    if output_reweighted_dataset_path is not None:
        destination = Path(output_reweighted_dataset_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_policyengine_us_time_period_dataset(
            build_policyengine_us_time_period_arrays(
                reweight_result.tables,
                period=period,
                **_infer_reduced_export_maps(reweight_result.tables),
            ),
            destination,
        )
        reweighted_dataset_path = str(destination)

    post_candidate_label = reweighted_dataset_path or f"{candidate_path}#reweighted"
    post_reports = {
        spec.name: _evaluate_us_reduced_benchmark_bundles(
            reweight_result.tables,
            baseline_bundle,
            spec,
            candidate_dataset=post_candidate_label,
            baseline_dataset=baseline_path,
            period=period,
        )
        for spec in evaluation_specs
    }

    benchmark_deltas = {
        spec.name: {
            "pre_mean_measure_mare": float(pre_reports[spec.name].summary["mean_measure_mare"]),
            "post_mean_measure_mare": float(
                post_reports[spec.name].summary["mean_measure_mare"]
            ),
            "delta_mean_measure_mare": float(
                post_reports[spec.name].summary["mean_measure_mare"]
                - pre_reports[spec.name].summary["mean_measure_mare"]
            ),
        }
        for spec in evaluation_specs
    }
    reweighting_summary = {
        "target_count": reweight_result.diagnostics.target_count,
        "constraint_count": reweight_result.diagnostics.constraint_count,
        "iterations": reweight_result.diagnostics.iterations,
        "converged": reweight_result.diagnostics.converged,
        "mean_abs_relative_error": reweight_result.diagnostics.mean_abs_relative_error,
        "max_abs_relative_error": reweight_result.diagnostics.max_abs_relative_error,
        "skipped_targets": list(reweight_result.compilation.skipped_targets),
    }
    return USMicroplexReducedCalibrationReport(
        calibration_spec=calibration_spec,
        evaluation_specs=evaluation_specs,
        candidate_dataset=str(candidate_path),
        baseline_dataset=str(baseline_path),
        period=int(period),
        target_count=len(calibration_targets),
        reweighting_summary=reweighting_summary,
        pre_reports=pre_reports,
        post_reports=post_reports,
        benchmark_deltas=benchmark_deltas,
        reweighted_dataset_path=reweighted_dataset_path,
    )


def _validate_reduced_benchmark_spec(spec: USMicroplexReducedBenchmarkSpec) -> None:
    if spec.top_k <= 0:
        raise ValueError("Reduced benchmark top_k must be positive")
    if not spec.measures:
        raise ValueError("Reduced benchmark spec requires at least one measure")
    for measure in spec.measures:
        if measure.aggregation in {"weighted_sum", "weighted_mean"} and not measure.variable:
            raise ValueError(
                f"Reduced benchmark measure '{measure.name}' requires variable for {measure.aggregation}"
            )
    seen_output_names: set[str] = set()
    for dimension in spec.dimensions:
        output_name = dimension.output_name
        if output_name in seen_output_names:
            raise ValueError(
                f"Reduced benchmark has duplicate dimension output name '{output_name}'"
            )
        seen_output_names.add(output_name)
        if dimension.bins is not None and len(dimension.bins) < 2:
            raise ValueError(
                f"Reduced benchmark dimension '{dimension.variable}' requires at least two bin edges"
            )
        if (
            dimension.bins is not None
            and dimension.bin_labels is not None
            and len(dimension.bin_labels) != len(dimension.bins) - 1
        ):
            raise ValueError(
                f"Reduced benchmark dimension '{dimension.variable}' has mismatched bin labels"
            )


def _required_reduced_benchmark_variables(
    spec: USMicroplexReducedBenchmarkSpec,
) -> set[str]:
    variables = {dimension.variable for dimension in spec.dimensions}
    variables.update(
        measure.variable for measure in spec.measures if measure.variable is not None
    )
    return variables


def _group_reduced_benchmark_bundle(
    bundle: PolicyEngineUSEntityTableBundle,
    spec: USMicroplexReducedBenchmarkSpec,
) -> tuple[pd.DataFrame, int]:
    required_variables = _required_reduced_benchmark_variables(spec)
    table, weights = _materialize_entity_frame(
        bundle,
        spec.entity,
        required_variables,
    )
    frame = pd.DataFrame({"__weight__": weights})
    for dimension in spec.dimensions:
        if dimension.variable not in table.columns:
            raise KeyError(
                f"Reduced benchmark dimension '{dimension.variable}' is missing from {spec.entity} table"
            )
        frame[dimension.output_name] = _materialize_dimension(
            table[dimension.variable],
            dimension,
        )
    for measure in spec.measures:
        if measure.variable is None:
            continue
        if measure.variable not in table.columns:
            raise KeyError(
                f"Reduced benchmark measure variable '{measure.variable}' is missing from {spec.entity} table"
            )
        frame[f"__value__{measure.name}"] = pd.to_numeric(
            table[measure.variable],
            errors="coerce",
        ).fillna(0.0)

    group_columns = [dimension.output_name for dimension in spec.dimensions]
    if not group_columns:
        frame["__all__"] = "all"
        group_columns = ["__all__"]
    grouped = (
        frame.groupby(group_columns, dropna=False, observed=True)["__weight__"]
        .sum()
        .reset_index(name="__group_weight__")
    )
    result = grouped.copy()
    for measure in spec.measures:
        if measure.aggregation == "weighted_count":
            result[measure.name] = grouped["__group_weight__"]
            continue
        weighted_sum = (
            frame.assign(
                __weighted_value__=frame["__weight__"] * frame[f"__value__{measure.name}"]
            )
            .groupby(group_columns, dropna=False, observed=True)["__weighted_value__"]
            .sum()
            .reset_index(name="__weighted_sum__")
        )
        result = result.merge(weighted_sum, on=group_columns, how="left")
        if measure.aggregation == "weighted_sum":
            result[measure.name] = result["__weighted_sum__"].fillna(0.0)
        else:
            denominator = result["__group_weight__"].to_numpy(dtype=float)
            numerator = result["__weighted_sum__"].fillna(0.0).to_numpy(dtype=float)
            result[measure.name] = np.where(
                np.abs(denominator) > 1e-12,
                numerator / denominator,
                0.0,
            )
        result = result.drop(columns=["__weighted_sum__"])
    if "__all__" in result.columns:
        result = result.drop(columns=["__all__"])
    return result, int(len(table))


def _materialize_entity_frame(
    bundle: PolicyEngineUSEntityTableBundle,
    entity: USReducedBenchmarkEntity,
    variables: set[str],
) -> tuple[pd.DataFrame, pd.Series]:
    table, weights = _resolve_entity_table_and_weights(bundle, entity)
    table = table.copy()
    missing = sorted(variable for variable in variables if variable not in table.columns)
    if not missing:
        return table, weights

    household_lookup = bundle.households.set_index("household_id")
    if "household_id" in table.columns:
        for variable in missing:
            if variable in household_lookup.columns:
                table[variable] = table["household_id"].map(household_lookup[variable])
    return table, weights


def _entity_type_for_reduced_entity(entity: USReducedBenchmarkEntity) -> EntityType:
    mapping: dict[USReducedBenchmarkEntity, EntityType] = {
        "household": EntityType.HOUSEHOLD,
        "person": EntityType.PERSON,
        "tax_unit": EntityType.TAX_UNIT,
        "spm_unit": EntityType.SPM_UNIT,
        "family": EntityType.FAMILY,
        "marital_unit": EntityType.TAX_UNIT,
    }
    return mapping[entity]


def _dimension_filters_for_value(
    dimension: USMicroplexReducedDimensionSpec,
    value: Any,
) -> tuple[TargetFilter, ...]:
    if dimension.bins is None:
        filter_value = value
        if dimension.zero_pad is not None:
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(numeric):
                filter_value = int(round(float(numeric)))
        return (
            TargetFilter(
                feature=dimension.variable,
                operator=FilterOperator.EQ,
                value=filter_value,
            ),
        )

    if dimension.bin_labels is None:
        raise ValueError(
            f"Reduced calibration requires explicit bin labels for '{dimension.variable}'"
        )
    if value not in dimension.bin_labels:
        raise ValueError(
            f"Reduced calibration cannot map bucket '{value}' for '{dimension.variable}'"
        )
    index = dimension.bin_labels.index(value)
    lower = dimension.bins[index]
    upper = dimension.bins[index + 1]
    filters = [TargetFilter(dimension.variable, FilterOperator.GTE, lower)]
    if dimension.right:
        filters.append(TargetFilter(dimension.variable, FilterOperator.LTE, upper))
    else:
        filters.append(TargetFilter(dimension.variable, FilterOperator.LT, upper))
    return tuple(filters)


def _infer_reduced_export_maps(
    tables: PolicyEngineUSEntityTableBundle,
) -> dict[str, dict[str, str] | None]:
    return {
        "household_variable_map": _infer_export_map(
            tables.households,
            excluded_columns={"household_id", "household_weight", "weight"},
        ),
        "person_variable_map": _infer_export_map(
            tables.persons,
            excluded_columns={
                "person_id",
                "household_id",
                "weight",
                "tax_unit_id",
                "spm_unit_id",
                "family_id",
                "marital_unit_id",
            },
        ),
        "tax_unit_variable_map": _infer_export_map(
            tables.tax_units,
            excluded_columns={"tax_unit_id", "household_id", "household_weight", "weight"},
        ),
        "spm_unit_variable_map": _infer_export_map(
            tables.spm_units,
            excluded_columns={"spm_unit_id", "household_id", "household_weight", "weight"},
        ),
        "family_variable_map": _infer_export_map(
            tables.families,
            excluded_columns={"family_id", "household_id", "household_weight", "weight"},
        ),
        "marital_unit_variable_map": _infer_export_map(
            tables.marital_units,
            excluded_columns={
                "marital_unit_id",
                "household_id",
                "household_weight",
                "weight",
            },
        ),
    }


def _infer_export_map(
    table: pd.DataFrame | None,
    *,
    excluded_columns: set[str],
) -> dict[str, str] | None:
    if table is None:
        return None
    return {
        column: column
        for column in table.columns
        if column not in excluded_columns
    } or None


def _resolve_entity_table_and_weights(
    bundle: PolicyEngineUSEntityTableBundle,
    entity: USReducedBenchmarkEntity,
) -> tuple[pd.DataFrame, pd.Series]:
    household_weights = bundle.households.set_index("household_id")["household_weight"]
    if entity == "household":
        table = bundle.households.copy()
        return table, pd.to_numeric(table["household_weight"], errors="coerce").fillna(0.0)

    table_map: dict[USReducedBenchmarkEntity, pd.DataFrame | None] = {
        "person": bundle.persons,
        "tax_unit": bundle.tax_units,
        "spm_unit": bundle.spm_units,
        "family": bundle.families,
        "marital_unit": bundle.marital_units,
    }
    table = table_map[entity]
    if table is None:
        raise ValueError(f"Reduced benchmark entity '{entity}' is unavailable")
    table = table.copy()
    if entity == "person" and "weight" in table.columns:
        weights = pd.to_numeric(table["weight"], errors="coerce").fillna(0.0)
    else:
        if "household_id" not in table.columns:
            raise ValueError(
                f"Reduced benchmark entity '{entity}' requires household_id to infer weights"
            )
        weights = pd.to_numeric(
            table["household_id"].map(household_weights),
            errors="coerce",
        ).fillna(0.0)
    return table, weights


def _materialize_dimension(
    values: pd.Series,
    spec: USMicroplexReducedDimensionSpec,
) -> pd.Series:
    series = values.copy()
    if spec.bins is not None:
        series = pd.cut(
            pd.to_numeric(series, errors="coerce"),
            bins=spec.bins,
            labels=spec.bin_labels,
            right=spec.right,
            include_lowest=spec.include_lowest,
        )
    elif spec.zero_pad is not None:
        numeric = pd.to_numeric(series, errors="coerce")
        padded = pd.Series(pd.NA, index=series.index, dtype="object")
        valid = numeric.notna()
        if valid.any():
            padded.loc[valid] = (
                numeric.loc[valid].round().astype(int).astype(str).str.zfill(spec.zero_pad)
            )
        series = padded
    else:
        series = series.astype(object)
    series = series.astype(object)
    if spec.missing_label is not None:
        series = series.where(pd.notna(series), spec.missing_label)
    return series


def _json_compatible_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_compatible_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible_value(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _json_compatible_value(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, pd.Interval):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


__all__ = [
    "DEFAULT_ATOMIC_AGE_BINS",
    "DEFAULT_ATOMIC_AGE_LABELS",
    "USMicroplexReducedCalibrationReport",
    "USMicroplexReducedBenchmarkHarnessConfig",
    "USMicroplexReducedBenchmarkHarnessResult",
    "USMicroplexReducedBenchmarkReport",
    "USMicroplexReducedBenchmarkSpec",
    "USMicroplexReducedDimensionSpec",
    "USMicroplexReducedMeasureSpec",
    "calibrate_and_evaluate_us_reduced_benchmarks",
    "default_us_atomic_rung0_benchmarks",
    "default_us_atomic_rung1_benchmarks",
    "default_us_atomic_rung2_calibration",
    "evaluate_us_reduced_benchmark",
    "reduced_benchmark_to_calibration_targets",
    "run_us_microplex_reduced_benchmark_harness",
]
