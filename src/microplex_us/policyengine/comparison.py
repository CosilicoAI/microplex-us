"""Library helpers for scoring PE-US-compatible populations against target slices."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from microplex.targets import (
    TargetAggregation,
    TargetProvider,
    TargetQuery,
    TargetSet,
    TargetSpec,
)

from microplex_us.policyengine.us import (
    PolicyEngineUSEntityTableBundle,
    PolicyEngineUSVariableBinding,
    compile_supported_policyengine_us_household_linear_constraints,
    filter_supported_policyengine_us_targets,
    infer_policyengine_us_variable_bindings,
    load_policyengine_us_entity_tables,
    materialize_policyengine_us_variables_safely,
)


class PolicyEngineUSMaterializationError(RuntimeError):
    """Raised when PE-US-derived features cannot be materialized for scoring."""

    def __init__(self, label: str, failed_variables: dict[str, str]):
        self.label = label
        self.failed_variables = dict(failed_variables)
        details = ", ".join(
            f"{variable} ({reason})"
            for variable, reason in sorted(self.failed_variables.items())
        )
        super().__init__(
            f"{label} could not materialize required PolicyEngine US variables: {details}"
        )


def _freeze_cache_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            sorted((str(key), _freeze_cache_value(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple, set)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def _provider_cache_key(provider: TargetProvider) -> tuple[str, int]:
    return (f"{provider.__class__.__module__}.{provider.__class__.__qualname__}", id(provider))


def _query_cache_key(query: TargetQuery | None) -> tuple[Any, ...]:
    if query is None:
        return ("__none__",)
    return (
        query.period,
        query.entity.value if query.entity is not None else None,
        tuple(query.names),
        _freeze_cache_value(query.metadata_filters),
        _freeze_cache_value(query.provider_filters),
    )


def _target_cache_key(target: TargetSpec) -> tuple[Any, ...]:
    return (
        target.name,
        target.entity.value,
        target.period,
        target.measure,
        target.aggregation.value,
        tuple(
            (
                target_filter.feature,
                target_filter.operator.value,
                _freeze_cache_value(target_filter.value),
            )
            for target_filter in target.filters
        ),
        float(target.value),
        target.tolerance,
        target.source,
        target.units,
        target.description,
        _freeze_cache_value(target.metadata),
    )


def _target_set_cache_key(
    targets: TargetSet | list[TargetSpec] | tuple[TargetSpec, ...],
) -> tuple[Any, ...]:
    return tuple(_target_cache_key(target) for target in _normalize_target_list(targets))


@dataclass
class PolicyEngineUSComparisonCache:
    """In-memory cache for immutable PE-US comparison inputs."""

    target_sets: dict[tuple[Any, ...], TargetSet] = field(default_factory=dict)
    baseline_tables: dict[tuple[Any, ...], PolicyEngineUSEntityTableBundle] = field(
        default_factory=dict
    )
    baseline_reports: dict[tuple[Any, ...], PolicyEngineUSTargetEvaluationReport] = field(
        default_factory=dict
    )

    def load_target_set(
        self,
        provider: TargetProvider,
        query: TargetQuery | None,
    ) -> TargetSet:
        key = (_provider_cache_key(provider), _query_cache_key(query))
        target_set = self.target_sets.get(key)
        if target_set is None:
            target_set = provider.load_target_set(query)
            self.target_sets[key] = target_set
        return target_set

    def load_baseline_tables(
        self,
        baseline_dataset: str | Path | Any,
        *,
        period: int | str,
    ) -> PolicyEngineUSEntityTableBundle:
        dataset_key = str(baseline_dataset) if isinstance(baseline_dataset, Path) else baseline_dataset
        key = (dataset_key, period)
        tables = self.baseline_tables.get(key)
        if tables is None:
            tables = load_policyengine_us_entity_tables(
                baseline_dataset,
                period=period,
            )
            self.baseline_tables[key] = tables
        return tables

    def load_baseline_report(
        self,
        *,
        target_set: TargetSet,
        baseline_dataset: str | Path | Any,
        period: int | str,
        dataset_year: int | None,
        simulation_cls: Any | None,
        baseline_label: str,
        strict_materialization: bool,
    ) -> PolicyEngineUSTargetEvaluationReport:
        dataset_key = str(baseline_dataset) if isinstance(baseline_dataset, Path) else baseline_dataset
        key = (
            dataset_key,
            period,
            dataset_year,
            _freeze_cache_value(simulation_cls),
            baseline_label,
            strict_materialization,
            _target_set_cache_key(target_set),
        )
        report = self.baseline_reports.get(key)
        if report is None:
            baseline_tables = self.load_baseline_tables(
                baseline_dataset,
                period=period,
            )
            report = evaluate_policyengine_us_target_set(
                baseline_tables,
                target_set,
                period=period,
                dataset_year=dataset_year,
                simulation_cls=simulation_cls,
                label=baseline_label,
                strict_materialization=strict_materialization,
            )
            self.baseline_reports[key] = report
        return report


@dataclass(frozen=True)
class PolicyEngineUSTargetEvaluation:
    """Observed value and error for a single canonical target."""

    target: TargetSpec
    actual_value: float

    @property
    def absolute_error(self) -> float:
        return abs(self.actual_value - float(self.target.value))

    @property
    def relative_error(self) -> float | None:
        target_value = float(self.target.value)
        if target_value == 0.0:
            return None
        return (self.actual_value - target_value) / abs(target_value)


@dataclass
class PolicyEngineUSTargetEvaluationReport:
    """Summary of target-slice fit for one candidate dataset/bundle."""

    label: str
    period: int | str
    evaluations: list[PolicyEngineUSTargetEvaluation] = field(default_factory=list)
    unsupported_targets: list[TargetSpec] = field(default_factory=list)
    materialized_variables: tuple[str, ...] = ()
    materialization_failures: dict[str, str] = field(default_factory=dict)

    @property
    def mean_abs_relative_error(self) -> float | None:
        errors = [
            abs(evaluation.relative_error)
            for evaluation in self.evaluations
            if evaluation.relative_error is not None
        ]
        if not errors:
            return None
        return float(np.mean(errors))

    @property
    def max_abs_relative_error(self) -> float | None:
        errors = [
            abs(evaluation.relative_error)
            for evaluation in self.evaluations
            if evaluation.relative_error is not None
        ]
        if not errors:
            return None
        return float(max(errors))

    @property
    def supported_target_count(self) -> int:
        return len(self.evaluations)


@dataclass
class PolicyEngineUSTargetComparisonReport:
    """Side-by-side fit reports for a Microplex candidate and a PE baseline."""

    candidate: PolicyEngineUSTargetEvaluationReport
    baseline: PolicyEngineUSTargetEvaluationReport | None = None

    @property
    def mean_abs_relative_error_delta(self) -> float | None:
        if self.baseline is None:
            return None
        candidate_error = self.candidate.mean_abs_relative_error
        baseline_error = self.baseline.mean_abs_relative_error
        if candidate_error is None or baseline_error is None:
            return None
        return candidate_error - baseline_error


def evaluate_policyengine_us_target_set(
    tables: PolicyEngineUSEntityTableBundle,
    targets: TargetSet | list[TargetSpec] | tuple[TargetSpec, ...],
    *,
    period: int | str,
    dataset_year: int | None = None,
    simulation_cls: Any | None = None,
    label: str = "candidate",
    strict_materialization: bool = False,
) -> PolicyEngineUSTargetEvaluationReport:
    """Evaluate canonical targets against a PE-US-style entity-table bundle."""
    target_list = _normalize_target_list(targets)
    working_tables = tables
    bindings = infer_policyengine_us_variable_bindings(working_tables)
    materialization_result = materialize_policyengine_us_variables_safely(
        working_tables,
        variables=tuple(
            feature
            for target in target_list
            for feature in target.required_features
            if feature not in bindings
        ),
        period=period,
        dataset_year=dataset_year,
        simulation_cls=simulation_cls,
    )
    working_tables = materialization_result.tables
    bindings = {
        **bindings,
        **materialization_result.bindings,
    }
    materialized_variables = materialization_result.materialized_variables
    materialization_failures = materialization_result.failed_variables
    if strict_materialization and materialization_failures:
        raise PolicyEngineUSMaterializationError(label, materialization_failures)

    supported_targets = filter_supported_policyengine_us_targets(
        target_list,
        working_tables,
        bindings,
    )
    supported_target_keys = {
        _target_cache_key(target)
        for target in supported_targets
    }
    unsupported_targets = [
        target
        for target in target_list
        if _target_cache_key(target) not in supported_target_keys
    ]
    household_weights = _household_weights(working_tables)
    linear_targets = [
        target
        for target in supported_targets
        if target.aggregation is not TargetAggregation.MEAN
    ]
    mean_targets = [
        target for target in supported_targets if target.aggregation is TargetAggregation.MEAN
    ]
    linear_targets, compile_unsupported_targets, constraints = (
        compile_supported_policyengine_us_household_linear_constraints(
            linear_targets,
            working_tables,
            variable_bindings=bindings,
        )
    )
    unsupported_targets.extend(compile_unsupported_targets)
    evaluations: list[PolicyEngineUSTargetEvaluation] = []
    for target, constraint in zip(linear_targets, constraints, strict=True):
        evaluations.append(
            PolicyEngineUSTargetEvaluation(
                target=target,
                actual_value=float(np.dot(household_weights, constraint.coefficients)),
            )
        )
    for target in mean_targets:
        try:
            actual_value = _evaluate_target_value(
                target,
                tables=working_tables,
                bindings=bindings,
                household_weights=household_weights,
            )
        except ValueError as error:
            if "Cross-entity constraints are only supported" in str(error):
                unsupported_targets.append(target)
                continue
            raise
        evaluations.append(
            PolicyEngineUSTargetEvaluation(
                target=target,
                actual_value=actual_value,
            )
        )

    return PolicyEngineUSTargetEvaluationReport(
        label=label,
        period=period,
        evaluations=evaluations,
        unsupported_targets=unsupported_targets,
        materialized_variables=tuple(materialized_variables),
        materialization_failures=materialization_failures,
    )


def slice_policyengine_us_target_evaluation_report(
    report: PolicyEngineUSTargetEvaluationReport,
    targets: TargetSet | list[TargetSpec] | tuple[TargetSpec, ...],
) -> PolicyEngineUSTargetEvaluationReport:
    """Project a union target-evaluation report down to one target subset."""
    target_list = _normalize_target_list(targets)
    evaluations_by_name = {
        evaluation.target.name: evaluation for evaluation in report.evaluations
    }
    unsupported_by_name = {
        target.name: target for target in report.unsupported_targets
    }
    evaluations: list[PolicyEngineUSTargetEvaluation] = []
    unsupported_targets: list[TargetSpec] = []
    for target in target_list:
        evaluation = evaluations_by_name.get(target.name)
        if evaluation is not None:
            evaluations.append(evaluation)
            continue
        unsupported_targets.append(unsupported_by_name.get(target.name, target))
    return PolicyEngineUSTargetEvaluationReport(
        label=report.label,
        period=report.period,
        evaluations=evaluations,
        unsupported_targets=unsupported_targets,
        materialized_variables=report.materialized_variables,
        materialization_failures=dict(report.materialization_failures),
    )


def evaluate_policyengine_us_target_sets(
    tables: PolicyEngineUSEntityTableBundle,
    target_sets: dict[str, TargetSet],
    *,
    period: int | str,
    dataset_year: int | None = None,
    simulation_cls: Any | None = None,
    label: str = "candidate",
    strict_materialization: bool = False,
) -> dict[str, PolicyEngineUSTargetEvaluationReport]:
    """Evaluate the union of multiple target sets once, then slice the report back out."""
    union_targets: list[TargetSpec] = []
    seen_targets: dict[str, tuple[Any, ...]] = {}
    for target_set in target_sets.values():
        for target in _normalize_target_list(target_set):
            target_key = _target_cache_key(target)
            existing_key = seen_targets.get(target.name)
            if existing_key is None:
                seen_targets[target.name] = target_key
                union_targets.append(target)
                continue
            if existing_key != target_key:
                raise ValueError(
                    "PolicyEngine US target-set union encountered conflicting "
                    f"definitions for target '{target.name}'"
                )
    union_report = evaluate_policyengine_us_target_set(
        tables,
        union_targets,
        period=period,
        dataset_year=dataset_year,
        simulation_cls=simulation_cls,
        label=label,
        strict_materialization=strict_materialization,
    )
    return {
        name: slice_policyengine_us_target_evaluation_report(union_report, target_set)
        for name, target_set in target_sets.items()
    }


def evaluate_policyengine_us_target_query(
    tables: PolicyEngineUSEntityTableBundle,
    provider: TargetProvider,
    query: TargetQuery | None = None,
    *,
    dataset_year: int | None = None,
    simulation_cls: Any | None = None,
    label: str = "candidate",
    strict_materialization: bool = False,
) -> PolicyEngineUSTargetEvaluationReport:
    """Load canonical targets from a provider and evaluate them against tables."""
    target_set = provider.load_target_set(query)
    period = query.period if query is not None else 2024
    return evaluate_policyengine_us_target_set(
        tables,
        target_set,
        period=period,
        dataset_year=dataset_year,
        simulation_cls=simulation_cls,
        label=label,
        strict_materialization=strict_materialization,
    )


def compare_policyengine_us_target_query_to_baseline(
    candidate_tables: PolicyEngineUSEntityTableBundle,
    provider: TargetProvider,
    query: TargetQuery | None,
    *,
    baseline_dataset: str | Any,
    dataset_year: int | None = None,
    simulation_cls: Any | None = None,
    candidate_label: str = "microplex",
    baseline_label: str = "policyengine_baseline",
    strict_materialization: bool = True,
    cache: PolicyEngineUSComparisonCache | None = None,
) -> PolicyEngineUSTargetComparisonReport:
    """Compare a candidate PE-US bundle to a baseline PE dataset on one target slice."""
    target_set = (
        cache.load_target_set(provider, query)
        if cache is not None
        else provider.load_target_set(query)
    )
    period = query.period if query is not None else 2024
    baseline_report = (
        cache.load_baseline_report(
            target_set=target_set,
            baseline_dataset=baseline_dataset,
            period=period,
            dataset_year=dataset_year,
            simulation_cls=simulation_cls,
            baseline_label=baseline_label,
            strict_materialization=strict_materialization,
        )
        if cache is not None
        else evaluate_policyengine_us_target_set(
            load_policyengine_us_entity_tables(
                baseline_dataset,
                period=period,
            ),
            target_set,
            period=period,
            dataset_year=dataset_year,
            simulation_cls=simulation_cls,
            label=baseline_label,
            strict_materialization=strict_materialization,
        )
    )
    candidate_report = evaluate_policyengine_us_target_set(
        candidate_tables,
        target_set,
        period=period,
        dataset_year=dataset_year,
        simulation_cls=simulation_cls,
        label=candidate_label,
        strict_materialization=strict_materialization,
    )
    return PolicyEngineUSTargetComparisonReport(
        candidate=candidate_report,
        baseline=baseline_report,
    )


def _normalize_target_list(
    targets: TargetSet | list[TargetSpec] | tuple[TargetSpec, ...],
) -> list[TargetSpec]:
    if isinstance(targets, TargetSet):
        return list(targets.targets)
    return list(targets)


def _household_weights(tables: PolicyEngineUSEntityTableBundle) -> np.ndarray:
    households = tables.households
    if "household_weight" in households.columns:
        values = households["household_weight"]
    elif "weight" in households.columns:
        values = households["weight"]
    else:
        raise ValueError(
            "Household table must contain 'household_weight' or 'weight' for evaluation"
        )
    return np.asarray(values, dtype=float)


def _evaluate_target_value(
    target: TargetSpec,
    *,
    tables: PolicyEngineUSEntityTableBundle,
    bindings: dict[str, PolicyEngineUSVariableBinding],
    household_weights: np.ndarray,
) -> float:
    if target.aggregation is TargetAggregation.MEAN:
        numerator_target = TargetSpec(
            name=f"{target.name}__numerator",
            entity=target.entity,
            value=0.0,
            period=target.period,
            measure=target.measure,
            aggregation=TargetAggregation.SUM,
            filters=target.filters,
            tolerance=target.tolerance,
            source=target.source,
            units=target.units,
            description=target.description,
            metadata=dict(target.metadata),
        )
        denominator_target = TargetSpec(
            name=f"{target.name}__denominator",
            entity=target.entity,
            value=0.0,
            period=target.period,
            aggregation=TargetAggregation.COUNT,
            filters=target.filters,
            tolerance=target.tolerance,
            source=target.source,
            units=target.units,
            description=target.description,
            metadata=dict(target.metadata),
        )
        numerator = _evaluate_linear_target(
            numerator_target,
            tables=tables,
            bindings=bindings,
            household_weights=household_weights,
        )
        denominator = _evaluate_linear_target(
            denominator_target,
            tables=tables,
            bindings=bindings,
            household_weights=household_weights,
        )
        if denominator == 0.0:
            return float("nan")
        return numerator / denominator

    return _evaluate_linear_target(
        target,
        tables=tables,
        bindings=bindings,
        household_weights=household_weights,
    )


def _evaluate_linear_target(
    target: TargetSpec,
    *,
    tables: PolicyEngineUSEntityTableBundle,
    bindings: dict[str, PolicyEngineUSVariableBinding],
    household_weights: np.ndarray,
) -> float:
    supported_targets, unsupported_targets, constraints = (
        compile_supported_policyengine_us_household_linear_constraints(
            [target],
            tables,
            variable_bindings=bindings,
        )
    )
    if unsupported_targets or not supported_targets:
        raise ValueError(
            "Cross-entity constraints are only supported against household targets "
            "or household metadata"
        )
    constraint = constraints[0]
    return float(np.dot(household_weights, constraint.coefficients))
