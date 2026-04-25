"""Calibration harness for PE parity experiments over canonical target specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from microplex.core import EntityType
from microplex.targets import (
    FilterOperator,
    TargetAggregation,
    TargetFilter,
    TargetProvider,
    TargetQuery,
    TargetSpec,
)

from microplex_us.target_registry import (
    TargetCategory,
    TargetLevel,
    TargetRegistry,
    get_registry,
    target_available_in_cps,
    target_category,
    target_group_name,
    target_level,
    target_requires_imputation,
)


@dataclass
class CalibrationResult:
    """Result of a calibration run."""

    weights: np.ndarray
    targets_used: list[str]
    errors: dict[str, float]
    iterations: int
    converged: bool
    weight_stats: dict[str, float]

    @property
    def mean_error(self) -> float:
        return float(np.mean(list(self.errors.values()))) if self.errors else 0.0

    @property
    def max_error(self) -> float:
        return max(self.errors.values()) if self.errors else 0.0

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Calibration Result:",
            f"  Targets: {len(self.targets_used)}",
            f"  Converged: {self.converged} ({self.iterations} iterations)",
            f"  Mean error: {self.mean_error:.2f}%",
            f"  Max error: {self.max_error:.2f}%",
            f"  Weight CV: {self.weight_stats.get('cv', 0):.2f}",
        ]
        return "\n".join(lines)


class CalibrationHarness:
    """Harness for calibration experiments over one entity frame at a time."""

    def __init__(
        self,
        registry: TargetRegistry | None = None,
        *,
        target_provider: TargetProvider | None = None,
    ):
        if target_provider is None:
            self.registry = registry or get_registry()
            self.target_provider = self.registry
        else:
            self.registry = registry
            self.target_provider = target_provider
        self._results: dict[str, CalibrationResult] = {}

    def select_targets(
        self,
        *,
        categories: list[TargetCategory] | None = None,
        levels: list[TargetLevel] | None = None,
        groups: list[str] | None = None,
        only_available: bool = False,
        entity: EntityType | str | None = None,
        period: int | str | None = None,
        provider_filters: dict[str, Any] | None = None,
    ) -> list[TargetSpec]:
        """Select canonical targets from the configured provider."""
        query = TargetQuery(
            period=period,
            entity=entity,
            provider_filters=dict(provider_filters or {}),
        )
        targets = self.target_provider.load_target_set(query).targets
        return [
            target
            for target in targets
            if _matches_us_target_filters(
                target,
                categories=categories,
                levels=levels,
                groups=groups,
                only_available=only_available,
            )
        ]

    def get_target_vector(
        self,
        df: pd.DataFrame,
        targets: list[TargetSpec],
        *,
        entity: EntityType | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build a design matrix and target vector from canonical targets."""
        resolved_entity = _resolve_entity(entity)
        n_rows = len(df)
        design_rows: list[np.ndarray] = []
        target_values: list[float] = []
        target_names: list[str] = []

        for spec in targets:
            if resolved_entity is not None and spec.entity is not resolved_entity:
                continue

            if any(feature not in df.columns for feature in spec.required_features):
                continue

            row = _build_constraint_row(df, spec)
            design_rows.append(row)
            target_values.append(spec.value)
            target_names.append(spec.name)

        design_matrix = (
            np.column_stack(design_rows) if design_rows else np.zeros((n_rows, 0))
        )
        target_vector = np.array(target_values, dtype=float)
        return design_matrix, target_vector, target_names

    def calibrate(
        self,
        df: pd.DataFrame,
        targets: list[TargetSpec],
        weight_col: str = "weight",
        *,
        entity: EntityType | str | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        bounds: tuple[float, float] = (0.01, 100.0),
        verbose: bool = True,
    ) -> CalibrationResult:
        """Run IPF calibration against canonical targets for one entity frame."""
        if weight_col in df.columns:
            weights = df[weight_col].to_numpy(dtype=float, copy=True)
        else:
            weights = np.ones(len(df), dtype=float)

        design_matrix, target_vec, names = self.get_target_vector(
            df,
            targets,
            entity=entity,
        )
        n_samples, n_targets = design_matrix.shape

        if verbose:
            print(f"Calibrating {n_samples:,} samples to {n_targets} targets")

        if n_targets == 0:
            return CalibrationResult(
                weights=weights,
                targets_used=[],
                errors={},
                iterations=0,
                converged=True,
                weight_stats=_weight_stats(weights),
            )

        converged = False
        for iteration in range(max_iter):
            old_weights = weights.copy()

            for target_index in range(n_targets):
                if target_vec[target_index] == 0:
                    continue

                current = np.sum(weights * design_matrix[:, target_index])
                if current <= 0:
                    continue

                factor = target_vec[target_index] / current
                factor = np.clip(factor, bounds[0], bounds[1])
                mask = design_matrix[:, target_index] > 0
                weights[mask] *= factor

            max_change = np.max(np.abs(weights - old_weights) / (old_weights + 1e-10))
            if max_change < tol:
                converged = True
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

        errors: dict[str, float] = {}
        if verbose:
            print(f"\n{'Target':<40} {'Computed':>15} {'Target':>15} {'Error':>10}")
            print("-" * 85)

        for target_index, name in enumerate(names):
            computed = float(np.sum(weights * design_matrix[:, target_index]))
            target = float(target_vec[target_index])
            if target != 0:
                error = abs(computed - target) / abs(target) * 100
            else:
                error = 0 if computed == 0 else 100
            errors[name] = min(error, 100.0)

            if verbose:
                if abs(target) > 1e9:
                    computed_str = f"${computed / 1e9:.1f}B"
                    target_str = f"${target / 1e9:.1f}B"
                elif abs(target) > 1e6:
                    computed_str = f"{computed / 1e6:.1f}M"
                    target_str = f"{target / 1e6:.1f}M"
                else:
                    computed_str = f"{computed:,.0f}"
                    target_str = f"{target:,.0f}"
                print(f"{name:<40} {computed_str:>15} {target_str:>15} {error:>9.1f}%")

        return CalibrationResult(
            weights=weights,
            targets_used=names,
            errors=errors,
            iterations=iteration + 1,
            converged=converged,
            weight_stats=_weight_stats(weights),
        )

    def run_experiment(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        categories: list[TargetCategory] | None = None,
        levels: list[TargetLevel] | None = None,
        groups: list[str] | None = None,
        only_available: bool = False,
        entity: EntityType | str | None = None,
        period: int | str | None = None,
        provider_filters: dict[str, Any] | None = None,
        **calibrate_kwargs,
    ) -> CalibrationResult:
        """Run a calibration experiment over a filtered target subset."""
        selected = self.select_targets(
            categories=categories,
            levels=levels,
            groups=groups,
            only_available=only_available,
            entity=entity,
            period=period,
            provider_filters=provider_filters,
        )
        selected = [
            target
            for target in selected
            if not (
                target.value == 0 and target.aggregation is not TargetAggregation.COUNT
            )
        ]

        print(f"\n=== Experiment: {name} ===")
        print(f"Selected {len(selected)} targets")

        result = self.calibrate(
            df,
            selected,
            entity=entity,
            **calibrate_kwargs,
        )
        self._results[name] = result
        return result

    def compare_experiments(self) -> pd.DataFrame:
        """Compare results across experiments."""
        records = []
        for name, result in self._results.items():
            records.append(
                {
                    "experiment": name,
                    "n_targets": len(result.targets_used),
                    "converged": result.converged,
                    "iterations": result.iterations,
                    "mean_error": result.mean_error,
                    "max_error": result.max_error,
                    "weight_cv": result.weight_stats["cv"],
                    "weight_max": result.weight_stats["max"],
                    "zero_weights": result.weight_stats["zero_count"],
                }
            )
        return pd.DataFrame(records)

    def print_target_coverage(
        self,
        df: pd.DataFrame,
        *,
        entity: EntityType | str | None = None,
    ) -> None:
        """Print which canonical targets can be computed from the given frame."""
        print("=" * 70)
        print("TARGET COVERAGE ANALYSIS")
        print("=" * 70)

        all_targets = self.select_targets(entity=entity)
        columns = set(df.columns)

        available: list[TargetSpec] = []
        missing_column: list[TargetSpec] = []
        needs_imputation: list[TargetSpec] = []

        for target in all_targets:
            if any(feature not in columns for feature in target.required_features):
                missing_column.append(target)
            elif target_requires_imputation(target):
                needs_imputation.append(target)
            else:
                available.append(target)

        print(f"\nAvailable ({len(available)} targets):")
        for category in TargetCategory:
            count = sum(1 for target in available if target_category(target) is category)
            if count:
                print(f"  {category.value}: {count}")

        print(f"\nMissing column ({len(missing_column)} targets):")
        missing_features = {
            feature
            for target in missing_column
            for feature in target.required_features
            if feature not in columns
        }
        for feature in sorted(missing_features):
            count = sum(1 for target in missing_column if feature in target.required_features)
            print(f"  {feature}: {count} targets")

        print(f"\nRequires imputation ({len(needs_imputation)} targets):")
        for category in TargetCategory:
            count = sum(
                1 for target in needs_imputation if target_category(target) is category
            )
            if count:
                print(f"  {category.value}: {count}")


def run_pe_parity_suite(
    df: pd.DataFrame,
    weight_col: str = "weight",
    *,
    entity: EntityType | str = EntityType.PERSON,
) -> pd.DataFrame:
    """Run the full PE parity calibration suite for a single entity frame."""
    harness = CalibrationHarness()
    harness.print_target_coverage(df, entity=entity)

    print("\n" + "=" * 70)
    print("RUNNING CALIBRATION EXPERIMENTS")
    print("=" * 70)

    harness.run_experiment(
        df,
        "states_only",
        groups=["state_population"],
        entity=entity,
        weight_col=weight_col,
        verbose=True,
    )
    harness.run_experiment(
        df,
        "income_available",
        categories=[TargetCategory.INCOME],
        only_available=True,
        entity=entity,
        weight_col=weight_col,
        verbose=True,
    )
    harness.run_experiment(
        df,
        "benefits_only",
        groups=["benefit_programs"],
        entity=entity,
        weight_col=weight_col,
        verbose=True,
    )
    harness.run_experiment(
        df,
        "full_available",
        groups=["state_population", "irs_soi_income", "benefit_programs"],
        only_available=True,
        entity=entity,
        weight_col=weight_col,
        verbose=True,
    )
    harness.run_experiment(
        df,
        "all_targets",
        only_available=False,
        entity=entity,
        weight_col=weight_col,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)

    comparison = harness.compare_experiments()
    print(comparison.to_string(index=False))
    return comparison


def _resolve_entity(entity: EntityType | str | None) -> EntityType | None:
    if entity is None or isinstance(entity, EntityType):
        return entity
    return EntityType(entity)


def _weight_stats(weights: np.ndarray) -> dict[str, float]:
    mean_weight = float(np.mean(weights)) if len(weights) else 0.0
    std_weight = float(np.std(weights)) if len(weights) else 0.0
    return {
        "mean": mean_weight,
        "std": std_weight,
        "cv": std_weight / mean_weight if mean_weight > 0 else 0.0,
        "min": float(np.min(weights)) if len(weights) else 0.0,
        "max": float(np.max(weights)) if len(weights) else 0.0,
        "zero_count": int(np.sum(weights == 0)),
    }


def _matches_us_target_filters(
    target: TargetSpec,
    *,
    categories: list[TargetCategory] | None = None,
    levels: list[TargetLevel] | None = None,
    groups: list[str] | None = None,
    only_available: bool = False,
) -> bool:
    if categories and target_category(target) not in categories:
        return False
    if levels and target_level(target) not in levels:
        return False
    if groups and target_group_name(target) not in groups:
        return False
    if only_available and not target_available_in_cps(target):
        return False
    return True


def _build_constraint_row(df: pd.DataFrame, spec: TargetSpec) -> np.ndarray:
    if spec.aggregation is TargetAggregation.MEAN:
        raise NotImplementedError("Mean targets are not supported by this harness")

    mask = np.ones(len(df), dtype=bool)
    for target_filter in spec.filters:
        mask &= _evaluate_filter(df[target_filter.feature], target_filter)

    if spec.aggregation is TargetAggregation.COUNT:
        return mask.astype(float)

    if spec.measure is None:
        raise ValueError(f"Sum target {spec.name} is missing a measure")

    values = df[spec.measure].fillna(0).to_numpy(dtype=float, copy=False)
    return mask.astype(float) * values


def _evaluate_filter(series: pd.Series, target_filter: TargetFilter) -> np.ndarray:
    operator = target_filter.operator
    value = target_filter.value

    if operator is FilterOperator.EQ:
        return (series == value).to_numpy(dtype=bool, copy=False)
    if operator is FilterOperator.NE:
        return (series != value).to_numpy(dtype=bool, copy=False)
    if operator is FilterOperator.GT:
        return (series > value).to_numpy(dtype=bool, copy=False)
    if operator is FilterOperator.GTE:
        return (series >= value).to_numpy(dtype=bool, copy=False)
    if operator is FilterOperator.LT:
        return (series < value).to_numpy(dtype=bool, copy=False)
    if operator is FilterOperator.LTE:
        return (series <= value).to_numpy(dtype=bool, copy=False)
    if operator is FilterOperator.IN:
        return series.isin(value).to_numpy(dtype=bool, copy=False)
    if operator is FilterOperator.NOT_IN:
        return (~series.isin(value)).to_numpy(dtype=bool, copy=False)
    raise ValueError(f"Unsupported filter operator: {operator}")
