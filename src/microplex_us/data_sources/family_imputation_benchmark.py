"""Holdout benchmarks for decomposable-family imputers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from microplex_us.data_sources.share_imputation import (
    fit_grouped_share_model,
    predict_grouped_component_shares,
)


@dataclass(frozen=True)
class DecomposableFamilyBenchmarkSpec:
    """Describe a decomposable-family holdout benchmark."""

    total_column: str
    component_columns: tuple[str, ...]
    grouped_feature_sets: tuple[tuple[str, ...], ...]
    qrf_condition_vars: tuple[str, ...]
    implicit_component_column: str | None = None
    weight_column: str = "weight"
    group_eval_columns: tuple[str, ...] = ()
    qrf_n_estimators: int = 100

    @property
    def explicit_component_columns(self) -> tuple[str, ...]:
        if self.implicit_component_column is None:
            return self.component_columns
        return tuple(
            column
            for column in self.component_columns
            if column != self.implicit_component_column
        )


@dataclass(frozen=True)
class FamilyImputationMethodBenchmark:
    """Aggregate benchmark metrics for one imputation method."""

    component_total_relative_error: dict[str, float]
    component_support_relative_error: dict[str, float]
    component_group_sum_mare: dict[str, float]
    mean_component_total_relative_error: float
    mean_component_support_relative_error: float
    mean_component_group_sum_mare: float | None


@dataclass(frozen=True)
class FamilyImputationBenchmarkResult:
    """Comparable holdout metrics for a decomposable family."""

    spec: DecomposableFamilyBenchmarkSpec
    row_count: int
    train_row_count: int
    test_row_count: int
    methods: dict[str, FamilyImputationMethodBenchmark]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spec"] = asdict(self.spec)
        return payload


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)


def _build_positive_family_frame(
    reference: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
) -> pd.DataFrame:
    required_columns = {
        spec.total_column,
        spec.weight_column,
        *spec.component_columns,
        *spec.qrf_condition_vars,
        *spec.group_eval_columns,
    }
    for feature_set in spec.grouped_feature_sets:
        required_columns.update(feature_set)
    frame = reference.loc[:, list(required_columns)].copy()
    frame[spec.total_column] = _numeric_series(frame, spec.total_column)
    frame[spec.weight_column] = _numeric_series(frame, spec.weight_column).clip(lower=0.0)
    for column in spec.component_columns:
        frame[column] = _numeric_series(frame, column).clip(lower=0.0)
    positive_mask = frame[spec.total_column] > 0.0
    frame = frame.loc[positive_mask].copy()
    if frame.empty:
        raise ValueError("Benchmark requires at least one positive family-total row")
    return frame.reset_index(drop=True)


def _split_train_test(
    frame: pd.DataFrame,
    *,
    train_frac: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1")
    if len(frame) < 2:
        raise ValueError("Benchmark requires at least two rows to create a holdout split")
    rng = np.random.default_rng(random_seed)
    mask = rng.random(len(frame)) < train_frac
    if mask.all():
        mask[rng.integers(len(frame))] = False
    if (~mask).all():
        mask[rng.integers(len(frame))] = True
    return frame.loc[mask].reset_index(drop=True), frame.loc[~mask].reset_index(drop=True)


def _weighted_component_totals(
    frame: pd.DataFrame,
    *,
    component_columns: tuple[str, ...],
    weight_column: str,
) -> dict[str, float]:
    weights = _numeric_series(frame, weight_column)
    return {
        column: float((_numeric_series(frame, column) * weights).sum())
        for column in component_columns
    }


def _weighted_component_support(
    frame: pd.DataFrame,
    *,
    component_columns: tuple[str, ...],
    weight_column: str,
) -> dict[str, float]:
    weights = _numeric_series(frame, weight_column)
    return {
        column: float(weights[_numeric_series(frame, column) > 0.0].sum())
        for column in component_columns
    }


def _relative_error(candidate: float, baseline: float) -> float:
    baseline_abs = abs(float(baseline))
    if baseline_abs <= 1e-9:
        return 0.0 if abs(float(candidate)) <= 1e-9 else 1.0
    return float(abs(float(candidate) - float(baseline)) / baseline_abs)


def _component_group_sum_mare(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    component_columns: tuple[str, ...],
    weight_column: str,
    group_columns: tuple[str, ...],
) -> dict[str, float]:
    if not group_columns:
        return {}
    actual_work = actual.loc[:, [*group_columns, weight_column, *component_columns]].copy()
    predicted_work = predicted.loc[:, [*group_columns, weight_column, *component_columns]].copy()
    actual_work[weight_column] = _numeric_series(actual_work, weight_column)
    predicted_work[weight_column] = _numeric_series(predicted_work, weight_column)

    results: dict[str, float] = {}
    for column in component_columns:
        actual_work[f"__{column}_weighted"] = (
            _numeric_series(actual_work, column) * actual_work[weight_column]
        )
        predicted_work[f"__{column}_weighted"] = (
            _numeric_series(predicted_work, column) * predicted_work[weight_column]
        )
        actual_grouped = (
            actual_work.groupby(list(group_columns), dropna=False, observed=False)[
                f"__{column}_weighted"
            ]
            .sum()
            .reset_index(name="actual")
        )
        predicted_grouped = (
            predicted_work.groupby(list(group_columns), dropna=False, observed=False)[
                f"__{column}_weighted"
            ]
            .sum()
            .reset_index(name="predicted")
        )
        merged = actual_grouped.merge(
            predicted_grouped,
            on=list(group_columns),
            how="outer",
            sort=False,
        ).fillna(0.0)
        if merged.empty:
            results[column] = 0.0
            continue
        errors = [
            _relative_error(row.predicted, row.actual)
            for row in merged.itertuples(index=False)
        ]
        results[column] = float(np.mean(errors)) if errors else 0.0
    return results


def _overall_component_shares(
    train_frame: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
) -> dict[str, float]:
    totals = _weighted_component_totals(
        train_frame,
        component_columns=spec.component_columns,
        weight_column=spec.weight_column,
    )
    total_sum = sum(totals.values())
    if total_sum <= 1e-9:
        uniform = 1.0 / len(spec.component_columns)
        return {column: uniform for column in spec.component_columns}
    return {column: value / total_sum for column, value in totals.items()}


def reconcile_component_predictions_to_total(
    predictions: pd.DataFrame,
    *,
    family_total: pd.Series,
    component_columns: tuple[str, ...],
    fallback_shares: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Project component predictions onto the observed family total."""

    result = pd.DataFrame(index=predictions.index)
    total = pd.to_numeric(family_total, errors="coerce").fillna(0.0).clip(lower=0.0)
    fallback = fallback_shares or {
        column: 1.0 / len(component_columns)
        for column in component_columns
    }
    for column in component_columns:
        result[column] = _numeric_series(predictions, column).clip(lower=0.0)

    positive_total = total > 0.0
    row_sum = result.loc[:, list(component_columns)].sum(axis=1)
    positive_rows = positive_total & (row_sum > 0.0)
    zero_rows = positive_total & ~positive_rows

    if positive_rows.any():
        result.loc[positive_rows, list(component_columns)] = result.loc[
            positive_rows,
            list(component_columns),
        ].div(row_sum.loc[positive_rows], axis=0).mul(total.loc[positive_rows], axis=0)

    if zero_rows.any():
        for column in component_columns:
            result.loc[zero_rows, column] = total.loc[zero_rows] * float(
                fallback.get(column, 0.0)
            )

    if (~positive_total).any():
        result.loc[~positive_total, list(component_columns)] = 0.0

    return result.loc[:, list(component_columns)]


def _grouped_share_predict(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
) -> pd.DataFrame:
    model = fit_grouped_share_model(
        train_frame,
        explicit_component_columns=spec.explicit_component_columns,
        implicit_component_column=spec.implicit_component_column,
        feature_sets=spec.grouped_feature_sets,
        weight_column=spec.weight_column,
    )
    feature_columns = sorted({column for group in spec.grouped_feature_sets for column in group})
    shares = predict_grouped_component_shares(
        test_frame.loc[:, feature_columns].copy(),
        model,
    )
    result = pd.DataFrame(index=test_frame.index)
    family_total = _numeric_series(test_frame, spec.total_column)
    for column in spec.component_columns:
        result[column] = family_total * shares[column]
    return result.loc[:, list(spec.component_columns)]


def _default_qrf_factory(*, condition_vars: list[str], target_vars: list[str], n_estimators: int):
    from microplex_us.pipelines.us import ColumnwiseQRFDonorImputer

    return ColumnwiseQRFDonorImputer(
        condition_vars=condition_vars,
        target_vars=target_vars,
        n_estimators=n_estimators,
    )


def _qrf_predict(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
    random_seed: int,
    qrf_factory: Callable[..., Any] | None,
) -> pd.DataFrame:
    factory = qrf_factory or _default_qrf_factory
    imputer = factory(
        condition_vars=list(spec.qrf_condition_vars),
        target_vars=list(spec.component_columns),
        n_estimators=spec.qrf_n_estimators,
    )
    fit_frame = train_frame.loc[
        :,
        [*spec.qrf_condition_vars, *spec.component_columns, spec.weight_column],
    ].copy()
    imputer.fit(
        fit_frame,
        weight_col=spec.weight_column,
        epochs=None,
        batch_size=None,
        learning_rate=None,
        verbose=False,
    )
    generated = imputer.generate(
        test_frame.loc[:, list(spec.qrf_condition_vars)].copy(),
        seed=random_seed,
    )
    return reconcile_component_predictions_to_total(
        generated,
        family_total=test_frame[spec.total_column],
        component_columns=spec.component_columns,
        fallback_shares=_overall_component_shares(train_frame, spec=spec),
    )


def _summarize_method(
    actual: pd.DataFrame,
    predicted_components: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
) -> FamilyImputationMethodBenchmark:
    actual_eval = actual.loc[:, [spec.weight_column, *spec.component_columns, *spec.group_eval_columns]].copy()
    predicted_eval = actual.loc[:, [spec.weight_column, *spec.group_eval_columns]].copy()
    for column in spec.component_columns:
        predicted_eval[column] = _numeric_series(predicted_components, column)

    actual_totals = _weighted_component_totals(
        actual_eval,
        component_columns=spec.component_columns,
        weight_column=spec.weight_column,
    )
    predicted_totals = _weighted_component_totals(
        predicted_eval,
        component_columns=spec.component_columns,
        weight_column=spec.weight_column,
    )
    total_relative_error = {
        column: _relative_error(predicted_totals[column], actual_totals[column])
        for column in spec.component_columns
    }

    actual_support = _weighted_component_support(
        actual_eval,
        component_columns=spec.component_columns,
        weight_column=spec.weight_column,
    )
    predicted_support = _weighted_component_support(
        predicted_eval,
        component_columns=spec.component_columns,
        weight_column=spec.weight_column,
    )
    support_relative_error = {
        column: _relative_error(predicted_support[column], actual_support[column])
        for column in spec.component_columns
    }

    group_sum_mare = _component_group_sum_mare(
        actual_eval,
        predicted_eval,
        component_columns=spec.component_columns,
        weight_column=spec.weight_column,
        group_columns=spec.group_eval_columns,
    )

    return FamilyImputationMethodBenchmark(
        component_total_relative_error=total_relative_error,
        component_support_relative_error=support_relative_error,
        component_group_sum_mare=group_sum_mare,
        mean_component_total_relative_error=float(np.mean(list(total_relative_error.values()))),
        mean_component_support_relative_error=float(
            np.mean(list(support_relative_error.values()))
        ),
        mean_component_group_sum_mare=(
            float(np.mean(list(group_sum_mare.values()))) if group_sum_mare else None
        ),
    )


def benchmark_decomposable_family_imputers(
    reference: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
    train_frac: float = 0.8,
    random_seed: int = 42,
    qrf_factory: Callable[..., Any] | None = None,
) -> FamilyImputationBenchmarkResult:
    """Benchmark grouped-share and QRF imputers on a holdout split."""

    frame = _build_positive_family_frame(reference, spec=spec)
    train_frame, test_frame = _split_train_test(
        frame,
        train_frac=train_frac,
        random_seed=random_seed,
    )

    grouped_predictions = _grouped_share_predict(
        train_frame,
        test_frame,
        spec=spec,
    )
    qrf_predictions = _qrf_predict(
        train_frame,
        test_frame,
        spec=spec,
        random_seed=random_seed,
        qrf_factory=qrf_factory,
    )

    methods = {
        "grouped_share": _summarize_method(
            test_frame,
            grouped_predictions,
            spec=spec,
        ),
        "qrf": _summarize_method(
            test_frame,
            qrf_predictions,
            spec=spec,
        ),
    }
    return FamilyImputationBenchmarkResult(
        spec=spec,
        row_count=int(len(frame)),
        train_row_count=int(len(train_frame)),
        test_row_count=int(len(test_frame)),
        methods=methods,
    )
