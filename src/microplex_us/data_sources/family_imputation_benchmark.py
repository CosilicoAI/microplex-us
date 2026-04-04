"""Holdout benchmarks for decomposable-family imputers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from microplex.calibration import Calibrator
from sklearn.ensemble import RandomForestRegressor

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
    forest_condition_vars: tuple[str, ...] = ()
    forest_n_estimators: int = 200
    forest_min_samples_leaf: int = 5
    reweight_feature_sets: tuple[tuple[str, ...], ...] = ()
    reweight_method: str = "ipf"
    reweight_max_iter: int = 100
    reweight_tol: float = 1e-6
    reweight_initial_weight_mode: str = "observed"

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
    post_reweight_component_total_relative_error: dict[str, float] | None = None
    post_reweight_component_support_relative_error: dict[str, float] | None = None
    post_reweight_component_group_sum_mare: dict[str, float] | None = None
    post_reweight_mean_component_total_relative_error: float | None = None
    post_reweight_mean_component_support_relative_error: float | None = None
    post_reweight_mean_component_group_sum_mare: float | None = None
    post_reweight_total_error_degradation: dict[str, float] | None = None
    post_reweight_mean_component_total_error_degradation: float | None = None
    oracle_post_reweight_component_total_relative_error: dict[str, float] | None = None
    oracle_post_reweight_mean_component_total_relative_error: float | None = None
    post_reweight_excess_over_oracle_total_error: dict[str, float] | None = None
    post_reweight_mean_component_total_error_excess_over_oracle: float | None = None
    reweighting_summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class FamilyImputationBenchmarkResult:
    """Comparable holdout metrics for a decomposable family."""

    spec: DecomposableFamilyBenchmarkSpec
    row_count: int
    train_row_count: int
    eval_row_count: int
    target_row_count: int
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


def _encode_condition_frames(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    condition_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoded_train = pd.DataFrame(index=train_frame.index)
    encoded_test = pd.DataFrame(index=test_frame.index)
    for column in condition_columns:
        train_series = train_frame[column]
        test_series = test_frame[column]
        if pd.api.types.is_numeric_dtype(train_series):
            encoded_train[column] = pd.to_numeric(train_series, errors="coerce").fillna(0.0)
            encoded_test[column] = pd.to_numeric(test_series, errors="coerce").fillna(0.0)
            continue
        train_values = train_series.astype("string").fillna("__MISSING__")
        test_values = test_series.astype("string").fillna("__MISSING__")
        categories = pd.Index(train_values.unique(), dtype="object")
        encoded_train[column] = pd.Categorical(train_values, categories=categories).codes.astype(float)
        encoded_test[column] = pd.Categorical(test_values, categories=categories).codes.astype(float)
    return encoded_train, encoded_test


def _split_train_eval_target(
    frame: pd.DataFrame,
    *,
    train_frac: float,
    target_frac: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1")
    if not 0.0 < target_frac < 1.0:
        raise ValueError("target_frac must be between 0 and 1")
    if train_frac + target_frac >= 1.0:
        raise ValueError("train_frac + target_frac must leave room for eval rows")
    if len(frame) < 3:
        raise ValueError("Benchmark requires at least three rows to create train/eval/target splits")
    rng = np.random.default_rng(random_seed)
    shuffled = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
    n_rows = len(shuffled)
    n_train = max(1, int(np.floor(n_rows * train_frac)))
    n_target = max(1, int(np.floor(n_rows * target_frac)))
    if n_train + n_target >= n_rows:
        n_target = max(1, n_rows - n_train - 1)
    if n_train + n_target >= n_rows:
        n_train = max(1, n_rows - n_target - 1)
    train_frame = shuffled.iloc[:n_train].reset_index(drop=True)
    target_frame = shuffled.iloc[n_train : n_train + n_target].reset_index(drop=True)
    eval_frame = shuffled.iloc[n_train + n_target :].reset_index(drop=True)
    if eval_frame.empty:
        raise ValueError("Benchmark split produced no eval rows")
    return train_frame, eval_frame, target_frame


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


def _combined_categorical_column(
    frame: pd.DataFrame,
    feature_set: tuple[str, ...],
) -> pd.Series:
    if len(feature_set) == 1:
        return frame[feature_set[0]].astype("string").fillna("__MISSING__")
    combined = frame.loc[:, list(feature_set)].astype("string").fillna("__MISSING__")
    return combined.agg("||".join, axis=1)


def _build_reweighting_targets(
    frame: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
    required_categories_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    if not spec.reweight_feature_sets:
        return frame.copy(), {}
    prepared = frame.copy()
    required_prepared = (
        required_categories_frame.copy() if required_categories_frame is not None else None
    )
    targets: dict[str, dict[str, float]] = {}
    weights = _numeric_series(prepared, spec.weight_column)
    for feature_set in spec.reweight_feature_sets:
        target_column = "__reweight__" + "__".join(feature_set)
        prepared[target_column] = _combined_categorical_column(prepared, feature_set)
        if required_prepared is not None:
            required_prepared[target_column] = _combined_categorical_column(
                required_prepared,
                feature_set,
            )
        grouped = (
            pd.DataFrame({target_column: prepared[target_column], "__weight": weights})
            .groupby(target_column, dropna=False, observed=False)["__weight"]
            .sum()
        )
        target_values = {
            str(category): float(total)
            for category, total in grouped.items()
        }
        if required_prepared is not None:
            required_categories = (
                required_prepared[target_column]
                .astype("string")
                .fillna("__MISSING__")
                .unique()
                .tolist()
            )
            for category in required_categories:
                target_values.setdefault(str(category), 0.0)
        targets[target_column] = target_values
    return prepared, targets


def _apply_reweighting(
    target_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    if not spec.reweight_feature_sets:
        return None
    target_prepared, targets = _build_reweighting_targets(
        target_frame,
        spec=spec,
        required_categories_frame=eval_frame,
    )
    eval_prepared, _ = _build_reweighting_targets(eval_frame, spec=spec)
    if spec.reweight_initial_weight_mode == "uniform":
        eval_prepared[spec.weight_column] = 1.0
    elif spec.reweight_initial_weight_mode != "observed":
        raise ValueError(
            "reweight_initial_weight_mode must be 'observed' or 'uniform'"
        )
    calibrator = Calibrator(
        method=spec.reweight_method,
        tol=spec.reweight_tol,
        max_iter=spec.reweight_max_iter,
    )
    reweighted = calibrator.fit_transform(
        eval_prepared,
        marginal_targets=targets,
        weight_col=spec.weight_column,
    )
    validation = calibrator.validate(eval_prepared, weight_col=spec.weight_column)
    keep_columns = [spec.weight_column, *spec.group_eval_columns, *spec.component_columns]
    return reweighted.loc[:, keep_columns].copy(), {
        "converged": bool(validation["converged"]),
        "max_error": float(validation["max_error"]),
        "n_iterations": int(calibrator.n_iterations_),
        "target_count": int(sum(len(values) for values in targets.values())),
        "target_columns": sorted(targets.keys()),
        "method": spec.reweight_method,
        "initial_weight_mode": spec.reweight_initial_weight_mode,
        "target_row_count": int(len(target_prepared)),
        "eval_row_count": int(len(eval_prepared)),
    }


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


def _component_share_targets(
    frame: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
) -> pd.DataFrame:
    total = _numeric_series(frame, spec.total_column)
    safe_total = total.where(total > 0.0, 1.0)
    shares = pd.DataFrame(index=frame.index)
    for column in spec.component_columns:
        shares[column] = (_numeric_series(frame, column) / safe_total).clip(lower=0.0)
    row_sum = shares.sum(axis=1)
    overfull = row_sum > 1.0
    if overfull.any():
        shares.loc[overfull, list(spec.component_columns)] = shares.loc[
            overfull,
            list(spec.component_columns),
        ].div(row_sum.loc[overfull], axis=0)
    shares.loc[total <= 0.0, list(spec.component_columns)] = 0.0
    return shares.loc[:, list(spec.component_columns)]


def _normalize_share_predictions(
    shares: pd.DataFrame,
    *,
    component_columns: tuple[str, ...],
    fallback_shares: dict[str, float],
) -> pd.DataFrame:
    result = pd.DataFrame(index=shares.index)
    for column in component_columns:
        result[column] = _numeric_series(shares, column).clip(lower=0.0)
    row_sum = result.sum(axis=1)
    positive = row_sum > 0.0
    if positive.any():
        result.loc[positive, list(component_columns)] = result.loc[
            positive,
            list(component_columns),
        ].div(row_sum.loc[positive], axis=0)
    zero_rows = ~positive
    if zero_rows.any():
        for column in component_columns:
            result.loc[zero_rows, column] = float(fallback_shares[column])
    return result.loc[:, list(component_columns)]


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
    encoded_train, encoded_test = _encode_condition_frames(
        train_frame,
        test_frame,
        condition_columns=spec.qrf_condition_vars,
    )
    imputer = factory(
        condition_vars=list(spec.qrf_condition_vars),
        target_vars=list(spec.component_columns),
        n_estimators=spec.qrf_n_estimators,
    )
    fit_frame = encoded_train.copy()
    for column in spec.component_columns:
        fit_frame[column] = _numeric_series(train_frame, column)
    fit_frame[spec.weight_column] = _numeric_series(train_frame, spec.weight_column)
    imputer.fit(
        fit_frame,
        weight_col=spec.weight_column,
        epochs=None,
        batch_size=None,
        learning_rate=None,
        verbose=False,
    )
    generated = imputer.generate(
        encoded_test.copy(),
        seed=random_seed,
    )
    return reconcile_component_predictions_to_total(
        generated,
        family_total=test_frame[spec.total_column],
        component_columns=spec.component_columns,
        fallback_shares=_overall_component_shares(train_frame, spec=spec),
    )


def _forest_share_predict(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
    random_seed: int,
) -> pd.DataFrame:
    condition_columns = (
        spec.forest_condition_vars if spec.forest_condition_vars else spec.qrf_condition_vars
    )
    encoded_train, encoded_test = _encode_condition_frames(
        train_frame,
        test_frame,
        condition_columns=condition_columns,
    )
    model = RandomForestRegressor(
        n_estimators=spec.forest_n_estimators,
        min_samples_leaf=spec.forest_min_samples_leaf,
        random_state=random_seed,
        n_jobs=-1,
    )
    train_targets = _component_share_targets(train_frame, spec=spec)
    train_weights = _numeric_series(train_frame, spec.weight_column)
    if train_weights.sum() > 0.0:
        model.fit(
            encoded_train.to_numpy(dtype=float),
            train_targets.to_numpy(dtype=float),
            sample_weight=train_weights.to_numpy(dtype=float),
        )
    else:
        model.fit(
            encoded_train.to_numpy(dtype=float),
            train_targets.to_numpy(dtype=float),
        )
    predicted_shares = pd.DataFrame(
        model.predict(encoded_test.to_numpy(dtype=float)),
        index=test_frame.index,
        columns=list(spec.component_columns),
    )
    normalized = _normalize_share_predictions(
        predicted_shares,
        component_columns=spec.component_columns,
        fallback_shares=_overall_component_shares(train_frame, spec=spec),
    )
    family_total = _numeric_series(test_frame, spec.total_column)
    result = pd.DataFrame(index=test_frame.index)
    for column in spec.component_columns:
        result[column] = normalized[column] * family_total
    return result.loc[:, list(spec.component_columns)]


def _summarize_method(
    actual_eval: pd.DataFrame,
    target_eval: pd.DataFrame,
    predicted_components: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
) -> FamilyImputationMethodBenchmark:
    passthrough_columns = {
        spec.weight_column,
        *spec.group_eval_columns,
        *spec.component_columns,
    }
    for feature_set in spec.reweight_feature_sets:
        passthrough_columns.update(feature_set)
    actual_eval = actual_eval.loc[:, list(passthrough_columns)].copy()
    target_eval = target_eval.loc[:, list(passthrough_columns)].copy()
    predicted_eval = actual_eval.loc[:, [column for column in passthrough_columns if column not in spec.component_columns]].copy()
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

    target_totals = _weighted_component_totals(
        target_eval,
        component_columns=spec.component_columns,
        weight_column=spec.weight_column,
    )
    target_support = _weighted_component_support(
        target_eval,
        component_columns=spec.component_columns,
        weight_column=spec.weight_column,
    )

    post_reweight_total_error = None
    post_reweight_support_error = None
    post_reweight_group_sum_mare = None
    post_reweight_mean_total_error = None
    post_reweight_mean_support_error = None
    post_reweight_mean_group_sum_mare = None
    post_reweight_total_error_degradation = None
    post_reweight_mean_total_error_degradation = None
    oracle_post_reweight_total_error = None
    oracle_post_reweight_mean_total_error = None
    post_reweight_excess_over_oracle_total_error = None
    post_reweight_mean_total_error_excess_over_oracle = None
    reweighting_summary = None
    reweighted_result = _apply_reweighting(
        target_eval,
        predicted_eval,
        spec=spec,
    )
    if reweighted_result is not None:
        reweighted_eval, reweighting_summary = reweighted_result
        reweighted_totals = _weighted_component_totals(
            reweighted_eval,
            component_columns=spec.component_columns,
            weight_column=spec.weight_column,
        )
        post_reweight_total_error = {
            column: _relative_error(reweighted_totals[column], target_totals[column])
            for column in spec.component_columns
        }
        reweighted_support = _weighted_component_support(
            reweighted_eval,
            component_columns=spec.component_columns,
            weight_column=spec.weight_column,
        )
        post_reweight_support_error = {
            column: _relative_error(reweighted_support[column], target_support[column])
            for column in spec.component_columns
        }
        post_reweight_group_sum_mare = _component_group_sum_mare(
            target_eval,
            reweighted_eval,
            component_columns=spec.component_columns,
            weight_column=spec.weight_column,
            group_columns=spec.group_eval_columns,
        )
        post_reweight_mean_total_error = float(
            np.mean(list(post_reweight_total_error.values()))
        )
        post_reweight_mean_support_error = float(
            np.mean(list(post_reweight_support_error.values()))
        )
        post_reweight_mean_group_sum_mare = (
            float(np.mean(list(post_reweight_group_sum_mare.values())))
            if post_reweight_group_sum_mare
            else None
        )
        oracle_reweighted_result = _apply_reweighting(
            target_eval,
            actual_eval,
            spec=spec,
        )
        if oracle_reweighted_result is not None:
            oracle_reweighted_eval, _oracle_summary = oracle_reweighted_result
            oracle_reweighted_totals = _weighted_component_totals(
                oracle_reweighted_eval,
                component_columns=spec.component_columns,
                weight_column=spec.weight_column,
            )
            oracle_post_reweight_total_error = {
                column: _relative_error(
                    oracle_reweighted_totals[column],
                    target_totals[column],
                )
                for column in spec.component_columns
            }
            oracle_post_reweight_mean_total_error = float(
                np.mean(list(oracle_post_reweight_total_error.values()))
            )
            post_reweight_excess_over_oracle_total_error = {
                column: post_reweight_total_error[column]
                - oracle_post_reweight_total_error[column]
                for column in spec.component_columns
            }
            post_reweight_mean_total_error_excess_over_oracle = float(
                np.mean(list(post_reweight_excess_over_oracle_total_error.values()))
            )
        post_reweight_total_error_degradation = {
            column: post_reweight_total_error[column] - total_relative_error[column]
            for column in spec.component_columns
        }
        post_reweight_mean_total_error_degradation = float(
            np.mean(list(post_reweight_total_error_degradation.values()))
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
        post_reweight_component_total_relative_error=post_reweight_total_error,
        post_reweight_component_support_relative_error=post_reweight_support_error,
        post_reweight_component_group_sum_mare=post_reweight_group_sum_mare,
        post_reweight_mean_component_total_relative_error=post_reweight_mean_total_error,
        post_reweight_mean_component_support_relative_error=post_reweight_mean_support_error,
        post_reweight_mean_component_group_sum_mare=post_reweight_mean_group_sum_mare,
        post_reweight_total_error_degradation=post_reweight_total_error_degradation,
        post_reweight_mean_component_total_error_degradation=post_reweight_mean_total_error_degradation,
        oracle_post_reweight_component_total_relative_error=oracle_post_reweight_total_error,
        oracle_post_reweight_mean_component_total_relative_error=oracle_post_reweight_mean_total_error,
        post_reweight_excess_over_oracle_total_error=post_reweight_excess_over_oracle_total_error,
        post_reweight_mean_component_total_error_excess_over_oracle=post_reweight_mean_total_error_excess_over_oracle,
        reweighting_summary=reweighting_summary,
    )


def benchmark_decomposable_family_imputers(
    reference: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
    train_frac: float = 0.8,
    target_frac: float = 0.1,
    random_seed: int = 42,
    qrf_factory: Callable[..., Any] | None = None,
) -> FamilyImputationBenchmarkResult:
    """Benchmark grouped-share and QRF imputers on a holdout split."""

    frame = _build_positive_family_frame(reference, spec=spec)
    train_frame, eval_frame, target_frame = _split_train_eval_target(
        frame,
        train_frac=train_frac,
        target_frac=target_frac,
        random_seed=random_seed,
    )

    grouped_predictions = _grouped_share_predict(
        train_frame,
        eval_frame,
        spec=spec,
    )
    qrf_predictions = _qrf_predict(
        train_frame,
        eval_frame,
        spec=spec,
        random_seed=random_seed,
        qrf_factory=qrf_factory,
    )

    methods = {
        "grouped_share": _summarize_method(
            eval_frame,
            target_frame,
            grouped_predictions,
            spec=spec,
        ),
        "forest_share": _summarize_method(
            eval_frame,
            target_frame,
            _forest_share_predict(
                train_frame,
                eval_frame,
                spec=spec,
                random_seed=random_seed,
            ),
            spec=spec,
        ),
        "qrf": _summarize_method(
            eval_frame,
            target_frame,
            qrf_predictions,
            spec=spec,
        ),
    }
    return FamilyImputationBenchmarkResult(
        spec=spec,
        row_count=int(len(frame)),
        train_row_count=int(len(train_frame)),
        eval_row_count=int(len(eval_frame)),
        target_row_count=int(len(target_frame)),
        methods=methods,
    )
