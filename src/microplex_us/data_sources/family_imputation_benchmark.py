"""Holdout benchmarks for decomposable-family imputers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import pandas as pd
from microplex.calibration import Calibrator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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
    support_gate_probability_threshold: float = 0.2
    forest_share_min_component_share: float = 0.05
    qrf_support_augmentation_max_extra_components: int = 1
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
    pre_target_component_total_relative_error: dict[str, float] | None = None
    pre_target_mean_component_total_relative_error: float | None = None
    post_reweight_component_total_relative_error: dict[str, float] | None = None
    post_reweight_component_support_relative_error: dict[str, float] | None = None
    post_reweight_component_group_sum_mare: dict[str, float] | None = None
    post_reweight_mean_component_total_relative_error: float | None = None
    post_reweight_mean_component_support_relative_error: float | None = None
    post_reweight_mean_component_group_sum_mare: float | None = None
    post_reweight_total_error_lift: dict[str, float] | None = None
    post_reweight_mean_component_total_error_lift: float | None = None
    oracle_pre_target_component_total_relative_error: dict[str, float] | None = None
    oracle_pre_target_mean_component_total_relative_error: float | None = None
    oracle_post_reweight_component_total_relative_error: dict[str, float] | None = None
    oracle_post_reweight_mean_component_total_relative_error: float | None = None
    oracle_post_reweight_total_error_lift: dict[str, float] | None = None
    oracle_post_reweight_mean_component_total_error_lift: float | None = None
    post_reweight_excess_over_oracle_total_error: dict[str, float] | None = None
    post_reweight_mean_component_total_error_excess_over_oracle: float | None = None
    reweighting_summary: dict[str, Any] | None = None
    repeat_metric_summary: dict[str, dict[str, float]] | None = None


@dataclass(frozen=True)
class FamilyImputationBenchmarkResult:
    """Comparable holdout metrics for a decomposable family."""

    spec: DecomposableFamilyBenchmarkSpec
    row_count: int
    train_row_count: int
    eval_row_count: int
    target_row_count: int
    methods: dict[str, FamilyImputationMethodBenchmark]
    repeat_count: int = 1
    split_seeds: tuple[int, ...] = ()
    repeat_summaries: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spec"] = asdict(self.spec)
        return payload


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)


def _median_or_none(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(np.median(numeric))


def _max_or_none(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(max(numeric))


def _aggregate_numeric_dicts(
    dicts: list[dict[str, float] | None],
) -> dict[str, float] | None:
    keys = sorted({key for mapping in dicts if mapping is not None for key in mapping})
    if not keys:
        return None
    aggregated = {
        key: _median_or_none(
            [
                None if mapping is None else mapping.get(key)
                for mapping in dicts
            ]
        )
        for key in keys
    }
    return {key: float(value) for key, value in aggregated.items() if value is not None}


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
    initial_weights = _numeric_series(eval_prepared, spec.weight_column).copy()
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
    final_weights = _numeric_series(reweighted, spec.weight_column)
    denom = initial_weights.abs().clip(lower=1e-9)
    relative_change = (final_weights - initial_weights).abs() / denom
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
        "initial_total_weight": float(initial_weights.sum()),
        "final_total_weight": float(final_weights.sum()),
        "mean_abs_relative_weight_change": float(relative_change.mean()),
        "max_abs_relative_weight_change": float(relative_change.max()),
        "share_rows_changed_gt_1pct": float((relative_change > 0.01).mean()),
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


def _component_support_targets(
    frame: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
) -> pd.DataFrame:
    support = pd.DataFrame(index=frame.index)
    for column in spec.component_columns:
        support[column] = (_numeric_series(frame, column) > 0.0).astype(int)
    return support.loc[:, list(spec.component_columns)]


def _fit_support_probability_model(
    encoded_train: pd.DataFrame,
    encoded_test: pd.DataFrame,
    *,
    target: pd.Series,
    sample_weight: pd.Series,
    n_estimators: int,
    min_samples_leaf: int,
    random_seed: int,
) -> np.ndarray:
    target = pd.to_numeric(target, errors="coerce").fillna(0.0).astype(int)
    unique_values = sorted(set(target.tolist()))
    if len(unique_values) <= 1:
        return np.full(len(encoded_test), float(unique_values[0] if unique_values else 0.0))

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    classifier.fit(
        encoded_train.to_numpy(dtype=float),
        target.to_numpy(dtype=int),
        sample_weight=sample_weight.to_numpy(dtype=float),
    )
    classes = classifier.classes_.tolist()
    positive_index = classes.index(1)
    probabilities = classifier.predict_proba(encoded_test.to_numpy(dtype=float))
    return probabilities[:, positive_index]


def _predict_active_component_counts(
    encoded_train: pd.DataFrame,
    encoded_test: pd.DataFrame,
    *,
    support_targets: pd.DataFrame,
    sample_weight: pd.Series,
    n_estimators: int,
    min_samples_leaf: int,
    random_seed: int,
) -> np.ndarray:
    active_counts = support_targets.sum(axis=1).clip(lower=1).astype(int)
    unique_values = sorted(set(active_counts.tolist()))
    if len(unique_values) <= 1:
        return np.full(
            len(encoded_test),
            int(unique_values[0] if unique_values else 1),
            dtype=int,
        )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    classifier.fit(
        encoded_train.to_numpy(dtype=float),
        active_counts.to_numpy(dtype=int),
        sample_weight=sample_weight.to_numpy(dtype=float),
    )
    return classifier.predict(encoded_test.to_numpy(dtype=float)).astype(int)


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


def _sparsify_normalized_share_predictions(
    shares: pd.DataFrame,
    *,
    component_columns: tuple[str, ...],
    min_component_share: float,
) -> pd.DataFrame:
    sparsified = pd.DataFrame(0.0, index=shares.index, columns=list(component_columns))
    for index in shares.index:
        share_row = (
            shares.loc[index, list(component_columns)]
            .astype(float)
            .clip(lower=0.0)
        )
        selected = share_row[share_row >= min_component_share].index
        if len(selected) == 0:
            selected = share_row.sort_values(ascending=False).index[:1]
        selected_scores = share_row.loc[selected]
        if float(selected_scores.sum()) <= 0.0:
            selected_scores = pd.Series(0.0, index=selected, dtype=float)
            selected_scores.iloc[0] = 1.0
        sparsified.loc[index, selected] = selected_scores.to_numpy(dtype=float)

    return _normalize_share_predictions(
        sparsified,
        component_columns=component_columns,
        fallback_shares={
            column: 1.0 / len(component_columns)
            for column in component_columns
        },
    )


def _mask_share_predictions_to_supported_components(
    predicted_shares: pd.DataFrame,
    support_probabilities: pd.DataFrame,
    predicted_active_counts: np.ndarray,
    *,
    component_columns: tuple[str, ...],
    support_gate_probability_threshold: float,
) -> pd.DataFrame:
    masked = pd.DataFrame(0.0, index=predicted_shares.index, columns=list(component_columns))
    component_count = len(component_columns)

    for position, index in enumerate(predicted_shares.index):
        share_row = (
            predicted_shares.loc[index, list(component_columns)]
            .astype(float)
            .clip(lower=0.0)
        )
        probability_row = (
            support_probabilities.loc[index, list(component_columns)]
            .astype(float)
            .clip(lower=0.0)
        )
        desired_count = int(np.clip(predicted_active_counts[position], 1, component_count))
        confident_count = int((probability_row >= support_gate_probability_threshold).sum())
        keep_count = max(1, min(component_count, max(desired_count, confident_count)))
        selected = probability_row.sort_values(ascending=False).index[:keep_count]
        selected_scores = share_row.loc[selected]
        if float(selected_scores.sum()) <= 0.0:
            selected_scores = probability_row.loc[selected]
        if float(selected_scores.sum()) <= 0.0:
            selected_scores = pd.Series(0.0, index=selected, dtype=float)
            selected_scores.iloc[0] = 1.0
        masked.loc[index, selected] = selected_scores.to_numpy(dtype=float)

    return masked.loc[:, list(component_columns)]


def _mask_share_predictions_to_binary_support(
    predicted_shares: pd.DataFrame,
    support_mask: pd.DataFrame,
    *,
    component_columns: tuple[str, ...],
) -> pd.DataFrame:
    masked = pd.DataFrame(0.0, index=predicted_shares.index, columns=list(component_columns))

    for index in predicted_shares.index:
        share_row = (
            predicted_shares.loc[index, list(component_columns)]
            .astype(float)
            .clip(lower=0.0)
        )
        selected = [
            column
            for column in component_columns
            if float(support_mask.loc[index, column]) > 0.0
        ]
        if not selected:
            selected = [share_row.sort_values(ascending=False).index[0]]
        selected_scores = share_row.loc[selected]
        if float(selected_scores.sum()) <= 0.0:
            selected_scores = pd.Series(0.0, index=selected, dtype=float)
            selected_scores.iloc[0] = 1.0
        masked.loc[index, selected] = selected_scores.to_numpy(dtype=float)

    return masked.loc[:, list(component_columns)]


def _augment_sparse_shares_with_support_prior(
    sparse_shares: pd.DataFrame,
    base_share_scores: pd.DataFrame,
    support_mask: pd.DataFrame,
    *,
    component_columns: tuple[str, ...],
    max_extra_components: int,
) -> pd.DataFrame:
    augmented = sparse_shares.loc[:, list(component_columns)].copy()
    if max_extra_components <= 0:
        return augmented

    for index in augmented.index:
        sparse_row = (
            augmented.loc[index, list(component_columns)].astype(float).clip(lower=0.0)
        )
        active_components = [
            column for column in component_columns if float(sparse_row[column]) > 0.0
        ]
        supported_components = [
            column
            for column in component_columns
            if float(support_mask.loc[index, column]) > 0.0
        ]
        missing_supported = [
            column for column in supported_components if column not in active_components
        ]
        if not missing_supported:
            continue

        extra_budget = min(
            max_extra_components,
            max(0, len(supported_components) - len(active_components)),
        )
        if extra_budget <= 0:
            continue

        base_row = (
            base_share_scores.loc[index, list(component_columns)]
            .astype(float)
            .clip(lower=0.0)
        )
        selected = (
            base_row.loc[missing_supported]
            .sort_values(ascending=False)
            .index[:extra_budget]
            .tolist()
        )
        if not selected:
            continue

        selected_scores = base_row.loc[selected]
        if float(selected_scores.sum()) <= 0.0:
            selected_scores = pd.Series(0.0, index=selected, dtype=float)
            selected_scores.iloc[0] = 1.0
        augmented.loc[index, selected] = selected_scores.to_numpy(dtype=float)

    return augmented.loc[:, list(component_columns)]


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


def _sparse_forest_share_predict(
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
    sparsified = _sparsify_normalized_share_predictions(
        normalized,
        component_columns=spec.component_columns,
        min_component_share=spec.forest_share_min_component_share,
    )
    family_total = _numeric_series(test_frame, spec.total_column)
    result = pd.DataFrame(index=test_frame.index)
    for column in spec.component_columns:
        result[column] = sparsified[column] * family_total
    return result.loc[:, list(spec.component_columns)]


def _support_gated_forest_share_predict(
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
    train_weights = _numeric_series(train_frame, spec.weight_column)
    positive_weights = (
        train_weights if float(train_weights.sum()) > 0.0 else pd.Series(1.0, index=train_frame.index)
    )
    predicted_shares = _component_share_targets(train_frame, spec=spec)
    share_model = RandomForestRegressor(
        n_estimators=spec.forest_n_estimators,
        min_samples_leaf=spec.forest_min_samples_leaf,
        random_state=random_seed,
        n_jobs=-1,
    )
    share_model.fit(
        encoded_train.to_numpy(dtype=float),
        predicted_shares.to_numpy(dtype=float),
        sample_weight=positive_weights.to_numpy(dtype=float),
    )
    raw_share_predictions = pd.DataFrame(
        share_model.predict(encoded_test.to_numpy(dtype=float)),
        index=test_frame.index,
        columns=list(spec.component_columns),
    )
    support_targets = _component_support_targets(train_frame, spec=spec)
    support_probabilities = pd.DataFrame(index=test_frame.index)
    for offset, column in enumerate(spec.component_columns, start=1):
        support_probabilities[column] = _fit_support_probability_model(
            encoded_train,
            encoded_test,
            target=support_targets[column],
            sample_weight=positive_weights,
            n_estimators=spec.forest_n_estimators,
            min_samples_leaf=spec.forest_min_samples_leaf,
            random_seed=random_seed + offset,
        )
    predicted_active_counts = _predict_active_component_counts(
        encoded_train,
        encoded_test,
        support_targets=support_targets,
        sample_weight=positive_weights,
        n_estimators=spec.forest_n_estimators,
        min_samples_leaf=spec.forest_min_samples_leaf,
        random_seed=random_seed + len(spec.component_columns) + 1,
    )
    masked_shares = _mask_share_predictions_to_supported_components(
        raw_share_predictions,
        support_probabilities,
        predicted_active_counts,
        component_columns=spec.component_columns,
        support_gate_probability_threshold=spec.support_gate_probability_threshold,
    )
    normalized = _normalize_share_predictions(
        masked_shares,
        component_columns=spec.component_columns,
        fallback_shares=_overall_component_shares(train_frame, spec=spec),
    )
    family_total = _numeric_series(test_frame, spec.total_column)
    result = pd.DataFrame(index=test_frame.index)
    for column in spec.component_columns:
        result[column] = normalized[column] * family_total
    return result.loc[:, list(spec.component_columns)]


def _qrf_support_masked_forest_share_predict(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
    random_seed: int,
    qrf_predictions: pd.DataFrame,
) -> pd.DataFrame:
    condition_columns = (
        spec.forest_condition_vars if spec.forest_condition_vars else spec.qrf_condition_vars
    )
    encoded_train, encoded_test = _encode_condition_frames(
        train_frame,
        test_frame,
        condition_columns=condition_columns,
    )
    train_weights = _numeric_series(train_frame, spec.weight_column)
    positive_weights = (
        train_weights
        if float(train_weights.sum()) > 0.0
        else pd.Series(1.0, index=train_frame.index)
    )
    predicted_shares = _component_share_targets(train_frame, spec=spec)
    share_model = RandomForestRegressor(
        n_estimators=spec.forest_n_estimators,
        min_samples_leaf=spec.forest_min_samples_leaf,
        random_state=random_seed,
        n_jobs=-1,
    )
    share_model.fit(
        encoded_train.to_numpy(dtype=float),
        predicted_shares.to_numpy(dtype=float),
        sample_weight=positive_weights.to_numpy(dtype=float),
    )
    raw_share_predictions = pd.DataFrame(
        share_model.predict(encoded_test.to_numpy(dtype=float)),
        index=test_frame.index,
        columns=list(spec.component_columns),
    )
    qrf_support_mask = pd.DataFrame(index=test_frame.index)
    for column in spec.component_columns:
        qrf_support_mask[column] = (_numeric_series(qrf_predictions, column) > 0.0).astype(float)
    masked_shares = _mask_share_predictions_to_binary_support(
        raw_share_predictions,
        qrf_support_mask,
        component_columns=spec.component_columns,
    )
    normalized = _normalize_share_predictions(
        masked_shares,
        component_columns=spec.component_columns,
        fallback_shares=_overall_component_shares(train_frame, spec=spec),
    )
    family_total = _numeric_series(test_frame, spec.total_column)
    result = pd.DataFrame(index=test_frame.index)
    for column in spec.component_columns:
        result[column] = normalized[column] * family_total
    return result.loc[:, list(spec.component_columns)]


def _qrf_augmented_sparse_forest_share_predict(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
    random_seed: int,
    qrf_predictions: pd.DataFrame,
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
    sparse = _sparsify_normalized_share_predictions(
        normalized,
        component_columns=spec.component_columns,
        min_component_share=spec.forest_share_min_component_share,
    )
    qrf_support_mask = pd.DataFrame(index=test_frame.index)
    for column in spec.component_columns:
        qrf_support_mask[column] = (_numeric_series(qrf_predictions, column) > 0.0).astype(float)
    augmented = _augment_sparse_shares_with_support_prior(
        sparse,
        normalized,
        qrf_support_mask,
        component_columns=spec.component_columns,
        max_extra_components=spec.qrf_support_augmentation_max_extra_components,
    )
    renormalized = _normalize_share_predictions(
        augmented,
        component_columns=spec.component_columns,
        fallback_shares=_overall_component_shares(train_frame, spec=spec),
    )
    family_total = _numeric_series(test_frame, spec.total_column)
    result = pd.DataFrame(index=test_frame.index)
    for column in spec.component_columns:
        result[column] = renormalized[column] * family_total
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
    if spec.reweight_initial_weight_mode == "uniform":
        start_weights = pd.Series(1.0, index=actual_eval.index, dtype=float)
    elif spec.reweight_initial_weight_mode == "observed":
        start_weights = _numeric_series(actual_eval, spec.weight_column)
    else:
        raise ValueError("reweight_initial_weight_mode must be 'observed' or 'uniform'")

    pre_target_eval = predicted_eval.copy()
    pre_target_eval[spec.weight_column] = start_weights.to_numpy(dtype=float)
    pre_target_totals = _weighted_component_totals(
        pre_target_eval,
        component_columns=spec.component_columns,
        weight_column=spec.weight_column,
    )
    pre_target_total_error = {
        column: _relative_error(pre_target_totals[column], target_totals[column])
        for column in spec.component_columns
    }
    pre_target_mean_total_error = float(
        np.mean(list(pre_target_total_error.values()))
    )

    oracle_pre_target_eval = actual_eval.copy()
    oracle_pre_target_eval[spec.weight_column] = start_weights.to_numpy(dtype=float)
    oracle_pre_target_totals = _weighted_component_totals(
        oracle_pre_target_eval,
        component_columns=spec.component_columns,
        weight_column=spec.weight_column,
    )
    oracle_pre_target_total_error = {
        column: _relative_error(oracle_pre_target_totals[column], target_totals[column])
        for column in spec.component_columns
    }
    oracle_pre_target_mean_total_error = float(
        np.mean(list(oracle_pre_target_total_error.values()))
    )

    post_reweight_total_error = None
    post_reweight_support_error = None
    post_reweight_group_sum_mare = None
    post_reweight_mean_total_error = None
    post_reweight_mean_support_error = None
    post_reweight_mean_group_sum_mare = None
    post_reweight_total_error_lift = None
    post_reweight_mean_total_error_lift = None
    oracle_post_reweight_total_error_lift = None
    oracle_post_reweight_mean_total_error_lift = None
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
            oracle_post_reweight_total_error_lift = {
                column: oracle_post_reweight_total_error[column]
                - oracle_pre_target_total_error[column]
                for column in spec.component_columns
            }
            oracle_post_reweight_mean_total_error_lift = float(
                np.mean(list(oracle_post_reweight_total_error_lift.values()))
            )
            post_reweight_excess_over_oracle_total_error = {
                column: post_reweight_total_error[column]
                - oracle_post_reweight_total_error[column]
                for column in spec.component_columns
            }
            post_reweight_mean_total_error_excess_over_oracle = float(
                np.mean(list(post_reweight_excess_over_oracle_total_error.values()))
            )
        post_reweight_total_error_lift = {
            column: post_reweight_total_error[column] - pre_target_total_error[column]
            for column in spec.component_columns
        }
        post_reweight_mean_total_error_lift = float(
            np.mean(list(post_reweight_total_error_lift.values()))
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
        pre_target_component_total_relative_error=pre_target_total_error,
        pre_target_mean_component_total_relative_error=pre_target_mean_total_error,
        post_reweight_component_total_relative_error=post_reweight_total_error,
        post_reweight_component_support_relative_error=post_reweight_support_error,
        post_reweight_component_group_sum_mare=post_reweight_group_sum_mare,
        post_reweight_mean_component_total_relative_error=post_reweight_mean_total_error,
        post_reweight_mean_component_support_relative_error=post_reweight_mean_support_error,
        post_reweight_mean_component_group_sum_mare=post_reweight_mean_group_sum_mare,
        post_reweight_total_error_lift=post_reweight_total_error_lift,
        post_reweight_mean_component_total_error_lift=post_reweight_mean_total_error_lift,
        oracle_pre_target_component_total_relative_error=oracle_pre_target_total_error,
        oracle_pre_target_mean_component_total_relative_error=oracle_pre_target_mean_total_error,
        oracle_post_reweight_component_total_relative_error=oracle_post_reweight_total_error,
        oracle_post_reweight_mean_component_total_relative_error=oracle_post_reweight_mean_total_error,
        oracle_post_reweight_total_error_lift=oracle_post_reweight_total_error_lift,
        oracle_post_reweight_mean_component_total_error_lift=oracle_post_reweight_mean_total_error_lift,
        post_reweight_excess_over_oracle_total_error=post_reweight_excess_over_oracle_total_error,
        post_reweight_mean_component_total_error_excess_over_oracle=post_reweight_mean_total_error_excess_over_oracle,
        reweighting_summary=reweighting_summary,
    )


_REPEAT_SCALAR_FIELDS = (
    "mean_component_total_relative_error",
    "mean_component_support_relative_error",
    "mean_component_group_sum_mare",
    "pre_target_mean_component_total_relative_error",
    "post_reweight_mean_component_total_relative_error",
    "post_reweight_mean_component_support_relative_error",
    "post_reweight_mean_component_group_sum_mare",
    "post_reweight_mean_component_total_error_lift",
    "oracle_pre_target_mean_component_total_relative_error",
    "oracle_post_reweight_mean_component_total_relative_error",
    "oracle_post_reweight_mean_component_total_error_lift",
    "post_reweight_mean_component_total_error_excess_over_oracle",
)


def _aggregate_reweighting_summaries(
    summaries: list[dict[str, Any] | None],
    *,
    repeat_count: int,
) -> dict[str, Any] | None:
    present = [summary for summary in summaries if summary is not None]
    if not present:
        return None
    first = present[0]
    aggregated: dict[str, Any] = {
        "method": first.get("method"),
        "initial_weight_mode": first.get("initial_weight_mode"),
        "target_columns": first.get("target_columns"),
        "target_count": first.get("target_count"),
        "target_row_count": first.get("target_row_count"),
        "eval_row_count": first.get("eval_row_count"),
        "repeat_count": repeat_count,
        "converged": all(bool(summary.get("converged")) for summary in present),
        "converged_count": int(sum(bool(summary.get("converged")) for summary in present)),
    }
    numeric_fields = (
        "max_error",
        "n_iterations",
        "initial_total_weight",
        "final_total_weight",
        "mean_abs_relative_weight_change",
        "max_abs_relative_weight_change",
        "share_rows_changed_gt_1pct",
    )
    for field_name in numeric_fields:
        values = [
            float(summary[field_name])
            for summary in present
            if summary.get(field_name) is not None
        ]
        if not values:
            continue
        aggregated[field_name] = float(np.median(values))
        aggregated[f"{field_name}_worst"] = float(max(values))
    return aggregated


def _aggregate_method_benchmarks(
    repeats: list[FamilyImputationMethodBenchmark],
) -> FamilyImputationMethodBenchmark:
    aggregated_values: dict[str, Any] = {}
    repeat_metric_summary: dict[str, dict[str, float]] = {}
    for field_info in fields(FamilyImputationMethodBenchmark):
        field_name = field_info.name
        if field_name in {"reweighting_summary", "repeat_metric_summary"}:
            continue
        values = [getattr(result, field_name) for result in repeats]
        sample = next((value for value in values if value is not None), None)
        if sample is None:
            aggregated_values[field_name] = None
            continue
        if isinstance(sample, dict):
            aggregated_values[field_name] = _aggregate_numeric_dicts(values)
            continue
        aggregated_values[field_name] = _median_or_none(values)
        if field_name in _REPEAT_SCALAR_FIELDS and aggregated_values[field_name] is not None:
            repeat_metric_summary[field_name] = {
                "median": float(aggregated_values[field_name]),
                "worst": float(_max_or_none(values)),
            }
    aggregated_values["reweighting_summary"] = _aggregate_reweighting_summaries(
        [result.reweighting_summary for result in repeats],
        repeat_count=len(repeats),
    )
    aggregated_values["repeat_metric_summary"] = repeat_metric_summary or None
    return FamilyImputationMethodBenchmark(**aggregated_values)


def _compact_repeat_summary(
    result: FamilyImputationBenchmarkResult,
    *,
    repeat_index: int,
    split_seed: int,
) -> dict[str, Any]:
    method_fields = (
        "mean_component_total_relative_error",
        "mean_component_support_relative_error",
        "mean_component_group_sum_mare",
        "pre_target_mean_component_total_relative_error",
        "post_reweight_mean_component_total_relative_error",
        "post_reweight_mean_component_support_relative_error",
        "post_reweight_mean_component_group_sum_mare",
        "post_reweight_mean_component_total_error_lift",
        "oracle_pre_target_mean_component_total_relative_error",
        "oracle_post_reweight_mean_component_total_relative_error",
        "oracle_post_reweight_mean_component_total_error_lift",
        "post_reweight_mean_component_total_error_excess_over_oracle",
    )
    methods = {
        method_name: {
            field_name: getattr(method_result, field_name)
            for field_name in method_fields
        }
        for method_name, method_result in result.methods.items()
    }
    return {
        "repeat_index": repeat_index,
        "split_seed": split_seed,
        "train_row_count": result.train_row_count,
        "eval_row_count": result.eval_row_count,
        "target_row_count": result.target_row_count,
        "methods": methods,
    }


def _benchmark_decomposable_family_imputers_once(
    reference: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
    train_frac: float,
    target_frac: float,
    random_seed: int,
    qrf_factory: Callable[..., Any] | None,
) -> FamilyImputationBenchmarkResult:
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
    qrf_support_masked_predictions = _qrf_support_masked_forest_share_predict(
        train_frame,
        eval_frame,
        spec=spec,
        random_seed=random_seed,
        qrf_predictions=qrf_predictions,
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
        "sparse_forest_share": _summarize_method(
            eval_frame,
            target_frame,
            _sparse_forest_share_predict(
                train_frame,
                eval_frame,
                spec=spec,
                random_seed=random_seed,
            ),
            spec=spec,
        ),
        "support_gated_forest_share": _summarize_method(
            eval_frame,
            target_frame,
            _support_gated_forest_share_predict(
                train_frame,
                eval_frame,
                spec=spec,
                random_seed=random_seed,
            ),
            spec=spec,
        ),
        "qrf_support_masked_forest_share": _summarize_method(
            eval_frame,
            target_frame,
            qrf_support_masked_predictions,
            spec=spec,
        ),
        "qrf_augmented_sparse_forest_share": _summarize_method(
            eval_frame,
            target_frame,
            _qrf_augmented_sparse_forest_share_predict(
                train_frame,
                eval_frame,
                spec=spec,
                random_seed=random_seed,
                qrf_predictions=qrf_predictions,
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


def benchmark_decomposable_family_imputers(
    reference: pd.DataFrame,
    *,
    spec: DecomposableFamilyBenchmarkSpec,
    train_frac: float = 0.8,
    target_frac: float = 0.1,
    random_seed: int = 42,
    repeat_count: int = 1,
    repeat_seed_step: int = 1,
    qrf_factory: Callable[..., Any] | None = None,
) -> FamilyImputationBenchmarkResult:
    """Benchmark decomposable-family imputers on one or more holdout splits."""

    if repeat_count < 1:
        raise ValueError("repeat_count must be at least 1")
    if repeat_seed_step < 1:
        raise ValueError("repeat_seed_step must be at least 1")

    repeat_results: list[FamilyImputationBenchmarkResult] = []
    split_seeds = tuple(
        int(random_seed + repeat_index * repeat_seed_step)
        for repeat_index in range(repeat_count)
    )
    for split_seed in split_seeds:
        repeat_results.append(
            _benchmark_decomposable_family_imputers_once(
                reference,
                spec=spec,
                train_frac=train_frac,
                target_frac=target_frac,
                random_seed=split_seed,
                qrf_factory=qrf_factory,
            )
        )

    first_result = repeat_results[0]
    methods = {
        method_name: _aggregate_method_benchmarks(
            [result.methods[method_name] for result in repeat_results]
        )
        for method_name in first_result.methods
    }
    repeat_summaries = tuple(
        _compact_repeat_summary(
            result,
            repeat_index=repeat_index,
            split_seed=split_seed,
        )
        for repeat_index, (result, split_seed) in enumerate(
            zip(repeat_results, split_seeds, strict=True)
        )
    )
    return FamilyImputationBenchmarkResult(
        spec=first_result.spec,
        row_count=first_result.row_count,
        train_row_count=first_result.train_row_count,
        eval_row_count=first_result.eval_row_count,
        target_row_count=first_result.target_row_count,
        methods=methods,
        repeat_count=repeat_count,
        split_seeds=split_seeds,
        repeat_summaries=repeat_summaries,
    )
