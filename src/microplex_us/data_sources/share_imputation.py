"""Reusable grouped-share imputation utilities for decomposable value families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import CategoricalDtype


@dataclass
class GroupedShareModel:
    """Weighted grouped-share model with hierarchical fallbacks."""

    explicit_component_columns: tuple[str, ...]
    implicit_component_column: str | None
    feature_sets: tuple[tuple[str, ...], ...]
    group_share_tables: dict[tuple[str, ...], pd.DataFrame]
    overall_explicit_shares: dict[str, float]

    @property
    def component_columns(self) -> tuple[str, ...]:
        if self.implicit_component_column is None:
            return self.explicit_component_columns
        return (*self.explicit_component_columns, self.implicit_component_column)


def _nonnegative_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0)


def _normalized_feature_frame(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    result = frame.loc[:, list(feature_columns)].copy()
    for column in feature_columns:
        series = result[column]
        if isinstance(series.dtype, CategoricalDtype):
            series = series.astype(object)
        result[column] = series.where(pd.notna(series), None)
    return result


def fit_grouped_share_model(
    reference: pd.DataFrame,
    *,
    explicit_component_columns: tuple[str, ...],
    implicit_component_column: str | None = None,
    feature_sets: tuple[tuple[str, ...], ...],
    weight_column: str = "weight",
) -> GroupedShareModel:
    """Fit weighted grouped shares over explicit components.

    The implicit component, when provided, is computed as the remaining share.
    """

    if not explicit_component_columns:
        raise ValueError("Grouped share model requires at least one explicit component")
    all_components = explicit_component_columns + (
        (implicit_component_column,) if implicit_component_column is not None else ()
    )
    component_total = sum(
        _nonnegative_series(reference, component)
        for component in all_components
    )
    positive_mask = component_total > 0.0
    if not positive_mask.any():
        raise ValueError("Grouped share model requires at least one positive training row")

    positive = reference.loc[positive_mask].copy()
    weights = _nonnegative_series(positive, weight_column)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        weights = pd.Series(1.0, index=positive.index, dtype=float)
        weight_sum = float(weights.sum())

    positive_total = component_total.loc[positive.index].where(
        component_total.loc[positive.index] > 0.0,
        1.0,
    )
    share_columns: list[str] = []
    work = positive.copy()
    for component in explicit_component_columns:
        share_column = f"__share_{component}"
        share_columns.append(share_column)
        work[share_column] = (
            _nonnegative_series(positive, component) / positive_total
        ).astype(float)
    work["__weight"] = weights.astype(float)

    overall_explicit_shares: dict[str, float] = {}
    for component, share_column in zip(
        explicit_component_columns,
        share_columns,
        strict=True,
    ):
        overall_explicit_shares[component] = float(
            np.average(work[share_column], weights=work["__weight"])
        )
    overall_sum = sum(overall_explicit_shares.values())
    if overall_sum > 1.0:
        overall_explicit_shares = {
            component: value / overall_sum
            for component, value in overall_explicit_shares.items()
        }

    group_share_tables: dict[tuple[str, ...], pd.DataFrame] = {}
    for feature_set in feature_sets:
        if not feature_set:
            continue
        features = _normalized_feature_frame(work, feature_set)
        grouped = pd.concat([features, work[share_columns], work[["__weight"]]], axis=1)
        weighted_columns = []
        for share_column in share_columns:
            weighted_column = f"{share_column}__weighted"
            grouped[weighted_column] = grouped[share_column] * grouped["__weight"]
            weighted_columns.append(weighted_column)
        aggregated = (
            grouped.groupby(list(feature_set), dropna=False, observed=False)[
                [*weighted_columns, "__weight"]
            ]
            .sum()
            .reset_index()
        )
        for component, share_column, weighted_column in zip(
            explicit_component_columns,
            share_columns,
            weighted_columns,
            strict=True,
        ):
            aggregated[component] = np.where(
                aggregated["__weight"] > 0.0,
                aggregated[weighted_column] / aggregated["__weight"],
                0.0,
            )
        group_share_tables[feature_set] = aggregated.loc[
            :,
            [*feature_set, *explicit_component_columns],
        ]

    return GroupedShareModel(
        explicit_component_columns=explicit_component_columns,
        implicit_component_column=implicit_component_column,
        feature_sets=feature_sets,
        group_share_tables=group_share_tables,
        overall_explicit_shares=overall_explicit_shares,
    )


def predict_grouped_component_shares(
    target: pd.DataFrame,
    model: GroupedShareModel,
) -> pd.DataFrame:
    """Predict MECE component shares for the target frame."""

    result = pd.DataFrame(index=target.index)
    explicit_columns = list(model.explicit_component_columns)
    unresolved = pd.Series(True, index=target.index, dtype=bool)

    for feature_set in model.feature_sets:
        if not feature_set:
            continue
        table = model.group_share_tables.get(feature_set)
        if table is None or table.empty:
            continue
        feature_frame = _normalized_feature_frame(target.loc[unresolved], feature_set)
        feature_frame["__row_id"] = feature_frame.index
        merged = feature_frame.merge(
            table,
            on=list(feature_set),
            how="left",
            sort=False,
        ).set_index("__row_id")
        merged = merged.reindex(feature_frame.index)
        matched_mask = merged[explicit_columns].notna().all(axis=1)
        if not matched_mask.any():
            continue
        matched_index = matched_mask.index[matched_mask]
        result.loc[matched_index, explicit_columns] = merged.loc[
            matched_index,
            explicit_columns,
        ]
        unresolved.loc[matched_index] = False

    for component in explicit_columns:
        default_series = pd.Series(
            model.overall_explicit_shares[component],
            index=result.index,
            dtype=float,
        )
        result[component] = pd.to_numeric(
            result.get(component, default_series),
            errors="coerce",
        ).fillna(model.overall_explicit_shares[component])
        result[component] = result[component].clip(lower=0.0, upper=1.0)

    explicit_sum = result[explicit_columns].sum(axis=1)
    overfull_mask = explicit_sum > 1.0
    if overfull_mask.any():
        result.loc[overfull_mask, explicit_columns] = result.loc[
            overfull_mask,
            explicit_columns,
        ].div(explicit_sum.loc[overfull_mask], axis=0)
        explicit_sum = result[explicit_columns].sum(axis=1)

    if model.implicit_component_column is not None:
        result[model.implicit_component_column] = (1.0 - explicit_sum).clip(
            lower=0.0,
            upper=1.0,
        )

    return result.loc[:, list(model.component_columns)]
