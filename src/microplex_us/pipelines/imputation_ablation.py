"""Ablation scoring for donor-imputation conditioning hypotheses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from microplex_us.variables import PUF_IRS_TAX_PREFERRED_CONDITION_VARS


@dataclass(frozen=True)
class ImputationAblationVariant:
    """Describe one imputation strategy under test."""

    name: str
    description: str
    condition_selection: str
    hard_gate_columns: tuple[str, ...] = ()
    primary_predictors: tuple[str, ...] = ()
    secondary_predictors: tuple[str, ...] = ()
    forbidden_predictors: tuple[str, ...] = ()
    support_mapping: str = "rank"
    semantic_guards: bool = False


@dataclass(frozen=True)
class ImputationAblationSliceSpec:
    """Joint slice used to test conditional imputation structure."""

    name: str
    columns: tuple[str, ...]
    min_weight: float = 0.0


@dataclass(frozen=True)
class ImputationTargetScore:
    """Target-level observed-vs-imputed score for one variant."""

    target: str
    row_count: int
    observed_positive_rate: float
    imputed_positive_rate: float
    support_precision: float
    support_recall: float
    support_f1: float
    mean_absolute_error: float
    weighted_mean_absolute_error: float
    weighted_total_relative_error: float


@dataclass(frozen=True)
class ImputationSliceScore:
    """Conditional distribution score for one target and joint slice."""

    target: str
    slice_name: str
    columns: tuple[str, ...]
    cell_count: int
    total_js_divergence: float | None
    support_js_divergence: float | None
    mean_abs_positive_rate_delta: float | None


@dataclass(frozen=True)
class ImputationAblationVariantScore:
    """All scores for one imputation ablation variant."""

    variant: ImputationAblationVariant
    target_scores: dict[str, ImputationTargetScore]
    slice_scores: tuple[ImputationSliceScore, ...]
    aggregate_metrics: dict[str, float | None]
    post_calibration_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ImputationAblationReport:
    """Comparable pre/post calibration imputation ablation scorecard."""

    row_count: int
    targets: tuple[str, ...]
    slice_specs: tuple[ImputationAblationSliceSpec, ...]
    variants: dict[str, ImputationAblationVariantScore]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_imputation_ablation_variants() -> tuple[ImputationAblationVariant, ...]:
    """Return the first hypothesis test variants for QRF conditioning."""
    pe_tax_predictors = tuple(PUF_IRS_TAX_PREFERRED_CONDITION_VARS)
    structural_gates = (
        "age_group",
        "tax_unit_is_joint",
        "is_tax_unit_head",
        "is_tax_unit_spouse",
        "is_tax_unit_dependent",
    )
    return (
        ImputationAblationVariant(
            name="broad_common_qrf",
            description="Current-style QRF using every compatible common predictor.",
            condition_selection="all_shared",
            support_mapping="rank",
            semantic_guards=False,
        ),
        ImputationAblationVariant(
            name="structured_pe_conditioning",
            description=(
                "PE-style structural gates, preferred tax-unit predictors, and "
                "zero-inflated support mapping."
            ),
            condition_selection="pe_prespecified",
            hard_gate_columns=structural_gates,
            primary_predictors=pe_tax_predictors,
            secondary_predictors=("state_fips", "employment_income", "income"),
            forbidden_predictors=("policyengine_output", "post_calibration_weight"),
            support_mapping="zero_inflated_positive",
            semantic_guards=True,
        ),
        ImputationAblationVariant(
            name="broad_common_with_guards",
            description=(
                "Broad common-predictor QRF with the same semantic guards, isolating "
                "guard effects from conditioning effects."
            ),
            condition_selection="all_shared",
            support_mapping="rank",
            semantic_guards=True,
        ),
        ImputationAblationVariant(
            name="rich_predictor_stress",
            description=(
                "Over-rich predictor set used to test whether more predictors alone "
                "beat explicit structure."
            ),
            condition_selection="all_shared",
            secondary_predictors=("state_fips", "education", "occupation", "survey_id"),
            support_mapping="rank",
            semantic_guards=False,
        ),
    )


def score_imputation_ablation_variants(
    *,
    observed_frame: pd.DataFrame,
    imputed_frames: Mapping[str, pd.DataFrame],
    target_variables: Sequence[str],
    slice_specs: Sequence[ImputationAblationSliceSpec] = (),
    variants: Sequence[ImputationAblationVariant] | None = None,
    weight_column: str | None = None,
    post_calibration_metrics: Mapping[str, Mapping[str, float]] | None = None,
) -> ImputationAblationReport:
    """Score imputed candidate frames against masked-observed truth proxies."""
    targets = tuple(dict.fromkeys(target_variables))
    if not targets:
        raise ValueError("target_variables must not be empty")
    _require_columns(observed_frame, targets)
    if weight_column is not None:
        _require_columns(observed_frame, (weight_column,))

    variant_defs = {
        variant.name: variant
        for variant in (
            tuple(variants)
            if variants is not None
            else default_imputation_ablation_variants()
        )
    }
    missing_variant_defs = set(imputed_frames) - set(variant_defs)
    if missing_variant_defs:
        for name in sorted(missing_variant_defs):
            variant_defs[name] = ImputationAblationVariant(
                name=name,
                description="Ad hoc imputation candidate supplied to the ablation scorer.",
                condition_selection="unspecified",
            )

    weights = _weight_series(observed_frame, weight_column)
    scores: dict[str, ImputationAblationVariantScore] = {}
    for variant_name, imputed_frame in imputed_frames.items():
        _validate_frame_pair(observed_frame, imputed_frame, targets)
        target_scores = {
            target: _score_target(
                observed_frame[target],
                imputed_frame[target],
                weights=weights,
                target=target,
            )
            for target in targets
        }
        slice_scores = tuple(
            _score_slice(
                observed_frame=observed_frame,
                imputed_frame=imputed_frame,
                weights=weights,
                target=target,
                slice_spec=slice_spec,
            )
            for target in targets
            for slice_spec in slice_specs
        )
        scores[variant_name] = ImputationAblationVariantScore(
            variant=variant_defs[variant_name],
            target_scores=target_scores,
            slice_scores=slice_scores,
            aggregate_metrics=_aggregate_variant_metrics(
                target_scores=target_scores,
                slice_scores=slice_scores,
            ),
            post_calibration_metrics=dict(
                (post_calibration_metrics or {}).get(variant_name, {})
            ),
        )

    return ImputationAblationReport(
        row_count=len(observed_frame),
        targets=targets,
        slice_specs=tuple(slice_specs),
        variants=scores,
    )


def _score_target(
    observed: pd.Series,
    imputed: pd.Series,
    *,
    weights: pd.Series,
    target: str,
) -> ImputationTargetScore:
    observed_numeric = _numeric_series(observed)
    imputed_numeric = _numeric_series(imputed)
    observed_positive = observed_numeric > 0.0
    imputed_positive = imputed_numeric > 0.0
    true_positive_weight = float(weights[observed_positive & imputed_positive].sum())
    imputed_positive_weight = float(weights[imputed_positive].sum())
    observed_positive_weight = float(weights[observed_positive].sum())
    precision = _safe_ratio(true_positive_weight, imputed_positive_weight)
    recall = _safe_ratio(true_positive_weight, observed_positive_weight)
    absolute_error = (imputed_numeric - observed_numeric).abs()
    observed_total = float((observed_numeric * weights).sum())
    imputed_total = float((imputed_numeric * weights).sum())
    return ImputationTargetScore(
        target=target,
        row_count=len(observed_numeric),
        observed_positive_rate=_safe_ratio(
            observed_positive_weight, float(weights.sum())
        ),
        imputed_positive_rate=_safe_ratio(
            imputed_positive_weight, float(weights.sum())
        ),
        support_precision=precision,
        support_recall=recall,
        support_f1=_safe_f1(precision, recall),
        mean_absolute_error=float(absolute_error.mean()),
        weighted_mean_absolute_error=_safe_ratio(
            float((absolute_error * weights).sum()),
            float(weights.sum()),
        ),
        weighted_total_relative_error=_relative_error(imputed_total, observed_total),
    )


def _score_slice(
    *,
    observed_frame: pd.DataFrame,
    imputed_frame: pd.DataFrame,
    weights: pd.Series,
    target: str,
    slice_spec: ImputationAblationSliceSpec,
) -> ImputationSliceScore:
    missing = [
        column for column in slice_spec.columns if column not in observed_frame.columns
    ]
    if missing:
        return ImputationSliceScore(
            target=target,
            slice_name=slice_spec.name,
            columns=slice_spec.columns,
            cell_count=0,
            total_js_divergence=None,
            support_js_divergence=None,
            mean_abs_positive_rate_delta=None,
        )

    observed_numeric = _numeric_series(observed_frame[target])
    imputed_numeric = _numeric_series(imputed_frame[target])
    cell_keys = _cell_keys(observed_frame, slice_spec.columns)
    cells = sorted(cell_keys.unique())
    observed_totals: list[float] = []
    imputed_totals: list[float] = []
    observed_support: list[float] = []
    imputed_support: list[float] = []
    positive_rate_deltas: list[float] = []
    for cell in cells:
        mask = cell_keys == cell
        cell_weight = float(weights[mask].sum())
        if cell_weight <= slice_spec.min_weight:
            continue
        observed_cell = observed_numeric[mask]
        imputed_cell = imputed_numeric[mask]
        cell_weights = weights[mask]
        observed_totals.append(float((observed_cell * cell_weights).sum()))
        imputed_totals.append(float((imputed_cell * cell_weights).sum()))
        observed_support_weight = float(cell_weights[observed_cell > 0.0].sum())
        imputed_support_weight = float(cell_weights[imputed_cell > 0.0].sum())
        observed_support.append(observed_support_weight)
        imputed_support.append(imputed_support_weight)
        positive_rate_deltas.append(
            abs(
                _safe_ratio(imputed_support_weight, cell_weight)
                - _safe_ratio(observed_support_weight, cell_weight)
            )
        )

    return ImputationSliceScore(
        target=target,
        slice_name=slice_spec.name,
        columns=slice_spec.columns,
        cell_count=len(observed_totals),
        total_js_divergence=_jensen_shannon_divergence(observed_totals, imputed_totals),
        support_js_divergence=_jensen_shannon_divergence(
            observed_support, imputed_support
        ),
        mean_abs_positive_rate_delta=(
            float(np.mean(positive_rate_deltas)) if positive_rate_deltas else None
        ),
    )


def _aggregate_variant_metrics(
    *,
    target_scores: Mapping[str, ImputationTargetScore],
    slice_scores: Sequence[ImputationSliceScore],
) -> dict[str, float | None]:
    return {
        "mean_weighted_mae": _mean_or_none(
            [score.weighted_mean_absolute_error for score in target_scores.values()]
        ),
        "mean_total_relative_error": _mean_or_none(
            [score.weighted_total_relative_error for score in target_scores.values()]
        ),
        "mean_support_f1": _mean_or_none(
            [score.support_f1 for score in target_scores.values()]
        ),
        "mean_slice_total_js_divergence": _mean_or_none(
            [score.total_js_divergence for score in slice_scores]
        ),
        "mean_slice_support_js_divergence": _mean_or_none(
            [score.support_js_divergence for score in slice_scores]
        ),
        "mean_slice_positive_rate_delta": _mean_or_none(
            [score.mean_abs_positive_rate_delta for score in slice_scores]
        ),
    }


def _validate_frame_pair(
    observed_frame: pd.DataFrame,
    imputed_frame: pd.DataFrame,
    targets: Sequence[str],
) -> None:
    if len(imputed_frame) != len(observed_frame):
        raise ValueError("observed_frame and imputed_frames must have the same length")
    if not imputed_frame.index.equals(observed_frame.index):
        raise ValueError("observed_frame and imputed_frames must have matching indexes")
    _require_columns(imputed_frame, targets)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing required columns: {missing}")


def _numeric_series(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )


def _weight_series(frame: pd.DataFrame, weight_column: str | None) -> pd.Series:
    if weight_column is None:
        return pd.Series(1.0, index=frame.index, dtype=float)
    return _numeric_series(frame[weight_column]).clip(lower=0.0)


def _cell_keys(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    keys = frame.loc[:, list(columns)].astype("string").fillna("__MISSING__")
    return keys.agg("|".join, axis=1)


def _relative_error(candidate: float, baseline: float) -> float:
    baseline_abs = abs(float(baseline))
    if baseline_abs <= 1e-9:
        return 0.0 if abs(float(candidate)) <= 1e-9 else 1.0
    return float(abs(float(candidate) - float(baseline)) / baseline_abs)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= 1e-12:
        return 0.0
    return float(numerator) / float(denominator)


def _safe_f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    if denominator <= 1e-12:
        return 0.0
    return float(2.0 * precision * recall / denominator)


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(np.mean(numeric))


def _jensen_shannon_divergence(
    observed_values: Sequence[float],
    imputed_values: Sequence[float],
) -> float | None:
    observed = np.asarray(observed_values, dtype=float).clip(min=0.0)
    imputed = np.asarray(imputed_values, dtype=float).clip(min=0.0)
    if observed.size == 0 or imputed.size == 0:
        return None
    observed_total = observed.sum()
    imputed_total = imputed.sum()
    if observed_total <= 1e-12 and imputed_total <= 1e-12:
        return 0.0
    if observed_total <= 1e-12 or imputed_total <= 1e-12:
        return 1.0
    observed_prob = observed / observed_total
    imputed_prob = imputed / imputed_total
    midpoint = 0.5 * (observed_prob + imputed_prob)
    return float(
        0.5 * _kl_divergence(observed_prob, midpoint)
        + 0.5 * _kl_divergence(imputed_prob, midpoint)
    )


def _kl_divergence(probabilities: np.ndarray, reference: np.ndarray) -> float:
    mask = probabilities > 0.0
    return float(
        np.sum(probabilities[mask] * np.log2(probabilities[mask] / reference[mask]))
    )
