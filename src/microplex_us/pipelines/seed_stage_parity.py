"""Audit seed/source-impute rows before synthesis and calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from microplex.core import EntityType

from microplex_us.pipelines.source_stage_parity import (
    _bundle_table,
    _compare_series,
    _entity_weights,
    _safe_ratio,
    _stringify_id_series,
    _summarize_series,
)
from microplex_us.policyengine.us import (
    PolicyEngineUSEntityTableBundle,
    load_policyengine_us_entity_tables,
)


@dataclass(frozen=True)
class SeedStageFocusVariableSpec:
    """One seed-stage variable comparison against the PE reference surface."""

    label: str
    seed_variable: str
    reference_variable: str | None = None
    value_kind: str = "auto"

    @property
    def resolved_reference_variable(self) -> str:
        return self.reference_variable or self.seed_variable


@dataclass(frozen=True)
class SeedStageBooleanLandingFeatureSpec:
    """One positive-support feature share among seed-stage positive rows."""

    label: str
    seed_variable: str
    reference_variable: str | None = None

    @property
    def resolved_reference_variable(self) -> str:
        return self.reference_variable or self.seed_variable


@dataclass(frozen=True)
class SeedStageCategoricalLandingFeatureSpec:
    """One categorical landing profile among seed-stage positive rows."""

    label: str
    seed_variable: str
    reference_variable: str | None = None
    transform: str = "identity"
    top_n: int = 10

    @property
    def resolved_reference_variable(self) -> str:
        return self.reference_variable or self.seed_variable


DEFAULT_SEED_STAGE_FOCUS_VARIABLES: tuple[SeedStageFocusVariableSpec, ...] = (
    SeedStageFocusVariableSpec(
        "self_employment_income",
        "self_employment_income",
        "self_employment_income_before_lsr",
        value_kind="numeric",
    ),
    SeedStageFocusVariableSpec(
        "partnership_s_corp_income",
        "partnership_s_corp_income",
        value_kind="numeric",
    ),
    SeedStageFocusVariableSpec(
        "health_savings_account_ald",
        "health_savings_account_ald",
        value_kind="numeric",
    ),
    SeedStageFocusVariableSpec(
        "self_employed_health_insurance_ald",
        "self_employed_health_insurance_ald",
        value_kind="numeric",
    ),
    SeedStageFocusVariableSpec(
        "self_employed_pension_contribution_ald",
        "self_employed_pension_contribution_ald",
        value_kind="numeric",
    ),
    SeedStageFocusVariableSpec(
        "taxable_interest_income",
        "taxable_interest_income",
        value_kind="numeric",
    ),
    SeedStageFocusVariableSpec(
        "non_qualified_dividend_income",
        "non_qualified_dividend_income",
        value_kind="numeric",
    ),
)

DEFAULT_SEED_STAGE_BOOLEAN_LANDING_FEATURES: tuple[
    SeedStageBooleanLandingFeatureSpec, ...
] = (
    SeedStageBooleanLandingFeatureSpec(
        "positive_self_employment_income",
        "self_employment_income",
        "self_employment_income_before_lsr",
    ),
    SeedStageBooleanLandingFeatureSpec(
        "positive_wage_income",
        "employment_income",
        "employment_income_before_lsr",
    ),
    SeedStageBooleanLandingFeatureSpec("has_esi", "has_esi"),
    SeedStageBooleanLandingFeatureSpec(
        "has_marketplace_health_coverage",
        "has_marketplace_health_coverage",
    ),
)

DEFAULT_SEED_STAGE_CATEGORICAL_LANDING_FEATURES: tuple[
    SeedStageCategoricalLandingFeatureSpec, ...
] = (
    SeedStageCategoricalLandingFeatureSpec(
        "age_bin",
        "age",
        "age",
        transform="age_bin",
    ),
    SeedStageCategoricalLandingFeatureSpec(
        "state_fips",
        "state_fips",
        "state_fips",
    ),
)

DEFAULT_SEED_STAGE_CANDIDATE_ONLY_LANDING_FEATURES: tuple[
    SeedStageCategoricalLandingFeatureSpec, ...
] = (
    SeedStageCategoricalLandingFeatureSpec(
        "employment_status",
        "employment_status",
    ),
)

_AGE_BIN_LABELS: tuple[str, ...] = tuple(
    [f"{start}-{start + 4}" for start in range(0, 85, 5)] + ["85+"]
)
_AGE_BIN_EDGES = np.array(list(range(0, 90, 5)) + [200], dtype=float)
_PERSON_PROJECTION_ORDER: tuple[tuple[EntityType, str], ...] = (
    (EntityType.HOUSEHOLD, "household_id"),
    (EntityType.TAX_UNIT, "tax_unit_id"),
    (EntityType.SPM_UNIT, "spm_unit_id"),
    (EntityType.FAMILY, "family_id"),
)


def build_us_seed_stage_parity_audit(
    seed_data: str | Path,
    reference_dataset: str | Path,
    *,
    period: int = 2024,
    focus_variables: tuple[SeedStageFocusVariableSpec | str, ...]
    | list[SeedStageFocusVariableSpec | str] = DEFAULT_SEED_STAGE_FOCUS_VARIABLES,
    boolean_landing_features: tuple[SeedStageBooleanLandingFeatureSpec, ...]
    | list[SeedStageBooleanLandingFeatureSpec] = DEFAULT_SEED_STAGE_BOOLEAN_LANDING_FEATURES,
    categorical_landing_features: tuple[SeedStageCategoricalLandingFeatureSpec, ...]
    | list[SeedStageCategoricalLandingFeatureSpec] = DEFAULT_SEED_STAGE_CATEGORICAL_LANDING_FEATURES,
    candidate_only_landing_features: tuple[
        SeedStageCategoricalLandingFeatureSpec, ...
    ]
    | list[SeedStageCategoricalLandingFeatureSpec] = DEFAULT_SEED_STAGE_CANDIDATE_ONLY_LANDING_FEATURES,
) -> dict[str, Any]:
    """Compare seed-stage donor landing against PE's person-level reference surface."""

    seed_path = Path(seed_data).resolve()
    reference_path = Path(reference_dataset).resolve()
    seed_rows = pd.read_parquet(seed_path)
    seed_weights = _seed_weight_series(seed_rows)

    focus_specs = _normalize_focus_variable_specs(focus_variables)
    boolean_specs = tuple(boolean_landing_features)
    categorical_specs = tuple(categorical_landing_features)
    candidate_only_specs = tuple(candidate_only_landing_features)

    reference_bundle = load_policyengine_us_entity_tables(reference_path, period=period)
    required_reference_variables = {
        spec.resolved_reference_variable for spec in focus_specs
    } | {
        spec.resolved_reference_variable for spec in boolean_specs
    } | {
        spec.resolved_reference_variable for spec in categorical_specs
    }
    reference_person_rows = _build_reference_person_projection(
        reference_bundle,
        required_reference_variables,
    )
    reference_weights = pd.to_numeric(
        reference_person_rows["weight"],
        errors="coerce",
    ).fillna(0.0)
    seed_total_weight = float(seed_weights.sum())
    reference_total_weight = float(reference_weights.sum())

    return {
        "schemaVersion": 1,
        "comparisonStage": "seed_source_impute",
        "period": int(period),
        "seedData": str(seed_path),
        "referenceDataset": str(reference_path),
        "weightScale": {
            "seed_total_weight": seed_total_weight,
            "reference_total_weight": reference_total_weight,
            "reference_to_seed_weight_scale": _safe_ratio(
                reference_total_weight,
                seed_total_weight,
            ),
        },
        "seedStructure": _seed_structure_summary(seed_rows, seed_weights),
        "referenceStructure": _reference_person_structure_summary(
            reference_person_rows,
            reference_weights,
            bundle=reference_bundle,
        ),
        "focusVariables": {
            spec.label: _seed_focus_variable_audit(
                seed_rows=seed_rows,
                seed_weights=seed_weights,
                reference_rows=reference_person_rows,
                reference_weights=reference_weights,
                focus_spec=spec,
                boolean_specs=boolean_specs,
                categorical_specs=categorical_specs,
                candidate_only_specs=candidate_only_specs,
            )
            for spec in focus_specs
        },
    }


def write_us_seed_stage_parity_audit(
    seed_data: str | Path,
    reference_dataset: str | Path,
    output_path: str | Path,
    *,
    period: int = 2024,
    focus_variables: tuple[SeedStageFocusVariableSpec | str, ...]
    | list[SeedStageFocusVariableSpec | str] = DEFAULT_SEED_STAGE_FOCUS_VARIABLES,
    boolean_landing_features: tuple[SeedStageBooleanLandingFeatureSpec, ...]
    | list[SeedStageBooleanLandingFeatureSpec] = DEFAULT_SEED_STAGE_BOOLEAN_LANDING_FEATURES,
    categorical_landing_features: tuple[SeedStageCategoricalLandingFeatureSpec, ...]
    | list[SeedStageCategoricalLandingFeatureSpec] = DEFAULT_SEED_STAGE_CATEGORICAL_LANDING_FEATURES,
    candidate_only_landing_features: tuple[
        SeedStageCategoricalLandingFeatureSpec, ...
    ]
    | list[SeedStageCategoricalLandingFeatureSpec] = DEFAULT_SEED_STAGE_CANDIDATE_ONLY_LANDING_FEATURES,
) -> Path:
    """Persist one seed/source-impute parity audit as JSON."""

    output = Path(output_path).resolve()
    payload = build_us_seed_stage_parity_audit(
        seed_data,
        reference_dataset,
        period=period,
        focus_variables=focus_variables,
        boolean_landing_features=boolean_landing_features,
        categorical_landing_features=categorical_landing_features,
        candidate_only_landing_features=candidate_only_landing_features,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def _seed_focus_variable_audit(
    *,
    seed_rows: pd.DataFrame,
    seed_weights: pd.Series,
    reference_rows: pd.DataFrame,
    reference_weights: pd.Series,
    focus_spec: SeedStageFocusVariableSpec,
    boolean_specs: tuple[SeedStageBooleanLandingFeatureSpec, ...],
    categorical_specs: tuple[SeedStageCategoricalLandingFeatureSpec, ...],
    candidate_only_specs: tuple[SeedStageCategoricalLandingFeatureSpec, ...],
) -> dict[str, Any]:
    seed_series = seed_rows.get(focus_spec.seed_variable)
    reference_series = reference_rows.get(focus_spec.resolved_reference_variable)
    payload: dict[str, Any] = {
        "seed_variable": focus_spec.seed_variable,
        "reference_variable": focus_spec.resolved_reference_variable,
        "seed_present": seed_series is not None,
        "reference_present": reference_series is not None,
    }
    if seed_series is None or reference_series is None:
        return payload

    payload["seed"] = _summarize_series(
        seed_series,
        weights=seed_weights,
        value_kind=focus_spec.value_kind,
    )
    payload["reference"] = _summarize_series(
        reference_series,
        weights=reference_weights,
        value_kind=focus_spec.value_kind,
    )
    payload["comparison"] = _compare_series(
        seed_series,
        reference_series,
        candidate_weights=seed_weights,
        reference_weights=reference_weights,
        value_kind=focus_spec.value_kind,
    )
    if payload["comparison"].get("type") == "numeric":
        weighted_sum_case = _undefined_ratio_case(
            payload["seed"]["weighted_sum"],
            payload["reference"]["weighted_sum"],
        )
        weighted_positive_share_case = _undefined_ratio_case(
            payload["seed"]["weighted_positive_share"],
            payload["reference"]["weighted_positive_share"],
        )
        payload["comparison"]["weighted_sum_ratio_defined"] = weighted_sum_case == "defined"
        payload["comparison"]["weighted_sum_ratio_case"] = weighted_sum_case
        payload["comparison"]["weighted_positive_share_ratio_defined"] = (
            weighted_positive_share_case == "defined"
        )
        payload["comparison"]["weighted_positive_share_ratio_case"] = (
            weighted_positive_share_case
        )
        payload["comparison"]["reference_scaled_weighted_sum_ratio"] = _safe_ratio(
            payload["seed"]["weighted_sum"]
            * _safe_ratio(float(reference_weights.sum()), float(seed_weights.sum())),
            payload["reference"]["weighted_sum"],
        )
        payload["comparison"]["reference_scaled_weighted_sum_ratio_defined"] = (
            weighted_sum_case == "defined"
        )
        payload["comparison"]["reference_scaled_weighted_sum_ratio_case"] = (
            weighted_sum_case
        )

    seed_positive = _positive_mask(seed_series)
    reference_positive = _positive_mask(reference_series)
    payload["positiveSupport"] = {
        "seed_positive_row_count": int(seed_positive.sum()),
        "reference_positive_row_count": int(reference_positive.sum()),
        "seed_positive_weight_share": _weighted_share(seed_positive, seed_weights),
        "reference_positive_weight_share": _weighted_share(
            reference_positive,
            reference_weights,
        ),
    }
    payload["positiveBooleanProfiles"] = {
        spec.label: _boolean_positive_profile(
            seed_rows=seed_rows,
            seed_weights=seed_weights,
            seed_positive=seed_positive,
            reference_rows=reference_rows,
            reference_weights=reference_weights,
            reference_positive=reference_positive,
            spec=spec,
        )
        for spec in boolean_specs
    }
    payload["positiveCategoricalProfiles"] = {
        spec.label: _categorical_positive_profile(
            seed_rows=seed_rows,
            seed_weights=seed_weights,
            seed_positive=seed_positive,
            reference_rows=reference_rows,
            reference_weights=reference_weights,
            reference_positive=reference_positive,
            spec=spec,
        )
        for spec in categorical_specs
    }
    payload["positiveCandidateOnlyProfiles"] = {
        spec.label: _candidate_only_positive_profile(
            seed_rows=seed_rows,
            seed_weights=seed_weights,
            seed_positive=seed_positive,
            spec=spec,
        )
        for spec in candidate_only_specs
    }
    return payload


def _build_reference_person_projection(
    bundle: PolicyEngineUSEntityTableBundle,
    required_variables: set[str],
) -> pd.DataFrame:
    persons = bundle.persons
    if persons is None:
        raise ValueError("Reference dataset must contain person rows")
    projected = persons.copy()
    projected["weight"] = _entity_weights(bundle, EntityType.PERSON).to_numpy(dtype=float)
    for variable in sorted(required_variables):
        if variable in projected.columns:
            continue
        for entity, id_column in _PERSON_PROJECTION_ORDER:
            table = _bundle_table(bundle, entity)
            if table is None or variable not in table.columns:
                continue
            if id_column not in projected.columns or id_column not in table.columns:
                continue
            lookup = pd.Series(
                table[variable].to_numpy(),
                index=_stringify_id_series(table[id_column]),
            )
            projected[variable] = _stringify_id_series(projected[id_column]).map(lookup)
            break
    return projected


def _seed_structure_summary(rows: pd.DataFrame, weights: pd.Series) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "row_count": int(len(rows)),
    }
    for column in ("person_id", "household_id", "tax_unit_id"):
        if column in rows.columns:
            summary[f"{column}_count"] = int(rows[column].nunique(dropna=True))
    if "household_id" in rows.columns:
        household_sizes = rows.groupby("household_id", observed=True).size()
        household_sizes.index = pd.Index(
            _stringify_id_series(pd.Series(household_sizes.index)).tolist()
        )
        household_weights = _seed_household_weights(rows).reindex(household_sizes.index)
        summary["mean_rows_per_household"] = float(household_sizes.mean())
        summary["weighted_mean_rows_per_household"] = _weighted_mean(
            household_sizes.astype(float).to_numpy(),
            pd.to_numeric(household_weights, errors="coerce").fillna(0.0).to_numpy(
                dtype=float
            ),
        )
    summary["total_weight"] = float(weights.sum())
    return summary


def _reference_person_structure_summary(
    rows: pd.DataFrame,
    weights: pd.Series,
    *,
    bundle: PolicyEngineUSEntityTableBundle,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"person_row_count": int(len(rows))}
    for column in ("person_id", "household_id", "tax_unit_id"):
        if column in rows.columns:
            summary[f"{column}_count"] = int(rows[column].nunique(dropna=True))
    if "household_id" in rows.columns:
        household_sizes = rows.groupby("household_id", observed=True).size()
        household_sizes.index = pd.Index(
            _stringify_id_series(pd.Series(household_sizes.index)).tolist()
        )
        household_weights = _reference_household_weights(bundle).reindex(household_sizes.index)
        summary["mean_rows_per_household"] = float(household_sizes.mean())
        summary["weighted_mean_rows_per_household"] = _weighted_mean(
            household_sizes.astype(float).to_numpy(),
            pd.to_numeric(household_weights, errors="coerce").fillna(0.0).to_numpy(
                dtype=float
            ),
        )
    summary["total_weight"] = float(weights.sum())
    return summary


def _boolean_positive_profile(
    *,
    seed_rows: pd.DataFrame,
    seed_weights: pd.Series,
    seed_positive: pd.Series,
    reference_rows: pd.DataFrame,
    reference_weights: pd.Series,
    reference_positive: pd.Series,
    spec: SeedStageBooleanLandingFeatureSpec,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "seed_present": spec.seed_variable in seed_rows.columns,
        "reference_present": spec.resolved_reference_variable in reference_rows.columns,
    }
    if spec.seed_variable in seed_rows.columns:
        seed_feature = _positive_mask(seed_rows[spec.seed_variable])
        payload["seed_positive_share"] = _conditional_weight_share(
            seed_feature,
            seed_positive,
            seed_weights,
        )
    if spec.resolved_reference_variable in reference_rows.columns:
        reference_feature = _positive_mask(reference_rows[spec.resolved_reference_variable])
        payload["reference_positive_share"] = _conditional_weight_share(
            reference_feature,
            reference_positive,
            reference_weights,
        )
    if (
        "seed_positive_share" in payload
        and "reference_positive_share" in payload
    ):
        payload["share_delta"] = (
            payload["seed_positive_share"] - payload["reference_positive_share"]
        )
    return payload


def _categorical_positive_profile(
    *,
    seed_rows: pd.DataFrame,
    seed_weights: pd.Series,
    seed_positive: pd.Series,
    reference_rows: pd.DataFrame,
    reference_weights: pd.Series,
    reference_positive: pd.Series,
    spec: SeedStageCategoricalLandingFeatureSpec,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "seed_present": spec.seed_variable in seed_rows.columns,
        "reference_present": spec.resolved_reference_variable in reference_rows.columns,
    }
    if spec.seed_variable in seed_rows.columns:
        seed_series = _transform_profile_series(seed_rows[spec.seed_variable], spec.transform)
        payload["seed"] = _categorical_subset_summary(
            seed_series,
            seed_weights,
            seed_positive,
            top_n=spec.top_n,
        )
    if spec.resolved_reference_variable in reference_rows.columns:
        reference_series = _transform_profile_series(
            reference_rows[spec.resolved_reference_variable],
            spec.transform,
        )
        payload["reference"] = _categorical_subset_summary(
            reference_series,
            reference_weights,
            reference_positive,
            top_n=spec.top_n,
        )
    return payload


def _candidate_only_positive_profile(
    *,
    seed_rows: pd.DataFrame,
    seed_weights: pd.Series,
    seed_positive: pd.Series,
    spec: SeedStageCategoricalLandingFeatureSpec,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"seed_present": spec.seed_variable in seed_rows.columns}
    if spec.seed_variable in seed_rows.columns:
        seed_series = _transform_profile_series(seed_rows[spec.seed_variable], spec.transform)
        payload["seed"] = _categorical_subset_summary(
            seed_series,
            seed_weights,
            seed_positive,
            top_n=spec.top_n,
        )
    return payload


def _categorical_subset_summary(
    series: pd.Series,
    weights: pd.Series,
    mask: pd.Series,
    *,
    top_n: int,
) -> dict[str, Any]:
    subset_series = series.loc[mask].reset_index(drop=True)
    subset_weights = pd.to_numeric(
        weights.loc[mask],
        errors="coerce",
    ).fillna(0.0).reset_index(drop=True)
    summary = _summarize_series(
        subset_series,
        weights=subset_weights,
        value_kind="categorical",
    )
    summary["top_values"] = list(summary.get("top_values", []))[: int(top_n)]
    return summary


def _normalize_focus_variable_specs(
    specs: tuple[SeedStageFocusVariableSpec | str, ...]
    | list[SeedStageFocusVariableSpec | str],
) -> tuple[SeedStageFocusVariableSpec, ...]:
    result: list[SeedStageFocusVariableSpec] = []
    for spec in specs:
        if isinstance(spec, SeedStageFocusVariableSpec):
            result.append(spec)
            continue
        result.append(
            SeedStageFocusVariableSpec(
                label=str(spec),
                seed_variable=str(spec),
            )
        )
    return tuple(result)


def _seed_weight_column(rows: pd.DataFrame) -> str:
    for candidate in ("hh_weight", "household_weight", "weight"):
        if candidate in rows.columns:
            return candidate
    raise ValueError("Seed rows must contain hh_weight, household_weight, or weight")


def _seed_weight_series(rows: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(rows[_seed_weight_column(rows)], errors="coerce").fillna(0.0)


def _seed_household_weights(rows: pd.DataFrame) -> pd.Series:
    if "household_id" not in rows.columns:
        raise ValueError("Seed rows must contain household_id to summarize households")
    household_ids = _stringify_id_series(rows["household_id"])
    if "hh_weight" in rows.columns:
        values = pd.to_numeric(rows["hh_weight"], errors="coerce").fillna(0.0)
    elif "household_weight" in rows.columns:
        values = pd.to_numeric(rows["household_weight"], errors="coerce").fillna(0.0)
    else:
        values = pd.to_numeric(rows["weight"], errors="coerce").fillna(0.0)
    grouped = (
        pd.DataFrame({"household_id": household_ids, "weight": values.to_numpy(dtype=float)})
        .groupby("household_id", observed=True)["weight"]
        .mean()
    )
    return grouped


def _reference_household_weights(bundle: PolicyEngineUSEntityTableBundle) -> pd.Series:
    households = bundle.households
    if households is None or "household_id" not in households.columns:
        return pd.Series(dtype=float)
    household_ids = _stringify_id_series(households["household_id"])
    weights = _entity_weights(bundle, EntityType.HOUSEHOLD)
    return pd.Series(weights.to_numpy(dtype=float), index=household_ids)


def _positive_mask(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(0.0).gt(0.0)


def _conditional_weight_share(
    feature_mask: pd.Series,
    positive_mask: pd.Series,
    weights: pd.Series,
) -> float:
    aligned_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    positive_weight = float(aligned_weights.loc[positive_mask].sum())
    if positive_weight <= 0.0:
        return 0.0
    return float(aligned_weights.loc[positive_mask & feature_mask].sum() / positive_weight)


def _weighted_share(mask: pd.Series, weights: pd.Series) -> float:
    return _safe_ratio(
        float(pd.to_numeric(weights, errors="coerce").fillna(0.0).loc[mask].sum()),
        float(pd.to_numeric(weights, errors="coerce").fillna(0.0).sum()),
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return 0.0
    return float(np.dot(values, weights) / total_weight)


def _undefined_ratio_case(candidate_value: float, reference_value: float) -> str:
    if float(reference_value) != 0.0:
        return "defined"
    if float(candidate_value) != 0.0:
        return "candidate_nonzero_reference_zero"
    return "both_zero"


def _transform_profile_series(series: pd.Series, transform: str) -> pd.Series:
    if transform == "identity":
        return series
    if transform == "age_bin":
        numeric = pd.to_numeric(series, errors="coerce")
        binned = pd.cut(
            numeric,
            bins=_AGE_BIN_EDGES,
            labels=_AGE_BIN_LABELS,
            right=False,
            include_lowest=True,
        )
        return binned.astype("string")
    raise ValueError(f"Unsupported profile transform: {transform}")
