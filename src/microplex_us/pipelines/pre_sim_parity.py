"""Audit candidate PE-US datasets against PE's pre-sim input surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microplex_us.policyengine.us import (
    _decode_policyengine_array,
    _load_policyengine_us_period_arrays,
    load_policyengine_us_entity_tables,
)

DEFAULT_PRE_SIM_FOCUS_VARIABLES: tuple[str, ...] = (
    "age",
    "state_fips",
    "county_fips",
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "dividend_income",
    "interest_income",
    "long_term_capital_gains_before_response",
    "partnership_s_corp_income",
    "farm_income",
    "rent",
    "real_estate_taxes",
    "net_worth",
    "is_household_head",
    "is_hispanic",
    "is_disabled",
    "has_esi",
    "has_marketplace_health_coverage",
)

DEFAULT_CRITICAL_REFERENCE_VARIABLES: tuple[str, ...] = (
    "county_fips",
    "cps_race",
    "is_household_head",
    "is_hispanic",
    "is_disabled",
    "rent",
    "real_estate_taxes",
    "net_worth",
    "has_esi",
    "has_marketplace_health_coverage",
)

_AGE_BIN_LABELS: tuple[str, ...] = tuple(
    [f"{start}-{start + 4}" for start in range(0, 85, 5)] + ["85+"]
)
_AGE_BIN_EDGES = np.array(list(range(0, 90, 5)) + [200], dtype=float)


def build_us_pre_sim_parity_audit(
    candidate_dataset: str | Path,
    reference_dataset: str | Path,
    *,
    period: int = 2024,
    focus_variables: tuple[str, ...] | list[str] = DEFAULT_PRE_SIM_FOCUS_VARIABLES,
    critical_reference_variables: tuple[str, ...]
    | list[str] = DEFAULT_CRITICAL_REFERENCE_VARIABLES,
) -> dict[str, Any]:
    """Compare one candidate PE-US dataset to PE's own pre-sim input dataset."""

    candidate_path = Path(candidate_dataset).resolve()
    reference_path = Path(reference_dataset).resolve()
    period_key = str(period)
    candidate_arrays = _load_policyengine_us_period_arrays(
        candidate_path,
        period_key=period_key,
        variables=None,
    )
    reference_arrays = _load_policyengine_us_period_arrays(
        reference_path,
        period_key=period_key,
        variables=None,
    )
    candidate_variables = set(candidate_arrays)
    reference_variables = set(reference_arrays)
    common_variables = sorted(candidate_variables & reference_variables)
    missing_in_candidate = sorted(reference_variables - candidate_variables)
    extra_in_candidate = sorted(candidate_variables - reference_variables)

    candidate_bundle = load_policyengine_us_entity_tables(candidate_path, period=period)
    reference_bundle = load_policyengine_us_entity_tables(reference_path, period=period)

    focus = tuple(dict.fromkeys(str(variable) for variable in focus_variables))
    critical = tuple(
        dict.fromkeys(str(variable) for variable in critical_reference_variables)
    )

    return {
        "period": period,
        "candidate_dataset": str(candidate_path),
        "reference_dataset": str(reference_path),
        "schema": {
            "candidate_variable_count": len(candidate_variables),
            "reference_variable_count": len(reference_variables),
            "common_variable_count": len(common_variables),
            "missing_in_candidate_count": len(missing_in_candidate),
            "extra_in_candidate_count": len(extra_in_candidate),
            "schema_recall": _safe_ratio(len(common_variables), len(reference_variables)),
            "schema_precision": _safe_ratio(len(common_variables), len(candidate_variables)),
            "missing_in_candidate": missing_in_candidate,
            "extra_in_candidate": extra_in_candidate,
            "missing_critical_reference_variables": [
                variable
                for variable in critical
                if variable in reference_variables and variable not in candidate_variables
            ],
        },
        "entity_structure": {
            "candidate": _entity_structure_summary(candidate_bundle),
            "reference": _entity_structure_summary(reference_bundle),
        },
        "focus_variables": {
            variable: _variable_comparison(
                variable=variable,
                candidate_arrays=candidate_arrays,
                reference_arrays=reference_arrays,
            )
            for variable in focus
        },
        "state_age_support": _state_age_support_comparison(
            candidate_bundle=candidate_bundle,
            reference_bundle=reference_bundle,
        ),
    }


def write_us_pre_sim_parity_audit(
    candidate_dataset: str | Path,
    reference_dataset: str | Path,
    output_path: str | Path,
    *,
    period: int = 2024,
    focus_variables: tuple[str, ...] | list[str] = DEFAULT_PRE_SIM_FOCUS_VARIABLES,
    critical_reference_variables: tuple[str, ...]
    | list[str] = DEFAULT_CRITICAL_REFERENCE_VARIABLES,
) -> Path:
    """Build and persist one PE pre-sim parity audit as JSON."""

    output = Path(output_path).resolve()
    payload = build_us_pre_sim_parity_audit(
        candidate_dataset,
        reference_dataset,
        period=period,
        focus_variables=focus_variables,
        critical_reference_variables=critical_reference_variables,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return output


def _entity_structure_summary(bundle) -> dict[str, Any]:
    households = bundle.households
    persons = bundle.persons
    tax_units = bundle.tax_units
    families = bundle.families
    spm_units = bundle.spm_units
    marital_units = bundle.marital_units

    summary: dict[str, Any] = {
        "household_rows": int(len(households)),
        "person_rows": int(len(persons)) if persons is not None else 0,
        "tax_unit_rows": int(len(tax_units)) if tax_units is not None else 0,
        "family_rows": int(len(families)) if families is not None else 0,
        "spm_unit_rows": int(len(spm_units)) if spm_units is not None else 0,
        "marital_unit_rows": int(len(marital_units)) if marital_units is not None else 0,
    }
    if persons is None:
        return summary

    household_sizes = persons.groupby("household_id", observed=True).size()
    summary["mean_household_size"] = float(household_sizes.mean())
    summary["share_multi_person_households"] = float((household_sizes >= 2).mean())

    if "tax_unit_id" in persons.columns:
        grouped = persons.dropna(subset=["tax_unit_id"])
        if not grouped.empty:
            tax_unit_sizes = grouped.groupby("tax_unit_id", observed=True).size()
            summary["mean_tax_unit_size"] = float(tax_unit_sizes.mean())
            summary["share_multi_person_tax_units"] = float((tax_unit_sizes >= 2).mean())

    if "is_household_head" in persons.columns:
        head_flags = _boolean_series(persons["is_household_head"])
        head_counts = (
            persons.assign(_is_head=head_flags)
            .groupby("household_id", observed=True)["_is_head"]
            .sum(min_count=1)
        )
        summary["households_with_exactly_one_head_share"] = float(
            (head_counts == 1).mean()
        )
        summary["households_with_no_head_count"] = int((head_counts == 0).sum())

    return summary


def _state_age_support_comparison(*, candidate_bundle, reference_bundle) -> dict[str, Any]:
    candidate = _state_age_support(candidate_bundle)
    reference = _state_age_support(reference_bundle)
    candidate_cells = set(candidate["cell_counts"])
    reference_cells = set(reference["cell_counts"])
    missing_cells = reference_cells - candidate_cells
    missing_by_state: dict[str, int] = {}
    for state, _age_bin in missing_cells:
        missing_by_state[state] = missing_by_state.get(state, 0) + 1
    top_missing_states = sorted(
        (
            {"state_fips": state, "missing_cell_count": count}
            for state, count in missing_by_state.items()
        ),
        key=lambda row: (-row["missing_cell_count"], row["state_fips"]),
    )[:10]
    return {
        "candidate": {
            "nonempty_cell_count": len(candidate_cells),
            "state_count": candidate["state_count"],
            "support_rate": candidate["support_rate"],
        },
        "reference": {
            "nonempty_cell_count": len(reference_cells),
            "state_count": reference["state_count"],
            "support_rate": reference["support_rate"],
        },
        "support_recall": _safe_ratio(len(candidate_cells & reference_cells), len(reference_cells)),
        "missing_cell_count": len(missing_cells),
        "top_missing_states": top_missing_states,
    }


def _state_age_support(bundle) -> dict[str, Any]:
    persons = bundle.persons
    households = bundle.households
    if persons is None or "age" not in persons.columns or "state_fips" not in households.columns:
        return {"cell_counts": {}, "state_count": 0, "support_rate": 0.0}

    merged = persons[["household_id", "age"]].merge(
        households[["household_id", "state_fips"]],
        on="household_id",
        how="left",
    )
    merged = merged.dropna(subset=["age", "state_fips"]).copy()
    if merged.empty:
        return {"cell_counts": {}, "state_count": 0, "support_rate": 0.0}

    merged["state_fips"] = merged["state_fips"].astype(int).astype(str).str.zfill(2)
    merged["age_bin"] = pd.cut(
        merged["age"].astype(float),
        bins=_AGE_BIN_EDGES,
        labels=_AGE_BIN_LABELS,
        right=False,
        include_lowest=True,
    )
    merged = merged.dropna(subset=["age_bin"])
    cell_counts = (
        merged.groupby(["state_fips", "age_bin"], observed=True)
        .size()
        .astype(int)
        .to_dict()
    )
    state_count = int(merged["state_fips"].nunique())
    total_possible = state_count * len(_AGE_BIN_LABELS)
    return {
        "cell_counts": cell_counts,
        "state_count": state_count,
        "support_rate": _safe_ratio(len(cell_counts), total_possible),
    }


def _variable_comparison(
    *,
    variable: str,
    candidate_arrays: dict[str, np.ndarray],
    reference_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    candidate_present = variable in candidate_arrays
    reference_present = variable in reference_arrays
    result: dict[str, Any] = {
        "candidate_present": candidate_present,
        "reference_present": reference_present,
    }
    if not reference_present and not candidate_present:
        return result
    if reference_present:
        reference_values = _decode_policyengine_array(reference_arrays[variable])
        result["reference"] = _summarize_values(reference_values)
    if candidate_present:
        candidate_values = _decode_policyengine_array(candidate_arrays[variable])
        result["candidate"] = _summarize_values(candidate_values)
    if candidate_present and reference_present:
        result["comparison"] = _compare_values(candidate_values, reference_values)
    return result


def _summarize_values(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    if array.dtype.kind in {"U", "S", "O"}:
        series = pd.Series(array.astype(str))
        normalized = series.replace({"": pd.NA, "nan": pd.NA})
        value_counts = normalized.dropna().value_counts()
        return {
            "kind": "categorical",
            "n": int(len(series)),
            "nonnull_share": _safe_ratio(int(normalized.notna().sum()), len(series)),
            "unique_count": int(normalized.nunique(dropna=True)),
            "top_values": [
                {"value": str(value), "count": int(count)}
                for value, count in value_counts.head(10).items()
            ],
        }

    numeric = pd.Series(array).replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {"kind": "numeric", "n": int(len(array)), "nonnull_share": 0.0}

    unique_count = int(numeric.nunique())
    is_categorical = unique_count <= 64 and numeric.dtype.kind in {"i", "u", "b"}
    if is_categorical:
        value_counts = numeric.astype(int).astype(str).value_counts()
        return {
            "kind": "categorical",
            "n": int(len(array)),
            "nonnull_share": _safe_ratio(int(len(numeric)), len(array)),
            "unique_count": unique_count,
            "top_values": [
                {"value": str(value), "count": int(count)}
                for value, count in value_counts.head(10).items()
            ],
        }

    numeric_values = numeric.astype(float)
    return {
        "kind": "numeric",
        "n": int(len(array)),
        "nonnull_share": _safe_ratio(int(len(numeric_values)), len(array)),
        "zero_share": float((numeric_values == 0).mean()),
        "positive_share": float((numeric_values > 0).mean()),
        "negative_share": float((numeric_values < 0).mean()),
        "mean": float(numeric_values.mean()),
        "p50": float(np.quantile(numeric_values, 0.5)),
        "p90": float(np.quantile(numeric_values, 0.9)),
        "p99": float(np.quantile(numeric_values, 0.99)),
        "sum": float(numeric_values.sum()),
    }


def _compare_values(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate_array = np.asarray(candidate)
    reference_array = np.asarray(reference)
    if candidate_array.dtype.kind in {"U", "S", "O"} or reference_array.dtype.kind in {
        "U",
        "S",
        "O",
    }:
        return _compare_categorical(candidate_array.astype(str), reference_array.astype(str))

    candidate_numeric = pd.Series(candidate_array).replace([np.inf, -np.inf], np.nan).dropna()
    reference_numeric = pd.Series(reference_array).replace([np.inf, -np.inf], np.nan).dropna()
    candidate_unique = int(candidate_numeric.nunique()) if not candidate_numeric.empty else 0
    reference_unique = int(reference_numeric.nunique()) if not reference_numeric.empty else 0
    if max(candidate_unique, reference_unique) <= 64 and (
        candidate_numeric.dtype.kind in {"i", "u", "b"}
        or reference_numeric.dtype.kind in {"i", "u", "b"}
    ):
        return _compare_categorical(
            candidate_numeric.astype(int).astype(str).to_numpy(),
            reference_numeric.astype(int).astype(str).to_numpy(),
        )
    return _compare_numeric(candidate_numeric.to_numpy(dtype=float), reference_numeric.to_numpy(dtype=float))


def _compare_categorical(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate_series = pd.Series(candidate).replace({"": pd.NA, "nan": pd.NA}).dropna()
    reference_series = pd.Series(reference).replace({"": pd.NA, "nan": pd.NA}).dropna()
    candidate_support = set(candidate_series.astype(str))
    reference_support = set(reference_series.astype(str))
    missing = sorted(reference_support - candidate_support)
    return {
        "type": "categorical",
        "support_recall": _safe_ratio(len(candidate_support & reference_support), len(reference_support)),
        "support_precision": _safe_ratio(len(candidate_support & reference_support), len(candidate_support)),
        "missing_reference_values": missing[:20],
    }


def _compare_numeric(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if len(reference) == 0:
        return {"type": "numeric", "positive_share_ratio": 0.0, "row_count_ratio": 0.0}
    candidate_positive = candidate > 0
    reference_positive = reference > 0
    return {
        "type": "numeric",
        "row_count_ratio": _safe_ratio(len(candidate), len(reference)),
        "candidate_zero_share": float((candidate == 0).mean()) if len(candidate) else 0.0,
        "reference_zero_share": float((reference == 0).mean()),
        "candidate_positive_share": float(candidate_positive.mean()) if len(candidate) else 0.0,
        "reference_positive_share": float(reference_positive.mean()),
        "positive_share_ratio": _safe_ratio(
            float(candidate_positive.mean()) if len(candidate) else 0.0,
            float(reference_positive.mean()),
        ),
    }


def _boolean_series(values: pd.Series) -> pd.Series:
    if values.dtype.kind == "b":
        return values.fillna(False)
    if values.dtype.kind in {"i", "u", "f"}:
        return values.fillna(0).astype(float) > 0
    normalized = values.astype(str).str.lower()
    return normalized.isin({"1", "true", "t", "yes"})


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dataset")
    parser.add_argument("reference_dataset")
    parser.add_argument("output_path")
    parser.add_argument("--period", type=int, default=2024)
    args = parser.parse_args()

    output = write_us_pre_sim_parity_audit(
        args.candidate_dataset,
        args.reference_dataset,
        args.output_path,
        period=args.period,
    )
    print(output)


if __name__ == "__main__":
    main()
