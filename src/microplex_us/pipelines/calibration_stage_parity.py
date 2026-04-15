"""Audit synthetic vs calibrated stage outputs, with optional PE reference context."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microplex_us.pipelines.pre_sim_parity import (
    DEFAULT_PRE_SIM_FOCUS_VARIABLES,
    PreSimParityVariableSpec,
)
from microplex_us.pipelines.source_stage_parity import (
    _compare_series,
    _resolve_bundle_variable,
    _summarize_series,
)
from microplex_us.policyengine.us import load_policyengine_us_entity_tables


def build_us_calibration_stage_parity_audit(
    synthetic_data: str | Path,
    calibrated_data: str | Path,
    *,
    reference_dataset: str | Path | None = None,
    period: int = 2024,
    focus_variables: tuple[PreSimParityVariableSpec | str, ...]
    | list[PreSimParityVariableSpec | str] = DEFAULT_PRE_SIM_FOCUS_VARIABLES,
) -> dict[str, Any]:
    """Compare synthetic vs calibrated stage rows, with optional PE reference."""

    synthetic_path = Path(synthetic_data).resolve()
    calibrated_path = Path(calibrated_data).resolve()
    synthetic_rows = pd.read_parquet(synthetic_path)
    calibrated_rows = pd.read_parquet(calibrated_path)
    focus_specs = _normalize_focus_variable_specs(focus_variables)
    reference_bundle = (
        load_policyengine_us_entity_tables(Path(reference_dataset).resolve(), period=period)
        if reference_dataset is not None
        else None
    )

    return {
        "schemaVersion": 1,
        "comparisonStage": "calibration",
        "period": int(period),
        "synthetic_data": str(synthetic_path),
        "calibrated_data": str(calibrated_path),
        "reference_dataset": str(Path(reference_dataset).resolve())
        if reference_dataset is not None
        else None,
        "rowStructure": {
            "synthetic": _row_structure_summary(synthetic_rows),
            "calibrated": _row_structure_summary(calibrated_rows),
        },
        "weightDiagnostics": {
            "synthetic": _household_weight_diagnostics(synthetic_rows),
            "calibrated": _household_weight_diagnostics(calibrated_rows),
        },
        "focusVariables": {
            spec.label: _calibration_variable_comparison(
                synthetic_rows=synthetic_rows,
                calibrated_rows=calibrated_rows,
                reference_bundle=reference_bundle,
                spec=spec,
            )
            for spec in focus_specs
        },
    }


def write_us_calibration_stage_parity_audit(
    synthetic_data: str | Path,
    calibrated_data: str | Path,
    output_path: str | Path,
    *,
    reference_dataset: str | Path | None = None,
    period: int = 2024,
    focus_variables: tuple[PreSimParityVariableSpec | str, ...]
    | list[PreSimParityVariableSpec | str] = DEFAULT_PRE_SIM_FOCUS_VARIABLES,
) -> Path:
    """Persist one calibration-stage parity audit as JSON."""

    output = Path(output_path).resolve()
    payload = build_us_calibration_stage_parity_audit(
        synthetic_data,
        calibrated_data,
        reference_dataset=reference_dataset,
        period=period,
        focus_variables=focus_variables,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def _calibration_variable_comparison(
    *,
    synthetic_rows: pd.DataFrame,
    calibrated_rows: pd.DataFrame,
    reference_bundle,
    spec: PreSimParityVariableSpec,
) -> dict[str, Any]:
    synthetic_weights = _stage_weight_series(synthetic_rows)
    calibrated_weights = _stage_weight_series(calibrated_rows)

    result: dict[str, Any] = {
        "candidate_variable": spec.candidate_variable,
        "reference_variable": spec.resolved_reference_variable,
        "synthetic_present": spec.candidate_variable in synthetic_rows.columns,
        "calibrated_present": spec.candidate_variable in calibrated_rows.columns,
        "reference_present": False,
    }

    if spec.candidate_variable in synthetic_rows.columns:
        result["synthetic"] = _summarize_series(
            synthetic_rows[spec.candidate_variable],
            weights=synthetic_weights,
            value_kind=spec.value_kind,
        )
    if spec.candidate_variable in calibrated_rows.columns:
        result["calibrated"] = _summarize_series(
            calibrated_rows[spec.candidate_variable],
            weights=calibrated_weights,
            value_kind=spec.value_kind,
        )
    if (
        spec.candidate_variable in synthetic_rows.columns
        and spec.candidate_variable in calibrated_rows.columns
    ):
        result["calibrated_vs_synthetic"] = _compare_series(
            calibrated_rows[spec.candidate_variable],
            synthetic_rows[spec.candidate_variable],
            candidate_weights=calibrated_weights,
            reference_weights=synthetic_weights,
            value_kind=spec.value_kind,
        )

    if reference_bundle is not None:
        reference_entry = _resolve_bundle_variable(
            reference_bundle,
            spec.resolved_reference_variable,
        )
        if reference_entry is not None:
            result["reference_present"] = True
            result["reference_entity"] = reference_entry["entity"].value
            result["reference"] = _summarize_series(
                reference_entry["series"],
                weights=reference_entry["weights"],
                value_kind=spec.value_kind,
            )
            if spec.candidate_variable in calibrated_rows.columns:
                result["calibrated_vs_reference"] = _compare_series(
                    calibrated_rows[spec.candidate_variable],
                    reference_entry["series"],
                    candidate_weights=calibrated_weights,
                    reference_weights=reference_entry["weights"],
                    value_kind=spec.value_kind,
                )
            if spec.candidate_variable in synthetic_rows.columns:
                result["synthetic_vs_reference"] = _compare_series(
                    synthetic_rows[spec.candidate_variable],
                    reference_entry["series"],
                    candidate_weights=synthetic_weights,
                    reference_weights=reference_entry["weights"],
                    value_kind=spec.value_kind,
                )

    return result


def _row_structure_summary(rows: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"row_count": int(len(rows))}
    if "household_id" in rows.columns:
        household_ids = pd.to_numeric(rows["household_id"], errors="coerce")
        summary["household_count"] = int(household_ids.nunique(dropna=True))
        if summary["household_count"] > 0:
            rows_per_household = rows.groupby("household_id", observed=True).size()
            summary["mean_rows_per_household"] = float(rows_per_household.mean())
    return summary


def _household_weight_diagnostics(rows: pd.DataFrame) -> dict[str, Any]:
    if "household_id" not in rows.columns:
        weights = _stage_weight_series(rows)
        return _weight_summary(weights)
    households = (
        rows.loc[:, ["household_id", _stage_weight_column(rows)]]
        .dropna(subset=["household_id"])
        .drop_duplicates(subset=["household_id"])
    )
    weights = pd.to_numeric(
        households[_stage_weight_column(rows)],
        errors="coerce",
    ).fillna(0.0)
    summary = _weight_summary(weights)
    summary["household_count"] = int(len(households))
    return summary


def _weight_summary(weights: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float)
    if values.empty:
        return {
            "total_weight": 0.0,
            "mean_weight": 0.0,
            "p50_weight": 0.0,
            "p90_weight": 0.0,
            "p99_weight": 0.0,
            "max_weight": 0.0,
            "effective_sample_size": 0.0,
        }
    total_weight = float(values.sum())
    ess = 0.0
    denom = float(np.square(values.to_numpy(dtype=float)).sum())
    if denom > 0.0:
        ess = float((total_weight**2) / denom)
    return {
        "total_weight": total_weight,
        "mean_weight": float(values.mean()),
        "p50_weight": float(values.quantile(0.5)),
        "p90_weight": float(values.quantile(0.9)),
        "p99_weight": float(values.quantile(0.99)),
        "max_weight": float(values.max()),
        "effective_sample_size": ess,
    }


def _stage_weight_column(rows: pd.DataFrame) -> str:
    for candidate in ("weight", "household_weight"):
        if candidate in rows.columns:
            return candidate
    raise ValueError("Stage rows must contain either 'weight' or 'household_weight'")


def _stage_weight_series(rows: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(rows[_stage_weight_column(rows)], errors="coerce").fillna(0.0)


def _normalize_focus_variable_specs(
    focus_variables: tuple[PreSimParityVariableSpec | str, ...]
    | list[PreSimParityVariableSpec | str],
) -> tuple[PreSimParityVariableSpec, ...]:
    specs: list[PreSimParityVariableSpec] = []
    seen_labels: set[str] = set()
    for variable in focus_variables:
        spec = (
            variable
            if isinstance(variable, PreSimParityVariableSpec)
            else PreSimParityVariableSpec(str(variable), str(variable))
        )
        if spec.label in seen_labels:
            continue
        seen_labels.add(spec.label)
        specs.append(spec)
    return tuple(specs)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("synthetic_data")
    parser.add_argument("calibrated_data")
    parser.add_argument("output_path")
    parser.add_argument("--reference-dataset")
    parser.add_argument("--period", type=int, default=2024)
    args = parser.parse_args()

    output = write_us_calibration_stage_parity_audit(
        args.synthetic_data,
        args.calibrated_data,
        args.output_path,
        reference_dataset=args.reference_dataset,
        period=args.period,
    )
    print(output)


if __name__ == "__main__":
    main()
