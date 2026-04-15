"""Summarize child-linked AGI component drift across artifact stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DEFAULT_VARIABLES = (
    "total_person_income",
    "income",
    "employment_income",
    "wage_income",
    "self_employment_income",
    "gross_social_security",
    "ssi",
    "public_assistance",
    "pension_income",
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "taxable_pension_income",
    "dividend_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "rental_income",
    "partnership_s_corp_income",
)

DEFAULT_STAGE_FILES = {
    "seed": "seed_data.parquet",
    "calibrated": "calibrated_data.parquet",
    "synthetic": "synthetic_data.parquet",
}


def _resolve_artifact_dir(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        if (candidate / "manifest.json").exists():
            return candidate
        for filename in DEFAULT_STAGE_FILES.values():
            if (candidate / filename).exists():
                return candidate
        manifest = next(candidate.glob("**/manifest.json"), None)
        if manifest is None:
            raise FileNotFoundError(f"No manifest.json found under {candidate}")
        return manifest.parent
    if candidate.name == "manifest.json":
        return candidate.parent
    raise FileNotFoundError(f"Expected an artifact directory or manifest.json, got {candidate}")


def _summarize_variable(frame: pd.DataFrame, variable: str) -> dict[str, float]:
    series = pd.to_numeric(
        frame.get(variable, pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    count = int(series.shape[0])
    if count == 0:
        return {"count": 0, "sum": 0.0, "mean": 0.0, "nonzero_share": 0.0}
    nonzero = (series != 0).sum()
    return {
        "count": count,
        "sum": float(series.sum()),
        "mean": float(series.mean()),
        "nonzero_share": float(nonzero / count),
    }


def _summarize_frame(frame: pd.DataFrame, variables: Iterable[str]) -> dict[str, Any]:
    age = pd.to_numeric(frame.get("age", pd.Series([], dtype=float)), errors="coerce")
    if "is_tax_unit_dependent" in frame.columns:
        is_dependent = pd.to_numeric(frame["is_tax_unit_dependent"], errors="coerce")
    else:
        is_dependent = pd.to_numeric(
            frame.get("is_dependent", pd.Series([], dtype=float)), errors="coerce"
        )
    subsets = {
        "all": frame.index,
        "under_20": frame.index[age.fillna(-1) < 20],
        "dependents_under_20": frame.index[
            (age.fillna(-1) < 20) & (is_dependent.fillna(0) > 0)
        ],
        "adults": frame.index[age.fillna(-1) >= 20],
    }
    result: dict[str, Any] = {
        "row_count": int(frame.shape[0]),
        "subsets": {},
        "tax_unit_subsets": {},
    }
    for subset_name, index in subsets.items():
        subset = frame.loc[index]
        result["subsets"][subset_name] = {
            variable: _summarize_variable(subset, variable)
            for variable in variables
        }
    if "tax_unit_id" in frame.columns:
        tax_unit_ids = frame["tax_unit_id"].astype(str)
        tax_unit_flags = pd.DataFrame(
            {
                "tax_unit_id": tax_unit_ids,
                "has_child": age.fillna(-1).lt(20).groupby(tax_unit_ids).transform("max"),
            }
        )
        available_vars = [var for var in variables if var in frame.columns]
        tax_unit_agg = frame.loc[:, ["tax_unit_id", *available_vars]].copy()
        tax_unit_agg = tax_unit_agg.groupby("tax_unit_id").sum(numeric_only=True)
        tax_unit_flags = (
            tax_unit_flags.drop_duplicates("tax_unit_id")
            .set_index("tax_unit_id")
            .reindex(tax_unit_agg.index)
        )
        tax_unit_agg["has_child"] = tax_unit_flags["has_child"].fillna(0).astype(float)
        tax_subsets = {
            "all": tax_unit_agg.index,
            "with_children": tax_unit_agg.index[tax_unit_agg["has_child"] > 0],
            "without_children": tax_unit_agg.index[tax_unit_agg["has_child"] == 0],
        }
        result["tax_unit_row_count"] = int(tax_unit_agg.shape[0])
        for subset_name, index in tax_subsets.items():
            subset = tax_unit_agg.loc[index]
            result["tax_unit_subsets"][subset_name] = {
                variable: _summarize_variable(subset, variable)
                for variable in variables
            }
    return result


def summarize_child_tax_unit_agi_drift(
    artifact_path: str | Path,
    *,
    variables: Iterable[str] = DEFAULT_VARIABLES,
    stage_files: dict[str, str] = DEFAULT_STAGE_FILES,
) -> dict[str, Any]:
    """Summarize child-linked AGI component drift for one artifact bundle."""
    artifact_dir = _resolve_artifact_dir(artifact_path)
    payload: dict[str, Any] = {
        "artifact_path": str(artifact_dir),
        "variables": list(variables),
        "stages": {},
    }
    for stage, filename in stage_files.items():
        file_path = artifact_dir / filename
        if not file_path.exists():
            continue
        frame = pd.read_parquet(file_path)
        payload["stages"][stage] = _summarize_frame(frame, variables)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize child-linked AGI component drift for one artifact."
    )
    parser.add_argument("artifact", help="Artifact directory or manifest.json path.")
    parser.add_argument("--out", help="Optional JSON output path.")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=list(DEFAULT_VARIABLES),
        help="Variables to summarize.",
    )
    args = parser.parse_args(argv)

    payload = summarize_child_tax_unit_agi_drift(
        args.artifact,
        variables=tuple(args.variables),
    )
    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).expanduser().write_text(output)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
