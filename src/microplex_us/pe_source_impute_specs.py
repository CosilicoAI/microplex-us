"""Shared PE source-impute donor block specs loaded from manifest data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd
from microplex.core import SourceArchetype


@dataclass(frozen=True)
class PERawIndicatorSpec:
    """One manifest-backed raw indicator rule."""

    column: str
    equals: str | int | float | bool


@dataclass(frozen=True)
class PESourceImputeRawLoaderSpec:
    """Declarative raw-file extraction contract for one donor block."""

    filename: str
    delimiter: str | None
    usecols: tuple[str, ...]
    direct_columns: dict[str, str]
    sum_columns_contains: dict[str, str]
    indicator_columns: dict[str, PERawIndicatorSpec]
    int_columns: tuple[str, ...]
    household_id_parts: tuple[str, ...]
    person_id_parts: tuple[str, ...]
    constant_columns: dict[str, str | int | float | bool]
    copy_columns: dict[str, str]


@dataclass(frozen=True)
class PEPolicyengineDatasetLoaderSpec:
    """Declarative subprocess dataset-loader contract for one donor block."""

    module: str
    class_name: str
    builder_kind: str
    household_index_key: str | None
    person_household_key: str | None
    person_id_key: str | None
    length_source_key: str | None
    direct_person_columns: dict[str, str]
    boolean_person_columns: dict[str, str]
    row_indexed_person_columns: dict[str, str]
    mapped_row_person_columns: dict[str, str]
    mapped_value_tables: dict[str, dict[str, int]]
    fallback_person_columns: dict[str, tuple[str, ...]]
    copy_person_columns: dict[str, str]
    constant_person_columns: dict[str, str | int | float | bool]
    income_sum_columns: tuple[str, ...]
    int_person_columns: tuple[str, ...]
    sex_from_boolean_source: str | None
    sex_true_value: int | None
    sex_false_value: int | None
    generated_household_ids: bool
    person_id_from_household_id: bool


@dataclass(frozen=True)
class PESourceImputeBlockSpec:
    """Declarative contract for one PE donor-survey block."""

    key: str
    survey_name: str
    block_name: str | None
    default_year: int
    archetype: SourceArchetype | None
    dataset_loader: PEPolicyengineDatasetLoaderSpec | None
    raw_loader: PESourceImputeRawLoaderSpec | None
    required_monthcode: int | None
    annualized_variables: tuple[str, ...]
    household_count_variables: tuple[str, ...]
    household_variables: tuple[str, ...]
    person_variables: tuple[str, ...]
    target_variables: tuple[str, ...]
    predictors: tuple[str, ...]

    @property
    def descriptor_name(self) -> str:
        if self.block_name is None:
            return self.survey_name
        return f"{self.survey_name}_{self.block_name}"

    def source_name(self, year: int) -> str:
        return f"{self.descriptor_name}_{year}"


def _manifest_path() -> Path:
    return Path(__file__).resolve().parent / "manifests" / "pe_source_impute_blocks.json"


def _archetype_from_name(value: str | None) -> SourceArchetype | None:
    if value is None:
        return None
    return SourceArchetype(value)


def _raw_indicator_from_payload(payload: dict[str, Any]) -> PERawIndicatorSpec:
    return PERawIndicatorSpec(
        column=str(payload["column"]),
        equals=payload["equals"],
    )


def _raw_loader_from_payload(
    payload: dict[str, Any] | None,
) -> PESourceImputeRawLoaderSpec | None:
    if payload is None:
        return None
    return PESourceImputeRawLoaderSpec(
        filename=str(payload["filename"]),
        delimiter=payload.get("delimiter"),
        usecols=tuple(payload.get("usecols", ())),
        direct_columns={str(key): str(value) for key, value in payload.get("direct_columns", {}).items()},
        sum_columns_contains={
            str(key): str(value)
            for key, value in payload.get("sum_columns_contains", {}).items()
        },
        indicator_columns={
            str(key): _raw_indicator_from_payload(value)
            for key, value in payload.get("indicator_columns", {}).items()
        },
        int_columns=tuple(payload.get("int_columns", ())),
        household_id_parts=tuple(payload.get("household_id_parts", ())),
        person_id_parts=tuple(payload.get("person_id_parts", ())),
        constant_columns={
            str(key): value
            for key, value in payload.get("constant_columns", {}).items()
        },
        copy_columns={str(key): str(value) for key, value in payload.get("copy_columns", {}).items()},
    )


def _dataset_loader_from_payload(
    payload: dict[str, Any] | None,
) -> PEPolicyengineDatasetLoaderSpec | None:
    if payload is None:
        return None
    return PEPolicyengineDatasetLoaderSpec(
        module=str(payload["module"]),
        class_name=str(payload["class_name"]),
        builder_kind=str(payload["builder_kind"]),
        household_index_key=payload.get("household_index_key"),
        person_household_key=payload.get("person_household_key"),
        person_id_key=payload.get("person_id_key"),
        length_source_key=payload.get("length_source_key"),
        direct_person_columns={
            str(key): str(value)
            for key, value in payload.get("direct_person_columns", {}).items()
        },
        boolean_person_columns={
            str(key): str(value)
            for key, value in payload.get("boolean_person_columns", {}).items()
        },
        row_indexed_person_columns={
            str(key): str(value)
            for key, value in payload.get("row_indexed_person_columns", {}).items()
        },
        mapped_row_person_columns={
            str(key): str(value)
            for key, value in payload.get("mapped_row_person_columns", {}).items()
        },
        mapped_value_tables={
            str(key): {
                str(mapped_key): int(mapped_value)
                for mapped_key, mapped_value in value.items()
            }
            for key, value in payload.get("mapped_value_tables", {}).items()
        },
        fallback_person_columns={
            str(key): tuple(str(item) for item in value)
            for key, value in payload.get("fallback_person_columns", {}).items()
        },
        copy_person_columns={
            str(key): str(value)
            for key, value in payload.get("copy_person_columns", {}).items()
        },
        constant_person_columns={
            str(key): value
            for key, value in payload.get("constant_person_columns", {}).items()
        },
        income_sum_columns=tuple(payload.get("income_sum_columns", ())),
        int_person_columns=tuple(payload.get("int_person_columns", ())),
        sex_from_boolean_source=payload.get("sex_from_boolean_source"),
        sex_true_value=(
            None if payload.get("sex_true_value") is None else int(payload["sex_true_value"])
        ),
        sex_false_value=(
            None if payload.get("sex_false_value") is None else int(payload["sex_false_value"])
        ),
        generated_household_ids=bool(payload.get("generated_household_ids", False)),
        person_id_from_household_id=bool(payload.get("person_id_from_household_id", False)),
    )


def _spec_from_payload(key: str, payload: dict[str, Any]) -> PESourceImputeBlockSpec:
    return PESourceImputeBlockSpec(
        key=key,
        survey_name=str(payload["survey_name"]),
        block_name=payload.get("block_name"),
        default_year=int(payload["default_year"]),
        archetype=_archetype_from_name(payload.get("archetype")),
        dataset_loader=_dataset_loader_from_payload(payload.get("dataset_loader")),
        raw_loader=_raw_loader_from_payload(payload.get("raw_loader")),
        required_monthcode=(
            None
            if payload.get("required_monthcode") is None
            else int(payload["required_monthcode"])
        ),
        annualized_variables=tuple(payload.get("annualized_variables", ())),
        household_count_variables=tuple(payload.get("household_count_variables", ())),
        household_variables=tuple(payload["household_variables"]),
        person_variables=tuple(payload["person_variables"]),
        target_variables=tuple(payload["target_variables"]),
        predictors=tuple(payload["predictors"]),
    )


@cache
def load_pe_source_impute_block_specs() -> dict[str, PESourceImputeBlockSpec]:
    """Load the PE donor-block spec manifest."""
    with _manifest_path().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    blocks = payload.get("blocks", {})
    return {
        key: _spec_from_payload(key, value)
        for key, value in blocks.items()
    }


def get_pe_source_impute_block_spec(key: str) -> PESourceImputeBlockSpec:
    """Return one named PE donor-block spec."""
    specs = load_pe_source_impute_block_specs()
    try:
        return specs[key]
    except KeyError as error:
        available = ", ".join(sorted(specs))
        raise KeyError(f"Unknown PE source-impute block '{key}'. Expected one of: {available}") from error


def resolve_sipp_source_impute_block_spec(block: str) -> PESourceImputeBlockSpec:
    """Resolve one SIPP donor block by short block name."""
    return get_pe_source_impute_block_spec(f"sipp_{block}")


def resolve_pe_source_impute_block_key(
    *,
    donor_source_name: str | None,
    donor_block: tuple[str, ...],
) -> str | None:
    """Map a donor source name and target block to one manifest block key."""
    normalized_name = (donor_source_name or "").strip().lower()
    block_set = set(donor_block)
    for key, spec in load_pe_source_impute_block_specs().items():
        if spec.survey_name not in normalized_name:
            continue
        if block_set <= set(spec.target_variables):
            return key
    return None


_HOUSEHOLD_COUNT_AGE_THRESHOLDS = {
    "count_under_18": 18,
    "count_under_6": 6,
}


def apply_pe_source_impute_loader_postprocess(
    frame: pd.DataFrame,
    spec: PESourceImputeBlockSpec,
    *,
    month_column: str = "MONTHCODE",
    household_key: str = "household_id",
    age_column: str = "age",
) -> pd.DataFrame:
    """Apply manifest-backed donor-row postprocessing for one PE block."""
    result = frame.copy()
    if spec.required_monthcode is not None and month_column in result.columns:
        monthcode = pd.to_numeric(result[month_column], errors="coerce")
        result = result[monthcode.eq(spec.required_monthcode)].copy()
    for variable in spec.annualized_variables:
        if variable not in result.columns:
            continue
        result[variable] = pd.to_numeric(result[variable], errors="coerce").fillna(0.0) * 12.0
    if household_key not in result.columns or age_column not in result.columns:
        return result
    ages = pd.to_numeric(result[age_column], errors="coerce").fillna(0.0)
    for variable in spec.household_count_variables:
        threshold = _HOUSEHOLD_COUNT_AGE_THRESHOLDS.get(variable)
        if threshold is None:
            continue
        result[variable] = (
            ages.lt(threshold)
            .groupby(result[household_key], dropna=False)
            .transform("sum")
            .astype(float)
        )
    return result


def prepare_pe_source_impute_condition_frame(
    frame: pd.DataFrame,
    spec: PESourceImputeBlockSpec,
) -> pd.DataFrame:
    """Derive the manifest-backed PE condition surface for one donor block."""
    prepared = frame.copy()
    zero = pd.Series(0.0, index=prepared.index, dtype=float)
    required = set(spec.predictors)

    def first_present(*columns: str) -> pd.Series:
        for column in columns:
            if column in prepared.columns:
                return (
                    pd.to_numeric(prepared[column], errors="coerce")
                    .fillna(0.0)
                    .astype(float)
                )
        return zero.copy()

    if "is_male" in required and "is_male" not in prepared.columns and "sex" in prepared.columns:
        sex = pd.to_numeric(prepared["sex"], errors="coerce").fillna(0)
        prepared["is_male"] = sex.eq(1).astype(float)
    elif "is_male" in required and "is_male" in prepared.columns:
        prepared["is_male"] = pd.to_numeric(prepared["is_male"], errors="coerce").fillna(0.0)

    if "is_female" in required and "is_female" not in prepared.columns and "sex" in prepared.columns:
        sex = pd.to_numeric(prepared["sex"], errors="coerce").fillna(0)
        prepared["is_female"] = sex.eq(2).astype(float)
    elif "is_female" in required and "is_female" in prepared.columns:
        prepared["is_female"] = pd.to_numeric(prepared["is_female"], errors="coerce").fillna(0.0)

    if "is_household_head" in required and "is_household_head" not in prepared.columns:
        if "is_head" in prepared.columns:
            prepared["is_household_head"] = (
                pd.to_numeric(prepared["is_head"], errors="coerce").fillna(0.0).astype(float)
            )

    if "tenure_type" in required and "tenure_type" not in prepared.columns and "tenure" in prepared.columns:
        prepared["tenure_type"] = (
            pd.to_numeric(prepared["tenure"], errors="coerce").fillna(0.0).astype(float)
        )

    if "social_security" in required and "social_security" not in prepared.columns:
        prepared["social_security"] = first_present("gross_social_security", "social_security")

    if "pension_income" in required and "pension_income" not in prepared.columns:
        prepared["pension_income"] = first_present("taxable_pension_income", "pension_income")

    if "interest_dividend_income" in required and "interest_dividend_income" not in prepared.columns:
        prepared["interest_dividend_income"] = (
            first_present("taxable_interest_income", "interest_income")
            + first_present("ordinary_dividend_income", "dividend_income")
        )

    if (
        "social_security_pension_income" in required
        and "social_security_pension_income" not in prepared.columns
    ):
        prepared["social_security_pension_income"] = (
            first_present("social_security", "gross_social_security")
            + first_present("pension_income", "taxable_pension_income")
        )

    if "is_married" in required and "is_married" not in prepared.columns:
        if "filing_status" in prepared.columns:
            filing_status = prepared["filing_status"].astype(str)
            prepared["is_married"] = filing_status.eq("JOINT").astype(float)
        elif "marital_status" in prepared.columns:
            marital_status = (
                pd.to_numeric(prepared["marital_status"], errors="coerce").fillna(0).astype(int)
            )
            prepared["is_married"] = marital_status.isin({1, 2}).astype(float)

    household_key = next(
        (candidate for candidate in ("household_id", "spm_unit_id", "family_id") if candidate in prepared.columns),
        None,
    )
    if household_key is not None:
        household_groups = prepared.groupby(household_key, dropna=False)
        if "household_size" in required and "household_size" not in prepared.columns:
            prepared["household_size"] = (
                household_groups[household_key].transform("size").astype(float)
            )
        if "age" in prepared.columns:
            ages = pd.to_numeric(prepared["age"], errors="coerce").fillna(0.0)
            for variable in required & set(_HOUSEHOLD_COUNT_AGE_THRESHOLDS):
                if variable in prepared.columns:
                    continue
                threshold = _HOUSEHOLD_COUNT_AGE_THRESHOLDS[variable]
                prepared[variable] = (
                    ages.lt(threshold)
                    .groupby(prepared[household_key], dropna=False)
                    .transform("sum")
                    .astype(float)
                )

    if (
        "own_children_in_household" in required
        and "own_children_in_household" not in prepared.columns
        and "count_under_18" in prepared.columns
    ):
        prepared["own_children_in_household"] = (
            pd.to_numeric(prepared["count_under_18"], errors="coerce")
            .fillna(0.0)
            .gt(0.0)
            .astype(float)
        )

    return prepared
