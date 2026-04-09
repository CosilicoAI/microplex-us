"""Stage-matched raw source parity audits for CPS and PUF."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pandas as pd
from microplex.core import (
    EntityType,
    ObservationFrame,
    SourceQuery,
)

from microplex_us.data_sources.cps import CPSASECSourceProvider
from microplex_us.data_sources.puf import (
    PUF_UPRATING_MODE_INTERPOLATED,
    SOCIAL_SECURITY_SPLIT_STRATEGY_GROUPED_SHARE,
    PUFSourceProvider,
)
from microplex_us.policyengine.us import (
    PolicyEngineUSEntityTableBundle,
    _build_group_household_map,
    _decode_policyengine_array,
    _infer_policyengine_array_entity,
    _normalize_id_value,
    _normalize_weight_value,
    _policyengine_group_entity_type,
    _resolve_policyengine_us_tax_benefit_system,
    _resolve_prefixed_policyengine_table,
    load_policyengine_us_entity_tables,
)

_ENTITY_ORDER: tuple[EntityType, ...] = (
    EntityType.HOUSEHOLD,
    EntityType.PERSON,
    EntityType.TAX_UNIT,
    EntityType.SPM_UNIT,
    EntityType.FAMILY,
)
_ENTITY_ID_COLUMNS: dict[EntityType, str] = {
    EntityType.HOUSEHOLD: "household_id",
    EntityType.PERSON: "person_id",
    EntityType.TAX_UNIT: "tax_unit_id",
    EntityType.SPM_UNIT: "spm_unit_id",
    EntityType.FAMILY: "family_id",
}
_ENTITY_BASE_COLUMNS: dict[EntityType, set[str]] = {
    EntityType.HOUSEHOLD: {"household_id", "household_weight"},
    EntityType.PERSON: {"person_id", "household_id", "weight"},
    EntityType.TAX_UNIT: {"tax_unit_id", "household_id"},
    EntityType.SPM_UNIT: {"spm_unit_id", "household_id"},
    EntityType.FAMILY: {"family_id", "household_id"},
}
_HOUSEHOLD_SIZE_BUCKETS: tuple[str, ...] = ("1", "2", "3", "4", "5", "6", "7+")


@dataclass(frozen=True)
class SourceStageParityVariableSpec:
    """One semantic variable comparison between candidate and reference stages."""

    label: str
    candidate_variable: str
    reference_variable: str | None = None
    value_kind: str = "auto"

    @property
    def resolved_reference_variable(self) -> str:
        return self.reference_variable or self.candidate_variable


DEFAULT_CPS_SOURCE_STAGE_FOCUS_VARIABLES: tuple[SourceStageParityVariableSpec, ...] = (
    SourceStageParityVariableSpec("age", "age", value_kind="numeric"),
    SourceStageParityVariableSpec("state_fips", "state_fips", value_kind="categorical"),
    SourceStageParityVariableSpec("county_fips", "county_fips", value_kind="categorical"),
    SourceStageParityVariableSpec("cps_race", "cps_race", value_kind="categorical"),
    SourceStageParityVariableSpec("is_hispanic", "is_hispanic", value_kind="categorical"),
    SourceStageParityVariableSpec("is_disabled", "is_disabled", value_kind="categorical"),
    SourceStageParityVariableSpec("has_esi", "has_esi", value_kind="categorical"),
    SourceStageParityVariableSpec(
        "has_marketplace_health_coverage",
        "has_marketplace_health_coverage",
        value_kind="categorical",
    ),
    SourceStageParityVariableSpec(
        "employment_income",
        "wage_income",
        "employment_income",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec(
        "self_employment_income",
        "self_employment_income",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec(
        "taxable_interest_income",
        "interest_income",
        "taxable_interest_income",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec("rental_income", "rental_income", value_kind="numeric"),
    SourceStageParityVariableSpec(
        "medicare_part_b_premiums",
        "medicare_part_b_premiums",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec(
        "other_medical_expenses",
        "other_medical_expenses",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec(
        "over_the_counter_health_expenses",
        "over_the_counter_health_expenses",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec("receives_wic", "receives_wic", value_kind="categorical"),
    SourceStageParityVariableSpec("is_separated", "is_separated", value_kind="categorical"),
    SourceStageParityVariableSpec(
        "is_surviving_spouse",
        "is_surviving_spouse",
        value_kind="categorical",
    ),
    SourceStageParityVariableSpec(
        "social_security_retirement",
        "social_security_retirement",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec(
        "social_security_disability",
        "social_security_disability",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec(
        "social_security_survivors",
        "social_security_survivors",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec(
        "social_security_dependents",
        "social_security_dependents",
        value_kind="numeric",
    ),
)

DEFAULT_PUF_SOURCE_STAGE_FOCUS_VARIABLES: tuple[SourceStageParityVariableSpec, ...] = (
    SourceStageParityVariableSpec("age", "age", value_kind="numeric"),
    SourceStageParityVariableSpec("employment_income", "employment_income", value_kind="numeric"),
    SourceStageParityVariableSpec(
        "self_employment_income",
        "self_employment_income",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec(
        "taxable_interest_income",
        "taxable_interest_income",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec(
        "qualified_dividend_income",
        "qualified_dividend_income",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec(
        "non_qualified_dividend_income",
        "non_qualified_dividend_income",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec(
        "partnership_s_corp_income",
        "partnership_s_corp_income",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec("farm_income", "farm_income", value_kind="numeric"),
    SourceStageParityVariableSpec(
        "farm_operations_income",
        "farm_operations_income",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec(
        "farm_rent_income",
        "farm_rent_income",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec("rental_income", "rental_income", value_kind="numeric"),
    SourceStageParityVariableSpec("filing_status", "filing_status", value_kind="categorical"),
    SourceStageParityVariableSpec(
        "health_savings_account_ald",
        "health_savings_account_ald",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec(
        "self_employed_health_insurance_ald",
        "self_employed_health_insurance_ald",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec(
        "self_employed_pension_contribution_ald",
        "self_employed_pension_contribution_ald",
        value_kind="numeric",
    ),
    SourceStageParityVariableSpec(
        "pre_tax_contributions",
        "pre_tax_contributions",
        value_kind="numeric",
    ),
)


def observation_frame_to_policyengine_entity_bundle(
    frame: ObservationFrame,
) -> PolicyEngineUSEntityTableBundle:
    """Project a provider observation frame into a PE-style entity bundle."""

    households = _table_from_frame(frame, EntityType.HOUSEHOLD)
    persons = _table_from_frame(frame, EntityType.PERSON)
    if households is None or persons is None:
        raise ValueError(
            "Source-stage parity requires both household and person tables in the observation frame"
        )
    persons = persons.copy()
    households = households.copy()
    for entity in _ENTITY_ORDER:
        id_column = _ENTITY_ID_COLUMNS[entity]
        table = households if entity is EntityType.HOUSEHOLD else persons
        if id_column in table.columns:
            table[id_column] = _stringify_id_series(table[id_column])
    if "household_id" in persons.columns:
        persons["household_id"] = _stringify_id_series(persons["household_id"])
    if "household_id" in households.columns:
        households["household_id"] = _stringify_id_series(households["household_id"])

    tax_units = _group_table_from_persons(persons, "tax_unit_id")
    spm_units = _group_table_from_persons(persons, "spm_unit_id")
    families = _group_table_from_persons(persons, "family_id")
    marital_units = _group_table_from_persons(persons, "marital_unit_id")
    return PolicyEngineUSEntityTableBundle(
        households=households,
        persons=persons,
        tax_units=tax_units,
        spm_units=spm_units,
        families=families,
        marital_units=marital_units,
    )


def build_us_source_stage_parity_audit(
    candidate_bundle: PolicyEngineUSEntityTableBundle,
    reference_dataset: str | Path,
    *,
    source_id: str,
    period: int,
    focus_variables: tuple[SourceStageParityVariableSpec, ...]
    | list[SourceStageParityVariableSpec],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare one raw-source provider stage to a PE saved stage artifact."""

    reference_path = Path(reference_dataset).expanduser().resolve()
    reference_bundle = _load_reference_entity_bundle(reference_path, period=period)
    focus_specs = tuple(focus_variables)
    return {
        "schemaVersion": 1,
        "comparisonStage": "raw_source_provider",
        "sourceId": source_id,
        "period": int(period),
        "candidate": {
            "metadata": dict(metadata or {}),
        },
        "reference": {
            "datasetPath": str(reference_path),
        },
        "schema": _build_schema_summary(candidate_bundle, reference_bundle),
        "entityStructure": {
            "candidate": _entity_structure_summary(candidate_bundle),
            "reference": _entity_structure_summary(reference_bundle),
            "deltas": _numeric_deltas(
                _entity_structure_summary(candidate_bundle),
                _entity_structure_summary(reference_bundle),
            ),
        },
        "householdSizeDistribution": {
            "candidate": _weighted_household_size_distribution(candidate_bundle),
            "reference": _weighted_household_size_distribution(reference_bundle),
            "deltas": _distribution_deltas(
                _weighted_household_size_distribution(candidate_bundle),
                _weighted_household_size_distribution(reference_bundle),
            ),
        },
        "focusVariables": {
            spec.label: _focus_variable_comparison(
                candidate_bundle=candidate_bundle,
                reference_bundle=reference_bundle,
                spec=spec,
            )
            for spec in focus_specs
        },
    }


def write_us_source_stage_parity_audit(
    candidate_bundle: PolicyEngineUSEntityTableBundle,
    reference_dataset: str | Path,
    output_path: str | Path,
    *,
    source_id: str,
    period: int,
    focus_variables: tuple[SourceStageParityVariableSpec, ...]
    | list[SourceStageParityVariableSpec],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist one raw-source parity audit JSON."""

    destination = Path(output_path).expanduser().resolve()
    payload = build_us_source_stage_parity_audit(
        candidate_bundle,
        reference_dataset,
        source_id=source_id,
        period=period,
        focus_variables=focus_variables,
        metadata=metadata,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def build_us_cps_source_stage_parity_audit(
    reference_dataset: str | Path,
    *,
    year: int = 2023,
    cache_dir: str | Path | None = None,
    download: bool = True,
    sample_n: int | None = None,
    random_seed: int = 0,
    focus_variables: tuple[SourceStageParityVariableSpec, ...]
    | list[SourceStageParityVariableSpec] = DEFAULT_CPS_SOURCE_STAGE_FOCUS_VARIABLES,
) -> dict[str, Any]:
    """Run the raw CPS provider and compare it to a PE saved CPS artifact."""

    provider = CPSASECSourceProvider(
        year=year,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        download=download,
    )
    provider_filters: dict[str, Any] = {
        "year": int(year),
        "download": bool(download),
        "random_seed": int(random_seed),
    }
    if cache_dir is not None:
        provider_filters["cache_dir"] = str(Path(cache_dir).expanduser())
    if sample_n is not None:
        provider_filters["sample_n"] = int(sample_n)
    frame = provider.load_frame(SourceQuery(provider_filters=provider_filters))
    return build_us_source_stage_parity_audit(
        observation_frame_to_policyengine_entity_bundle(frame),
        reference_dataset,
        source_id="cps_asec",
        period=year,
        focus_variables=focus_variables,
        metadata={
            "candidateSourceName": frame.source.name,
            "providerFilters": provider_filters,
        },
    )


def write_us_cps_source_stage_parity_audit(
    reference_dataset: str | Path,
    output_path: str | Path,
    *,
    year: int = 2023,
    cache_dir: str | Path | None = None,
    download: bool = True,
    sample_n: int | None = None,
    random_seed: int = 0,
    focus_variables: tuple[SourceStageParityVariableSpec, ...]
    | list[SourceStageParityVariableSpec] = DEFAULT_CPS_SOURCE_STAGE_FOCUS_VARIABLES,
) -> Path:
    """Persist one raw CPS source-stage parity audit JSON."""

    payload = build_us_cps_source_stage_parity_audit(
        reference_dataset,
        year=year,
        cache_dir=cache_dir,
        download=download,
        sample_n=sample_n,
        random_seed=random_seed,
        focus_variables=focus_variables,
    )
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def build_us_puf_source_stage_parity_audit(
    reference_dataset: str | Path,
    *,
    target_year: int = 2024,
    cache_dir: str | Path | None = None,
    puf_path: str | Path | None = None,
    demographics_path: str | Path | None = None,
    sample_n: int | None = None,
    random_seed: int = 0,
    uprating_mode: str = PUF_UPRATING_MODE_INTERPOLATED,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
    impute_pre_tax_contributions: bool = False,
    pre_tax_training_year: int = 2024,
    require_pre_tax_contribution_model: bool = False,
    social_security_split_strategy: str = SOCIAL_SECURITY_SPLIT_STRATEGY_GROUPED_SHARE,
    focus_variables: tuple[SourceStageParityVariableSpec, ...]
    | list[SourceStageParityVariableSpec] = DEFAULT_PUF_SOURCE_STAGE_FOCUS_VARIABLES,
) -> dict[str, Any]:
    """Run the raw PUF provider and compare it to a PE saved PUF artifact."""

    provider = PUFSourceProvider(
        target_year=target_year,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        puf_path=puf_path,
        demographics_path=demographics_path,
        uprating_mode=uprating_mode,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
        impute_pre_tax_contributions=impute_pre_tax_contributions,
        pre_tax_training_year=pre_tax_training_year,
        require_pre_tax_contribution_model=require_pre_tax_contribution_model,
        social_security_split_strategy=social_security_split_strategy,
    )
    provider_filters: dict[str, Any] = {
        "target_year": int(target_year),
        "random_seed": int(random_seed),
        "uprating_mode": uprating_mode,
        "social_security_split_strategy": social_security_split_strategy,
        "impute_pre_tax_contributions": bool(impute_pre_tax_contributions),
        "pre_tax_training_year": int(pre_tax_training_year),
        "require_pre_tax_contribution_model": bool(require_pre_tax_contribution_model),
    }
    for key, value in (
        ("cache_dir", cache_dir),
        ("puf_path", puf_path),
        ("demographics_path", demographics_path),
        ("policyengine_us_data_repo", policyengine_us_data_repo),
        ("policyengine_us_data_python", policyengine_us_data_python),
    ):
        if value is not None:
            provider_filters[key] = str(Path(value).expanduser())
    if sample_n is not None:
        provider_filters["sample_n"] = int(sample_n)
    frame = provider.load_frame(SourceQuery(provider_filters=provider_filters))
    return build_us_source_stage_parity_audit(
        observation_frame_to_policyengine_entity_bundle(frame),
        reference_dataset,
        source_id="irs_soi_puf",
        period=target_year,
        focus_variables=focus_variables,
        metadata={
            "candidateSourceName": frame.source.name,
            "providerFilters": provider_filters,
        },
    )


def write_us_puf_source_stage_parity_audit(
    reference_dataset: str | Path,
    output_path: str | Path,
    *,
    target_year: int = 2024,
    cache_dir: str | Path | None = None,
    puf_path: str | Path | None = None,
    demographics_path: str | Path | None = None,
    sample_n: int | None = None,
    random_seed: int = 0,
    uprating_mode: str = PUF_UPRATING_MODE_INTERPOLATED,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
    impute_pre_tax_contributions: bool = False,
    pre_tax_training_year: int = 2024,
    require_pre_tax_contribution_model: bool = False,
    social_security_split_strategy: str = SOCIAL_SECURITY_SPLIT_STRATEGY_GROUPED_SHARE,
    focus_variables: tuple[SourceStageParityVariableSpec, ...]
    | list[SourceStageParityVariableSpec] = DEFAULT_PUF_SOURCE_STAGE_FOCUS_VARIABLES,
) -> Path:
    """Persist one raw PUF source-stage parity audit JSON."""

    payload = build_us_puf_source_stage_parity_audit(
        reference_dataset,
        target_year=target_year,
        cache_dir=cache_dir,
        puf_path=puf_path,
        demographics_path=demographics_path,
        sample_n=sample_n,
        random_seed=random_seed,
        uprating_mode=uprating_mode,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
        impute_pre_tax_contributions=impute_pre_tax_contributions,
        pre_tax_training_year=pre_tax_training_year,
        require_pre_tax_contribution_model=require_pre_tax_contribution_model,
        social_security_split_strategy=social_security_split_strategy,
        focus_variables=focus_variables,
    )
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def _table_from_frame(
    frame: ObservationFrame,
    entity: EntityType,
) -> pd.DataFrame | None:
    table = frame.tables.get(entity)
    if table is None:
        return None
    if isinstance(table, pd.DataFrame):
        return table
    to_pandas = getattr(table, "to_pandas", None)
    if callable(to_pandas):
        return cast(pd.DataFrame, to_pandas())
    raise TypeError(f"Unsupported table type for entity '{entity.value}'")


def _load_reference_entity_bundle(
    reference_dataset: str | Path,
    *,
    period: int,
) -> PolicyEngineUSEntityTableBundle:
    reference_path = Path(reference_dataset).expanduser().resolve()
    if _is_flat_policyengine_h5(reference_path):
        return _load_flat_policyengine_us_entity_tables(reference_path)
    return load_policyengine_us_entity_tables(reference_path, period=period)


def _is_flat_policyengine_h5(path: Path) -> bool:
    with h5py.File(path, "r") as handle:
        for value in handle.values():
            return isinstance(value, h5py.Dataset)
    return False


def _load_flat_policyengine_us_entity_tables(
    dataset: str | Path,
) -> PolicyEngineUSEntityTableBundle:
    source = Path(dataset).expanduser().resolve()
    with h5py.File(source, "r") as handle:
        arrays = {
            variable: np.asarray(values)
            for variable, values in handle.items()
            if isinstance(values, h5py.Dataset)
        }

    required_structural = {
        "household_id",
        "person_id",
        "person_household_id",
    }
    missing = sorted(required_structural - set(arrays))
    if missing:
        raise ValueError(
            "PolicyEngine flat dataset is missing required structural arrays: "
            + ", ".join(missing)
        )

    households = pd.DataFrame(
        {"household_id": _normalize_id_value(arrays["household_id"])}
    )
    household_weight = arrays.get("household_weight")
    households["household_weight"] = (
        _normalize_weight_value(household_weight)
        if household_weight is not None
        else np.ones(len(households), dtype=float)
    )

    persons = pd.DataFrame(
        {
            "person_id": _normalize_id_value(arrays["person_id"]),
            "household_id": _normalize_id_value(arrays["person_household_id"]),
        }
    )
    if "person_weight" in arrays:
        persons["weight"] = _normalize_weight_value(arrays["person_weight"])

    group_specs = (
        ("tax_unit", "tax_unit_id", "person_tax_unit_id"),
        ("spm_unit", "spm_unit_id", "person_spm_unit_id"),
        ("family", "family_id", "person_family_id"),
        ("marital_unit", "marital_unit_id", "person_marital_unit_id"),
    )
    group_tables: dict[str, pd.DataFrame | None] = {}
    entity_lengths = {
        EntityType.HOUSEHOLD: len(households),
        EntityType.PERSON: len(persons),
    }
    excluded_variable_names = {
        "household_id",
        "household_weight",
        "person_id",
        "person_household_id",
        "person_weight",
    }
    for group_name, id_column, membership_column in group_specs:
        group_ids = arrays.get(id_column)
        membership = arrays.get(membership_column)
        if membership is not None:
            persons[id_column] = _normalize_id_value(membership)
        if group_ids is None:
            group_tables[group_name] = None
            continue
        group_table = pd.DataFrame({id_column: _normalize_id_value(group_ids)})
        if membership is not None:
            group_table["household_id"] = group_table[id_column].map(
                _build_group_household_map(
                    group_name=group_name,
                    group_ids=pd.Series(_normalize_id_value(membership)),
                    household_ids=persons["household_id"],
                )
            )
        group_tables[group_name] = group_table
        entity_type = _policyengine_group_entity_type(group_name)
        if entity_type is not None:
            entity_lengths[entity_type] = len(group_table)
        excluded_variable_names.add(id_column)
        excluded_variable_names.add(membership_column)

    group_entity_to_table = {
        EntityType.TAX_UNIT: group_tables["tax_unit"],
        EntityType.SPM_UNIT: group_tables["spm_unit"],
        EntityType.FAMILY: group_tables["family"],
    }
    try:
        tax_benefit_system = _resolve_policyengine_us_tax_benefit_system(
            simulation_cls=None
        )
    except (ImportError, ValueError):
        tax_benefit_system = None
    for variable_name, values in arrays.items():
        if variable_name in excluded_variable_names:
            continue
        decoded = _decode_policyengine_array(values)
        prefixed_table = _resolve_prefixed_policyengine_table(
            variable_name=variable_name,
            households=households,
            persons=persons,
            group_tables=group_tables,
        )
        if prefixed_table is not None:
            prefixed_table[variable_name] = decoded
            continue
        try:
            entity = _infer_policyengine_array_entity(
                variable_name=variable_name,
                values=values,
                entity_lengths=entity_lengths,
                tax_benefit_system=tax_benefit_system,
            )
        except ValueError:
            continue
        if entity is EntityType.HOUSEHOLD:
            households[variable_name] = decoded
            continue
        if entity is EntityType.PERSON:
            persons[variable_name] = decoded
            continue
        group_table = group_entity_to_table.get(entity)
        if group_table is None:
            raise ValueError(
                f"Loaded variable '{variable_name}' for entity '{entity.value}' "
                "but no structural table exists for that entity"
            )
        group_table[variable_name] = decoded

    return PolicyEngineUSEntityTableBundle(
        households=households,
        persons=persons,
        tax_units=group_tables["tax_unit"],
        spm_units=group_tables["spm_unit"],
        families=group_tables["family"],
        marital_units=group_tables["marital_unit"],
    )


def _group_table_from_persons(
    persons: pd.DataFrame,
    id_column: str,
) -> pd.DataFrame | None:
    if id_column not in persons.columns:
        return None
    grouped = persons.dropna(subset=[id_column]).copy()
    if grouped.empty:
        return None
    grouped[id_column] = _stringify_id_series(grouped[id_column])
    return (
        grouped.groupby(id_column, observed=True)["household_id"]
        .first()
        .reset_index()
        .rename(columns={id_column: id_column})
    )


def _stringify_id_series(values: pd.Series) -> pd.Series:
    return values.astype(str)


def _build_schema_summary(
    candidate_bundle: PolicyEngineUSEntityTableBundle,
    reference_bundle: PolicyEngineUSEntityTableBundle,
) -> dict[str, Any]:
    entities: dict[str, Any] = {}
    for entity in _ENTITY_ORDER:
        candidate_table = _bundle_table(candidate_bundle, entity)
        reference_table = _bundle_table(reference_bundle, entity)
        entities[entity.value] = _entity_schema_summary(
            entity=entity,
            candidate_table=candidate_table,
            reference_table=reference_table,
        )
    return {"entities": entities}


def _entity_schema_summary(
    *,
    entity: EntityType,
    candidate_table: pd.DataFrame | None,
    reference_table: pd.DataFrame | None,
) -> dict[str, Any]:
    candidate_variables = (
        sorted(_variable_columns(candidate_table, entity))
        if candidate_table is not None
        else []
    )
    reference_variables = (
        sorted(_variable_columns(reference_table, entity))
        if reference_table is not None
        else []
    )
    common = sorted(set(candidate_variables) & set(reference_variables))
    missing = sorted(set(reference_variables) - set(candidate_variables))
    extra = sorted(set(candidate_variables) - set(reference_variables))
    return {
        "candidate_rows": int(len(candidate_table)) if candidate_table is not None else 0,
        "reference_rows": int(len(reference_table)) if reference_table is not None else 0,
        "candidate_variable_count": len(candidate_variables),
        "reference_variable_count": len(reference_variables),
        "common_variable_count": len(common),
        "missing_in_candidate_count": len(missing),
        "extra_in_candidate_count": len(extra),
        "missing_in_candidate": missing,
        "extra_in_candidate": extra,
    }


def _entity_structure_summary(bundle: PolicyEngineUSEntityTableBundle) -> dict[str, Any]:
    households = bundle.households
    persons = bundle.persons
    tax_units = bundle.tax_units
    summary: dict[str, Any] = {
        "household_rows": int(len(households)),
        "person_rows": int(len(persons)) if persons is not None else 0,
        "tax_unit_rows": int(len(tax_units)) if tax_units is not None else 0,
    }
    if persons is None or persons.empty:
        return summary

    household_sizes = persons.groupby("household_id", observed=True).size()
    household_weights = _household_weight_map(bundle)
    aligned_household_weights = (
        _stringify_id_series(household_sizes.index.to_series())
        .map(household_weights)
        .fillna(0.0)
    )
    summary["mean_household_size"] = float(household_sizes.mean())
    summary["weighted_mean_household_size"] = _weighted_mean(
        household_sizes.astype(float).to_numpy(),
        aligned_household_weights.to_numpy(dtype=float),
    )
    summary["share_multi_person_households"] = float((household_sizes >= 2).mean())
    summary["weighted_share_multi_person_households"] = _weighted_mean(
        (household_sizes >= 2).astype(float).to_numpy(),
        aligned_household_weights.to_numpy(dtype=float),
    )

    if tax_units is not None and not tax_units.empty:
        tax_units_per_household = tax_units.groupby("household_id", observed=True).size()
        tax_unit_weights = (
            _stringify_id_series(tax_units_per_household.index.to_series())
            .map(household_weights)
            .fillna(0.0)
        )
        summary["mean_tax_units_per_household"] = float(tax_units_per_household.mean())
        summary["weighted_mean_tax_units_per_household"] = _weighted_mean(
            tax_units_per_household.astype(float).to_numpy(),
            tax_unit_weights.to_numpy(dtype=float),
        )

    if "tax_unit_id" in persons.columns:
        person_tax_units = persons.dropna(subset=["tax_unit_id"]).copy()
        if not person_tax_units.empty:
            tax_unit_sizes = person_tax_units.groupby("tax_unit_id", observed=True).size()
            tax_unit_household_ids = (
                person_tax_units.groupby("tax_unit_id", observed=True)["household_id"].first()
            )
            tax_unit_weights = (
                _stringify_id_series(tax_unit_household_ids)
                .map(household_weights)
                .fillna(0.0)
            )
            summary["mean_tax_unit_size"] = float(tax_unit_sizes.mean())
            summary["weighted_mean_tax_unit_size"] = _weighted_mean(
                tax_unit_sizes.astype(float).to_numpy(),
                tax_unit_weights.to_numpy(dtype=float),
            )
            summary["share_multi_person_tax_units"] = float((tax_unit_sizes >= 2).mean())
            summary["weighted_share_multi_person_tax_units"] = _weighted_mean(
                (tax_unit_sizes >= 2).astype(float).to_numpy(),
                tax_unit_weights.to_numpy(dtype=float),
            )

    return summary


def _weighted_household_size_distribution(
    bundle: PolicyEngineUSEntityTableBundle,
) -> dict[str, Any]:
    persons = bundle.persons
    if persons is None or persons.empty:
        return {"shares": {}, "weighted_mean_household_size": 0.0}
    household_sizes = persons.groupby("household_id", observed=True).size()
    household_weights = (
        _stringify_id_series(household_sizes.index.to_series())
        .map(_household_weight_map(bundle))
        .fillna(0.0)
    )
    bucketed = household_sizes.apply(_household_size_bucket)
    totals = pd.DataFrame({"bucket": bucketed, "weight": household_weights}).groupby(
        "bucket",
        observed=True,
    )["weight"].sum()
    weight_sum = float(household_weights.sum())
    shares = {
        bucket: _safe_ratio(float(totals.get(bucket, 0.0)), weight_sum)
        for bucket in _HOUSEHOLD_SIZE_BUCKETS
    }
    return {
        "shares": shares,
        "weighted_mean_household_size": _weighted_mean(
            household_sizes.astype(float).to_numpy(),
            household_weights.to_numpy(dtype=float),
        ),
    }


def _focus_variable_comparison(
    *,
    candidate_bundle: PolicyEngineUSEntityTableBundle,
    reference_bundle: PolicyEngineUSEntityTableBundle,
    spec: SourceStageParityVariableSpec,
) -> dict[str, Any]:
    reference_entry = _resolve_bundle_variable(
        reference_bundle,
        spec.resolved_reference_variable,
    )
    candidate_entry = _resolve_bundle_variable(
        candidate_bundle,
        spec.candidate_variable,
        preferred_entity=reference_entry["entity"] if reference_entry is not None else None,
    )
    payload: dict[str, Any] = {
        "candidate_variable": spec.candidate_variable,
        "reference_variable": spec.resolved_reference_variable,
        "candidate_present": candidate_entry is not None,
        "reference_present": reference_entry is not None,
    }
    if candidate_entry is not None:
        payload["candidate_entity"] = candidate_entry["entity"].value
        payload["candidate"] = _summarize_series(
            candidate_entry["series"],
            weights=candidate_entry["weights"],
            value_kind=spec.value_kind,
        )
    if reference_entry is not None:
        payload["reference_entity"] = reference_entry["entity"].value
        payload["reference"] = _summarize_series(
            reference_entry["series"],
            weights=reference_entry["weights"],
            value_kind=spec.value_kind,
        )
    if candidate_entry is not None and reference_entry is not None:
        payload["comparison"] = _compare_series(
            candidate_entry["series"],
            reference_entry["series"],
            candidate_weights=candidate_entry["weights"],
            reference_weights=reference_entry["weights"],
            value_kind=spec.value_kind,
        )
    return payload


def _resolve_bundle_variable(
    bundle: PolicyEngineUSEntityTableBundle,
    variable: str,
    *,
    preferred_entity: EntityType | None = None,
) -> dict[str, Any] | None:
    search_order = (
        (preferred_entity,) + tuple(entity for entity in _ENTITY_ORDER if entity is not preferred_entity)
        if preferred_entity is not None
        else _ENTITY_ORDER
    )
    for entity in search_order:
        table = _bundle_table(bundle, entity)
        if table is None or variable not in table.columns:
            continue
        return {
            "entity": entity,
            "series": table[variable],
            "weights": _entity_weights(bundle, entity),
        }
    return None


def _bundle_table(
    bundle: PolicyEngineUSEntityTableBundle,
    entity: EntityType,
) -> pd.DataFrame | None:
    if entity is EntityType.HOUSEHOLD:
        return bundle.households
    if entity is EntityType.PERSON:
        return bundle.persons
    if entity is EntityType.TAX_UNIT:
        return bundle.tax_units
    if entity is EntityType.SPM_UNIT:
        return bundle.spm_units
    if entity is EntityType.FAMILY:
        return bundle.families
    return None


def _variable_columns(table: pd.DataFrame | None, entity: EntityType) -> set[str]:
    if table is None:
        return set()
    excluded = _ENTITY_BASE_COLUMNS.get(entity, set())
    return {
        column
        for column in table.columns
        if column not in excluded and not column.endswith("_id")
    }


def _entity_weights(
    bundle: PolicyEngineUSEntityTableBundle,
    entity: EntityType,
) -> pd.Series:
    table = _bundle_table(bundle, entity)
    if table is None:
        return pd.Series(dtype=float)
    if entity is EntityType.HOUSEHOLD:
        if "household_weight" in table.columns:
            return pd.to_numeric(table["household_weight"], errors="coerce").fillna(0.0)
        return pd.Series(np.ones(len(table), dtype=float), index=table.index)
    if entity is EntityType.PERSON and "weight" in table.columns:
        return pd.to_numeric(table["weight"], errors="coerce").fillna(0.0)
    if "household_id" in table.columns:
        household_weights = _household_weight_map(bundle)
        return _stringify_id_series(table["household_id"]).map(household_weights).fillna(0.0)
    return pd.Series(np.ones(len(table), dtype=float), index=table.index)


def _household_weight_map(bundle: PolicyEngineUSEntityTableBundle) -> pd.Series:
    households = bundle.households.copy()
    households["household_id"] = _stringify_id_series(households["household_id"])
    if "household_weight" in households.columns:
        weights = pd.to_numeric(households["household_weight"], errors="coerce").fillna(0.0)
    else:
        weights = pd.Series(np.ones(len(households), dtype=float), index=households.index)
    return pd.Series(weights.to_numpy(dtype=float), index=households["household_id"])


def _summarize_series(
    values: pd.Series,
    *,
    weights: pd.Series,
    value_kind: str = "auto",
) -> dict[str, Any]:
    series = values.reset_index(drop=True)
    weight_series = pd.to_numeric(weights, errors="coerce").fillna(0.0).reset_index(drop=True)
    if len(weight_series) != len(series):
        weight_series = pd.Series(np.ones(len(series), dtype=float))
    resolved_value_kind = _resolve_value_kind(series, value_kind)
    nonnull = series.notna()
    total_weight = float(weight_series.sum())
    nonnull_weight = float(weight_series[nonnull].sum())
    if resolved_value_kind == "categorical":
        return _summarize_categorical(series.astype("string"), weight_series, total_weight, nonnull_weight)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return _summarize_categorical(series.astype("string"), weight_series, total_weight, nonnull_weight)
    if resolved_value_kind == "auto":
        unique_count = int(numeric.dropna().nunique())
        if unique_count <= 64 and pd.api.types.is_integer_dtype(numeric.dropna()):
            return _summarize_categorical(
                numeric.round().astype("Int64").astype("string"),
                weight_series,
                total_weight,
                nonnull_weight,
            )
    if resolved_value_kind != "numeric" and (
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return _summarize_categorical(
            numeric.round().astype("Int64").astype("string"),
            weight_series,
            total_weight,
            nonnull_weight,
        )
    numeric_values = numeric.dropna().astype(float)
    numeric_weights = weight_series[numeric.notna()].astype(float)
    return {
        "kind": "numeric",
        "n": int(len(series)),
        "nonnull_share": _safe_ratio(int(numeric.notna().sum()), len(series)),
        "weighted_nonnull_share": _safe_ratio(nonnull_weight, total_weight),
        "zero_share": float((numeric_values == 0.0).mean()) if not numeric_values.empty else 0.0,
        "weighted_zero_share": _weighted_mean(
            (numeric_values == 0.0).astype(float).to_numpy(),
            numeric_weights.to_numpy(dtype=float),
        ),
        "positive_share": float((numeric_values > 0.0).mean()) if not numeric_values.empty else 0.0,
        "weighted_positive_share": _weighted_mean(
            (numeric_values > 0.0).astype(float).to_numpy(),
            numeric_weights.to_numpy(dtype=float),
        ),
        "negative_share": float((numeric_values < 0.0).mean()) if not numeric_values.empty else 0.0,
        "weighted_negative_share": _weighted_mean(
            (numeric_values < 0.0).astype(float).to_numpy(),
            numeric_weights.to_numpy(dtype=float),
        ),
        "mean": float(numeric_values.mean()) if not numeric_values.empty else 0.0,
        "weighted_mean": _weighted_mean(
            numeric_values.to_numpy(dtype=float),
            numeric_weights.to_numpy(dtype=float),
        ),
        "sum": float(numeric_values.sum()) if not numeric_values.empty else 0.0,
        "weighted_sum": float((numeric_values * numeric_weights).sum()),
    }


def _summarize_categorical(
    values: pd.Series,
    weights: pd.Series,
    total_weight: float,
    nonnull_weight: float,
) -> dict[str, Any]:
    normalized = values.replace({"": pd.NA, "nan": pd.NA}).dropna()
    if normalized.empty:
        return {
            "kind": "categorical",
            "n": int(len(values)),
            "nonnull_share": 0.0,
            "weighted_nonnull_share": _safe_ratio(nonnull_weight, total_weight),
            "unique_count": 0,
            "top_values": [],
        }
    aligned_weights = weights[values.replace({"": pd.NA, "nan": pd.NA}).notna()].astype(float)
    grouped = (
        pd.DataFrame({"value": normalized.astype(str), "weight": aligned_weights.to_numpy(dtype=float)})
        .groupby("value", observed=True)["weight"]
        .sum()
        .sort_values(ascending=False)
    )
    return {
        "kind": "categorical",
        "n": int(len(values)),
        "nonnull_share": _safe_ratio(int(normalized.notna().sum()), len(values)),
        "weighted_nonnull_share": _safe_ratio(nonnull_weight, total_weight),
        "unique_count": int(normalized.nunique(dropna=True)),
        "top_values": [
            {
                "value": str(index),
                "weighted_sum": float(weight),
                "weighted_share": _safe_ratio(float(weight), nonnull_weight),
            }
            for index, weight in grouped.head(10).items()
        ],
    }


def _compare_series(
    candidate: pd.Series,
    reference: pd.Series,
    *,
    candidate_weights: pd.Series,
    reference_weights: pd.Series,
    value_kind: str = "auto",
) -> dict[str, Any]:
    candidate_summary = _summarize_series(
        candidate,
        weights=candidate_weights,
        value_kind=value_kind,
    )
    reference_summary = _summarize_series(
        reference,
        weights=reference_weights,
        value_kind=value_kind,
    )
    if candidate_summary["kind"] != reference_summary["kind"]:
        return {
            "type": "mismatched",
            "candidate_kind": candidate_summary["kind"],
            "reference_kind": reference_summary["kind"],
        }
    if candidate_summary["kind"] == "categorical":
        candidate_support = _categorical_support(candidate)
        reference_support = _categorical_support(reference)
        missing = sorted(reference_support - candidate_support)
        return {
            "type": "categorical",
            "support_recall": _safe_ratio(
                len(candidate_support & reference_support),
                len(reference_support),
            ),
            "support_precision": _safe_ratio(
                len(candidate_support & reference_support),
                len(candidate_support),
            ),
            "missing_reference_values": missing[:20],
        }
    return {
        "type": "numeric",
        "weighted_mean_ratio": _safe_ratio(
            candidate_summary["weighted_mean"],
            reference_summary["weighted_mean"],
        ),
        "weighted_sum_ratio": _safe_ratio(
            candidate_summary["weighted_sum"],
            reference_summary["weighted_sum"],
        ),
        "weighted_positive_share_ratio": _safe_ratio(
            candidate_summary["weighted_positive_share"],
            reference_summary["weighted_positive_share"],
        ),
        "weighted_nonnull_share_delta": float(
            candidate_summary["weighted_nonnull_share"]
            - reference_summary["weighted_nonnull_share"]
        ),
    }


def _categorical_support(values: pd.Series) -> set[str]:
    normalized = (
        values.astype("string")
        .replace({"": pd.NA, "nan": pd.NA})
        .dropna()
        .astype(str)
    )
    return set(normalized.tolist())


def _resolve_value_kind(values: pd.Series, value_kind: str) -> str:
    if value_kind in {"numeric", "categorical"}:
        return value_kind
    if value_kind != "auto":
        raise ValueError(
            "Source-stage parity value_kind must be one of: auto, numeric, categorical"
        )
    if (
        pd.api.types.is_bool_dtype(values)
        or pd.api.types.is_object_dtype(values)
        or pd.api.types.is_string_dtype(values)
    ):
        return "categorical"
    return "auto"


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    weight_sum = float(np.asarray(weights, dtype=float).sum())
    if weight_sum <= 0.0:
        return float(np.asarray(values, dtype=float).mean())
    return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))


def _numeric_deltas(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key, candidate_value in candidate.items():
        reference_value = reference.get(key)
        if isinstance(candidate_value, (int, float)) and isinstance(reference_value, (int, float)):
            deltas[f"{key}_delta"] = float(candidate_value) - float(reference_value)
    return deltas


def _distribution_deltas(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    candidate_shares = dict(candidate.get("shares", {}))
    reference_shares = dict(reference.get("shares", {}))
    deltas = {
        f"share_{bucket}_delta": float(candidate_shares.get(bucket, 0.0))
        - float(reference_shares.get(bucket, 0.0))
        for bucket in _HOUSEHOLD_SIZE_BUCKETS
    }
    deltas["weighted_mean_household_size_delta"] = float(
        candidate.get("weighted_mean_household_size", 0.0)
    ) - float(reference.get("weighted_mean_household_size", 0.0))
    return deltas


def _household_size_bucket(size: int) -> str:
    return str(size) if size <= 6 else "7+"


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="source", required=True)

    cps_parser = subparsers.add_parser("cps", help="Audit raw CPS provider output")
    cps_parser.add_argument("reference_dataset")
    cps_parser.add_argument("output_path")
    cps_parser.add_argument("--year", type=int, default=2023)
    cps_parser.add_argument("--cache-dir")
    cps_parser.add_argument("--sample-n", type=int)
    cps_parser.add_argument("--random-seed", type=int, default=0)
    cps_parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)

    puf_parser = subparsers.add_parser("puf", help="Audit raw PUF provider output")
    puf_parser.add_argument("reference_dataset")
    puf_parser.add_argument("output_path")
    puf_parser.add_argument("--target-year", type=int, default=2024)
    puf_parser.add_argument("--cache-dir")
    puf_parser.add_argument("--puf-path")
    puf_parser.add_argument("--demographics-path")
    puf_parser.add_argument("--sample-n", type=int)
    puf_parser.add_argument("--random-seed", type=int, default=0)
    puf_parser.add_argument("--uprating-mode", default=PUF_UPRATING_MODE_INTERPOLATED)
    puf_parser.add_argument("--policyengine-us-data-repo")
    puf_parser.add_argument("--policyengine-us-data-python")
    puf_parser.add_argument(
        "--impute-pre-tax-contributions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    puf_parser.add_argument("--pre-tax-training-year", type=int, default=2024)
    puf_parser.add_argument(
        "--require-pre-tax-contribution-model",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    puf_parser.add_argument(
        "--social-security-split-strategy",
        default=SOCIAL_SECURITY_SPLIT_STRATEGY_GROUPED_SHARE,
    )
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()
    if args.source == "cps":
        output = write_us_cps_source_stage_parity_audit(
            args.reference_dataset,
            args.output_path,
            year=args.year,
            cache_dir=args.cache_dir,
            download=args.download,
            sample_n=args.sample_n,
            random_seed=args.random_seed,
        )
    else:
        output = write_us_puf_source_stage_parity_audit(
            args.reference_dataset,
            args.output_path,
            target_year=args.target_year,
            cache_dir=args.cache_dir,
            puf_path=args.puf_path,
            demographics_path=args.demographics_path,
            sample_n=args.sample_n,
            random_seed=args.random_seed,
            uprating_mode=args.uprating_mode,
            policyengine_us_data_repo=args.policyengine_us_data_repo,
            policyengine_us_data_python=args.policyengine_us_data_python,
            impute_pre_tax_contributions=args.impute_pre_tax_contributions,
            pre_tax_training_year=args.pre_tax_training_year,
            require_pre_tax_contribution_model=args.require_pre_tax_contribution_model,
            social_security_split_strategy=args.social_security_split_strategy,
        )
    print(output)


if __name__ == "__main__":
    main()
