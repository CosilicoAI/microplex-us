"""Thin US adapter for shared target-driven household reweighting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from microplex.core import EntityType
from microplex.targets import (
    EntityTableBinding,
    EntityTableBundle,
    TargetConstraintCompilationResult,
    TargetReweightingDiagnostics,
    TargetSpec,
    reweight_entity_table_bundle_targets,
)

from microplex_us.policyengine.us import PolicyEngineUSEntityTableBundle


@dataclass(frozen=True)
class USHouseholdTargetReweightingResult:
    """Result of applying shared target reweighting to a US PE table bundle."""

    tables: PolicyEngineUSEntityTableBundle
    compilation: TargetConstraintCompilationResult
    diagnostics: TargetReweightingDiagnostics


def reweight_us_household_targets(
    tables: PolicyEngineUSEntityTableBundle,
    *,
    targets: list[TargetSpec],
    max_iter: int = 8,
    tol: float = 1e-4,
    factor_bounds: tuple[float, float] = (0.5, 2.0),
) -> USHouseholdTargetReweightingResult:
    """Reweight US household-aligned PE tables using the shared core module."""
    bundle_result = reweight_entity_table_bundle_targets(
        _as_entity_table_bundle(tables),
        targets=targets,
        max_iter=max_iter,
        tol=tol,
        factor_bounds=factor_bounds,
    )
    updated_tables = _policyengine_us_bundle_from_entity_table_bundle(
        tables,
        bundle_result.bundle,
    )
    return USHouseholdTargetReweightingResult(
        tables=updated_tables,
        compilation=bundle_result.compilation,
        diagnostics=bundle_result.diagnostics,
    )


def _as_entity_table_bundle(tables: PolicyEngineUSEntityTableBundle) -> EntityTableBundle:
    bindings: dict[EntityType, EntityTableBinding] = {
        EntityType.HOUSEHOLD: EntityTableBinding(
            frame=tables.households,
            id_column="household_id",
        ),
    }
    for entity, frame, id_column, weight_link_column, synced_weight_column in (
        (EntityType.PERSON, tables.persons, "person_id", "household_id", "weight"),
        (
            EntityType.TAX_UNIT,
            tables.tax_units,
            "tax_unit_id",
            "household_id",
            "household_weight",
        ),
        (
            EntityType.SPM_UNIT,
            tables.spm_units,
            "spm_unit_id",
            "household_id",
            "household_weight",
        ),
        (
            EntityType.FAMILY,
            tables.families,
            "family_id",
            "household_id",
            "household_weight",
        ),
    ):
        if frame is None:
            continue
        resolved_link_column = (
            weight_link_column
            if weight_link_column in frame.columns
            else "person_household_id"
            if "person_household_id" in frame.columns
            else None
        )
        bindings[entity] = EntityTableBinding(
            frame=frame,
            id_column=id_column,
            weight_link_column=resolved_link_column,
            synced_weight_column=synced_weight_column if resolved_link_column else None,
        )
    return EntityTableBundle(
        weight_entity=EntityType.HOUSEHOLD,
        weight_column="household_weight",
        bindings=bindings,
    )


def _policyengine_us_bundle_from_entity_table_bundle(
    tables: PolicyEngineUSEntityTableBundle,
    bundle: EntityTableBundle,
) -> PolicyEngineUSEntityTableBundle:
    households = bundle.table_for(EntityType.HOUSEHOLD).copy()
    household_weights = households.set_index("household_id")["household_weight"]
    return PolicyEngineUSEntityTableBundle(
        households=households,
        persons=_table_or_synced_weights(
            bundle,
            tables.persons,
            EntityType.PERSON,
            household_weights,
            weight_column="weight",
        ),
        tax_units=_table_or_synced_weights(
            bundle,
            tables.tax_units,
            EntityType.TAX_UNIT,
            household_weights,
            weight_column="household_weight",
        ),
        spm_units=_table_or_synced_weights(
            bundle,
            tables.spm_units,
            EntityType.SPM_UNIT,
            household_weights,
            weight_column="household_weight",
        ),
        families=_table_or_synced_weights(
            bundle,
            tables.families,
            EntityType.FAMILY,
            household_weights,
            weight_column="household_weight",
        ),
        marital_units=_sync_entity_weights(
            tables.marital_units,
            household_weights,
            weight_column="household_weight",
        ),
    )


def _table_or_synced_weights(
    bundle: EntityTableBundle,
    fallback_frame: pd.DataFrame | None,
    entity: EntityType,
    household_weights: pd.Series,
    *,
    weight_column: str,
) -> pd.DataFrame | None:
    if fallback_frame is None:
        return None
    try:
        return bundle.table_for(entity).copy()
    except KeyError:
        return _sync_entity_weights(
            fallback_frame,
            household_weights,
            weight_column=weight_column,
        )


def _sync_entity_weights(
    frame: pd.DataFrame | None,
    household_weights: pd.Series,
    *,
    weight_column: str,
) -> pd.DataFrame | None:
    if frame is None:
        return None
    updated = frame.copy()
    household_id_column = (
        "person_household_id" if "person_household_id" in updated.columns else "household_id"
    )
    if household_id_column not in updated.columns:
        return updated
    updated[weight_column] = updated[household_id_column].map(household_weights)
    return updated
