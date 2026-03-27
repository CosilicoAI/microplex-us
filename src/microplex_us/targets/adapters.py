"""Adapters from US-specific target representations to core microplex target specs."""

from __future__ import annotations

from collections.abc import Iterable

from microplex.core import EntityType
from microplex.targets import (
    TargetAggregation,
    TargetFilter,
    TargetSet,
)
from microplex.targets import (
    TargetSpec as CanonicalTargetSpec,
)

from microplex_us.policyengine.us import PolicyEngineUSDBTarget

POLICYENGINE_US_COUNT_ENTITIES: dict[str, EntityType] = {
    "household_count": EntityType.HOUSEHOLD,
    "person_count": EntityType.PERSON,
    "tax_unit_count": EntityType.TAX_UNIT,
    "spm_unit_count": EntityType.SPM_UNIT,
    "family_count": EntityType.FAMILY,
}


def policyengine_db_target_to_canonical_spec(
    target: PolicyEngineUSDBTarget,
    *,
    default_entity: EntityType | str = EntityType.HOUSEHOLD,
    entity_overrides: dict[str, EntityType] | None = None,
) -> CanonicalTargetSpec:
    """Translate a PolicyEngine US DB target row into the canonical core spec."""
    resolved_default_entity = (
        default_entity
        if isinstance(default_entity, EntityType)
        else EntityType(default_entity)
    )
    resolved_entity = (
        (entity_overrides or {}).get(target.variable)
        or POLICYENGINE_US_COUNT_ENTITIES.get(target.variable)
        or resolved_default_entity
    )
    aggregation = (
        TargetAggregation.COUNT
        if target.variable.endswith("_count")
        else TargetAggregation.SUM
    )
    measure = None if aggregation is TargetAggregation.COUNT else target.variable
    filters = tuple(
        TargetFilter(
            feature=constraint.variable,
            operator=constraint.operation,
            value=constraint.value,
        )
        for constraint in target.constraints
    )

    return CanonicalTargetSpec(
        name=f"policyengine_us_target_{target.target_id}",
        entity=resolved_entity,
        value=target.value,
        period=target.period,
        measure=measure,
        aggregation=aggregation,
        filters=filters,
        tolerance=target.tolerance,
        source=target.source,
        description=target.notes,
        metadata={
            "target_id": target.target_id,
            "variable": target.variable,
            "stratum_id": target.stratum_id,
            "stratum_definition_hash": target.definition_hash,
            "parent_stratum_id": target.parent_stratum_id,
            "reform_id": target.reform_id,
            "active": target.active,
            "geo_level": target.geo_level,
            "geographic_id": target.geographic_id,
            "domain_variable": target.domain_variable,
            "domain_variables": target.domain_variables,
            "constraint_count": len(target.constraints),
        },
    )


def policyengine_db_targets_to_canonical_set(
    targets: Iterable[PolicyEngineUSDBTarget],
    *,
    default_entity: EntityType | str = EntityType.HOUSEHOLD,
    entity_overrides: dict[str, EntityType] | None = None,
) -> TargetSet:
    """Translate a sequence of PolicyEngine US DB targets into a canonical target set."""
    return TargetSet(
        [
            policyengine_db_target_to_canonical_spec(
                target,
                default_entity=default_entity,
                entity_overrides=entity_overrides,
            )
            for target in targets
        ]
    )
