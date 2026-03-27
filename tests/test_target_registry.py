"""Tests for the canonical US target registry."""

from microplex.core import EntityType
from microplex.targets import (
    TargetAggregation,
    TargetFilter,
    TargetProvider,
    TargetQuery,
    TargetSpec,
)

from microplex_us.target_registry import (
    TargetCategory,
    TargetGroup,
    TargetLevel,
    TargetRegistry,
)


class TestTargetRegistry:
    def test_registry_emits_canonical_target_specs(self):
        registry = TargetRegistry()

        target = registry.get_group("state_population").targets[0]

        assert isinstance(target, TargetSpec)
        assert target.entity is EntityType.PERSON
        assert target.aggregation is TargetAggregation.COUNT
        assert target.measure is None
        assert target.filters == (
            TargetFilter(feature="state_fips", operator="==", value="01"),
        )
        assert target.metadata["us_category"] == "geography"
        assert target.metadata["us_level"] == "state"
        assert target.metadata["us_group"] == "state_population"
        assert target.metadata["available_in_cps"] is True

    def test_registry_selects_targets_by_metadata(self):
        geography_target = TargetSpec(
            name="ca_people",
            entity=EntityType.PERSON,
            value=2.0,
            period=2024,
            aggregation=TargetAggregation.COUNT,
            filters=(TargetFilter(feature="state_fips", operator="==", value="06"),),
            metadata={
                "us_category": "geography",
                "us_level": "state",
                "us_group": "geography_targets",
                "available_in_cps": True,
                "requires_imputation": False,
            },
        )
        tax_target = TargetSpec(
            name="tax_claims",
            entity=EntityType.TAX_UNIT,
            value=1.0,
            period=2024,
            aggregation=TargetAggregation.COUNT,
            filters=(TargetFilter(feature="filing_status", operator="==", value="single"),),
            metadata={
                "us_category": "tax",
                "us_level": "national",
                "us_group": "tax_targets",
                "available_in_cps": False,
                "requires_imputation": True,
            },
        )
        registry = TargetRegistry(
            groups={
                "geography_targets": TargetGroup(
                    name="geography_targets",
                    category=TargetCategory.GEOGRAPHY,
                    targets=[geography_target],
                ),
                "tax_targets": TargetGroup(
                    name="tax_targets",
                    category=TargetCategory.TAX,
                    targets=[tax_target],
                ),
            },
            build_defaults=False,
        )

        selected = registry.select_targets(
            categories=[TargetCategory.GEOGRAPHY],
            levels=[TargetLevel.STATE],
            groups=["geography_targets"],
            only_available=True,
            entity=EntityType.PERSON,
        )

        assert selected == [geography_target]
        assert isinstance(registry, TargetProvider)
        assert registry.load_target_set(
            TargetQuery(
                entity=EntityType.PERSON,
                provider_filters={
                    "categories": [TargetCategory.GEOGRAPHY],
                    "levels": [TargetLevel.STATE],
                    "groups": ["geography_targets"],
                    "only_available": True,
                },
            )
        ).targets == [geography_target]
