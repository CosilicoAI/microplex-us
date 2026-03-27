"""Registry of US calibration targets expressed in the core microplex target IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd
from microplex.core import EntityType
from microplex.targets import (
    FilterOperator,
    TargetAggregation,
    TargetFilter,
    TargetQuery,
    TargetSet,
    TargetSpec,
    apply_target_query,
)

US_TARGET_CATEGORY_KEY = "us_category"
US_TARGET_LEVEL_KEY = "us_level"
US_TARGET_GROUP_KEY = "us_group"
US_TARGET_AVAILABLE_KEY = "available_in_cps"
US_TARGET_IMPUTATION_KEY = "requires_imputation"
US_TARGET_NOTES_KEY = "notes"


class TargetCategory(str, Enum):
    """High-level US calibration target categories."""

    GEOGRAPHY = "geography"
    INCOME = "income"
    BENEFITS = "benefits"
    DEMOGRAPHICS = "demographics"
    HEALTH = "health"
    TAX = "tax"


class TargetLevel(str, Enum):
    """Geographic level of a US target slice."""

    NATIONAL = "national"
    STATE = "state"
    CD = "cd"
    COUNTY = "county"
    TRACT = "tract"


def target_category(target: TargetSpec) -> TargetCategory | None:
    """Return the US category metadata for a target."""
    value = target.metadata.get(US_TARGET_CATEGORY_KEY)
    return TargetCategory(value) if value is not None else None


def target_level(target: TargetSpec) -> TargetLevel | None:
    """Return the US level metadata for a target."""
    value = target.metadata.get(US_TARGET_LEVEL_KEY)
    return TargetLevel(value) if value is not None else None


def target_group_name(target: TargetSpec) -> str | None:
    """Return the US group metadata for a target."""
    value = target.metadata.get(US_TARGET_GROUP_KEY)
    return str(value) if value is not None else None


def target_available_in_cps(target: TargetSpec) -> bool:
    """Whether the target is directly available in CPS-like source data."""
    return bool(target.metadata.get(US_TARGET_AVAILABLE_KEY, False))


def target_requires_imputation(target: TargetSpec) -> bool:
    """Whether the target depends on imputation or external modeling."""
    return bool(target.metadata.get(US_TARGET_IMPUTATION_KEY, False))


def target_notes(target: TargetSpec) -> str:
    """Free-form US notes metadata."""
    value = target.metadata.get(US_TARGET_NOTES_KEY)
    return str(value) if value is not None else ""


@dataclass
class TargetGroup:
    """A named US target family backed by canonical core targets."""

    name: str
    category: TargetCategory
    targets: list[TargetSpec] = field(default_factory=list)

    def add(self, target: TargetSpec) -> TargetGroup:
        self.targets.append(target)
        return self

    def __len__(self) -> int:
        return len(self.targets)


class TargetRegistry:
    """US target registry that emits canonical microplex targets."""

    def __init__(
        self,
        groups: dict[str, TargetGroup] | None = None,
        *,
        build_defaults: bool = True,
    ):
        self.groups: dict[str, TargetGroup] = dict(groups or {})
        if build_defaults:
            self._build_registry()

    def _build_registry(self) -> None:
        self._add_geography_targets()
        self._add_income_targets()
        self._add_benefit_targets()
        self._add_health_targets()
        self._add_tax_targets()
        self._add_demographic_targets()

    def _get_or_create_group(
        self,
        name: str,
        category: TargetCategory,
    ) -> TargetGroup:
        group = self.groups.get(name)
        if group is None:
            group = TargetGroup(name=name, category=category)
            self.groups[name] = group
        return group

    def _add_target(
        self,
        *,
        group_name: str,
        category: TargetCategory,
        level: TargetLevel,
        name: str,
        value: float,
        entity: EntityType,
        period: int = 2024,
        aggregation: TargetAggregation | str = TargetAggregation.SUM,
        measure: str | None = None,
        filters: tuple[TargetFilter, ...] = (),
        source: str = "",
        units: str = "",
        description: str = "",
        available_in_cps: bool = True,
        requires_imputation: bool = False,
        notes: str = "",
    ) -> TargetSpec:
        target = TargetSpec(
            name=name,
            entity=entity,
            value=value,
            period=period,
            measure=measure,
            aggregation=aggregation,
            filters=filters,
            source=source or None,
            units=units or None,
            description=description or None,
            metadata={
                US_TARGET_CATEGORY_KEY: category.value,
                US_TARGET_LEVEL_KEY: level.value,
                US_TARGET_GROUP_KEY: group_name,
                US_TARGET_AVAILABLE_KEY: available_in_cps,
                US_TARGET_IMPUTATION_KEY: requires_imputation,
                US_TARGET_NOTES_KEY: notes,
            },
        )
        self._get_or_create_group(group_name, category).add(target)
        return target

    def _add_geography_targets(self) -> None:
        census_2020 = {
            "01": 5024279,
            "02": 733391,
            "04": 7151502,
            "05": 3011524,
            "06": 39538223,
            "08": 5773714,
            "09": 3605944,
            "10": 989948,
            "11": 689545,
            "12": 21538187,
            "13": 10711908,
            "15": 1455271,
            "16": 1839106,
            "17": 12812508,
            "18": 6785528,
            "19": 3190369,
            "20": 2937880,
            "21": 4505836,
            "22": 4657757,
            "23": 1362359,
            "24": 6177224,
            "25": 7029917,
            "26": 10077331,
            "27": 5706494,
            "28": 2961279,
            "29": 6154913,
            "30": 1084225,
            "31": 1961504,
            "32": 3104614,
            "33": 1377529,
            "34": 9288994,
            "35": 2117522,
            "36": 20201249,
            "37": 10439388,
            "38": 779094,
            "39": 11799448,
            "40": 3959353,
            "41": 4237256,
            "42": 13002700,
            "44": 1097379,
            "45": 5118425,
            "46": 886667,
            "47": 6910840,
            "48": 29145505,
            "49": 3271616,
            "50": 643077,
            "51": 8631393,
            "53": 7705281,
            "54": 1793716,
            "55": 5893718,
            "56": 576851,
        }

        for fips, population in census_2020.items():
            self._add_target(
                group_name="state_population",
                category=TargetCategory.GEOGRAPHY,
                level=TargetLevel.STATE,
                name=f"population_{fips}",
                value=population,
                entity=EntityType.PERSON,
                aggregation=TargetAggregation.COUNT,
                filters=(
                    TargetFilter(
                        feature="state_fips",
                        operator=FilterOperator.EQ,
                        value=fips,
                    ),
                ),
                source="Census 2020",
                units="persons",
            )

        fips_to_abbr = {
            "01": "AL",
            "02": "AK",
            "04": "AZ",
            "05": "AR",
            "06": "CA",
            "08": "CO",
            "09": "CT",
            "10": "DE",
            "11": "DC",
            "12": "FL",
            "13": "GA",
            "15": "HI",
            "16": "ID",
            "17": "IL",
            "18": "IN",
            "19": "IA",
            "20": "KS",
            "21": "KY",
            "22": "LA",
            "23": "ME",
            "24": "MD",
            "25": "MA",
            "26": "MI",
            "27": "MN",
            "28": "MS",
            "29": "MO",
            "30": "MT",
            "31": "NE",
            "32": "NV",
            "33": "NH",
            "34": "NJ",
            "35": "NM",
            "36": "NY",
            "37": "NC",
            "38": "ND",
            "39": "OH",
            "40": "OK",
            "41": "OR",
            "42": "PA",
            "44": "RI",
            "45": "SC",
            "46": "SD",
            "47": "TN",
            "48": "TX",
            "49": "UT",
            "50": "VT",
            "51": "VA",
            "53": "WA",
            "54": "WV",
            "55": "WI",
            "56": "WY",
        }

        try:
            cd_df = pd.read_parquet("data/district_targets.parquet")
        except FileNotFoundError:
            cd_df = None

        if cd_df is not None:
            for _, row in cd_df.iterrows():
                fips_id = row["district_id"]
                state_fips, district_num = fips_id.split("-")
                state_abbr = fips_to_abbr.get(state_fips, state_fips)
                cd_id = f"{state_abbr}-AL" if district_num == "00" else f"{state_abbr}-{district_num}"
                self._add_target(
                    group_name="cd_population",
                    category=TargetCategory.GEOGRAPHY,
                    level=TargetLevel.CD,
                    name=f"cd_{fips_id}",
                    value=float(row["population"]),
                    entity=EntityType.PERSON,
                    aggregation=TargetAggregation.COUNT,
                    filters=(
                        TargetFilter(
                            feature="cd_id",
                            operator=FilterOperator.EQ,
                            value=cd_id,
                        ),
                    ),
                    source="Census ACS",
                    units="persons",
                )

        try:
            blocks = pd.read_parquet("data/block_probabilities.parquet")
        except FileNotFoundError:
            blocks = None

        if blocks is None:
            return

        sldu_col = "sldu_id" if "sldu_id" in blocks.columns else "sldu_geoid"
        if sldu_col in blocks.columns:
            sldu_pop = blocks.groupby(sldu_col)["population"].sum()
            for sldu_id, population in sldu_pop.items():
                if pd.notna(sldu_id) and population > 0:
                    self._add_target(
                        group_name="sldu_population",
                        category=TargetCategory.GEOGRAPHY,
                        level=TargetLevel.STATE,
                        name=f"sldu_{sldu_id}",
                        value=float(population),
                        entity=EntityType.PERSON,
                        aggregation=TargetAggregation.COUNT,
                        filters=(
                            TargetFilter(
                                feature="sldu_id",
                                operator=FilterOperator.EQ,
                                value=sldu_id,
                            ),
                        ),
                        source="Census",
                        units="persons",
                    )

        sldl_col = "sldl_id" if "sldl_id" in blocks.columns else "sldl_geoid"
        if sldl_col in blocks.columns:
            sldl_pop = blocks.groupby(sldl_col)["population"].sum()
            for sldl_id, population in sldl_pop.items():
                if pd.notna(sldl_id) and population > 0:
                    self._add_target(
                        group_name="sldl_population",
                        category=TargetCategory.GEOGRAPHY,
                        level=TargetLevel.STATE,
                        name=f"sldl_{sldl_id}",
                        value=float(population),
                        entity=EntityType.PERSON,
                        aggregation=TargetAggregation.COUNT,
                        filters=(
                            TargetFilter(
                                feature="sldl_id",
                                operator=FilterOperator.EQ,
                                value=sldl_id,
                            ),
                        ),
                        source="Census",
                        units="persons",
                    )

    def _add_income_targets(self) -> None:
        soi_income = {
            "employment_income": (9_022_352_941_000, "employment_income", True),
            "self_employment_income": (436_400_000_000, "self_employment_income", True),
            "social_security": (774_000_000_000, "social_security", True),
            "taxable_pension_income": (827_600_000_000, "taxable_pension_income", True),
            "tax_exempt_pension_income": (580_400_000_000, "tax_exempt_pension_income", True),
            "unemployment_compensation": (208_000_000_000, "unemployment_compensation", True),
            "dividend_income": (260_200_000_000, "dividend_income", False),
            "interest_income": (127_400_000_000, "interest_income", False),
            "rental_income": (46_000_000_000, "rental_income", True),
            "long_term_capital_gains": (1_137_000_000_000, "long_term_capital_gains", False),
            "short_term_capital_gains": (-72_000_000_000, "short_term_capital_gains", False),
            "partnership_s_corp_income": (976_000_000_000, "partnership_s_corp_income", False),
            "farm_income": (-26_141_944_000, "farm_income", False),
            "alimony_income": (8_500_000_000, "alimony_income", True),
        }

        for name, (value, measure, in_cps) in soi_income.items():
            self._add_target(
                group_name="irs_soi_income",
                category=TargetCategory.INCOME,
                level=TargetLevel.NATIONAL,
                name=name,
                value=value,
                entity=EntityType.PERSON,
                aggregation=TargetAggregation.SUM,
                measure=measure,
                source="IRS SOI",
                units="USD",
                available_in_cps=in_cps,
                requires_imputation=not in_cps,
                notes="" if in_cps else "Underreported in CPS, requires imputation",
            )

    def _add_benefit_targets(self) -> None:
        benefit_totals = {
            "snap_spending": (
                103_100_000_000,
                EntityType.HOUSEHOLD,
                "snap",
                TargetAggregation.SUM,
                (),
                "USD",
                "CBO",
            ),
            "snap_participation": (
                41_209_000,
                EntityType.PERSON,
                None,
                TargetAggregation.COUNT,
                (
                    TargetFilter(
                        feature="snap",
                        operator=FilterOperator.GT,
                        value=0,
                    ),
                ),
                "persons",
                "USDA",
            ),
            "ssi_spending": (
                78_500_000_000,
                EntityType.PERSON,
                "ssi",
                TargetAggregation.SUM,
                (),
                "USD",
                "CBO",
            ),
            "ssi_participation": (
                7_400_000,
                EntityType.PERSON,
                None,
                TargetAggregation.COUNT,
                (
                    TargetFilter(
                        feature="ssi",
                        operator=FilterOperator.GT,
                        value=0,
                    ),
                ),
                "persons",
                "SSA",
            ),
            "social_security_spending": (
                2_623_800_000_000,
                EntityType.PERSON,
                "social_security",
                TargetAggregation.SUM,
                (),
                "USD",
                "CBO",
            ),
            "social_security_participation": (
                66_000_000,
                EntityType.PERSON,
                None,
                TargetAggregation.COUNT,
                (
                    TargetFilter(
                        feature="social_security",
                        operator=FilterOperator.GT,
                        value=0,
                    ),
                ),
                "persons",
                "SSA",
            ),
            "eitc_spending": (
                72_700_000_000,
                EntityType.TAX_UNIT,
                "eitc",
                TargetAggregation.SUM,
                (),
                "USD",
                "Treasury",
            ),
            "unemployment_spending": (
                59_100_000_000,
                EntityType.PERSON,
                "unemployment_compensation",
                TargetAggregation.SUM,
                (),
                "USD",
                "CBO",
            ),
        }

        for name, (
            value,
            entity,
            measure,
            aggregation,
            filters,
            units,
            source,
        ) in benefit_totals.items():
            self._add_target(
                group_name="benefit_programs",
                category=TargetCategory.BENEFITS,
                level=TargetLevel.NATIONAL,
                name=name,
                value=value,
                entity=entity,
                aggregation=aggregation,
                measure=measure,
                filters=filters,
                source=source,
                units=units,
                available_in_cps=True,
            )

    def _add_health_targets(self) -> None:
        medicaid_categories = [
            "child",
            "aged",
            "disabled",
            "expansion_adults",
            "non_expansion_adults",
        ]

        for category in medicaid_categories:
            self._add_target(
                group_name="health_insurance",
                category=TargetCategory.HEALTH,
                level=TargetLevel.NATIONAL,
                name=f"medicaid_{category}_national",
                value=0,
                entity=EntityType.PERSON,
                aggregation=TargetAggregation.COUNT,
                filters=(
                    TargetFilter(
                        feature="medicaid",
                        operator=FilterOperator.GT,
                        value=0,
                    ),
                ),
                source="HHS/CMS",
                units="persons",
                available_in_cps=False,
                requires_imputation=True,
                notes="Requires eligibility modeling",
            )

        self._add_target(
            group_name="health_insurance",
            category=TargetCategory.HEALTH,
            level=TargetLevel.NATIONAL,
            name="chip_enrollment_national",
            value=0,
            entity=EntityType.PERSON,
            aggregation=TargetAggregation.COUNT,
            filters=(
                TargetFilter(
                    feature="chip",
                    operator=FilterOperator.GT,
                    value=0,
                ),
            ),
            source="CMS",
            units="persons",
            available_in_cps=False,
            requires_imputation=True,
        )

        self._add_target(
            group_name="health_insurance",
            category=TargetCategory.HEALTH,
            level=TargetLevel.NATIONAL,
            name="aca_enrollment_national",
            value=0,
            entity=EntityType.PERSON,
            aggregation=TargetAggregation.COUNT,
            filters=(
                TargetFilter(
                    feature="aca_enrolled",
                    operator=FilterOperator.GT,
                    value=0,
                ),
            ),
            source="CMS",
            units="persons",
            available_in_cps=False,
            requires_imputation=True,
        )

    def _add_tax_targets(self) -> None:
        tax_targets = {
            "income_tax_total": (
                4_412_800_000_000,
                TargetAggregation.SUM,
                "income_tax",
                (),
                "USD",
            ),
            "payroll_tax_total": (
                2_605_200_000_000,
                TargetAggregation.SUM,
                "payroll_tax",
                (),
                "USD",
            ),
            "eitc_claims": (
                25_000_000,
                TargetAggregation.COUNT,
                None,
                (
                    TargetFilter(
                        feature="eitc",
                        operator=FilterOperator.GT,
                        value=0,
                    ),
                ),
                "returns",
            ),
            "ctc_claims": (
                35_000_000,
                TargetAggregation.COUNT,
                None,
                (
                    TargetFilter(
                        feature="ctc",
                        operator=FilterOperator.GT,
                        value=0,
                    ),
                ),
                "returns",
            ),
        }

        for name, (value, aggregation, measure, filters, units) in tax_targets.items():
            self._add_target(
                group_name="tax_aggregates",
                category=TargetCategory.TAX,
                level=TargetLevel.NATIONAL,
                name=name,
                value=value,
                entity=EntityType.TAX_UNIT,
                aggregation=aggregation,
                measure=measure,
                filters=filters,
                source="CBO/IRS",
                units=units,
                available_in_cps=False,
                requires_imputation=True,
                notes="Requires tax calculation",
            )

    def _add_demographic_targets(self) -> None:
        filing_status = {
            "single": 75_000_000,
            "married_joint": 55_000_000,
            "married_separate": 3_000_000,
            "head_of_household": 22_000_000,
        }

        for status, count in filing_status.items():
            self._add_target(
                group_name="demographics",
                category=TargetCategory.DEMOGRAPHICS,
                level=TargetLevel.NATIONAL,
                name=f"filing_status_{status}",
                value=count,
                entity=EntityType.TAX_UNIT,
                aggregation=TargetAggregation.COUNT,
                filters=(
                    TargetFilter(
                        feature="filing_status",
                        operator=FilterOperator.EQ,
                        value=status,
                    ),
                ),
                source="IRS SOI",
                units="returns",
                available_in_cps=False,
                requires_imputation=True,
                notes="Requires tax unit modeling",
            )

    def get_group(self, name: str) -> TargetGroup | None:
        """Get a target group by name."""
        return self.groups.get(name)

    def get_all_targets(self) -> list[TargetSpec]:
        """Get all targets as a flat list."""
        all_targets: list[TargetSpec] = []
        for group in self.groups.values():
            all_targets.extend(group.targets)
        return all_targets

    def load_target_set(self, query: TargetQuery | None = None) -> TargetSet:
        """Load a canonical target set through the core provider protocol."""
        query = query or TargetQuery()
        provider_filters = query.provider_filters
        targets = self.select_targets(
            categories=provider_filters.get("categories"),
            levels=provider_filters.get("levels"),
            groups=provider_filters.get("groups"),
            only_available=bool(provider_filters.get("only_available", False)),
            entity=query.entity,
        )
        return apply_target_query(
            TargetSet(targets),
            TargetQuery(
                period=query.period,
                entity=query.entity,
                names=query.names,
                metadata_filters=query.metadata_filters,
            ),
        )

    def select_targets(
        self,
        *,
        categories: list[TargetCategory] | None = None,
        levels: list[TargetLevel] | None = None,
        groups: list[str] | None = None,
        only_available: bool = False,
        entity: EntityType | str | None = None,
    ) -> list[TargetSpec]:
        """Select canonical targets by US metadata and entity."""
        resolved_entity = (
            entity
            if entity is None or isinstance(entity, EntityType)
            else EntityType(entity)
        )

        selected: list[TargetSpec] = []
        for target in self.get_all_targets():
            if categories and target_category(target) not in categories:
                continue
            if levels and target_level(target) not in levels:
                continue
            if groups and target_group_name(target) not in groups:
                continue
            if only_available and not target_available_in_cps(target):
                continue
            if resolved_entity is not None and target.entity is not resolved_entity:
                continue
            selected.append(target)
        return selected

    def get_available_targets(self) -> list[TargetSpec]:
        """Get targets that are available in CPS data."""
        return [target for target in self.get_all_targets() if target_available_in_cps(target)]

    def get_targets_by_category(self, category: TargetCategory) -> list[TargetSpec]:
        """Get targets by US category metadata."""
        return [target for target in self.get_all_targets() if target_category(target) is category]

    def get_targets_by_level(self, level: TargetLevel) -> list[TargetSpec]:
        """Get targets by US level metadata."""
        return [target for target in self.get_all_targets() if target_level(target) is level]

    def summary(self) -> dict[str, Any]:
        """Get summary of registry contents."""
        all_targets = self.get_all_targets()
        available = self.get_available_targets()

        by_category = {category.value: len(self.get_targets_by_category(category)) for category in TargetCategory}
        by_level = {level.value: len(self.get_targets_by_level(level)) for level in TargetLevel}

        return {
            "total_targets": len(all_targets),
            "available_in_cps": len(available),
            "requires_imputation": sum(
                1 for target in all_targets if target_requires_imputation(target)
            ),
            "by_category": by_category,
            "by_level": by_level,
            "groups": {name: len(group) for name, group in self.groups.items()},
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the registry to a tabular summary."""
        records = []
        for target in self.get_all_targets():
            records.append(
                {
                    "name": target.name,
                    "entity": target.entity.value,
                    "category": target_category(target).value if target_category(target) else None,
                    "level": target_level(target).value if target_level(target) else None,
                    "group": target_group_name(target),
                    "value": target.value,
                    "measure": target.measure,
                    "aggregation": target.aggregation.value,
                    "source": target.source,
                    "units": target.units,
                    "available_in_cps": target_available_in_cps(target),
                    "requires_imputation": target_requires_imputation(target),
                    "notes": target_notes(target),
                    "filters": [
                        {
                            "feature": target_filter.feature,
                            "operator": target_filter.operator.value,
                            "value": target_filter.value,
                        }
                        for target_filter in target.filters
                    ],
                }
            )
        return pd.DataFrame(records)


def get_registry() -> TargetRegistry:
    """Get the default US target registry."""
    return TargetRegistry()


def print_registry_summary() -> None:
    """Print a summary of available US targets."""
    registry = get_registry()
    summary = registry.summary()

    print("=" * 70)
    print("MICROPLEX TARGET REGISTRY")
    print("=" * 70)
    print(f"\nTotal targets: {summary['total_targets']}")
    print(f"Available in CPS: {summary['available_in_cps']}")
    print(f"Requires imputation: {summary['requires_imputation']}")

    print("\nBy category:")
    for category, count in summary["by_category"].items():
        print(f"  {category}: {count}")

    print("\nBy level:")
    for level, count in summary["by_level"].items():
        print(f"  {level}: {count}")

    print("\nBy group:")
    for name, count in summary["groups"].items():
        print(f"  {name}: {count}")
