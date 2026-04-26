"""Named target-cell profiles for PolicyEngine US target selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyEngineUSTargetCell:
    """One exact target cell from the PolicyEngine US target DB."""

    variable: str
    geo_level: str | None = None
    domain_variable: str | None = None
    geographic_id: str | None = None

    def to_provider_filter(self) -> dict[str, str | None]:
        return {
            "variable": self.variable,
            "geo_level": self.geo_level,
            "domain_variable": self.domain_variable,
            "geographic_id": self.geographic_id,
        }


PE_NATIVE_BROAD_TARGET_CELLS: tuple[PolicyEngineUSTargetCell, ...] = (
    PolicyEngineUSTargetCell("aca_ptc", geo_level="national", domain_variable="aca_ptc"),
    PolicyEngineUSTargetCell("adjusted_gross_income", geo_level="national"),
    PolicyEngineUSTargetCell("alimony_expense", geo_level="national"),
    PolicyEngineUSTargetCell("alimony_income", geo_level="national"),
    PolicyEngineUSTargetCell("charitable_deduction", geo_level="national"),
    PolicyEngineUSTargetCell("child_support_expense", geo_level="national"),
    PolicyEngineUSTargetCell("child_support_received", geo_level="national"),
    PolicyEngineUSTargetCell("dividend_income", geo_level="national", domain_variable="dividend_income"),
    PolicyEngineUSTargetCell("eitc", geo_level="national"),
    PolicyEngineUSTargetCell("eitc", geo_level="national", domain_variable="eitc_child_count"),
    PolicyEngineUSTargetCell(
        "eitc",
        geo_level="national",
        domain_variable="adjusted_gross_income,eitc,eitc_child_count",
    ),
    PolicyEngineUSTargetCell(
        "health_insurance_premiums_without_medicare_part_b",
        geo_level="national",
    ),
    PolicyEngineUSTargetCell("income_tax", geo_level="national", domain_variable="income_tax"),
    PolicyEngineUSTargetCell(
        "income_tax_before_credits",
        geo_level="national",
        domain_variable="income_tax_before_credits",
    ),
    PolicyEngineUSTargetCell("income_tax_positive", geo_level="national"),
    PolicyEngineUSTargetCell("interest_deduction", geo_level="national"),
    PolicyEngineUSTargetCell("medicaid", geo_level="national"),
    PolicyEngineUSTargetCell("medical_expense_deduction", geo_level="national"),
    PolicyEngineUSTargetCell(
        "medical_expense_deduction",
        geo_level="national",
        domain_variable="medical_expense_deduction",
    ),
    PolicyEngineUSTargetCell("medicare_part_b_premiums", geo_level="national"),
    PolicyEngineUSTargetCell("net_capital_gains", geo_level="national", domain_variable="net_capital_gains"),
    PolicyEngineUSTargetCell("net_worth", geo_level="national"),
    PolicyEngineUSTargetCell("other_medical_expenses", geo_level="national"),
    PolicyEngineUSTargetCell("over_the_counter_health_expenses", geo_level="national"),
    PolicyEngineUSTargetCell("person_count", geo_level="national", domain_variable="aca_ptc"),
    PolicyEngineUSTargetCell("person_count", geo_level="national", domain_variable="age"),
    PolicyEngineUSTargetCell("person_count", geo_level="national", domain_variable="medicaid"),
    PolicyEngineUSTargetCell("person_count", geo_level="national", domain_variable="ssn_card_type"),
    PolicyEngineUSTargetCell("qualified_business_income_deduction", geo_level="national"),
    PolicyEngineUSTargetCell(
        "qualified_business_income_deduction",
        geo_level="national",
        domain_variable="qualified_business_income_deduction",
    ),
    PolicyEngineUSTargetCell(
        "qualified_dividend_income",
        geo_level="national",
        domain_variable="qualified_dividend_income",
    ),
    PolicyEngineUSTargetCell("real_estate_taxes", geo_level="national"),
    PolicyEngineUSTargetCell(
        "real_estate_taxes",
        geo_level="national",
        domain_variable="real_estate_taxes",
    ),
    PolicyEngineUSTargetCell("refundable_ctc", geo_level="national", domain_variable="refundable_ctc"),
    PolicyEngineUSTargetCell("rent", geo_level="national"),
    PolicyEngineUSTargetCell("rental_income", geo_level="national", domain_variable="rental_income"),
    PolicyEngineUSTargetCell("roth_ira_contributions", geo_level="national"),
    PolicyEngineUSTargetCell("salt", geo_level="national", domain_variable="salt"),
    PolicyEngineUSTargetCell("salt_deduction", geo_level="national"),
    PolicyEngineUSTargetCell(
        "self_employment_income",
        geo_level="national",
        domain_variable="self_employment_income",
    ),
    PolicyEngineUSTargetCell("snap", geo_level="national"),
    PolicyEngineUSTargetCell("social_security", geo_level="national"),
    PolicyEngineUSTargetCell("social_security_dependents", geo_level="national"),
    PolicyEngineUSTargetCell("social_security_disability", geo_level="national"),
    PolicyEngineUSTargetCell("social_security_retirement", geo_level="national"),
    PolicyEngineUSTargetCell("social_security_survivors", geo_level="national"),
    PolicyEngineUSTargetCell("spm_unit_capped_housing_subsidy", geo_level="national"),
    PolicyEngineUSTargetCell("spm_unit_capped_work_childcare_expenses", geo_level="national"),
    PolicyEngineUSTargetCell("ssi", geo_level="national"),
    PolicyEngineUSTargetCell("tanf", geo_level="national"),
    PolicyEngineUSTargetCell(
        "tax_exempt_interest_income",
        geo_level="national",
        domain_variable="tax_exempt_interest_income",
    ),
    PolicyEngineUSTargetCell("tax_unit_count", geo_level="national", domain_variable="aca_ptc"),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="dividend_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="eitc_child_count",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="adjusted_gross_income,eitc,eitc_child_count",
    ),
    PolicyEngineUSTargetCell("tax_unit_count", geo_level="national", domain_variable="income_tax"),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="income_tax_before_credits",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="medical_expense_deduction",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="net_capital_gains",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="qualified_business_income_deduction",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="qualified_dividend_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="real_estate_taxes",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="refundable_ctc",
    ),
    PolicyEngineUSTargetCell("tax_unit_count", geo_level="national", domain_variable="rental_income"),
    PolicyEngineUSTargetCell("tax_unit_count", geo_level="national", domain_variable="salt"),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="self_employment_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="tax_exempt_interest_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="tax_unit_partnership_s_corp_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="taxable_interest_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="taxable_ira_distributions",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="taxable_pension_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="taxable_social_security",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="national",
        domain_variable="unemployment_compensation",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_partnership_s_corp_income",
        geo_level="national",
        domain_variable="tax_unit_partnership_s_corp_income",
    ),
    PolicyEngineUSTargetCell(
        "taxable_interest_income",
        geo_level="national",
        domain_variable="taxable_interest_income",
    ),
    PolicyEngineUSTargetCell(
        "taxable_ira_distributions",
        geo_level="national",
        domain_variable="taxable_ira_distributions",
    ),
    PolicyEngineUSTargetCell(
        "taxable_pension_income",
        geo_level="national",
        domain_variable="taxable_pension_income",
    ),
    PolicyEngineUSTargetCell(
        "taxable_social_security",
        geo_level="national",
        domain_variable="taxable_social_security",
    ),
    PolicyEngineUSTargetCell("tip_income", geo_level="national"),
    PolicyEngineUSTargetCell("traditional_ira_contributions", geo_level="national"),
    PolicyEngineUSTargetCell("unemployment_compensation", geo_level="national"),
    PolicyEngineUSTargetCell(
        "unemployment_compensation",
        geo_level="national",
        domain_variable="unemployment_compensation",
    ),
    PolicyEngineUSTargetCell("aca_ptc", geo_level="state", domain_variable="aca_ptc"),
    PolicyEngineUSTargetCell("adjusted_gross_income", geo_level="state"),
    PolicyEngineUSTargetCell("dividend_income", geo_level="state", domain_variable="dividend_income"),
    PolicyEngineUSTargetCell("eitc", geo_level="state", domain_variable="eitc_child_count"),
    PolicyEngineUSTargetCell("household_count", geo_level="state", domain_variable="snap"),
    PolicyEngineUSTargetCell("income_tax", geo_level="state", domain_variable="income_tax"),
    PolicyEngineUSTargetCell(
        "income_tax_before_credits",
        geo_level="state",
        domain_variable="income_tax_before_credits",
    ),
    PolicyEngineUSTargetCell(
        "medical_expense_deduction",
        geo_level="state",
        domain_variable="medical_expense_deduction",
    ),
    PolicyEngineUSTargetCell("net_capital_gains", geo_level="state", domain_variable="net_capital_gains"),
    PolicyEngineUSTargetCell("person_count", geo_level="state", domain_variable="aca_ptc"),
    PolicyEngineUSTargetCell("person_count", geo_level="state", domain_variable="adjusted_gross_income"),
    PolicyEngineUSTargetCell("person_count", geo_level="state", domain_variable="age"),
    PolicyEngineUSTargetCell("person_count", geo_level="state", domain_variable="medicaid_enrolled"),
    PolicyEngineUSTargetCell(
        "qualified_business_income_deduction",
        geo_level="state",
        domain_variable="qualified_business_income_deduction",
    ),
    PolicyEngineUSTargetCell(
        "qualified_dividend_income",
        geo_level="state",
        domain_variable="qualified_dividend_income",
    ),
    PolicyEngineUSTargetCell("real_estate_taxes", geo_level="state", domain_variable="real_estate_taxes"),
    PolicyEngineUSTargetCell("refundable_ctc", geo_level="state", domain_variable="refundable_ctc"),
    PolicyEngineUSTargetCell("rental_income", geo_level="state", domain_variable="rental_income"),
    PolicyEngineUSTargetCell("salt", geo_level="state", domain_variable="salt"),
    PolicyEngineUSTargetCell(
        "self_employment_income",
        geo_level="state",
        domain_variable="self_employment_income",
    ),
    PolicyEngineUSTargetCell("snap", geo_level="state", domain_variable="snap"),
    PolicyEngineUSTargetCell("state_income_tax", geo_level="state"),
    PolicyEngineUSTargetCell(
        "tax_exempt_interest_income",
        geo_level="state",
        domain_variable="tax_exempt_interest_income",
    ),
    PolicyEngineUSTargetCell("tax_unit_count", geo_level="state", domain_variable="aca_ptc"),
    PolicyEngineUSTargetCell("tax_unit_count", geo_level="state", domain_variable="adjusted_gross_income"),
    PolicyEngineUSTargetCell("tax_unit_count", geo_level="state", domain_variable="dividend_income"),
    PolicyEngineUSTargetCell("tax_unit_count", geo_level="state", domain_variable="eitc_child_count"),
    PolicyEngineUSTargetCell("tax_unit_count", geo_level="state", domain_variable="income_tax"),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="income_tax_before_credits",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="medical_expense_deduction",
    ),
    PolicyEngineUSTargetCell("tax_unit_count", geo_level="state", domain_variable="net_capital_gains"),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="qualified_business_income_deduction",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="qualified_dividend_income",
    ),
    PolicyEngineUSTargetCell("tax_unit_count", geo_level="state", domain_variable="real_estate_taxes"),
    PolicyEngineUSTargetCell("tax_unit_count", geo_level="state", domain_variable="refundable_ctc"),
    PolicyEngineUSTargetCell("tax_unit_count", geo_level="state", domain_variable="rental_income"),
    PolicyEngineUSTargetCell("tax_unit_count", geo_level="state", domain_variable="salt"),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="self_employment_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="tax_exempt_interest_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="tax_unit_partnership_s_corp_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="taxable_interest_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="taxable_ira_distributions",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="taxable_pension_income",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="taxable_social_security",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_count",
        geo_level="state",
        domain_variable="unemployment_compensation",
    ),
    PolicyEngineUSTargetCell(
        "tax_unit_partnership_s_corp_income",
        geo_level="state",
        domain_variable="tax_unit_partnership_s_corp_income",
    ),
    PolicyEngineUSTargetCell(
        "taxable_interest_income",
        geo_level="state",
        domain_variable="taxable_interest_income",
    ),
    PolicyEngineUSTargetCell(
        "taxable_ira_distributions",
        geo_level="state",
        domain_variable="taxable_ira_distributions",
    ),
    PolicyEngineUSTargetCell(
        "taxable_pension_income",
        geo_level="state",
        domain_variable="taxable_pension_income",
    ),
    PolicyEngineUSTargetCell(
        "taxable_social_security",
        geo_level="state",
        domain_variable="taxable_social_security",
    ),
    PolicyEngineUSTargetCell(
        "unemployment_compensation",
        geo_level="state",
        domain_variable="unemployment_compensation",
    ),
)

_PE_NATIVE_BROAD_NO_STATE_ACA_EXCLUDED_CELLS = frozenset(
    {
        ("aca_ptc", "state", "aca_ptc", None),
        ("tax_unit_count", "state", "aca_ptc", None),
    }
)

PE_NATIVE_BROAD_NO_STATE_ACA_TARGET_CELLS: tuple[PolicyEngineUSTargetCell, ...] = tuple(
    cell
    for cell in PE_NATIVE_BROAD_TARGET_CELLS
    if (
        cell.variable,
        cell.geo_level,
        cell.domain_variable,
        cell.geographic_id,
    )
    not in _PE_NATIVE_BROAD_NO_STATE_ACA_EXCLUDED_CELLS
)

_TARGET_PROFILES: dict[str, tuple[PolicyEngineUSTargetCell, ...]] = {
    "pe_native_broad": PE_NATIVE_BROAD_TARGET_CELLS,
    "pe_native_broad_no_state_aca": PE_NATIVE_BROAD_NO_STATE_ACA_TARGET_CELLS,
}


def policyengine_us_target_profile_names() -> tuple[str, ...]:
    return tuple(sorted(_TARGET_PROFILES))


def resolve_policyengine_us_target_profile(
    name: str,
) -> tuple[PolicyEngineUSTargetCell, ...]:
    try:
        return _TARGET_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(policyengine_us_target_profile_names())
        raise ValueError(
            f"Unknown PolicyEngine US target profile '{name}'. Known profiles: {known}"
        ) from exc
