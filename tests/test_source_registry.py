"""Tests for declarative source-variable capability registry."""

from microplex.core import SourceVariableCapability

from microplex_us.source_registry import (
    PUF_SOURCE_VARIABLE_POLICY,
    SourceVariablePolicy,
    resolve_source_variable_capabilities,
)


def test_source_variable_policy_overrides_selected_fields():
    base = SourceVariableCapability(authoritative=True, usable_as_condition=True)
    policy = SourceVariablePolicy(authoritative=False)

    resolved = policy.apply(base)

    assert not resolved.authoritative
    assert resolved.usable_as_condition


def test_resolve_source_variable_capabilities_matches_source_prefix_and_year_suffix():
    capabilities = resolve_source_variable_capabilities(
        "irs_soi_puf_2024",
        (
            "state_fips",
            "income",
            "employment_status",
            "taxable_interest_income",
            "filing_status_code",
        ),
    )

    assert not capabilities["state_fips"].usable_as_condition
    assert not capabilities["income"].authoritative
    assert not capabilities["income"].usable_as_condition
    assert not capabilities["employment_status"].usable_as_condition
    assert "taxable_interest_income" not in capabilities
    assert capabilities["filing_status_code"].authoritative
    assert not capabilities["filing_status_code"].usable_as_condition


def test_puf_policy_spec_matches_provider_names():
    assert PUF_SOURCE_VARIABLE_POLICY.matches("irs_soi_puf")
    assert PUF_SOURCE_VARIABLE_POLICY.matches("irs_soi_puf_2024")
    assert not PUF_SOURCE_VARIABLE_POLICY.matches("cps_asec_2023")


def test_resolve_source_variable_capabilities_applies_generic_variable_semantics():
    capabilities = resolve_source_variable_capabilities(
        "cps_asec_2023",
        (
            "qualified_dividend_income",
            "non_qualified_dividend_income",
            "ordinary_dividend_income",
            "dividend_income",
        ),
    )

    assert not capabilities["dividend_income"].authoritative
    assert not capabilities["dividend_income"].usable_as_condition
    assert not capabilities["ordinary_dividend_income"].authoritative
    assert not capabilities["ordinary_dividend_income"].usable_as_condition
