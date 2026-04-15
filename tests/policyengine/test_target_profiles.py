from __future__ import annotations

from microplex_us.policyengine.target_profiles import (
    policyengine_us_target_profile_names,
    resolve_policyengine_us_target_profile,
)


def test_policyengine_us_target_profile_names_include_no_state_aca_variant() -> None:
    assert "pe_native_broad" in policyengine_us_target_profile_names()
    assert "pe_native_broad_no_state_aca" in policyengine_us_target_profile_names()


def test_no_state_aca_profile_excludes_only_state_aca_cells() -> None:
    broad = resolve_policyengine_us_target_profile("pe_native_broad")
    no_state_aca = resolve_policyengine_us_target_profile(
        "pe_native_broad_no_state_aca"
    )

    broad_cells = {
        (cell.variable, cell.geo_level, cell.domain_variable, cell.geographic_id)
        for cell in broad
    }
    no_state_aca_cells = {
        (cell.variable, cell.geo_level, cell.domain_variable, cell.geographic_id)
        for cell in no_state_aca
    }

    assert (
        "aca_ptc",
        "state",
        "aca_ptc",
        None,
    ) in broad_cells
    assert (
        "tax_unit_count",
        "state",
        "aca_ptc",
        None,
    ) in broad_cells
    assert (
        "aca_ptc",
        "national",
        "aca_ptc",
        None,
    ) in no_state_aca_cells
    assert (
        "tax_unit_count",
        "national",
        "aca_ptc",
        None,
    ) in no_state_aca_cells
    assert (
        "aca_ptc",
        "state",
        "aca_ptc",
        None,
    ) not in no_state_aca_cells
    assert (
        "tax_unit_count",
        "state",
        "aca_ptc",
        None,
    ) not in no_state_aca_cells

