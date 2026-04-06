"""Tests for shared PE source-impute donor block specs."""

from __future__ import annotations

from microplex.core import SourceArchetype

from microplex_us.pe_source_impute_specs import (
    get_pe_source_impute_block_spec,
    load_pe_source_impute_block_specs,
    resolve_pe_source_impute_block_key,
    resolve_sipp_source_impute_block_spec,
)


def test_load_pe_source_impute_block_specs_reads_manifest() -> None:
    specs = load_pe_source_impute_block_specs()

    assert set(specs) == {"acs", "sipp_tips", "sipp_assets", "scf"}
    assert specs["acs"].archetype is SourceArchetype.HOUSEHOLD_INCOME
    assert specs["scf"].archetype is SourceArchetype.WEALTH
    assert specs["sipp_assets"].target_variables == (
        "bank_account_assets",
        "stock_assets",
        "bond_assets",
    )


def test_resolve_pe_source_impute_block_key_uses_source_name_and_targets() -> None:
    assert (
        resolve_pe_source_impute_block_key(
            donor_source_name="acs_2022",
            donor_block=("rent",),
        )
        == "acs"
    )
    assert (
        resolve_pe_source_impute_block_key(
            donor_source_name="sipp_assets_2023",
            donor_block=("stock_assets", "bond_assets"),
        )
        == "sipp_assets"
    )
    assert (
        resolve_pe_source_impute_block_key(
            donor_source_name="scf_2022",
            donor_block=("tip_income",),
        )
        is None
    )


def test_resolve_sipp_source_impute_block_spec_and_named_lookup() -> None:
    tips = resolve_sipp_source_impute_block_spec("tips")
    scf = get_pe_source_impute_block_spec("scf")

    assert tips.key == "sipp_tips"
    assert tips.descriptor_name == "sipp_tips"
    assert scf.descriptor_name == "scf"
