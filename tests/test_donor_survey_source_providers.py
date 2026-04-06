"""Tests for PE-style donor survey source providers."""

from __future__ import annotations

import pandas as pd
from microplex.core import EntityType

import microplex_us.data_sources.donor_surveys as donor_surveys
from microplex_us.data_sources.donor_surveys import (
    ACSSourceProvider,
    DonorSurveyTables,
    SCFSourceProvider,
    SIPPSourceProvider,
)


def _acs_tables(**_kwargs) -> DonorSurveyTables:
    households = pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [100.0, 120.0],
            "state_fips": [6, 36],
            "tenure": [1, 2],
            "year": [2022, 2022],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": [11, 12, 21],
            "household_id": [1, 1, 2],
            "age": [45, 12, 68],
            "sex": [1, 2, 2],
            "is_male": [1.0, 0.0, 0.0],
            "is_household_head": [1.0, 0.0, 1.0],
            "tenure_type": [1, 1, 2],
            "employment_income": [50_000.0, 0.0, 12_000.0],
            "self_employment_income": [5_000.0, 0.0, 0.0],
            "social_security": [0.0, 0.0, 20_000.0],
            "taxable_pension_income": [0.0, 0.0, 15_000.0],
            "rent": [1_200.0, 0.0, 950.0],
            "real_estate_taxes": [3_000.0, 0.0, 0.0],
            "income": [55_000.0, 0.0, 47_000.0],
            "weight": [100.0, 100.0, 120.0],
            "year": [2022, 2022, 2022],
        }
    )
    return DonorSurveyTables(households=households, persons=persons)


def _sipp_tips_tables(**_kwargs) -> DonorSurveyTables:
    households = pd.DataFrame(
        {
            "household_id": ["100:1", "101:1"],
            "household_weight": [80.0, 90.0],
            "state_fips": [0, 0],
            "tenure": [0, 0],
            "year": [2023, 2023],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": ["100:1:1", "100:1:2", "101:1:1"],
            "household_id": ["100:1", "100:1", "101:1"],
            "age": [35, 8, 50],
            "sex": [1, 2, 2],
            "employment_income": [40_000.0, 0.0, 25_000.0],
            "income": [40_000.0, 0.0, 25_000.0],
            "tip_income": [900.0, 0.0, 250.0],
            "count_under_18": [1.0, 1.0, 0.0],
            "count_under_6": [0.0, 0.0, 0.0],
            "weight": [80.0, 80.0, 90.0],
            "year": [2023, 2023, 2023],
        }
    )
    return DonorSurveyTables(households=households, persons=persons)


def _sipp_assets_tables(**_kwargs) -> DonorSurveyTables:
    households = pd.DataFrame(
        {
            "household_id": ["100", "101"],
            "household_weight": [80.0, 90.0],
            "state_fips": [0, 0],
            "tenure": [0, 0],
            "year": [2023, 2023],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": ["100:1", "101:1"],
            "household_id": ["100", "101"],
            "age": [35, 50],
            "sex": [1, 2],
            "is_female": [0.0, 1.0],
            "is_married": [1.0, 0.0],
            "employment_income": [40_000.0, 25_000.0],
            "income": [40_000.0, 25_000.0],
            "count_under_18": [1.0, 0.0],
            "bank_account_assets": [2_500.0, 10_000.0],
            "stock_assets": [0.0, 4_000.0],
            "bond_assets": [0.0, 1_500.0],
            "weight": [80.0, 90.0],
            "year": [2023, 2023],
        }
    )
    return DonorSurveyTables(households=households, persons=persons)


def _scf_tables(**_kwargs) -> DonorSurveyTables:
    households = pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [10.0, 12.0],
            "state_fips": [0, 0],
            "tenure": [0, 0],
            "year": [2022, 2022],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": [1, 2],
            "household_id": [1, 2],
            "age": [45, 68],
            "sex": [1, 2],
            "is_female": [0.0, 1.0],
            "cps_race": [1, 2],
            "is_married": [1.0, 0.0],
            "own_children_in_household": [1.0, 0.0],
            "employment_income": [75_000.0, 0.0],
            "income": [75_000.0, 0.0],
            "interest_dividend_income": [1_200.0, 400.0],
            "social_security_pension_income": [0.0, 18_000.0],
            "net_worth": [350_000.0, 180_000.0],
            "auto_loan_balance": [8_000.0, 0.0],
            "auto_loan_interest": [550.0, 0.0],
            "weight": [10.0, 12.0],
            "year": [2022, 2022],
        }
    )
    return DonorSurveyTables(households=households, persons=persons)


def test_acs_source_provider_builds_observation_frame_from_injected_loader() -> None:
    provider = ACSSourceProvider(loader=_acs_tables)

    frame = provider.load_frame()

    assert frame.source.name == "acs_2022"
    assert frame.source.observes("rent", EntityType.PERSON)
    assert frame.source.allows_conditioning_on("state_fips") is True
    assert list(frame.tables[EntityType.HOUSEHOLD]["household_id"]) == [1, 2]


def test_sipp_and_scf_provider_fillers_are_not_usable_as_conditions() -> None:
    tips_provider = SIPPSourceProvider(block="tips", loader=_sipp_tips_tables)
    assets_provider = SIPPSourceProvider(block="assets", loader=_sipp_assets_tables)
    scf_provider = SCFSourceProvider(loader=_scf_tables)

    tips_frame = tips_provider.load_frame()
    assets_frame = assets_provider.load_frame()
    scf_frame = scf_provider.load_frame()

    assert tips_frame.source.name == "sipp_tips_2023"
    assert assets_frame.source.name == "sipp_assets_2023"
    assert scf_frame.source.name == "scf_2022"
    assert tips_frame.source.allows_conditioning_on("state_fips") is False
    assert assets_frame.source.is_authoritative_for("tenure") is False
    assert scf_frame.source.allows_conditioning_on("state_fips") is False
    assert scf_frame.source.observes("net_worth", EntityType.PERSON)


def test_sipp_tips_provider_uses_manifest_backed_raw_loader(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "pu2023_slim.csv"
    pd.DataFrame(
        {
            "SSUID": ["100", "100", "101"],
            "MONTHCODE": [1, 1, 2],
            "PNUM": [1, 2, 1],
            "WPFINWGT": [80.0, 80.0, 90.0],
            "TAGE": [35, 8, 50],
            "ESEX": [1, 2, 2],
            "TPTOTINC": [1000.0, 0.0, 500.0],
            "TXAMT1": [10.0, 0.0, 5.0],
            "TXAMT2": [2.0, 0.0, 0.0],
        }
    ).to_csv(path, index=False)

    monkeypatch.setattr(
        donor_surveys,
        "_download_policyengine_us_data_file",
        lambda **_kwargs: path,
    )

    frame = SIPPSourceProvider(block="tips").load_frame()
    persons = frame.tables[EntityType.PERSON]

    assert frame.source.name == "sipp_tips_2023"
    assert persons["tip_income"].tolist() == [144.0, 0.0, 60.0]
    assert persons["employment_income"].tolist() == [12000.0, 0.0, 6000.0]
    assert persons["count_under_18"].tolist() == [1.0, 1.0, 0.0]
    assert persons["count_under_6"].tolist() == [0.0, 0.0, 0.0]


def test_sipp_assets_provider_uses_manifest_backed_raw_loader(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "pu2023.csv"
    pd.DataFrame(
        {
            "SSUID": ["100", "100", "101"],
            "PNUM": [1, 2, 1],
            "MONTHCODE": [11, 12, 12],
            "WPFINWGT": [80.0, 80.0, 90.0],
            "TAGE": [10, 35, 50],
            "ESEX": [1, 2, 2],
            "EMS": [2, 1, 0],
            "TPTOTINC": [100.0, 200.0, 300.0],
            "TVAL_BANK": [1.0, 2.0, 3.0],
            "TVAL_STMF": [4.0, 5.0, 6.0],
            "TVAL_BOND": [7.0, 8.0, 9.0],
        }
    ).to_csv(path, index=False, sep="|")

    monkeypatch.setattr(
        donor_surveys,
        "_download_policyengine_us_data_file",
        lambda **_kwargs: path,
    )

    frame = SIPPSourceProvider(block="assets").load_frame()
    persons = frame.tables[EntityType.PERSON]

    assert frame.source.name == "sipp_assets_2023"
    assert persons["person_id"].tolist() == ["100:2", "101:1"]
    assert persons["employment_income"].tolist() == [2400.0, 3600.0]
    assert persons["is_female"].tolist() == [1.0, 1.0]
    assert persons["is_married"].tolist() == [1.0, 0.0]
    assert persons["count_under_18"].tolist() == [0.0, 0.0]
