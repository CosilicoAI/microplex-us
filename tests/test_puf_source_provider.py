"""Tests for PUF source-provider implementation."""

from __future__ import annotations

import pandas as pd
from microplex.core import EntityType, SourceArchetype, SourceProvider, SourceQuery

import microplex_us.data_sources.puf as puf_module
from microplex_us.data_sources import PUFSourceProvider, expand_to_persons
from microplex_us.data_sources.puf import _sample_tax_units


def test_expand_to_persons_preserves_joint_tax_unit_monetary_totals():
    tax_units = pd.DataFrame(
        {
            "filing_status": ["JOINT"],
            "employment_income": [100.0],
            "self_employment_income": [50.0],
            "taxable_interest_income": [20.0],
            "ordinary_dividend_income": [30.0],
            "qualified_dividend_income": [10.0],
            "gross_social_security": [40.0],
            "taxable_pension_income": [60.0],
            "unemployment_compensation": [80.0],
            "rental_income": [90.0],
            "weight": [1.0],
            "household_id": ["joint-household"],
            "year": [2024],
        }
    )

    persons = expand_to_persons(tax_units)
    head = persons.loc[persons["is_head"] == 1].iloc[0]
    spouse = persons.loc[persons["is_spouse"] == 1].iloc[0]

    assert head["employment_income"] == 60.0
    assert spouse["employment_income"] == 40.0
    assert head["self_employment_income"] == 30.0
    assert spouse["self_employment_income"] == 20.0
    assert head["taxable_interest_income"] == 10.0
    assert spouse["taxable_interest_income"] == 10.0
    assert head["ordinary_dividend_income"] == 15.0
    assert spouse["ordinary_dividend_income"] == 15.0
    assert head["non_qualified_dividend_income"] == 10.0
    assert spouse["non_qualified_dividend_income"] == 10.0
    assert persons["taxable_interest_income"].sum() == 20.0
    assert persons["ordinary_dividend_income"].sum() == 30.0
    assert persons["qualified_dividend_income"].sum() == 10.0
    assert persons["non_qualified_dividend_income"].sum() == 20.0
    assert persons["dividend_income"].sum() == 30.0
    assert persons["social_security"].sum() == 40.0
    assert persons["pension_income"].sum() == 60.0
    assert persons["income"].sum() == 470.0


def test_expand_to_persons_splits_negative_joint_self_employment_losses():
    tax_units = pd.DataFrame(
        {
            "filing_status": ["JOINT"],
            "self_employment_income": [-100.0],
            "weight": [1.0],
            "household_id": ["joint-household"],
            "year": [2024],
        }
    )

    persons = expand_to_persons(tax_units)
    head = persons.loc[persons["is_head"] == 1].iloc[0]
    spouse = persons.loc[persons["is_spouse"] == 1].iloc[0]

    assert head["self_employment_income"] == -60.0
    assert spouse["self_employment_income"] == -40.0
    assert persons["self_employment_income"].sum() == -100.0
    assert persons["income"].sum() == -100.0


def test_puf_source_provider_loads_observation_frame_from_local_files(tmp_path):
    puf = pd.DataFrame(
        {
            "RECID": [101, 202],
            "MARS": [2, 1],
            "XTOT": [2, 1],
            "S006": [100.0, 200.0],
            "E00200": [50_000.0, 20_000.0],
            "E00900": [0.0, 5_000.0],
            "AGE_HEAD": [45, 67],
            "GENDER": [1, 2],
        }
    )
    puf_path = tmp_path / "puf.csv"
    demographics_path = tmp_path / "demographics.csv"
    puf.to_csv(puf_path, index=False)
    pd.DataFrame({"RECID": [101, 202]}).to_csv(demographics_path, index=False)

    provider = PUFSourceProvider(
        puf_path=puf_path,
        demographics_path=demographics_path,
        target_year=2024,
    )
    frame = provider.load_frame(
        SourceQuery(period=2024, provider_filters={"sample_n": 1, "random_seed": 0})
    )

    assert isinstance(provider, SourceProvider)
    assert set(frame.tables) == {EntityType.HOUSEHOLD, EntityType.PERSON}
    assert len(frame.tables[EntityType.HOUSEHOLD]) == 1
    assert frame.tables[EntityType.PERSON]["household_id"].nunique() == 1
    assert frame.tables[EntityType.HOUSEHOLD]["year"].tolist() == [2024]
    assert frame.tables[EntityType.PERSON]["year"].nunique() == 1
    assert "income" in frame.tables[EntityType.PERSON].columns
    assert provider.descriptor.name.startswith("irs_soi_puf_")
    assert frame.source.archetype is SourceArchetype.TAX_MICRODATA


def test_puf_source_provider_sampling_respects_tax_unit_weights(tmp_path):
    puf = pd.DataFrame(
        {
            "RECID": [101, 202, 303],
            "MARS": [1, 1, 1],
            "XTOT": [1, 1, 1],
            "S006": [0.0, 0.0, 100.0],
            "E00200": [10_000.0, 20_000.0, 30_000.0],
            "AGE_HEAD": [45, 55, 65],
            "GENDER": [1, 2, 1],
        }
    )
    puf_path = tmp_path / "puf.csv"
    demographics_path = tmp_path / "demographics.csv"
    puf.to_csv(puf_path, index=False)
    pd.DataFrame({"RECID": [101, 202, 303]}).to_csv(demographics_path, index=False)

    provider = PUFSourceProvider(
        puf_path=puf_path,
        demographics_path=demographics_path,
        target_year=2024,
    )
    frame = provider.load_frame(
        SourceQuery(period=2024, provider_filters={"sample_n": 1, "random_seed": 0})
    )

    assert frame.tables[EntityType.HOUSEHOLD]["household_id"].tolist() == ["303"]
    assert frame.tables[EntityType.PERSON]["household_id"].nunique() == 1
    assert frame.tables[EntityType.PERSON]["household_id"].iloc[0] == "303"


def test_puf_source_provider_marks_placeholder_and_derived_variables_in_capabilities(
    tmp_path,
):
    puf = pd.DataFrame(
        {
            "RECID": [101],
            "MARS": [1],
            "XTOT": [1],
            "S006": [100.0],
            "E00200": [50_000.0],
            "AGE_HEAD": [45],
            "GENDER": [1],
        }
    )
    puf_path = tmp_path / "puf.csv"
    demographics_path = tmp_path / "demographics.csv"
    puf.to_csv(puf_path, index=False)
    pd.DataFrame({"RECID": [101]}).to_csv(demographics_path, index=False)

    provider = PUFSourceProvider(
        puf_path=puf_path,
        demographics_path=demographics_path,
        target_year=2024,
    )
    frame = provider.load_frame(SourceQuery(period=2024))
    descriptor = frame.source

    assert not descriptor.allows_conditioning_on("state_fips")
    assert not descriptor.is_authoritative_for("state_fips")
    assert not descriptor.allows_conditioning_on("income")
    assert not descriptor.is_authoritative_for("income")
    assert descriptor.is_authoritative_for("employment_income")
    assert not descriptor.allows_conditioning_on("employment_income")
    assert descriptor.allows_conditioning_on("age")


def test_puf_source_provider_does_not_duplicate_joint_tax_unit_financial_income(tmp_path):
    puf = pd.DataFrame(
        {
            "RECID": [101],
            "MARS": [2],
            "XTOT": [2],
            "S006": [100.0],
            "E00200": [100.0],
            "E00900": [50.0],
            "E00300": [20.0],
            "E00600": [30.0],
            "E00650": [10.0],
            "AGE_HEAD": [45],
            "GENDER": [1],
        }
    )
    puf_path = tmp_path / "puf.csv"
    demographics_path = tmp_path / "demographics.csv"
    puf.to_csv(puf_path, index=False)
    pd.DataFrame({"RECID": [101]}).to_csv(demographics_path, index=False)

    provider = PUFSourceProvider(
        puf_path=puf_path,
        demographics_path=demographics_path,
        target_year=2015,
    )
    frame = provider.load_frame(SourceQuery(period=2015))
    persons = frame.tables[EntityType.PERSON]

    assert len(persons) == 2
    assert persons["taxable_interest_income"].sum() == 20.0
    assert persons["ordinary_dividend_income"].sum() == 30.0
    assert persons["qualified_dividend_income"].sum() == 10.0
    assert persons["income"].sum() == 200.0


def test_puf_source_provider_maps_policyengine_medical_and_alimony_inputs(tmp_path):
    puf = pd.DataFrame(
        {
            "RECID": [101],
            "MARS": [1],
            "XTOT": [1],
            "S006": [100.0],
            "E00200": [50_000.0],
            "E00800": [2_000.0],
            "E17500": [1_000.0],
            "E26390": [700.0],
            "E26400": [200.0],
            "AGE_HEAD": [45],
            "GENDER": [1],
        }
    )
    puf_path = tmp_path / "puf.csv"
    demographics_path = tmp_path / "demographics.csv"
    puf.to_csv(puf_path, index=False)
    pd.DataFrame({"RECID": [101]}).to_csv(demographics_path, index=False)

    provider = PUFSourceProvider(
        puf_path=puf_path,
        demographics_path=demographics_path,
        target_year=2015,
    )
    frame = provider.load_frame(SourceQuery(period=2015))
    persons = frame.tables[EntityType.PERSON]

    assert persons["alimony_income"].sum() == 2_000.0
    assert persons["health_insurance_premiums_without_medicare_part_b"].sum() == 453.0
    assert persons["other_medical_expenses"].sum() == 325.0
    assert persons["medicare_part_b_premiums"].sum() == 137.0
    assert persons["over_the_counter_health_expenses"].sum() == 85.0
    assert persons["estate_income"].sum() == 500.0
    assert persons["income"].sum() == 52_000.0


def test_download_puf_prefers_existing_local_files_without_hub_lookup(tmp_path, monkeypatch):
    puf_path = tmp_path / "puf_2015.csv"
    demographics_path = tmp_path / "demographics_2015.csv"
    puf_path.write_text("RECID,MARS\n1,1\n")
    demographics_path.write_text("RECID\n1\n")

    def fail_download(*args, **kwargs):
        raise AssertionError("hf_hub_download should not be called when local files exist")

    monkeypatch.setattr(puf_module, "hf_hub_download", fail_download, raising=False)

    resolved_puf_path, resolved_demo_path = puf_module.download_puf(tmp_path)

    assert resolved_puf_path == puf_path
    assert resolved_demo_path == demographics_path


def test_map_puf_variables_seed_controls_age_imputation():
    puf = pd.DataFrame(
        {
            "RECID": [101, 202, 303],
            "MARS": [1, 1, 1],
            "XTOT": [1, 1, 1],
            "S006": [100.0, 100.0, 100.0],
            "E00200": [50_000.0, 150_000.0, 250_000.0],
            "E02400": [0.0, 10_000.0, 0.0],
            "E01400": [0.0, 0.0, 20_000.0],
        }
    )

    first = puf_module.map_puf_variables(puf, random_seed=17)
    second = puf_module.map_puf_variables(puf, random_seed=17)
    third = puf_module.map_puf_variables(puf, random_seed=18)

    assert first["age"].tolist() == second["age"].tolist()
    assert first["age"].tolist() != third["age"].tolist()


def test_puf_source_provider_age_imputation_is_reproducible_with_same_seed(tmp_path):
    puf = pd.DataFrame(
        {
            "RECID": [101, 202, 303],
            "MARS": [1, 2, 1],
            "XTOT": [1, 2, 1],
            "S006": [100.0, 120.0, 80.0],
            "E00200": [50_000.0, 75_000.0, 150_000.0],
            "E02400": [0.0, 8_000.0, 0.0],
            "E01400": [0.0, 0.0, 25_000.0],
            "GENDER": [1, 2, 1],
        }
    )
    puf_path = tmp_path / "puf.csv"
    demographics_path = tmp_path / "demographics.csv"
    puf.to_csv(puf_path, index=False)
    pd.DataFrame({"RECID": [101, 202, 303]}).to_csv(demographics_path, index=False)

    provider = PUFSourceProvider(
        puf_path=puf_path,
        demographics_path=demographics_path,
        target_year=2024,
    )
    query = SourceQuery(period=2024, provider_filters={"sample_n": 3, "random_seed": 7})
    first = provider.load_frame(query)
    second = provider.load_frame(query)

    first_persons = first.tables[EntityType.PERSON].sort_values("person_id").reset_index(drop=True)
    second_persons = second.tables[EntityType.PERSON].sort_values("person_id").reset_index(drop=True)

    assert first_persons["age"].tolist() == second_persons["age"].tolist()


def test_puf_sampling_falls_back_to_uniform_when_weighted_sampling_is_infeasible(
    monkeypatch,
):
    tax_units = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "weight": [10.0, 20.0, 30.0],
        }
    )

    original_sample = pd.DataFrame.sample

    def flaky_sample(self, *args, **kwargs):
        if kwargs.get("weights") is not None:
            raise ValueError(
                "Weighted sampling cannot be achieved with replace=False."
            )
        return original_sample(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "sample", flaky_sample)

    sampled = _sample_tax_units(
        tax_units,
        sample_n=2,
        random_seed=42,
    )

    assert len(sampled) == 2
