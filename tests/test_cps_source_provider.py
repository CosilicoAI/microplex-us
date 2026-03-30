"""Tests for CPS source-provider implementations."""

import zipfile

import pandas as pd
import polars as pl
from microplex.core import EntityType, SourceArchetype, SourceProvider, SourceQuery

from microplex_us.data_sources import CPSASECParquetSourceProvider
from microplex_us.data_sources.cps import (
    CPSASECSourceProvider,
    load_cps_asec,
    processed_cps_asec_cache_path,
)


def test_cps_parquet_source_provider_loads_observation_frame(tmp_path):
    households = pd.DataFrame(
        {
            "household_id": [1, 2],
            "state_fips": [6, 36],
            "household_weight": [1.0, 2.0],
        }
    )
    persons = pd.DataFrame(
        {
            "household_id": [1, 1, 2],
            "person_number": [1, 2, 1],
            "age": [34, 12, 52],
            "weight": [1.0, 1.0, 2.0],
        }
    )
    households.to_parquet(tmp_path / "cps_asec_households.parquet", index=False)
    persons.to_parquet(tmp_path / "cps_asec_persons.parquet", index=False)

    provider = CPSASECParquetSourceProvider(data_dir=tmp_path, year=2024)
    frame = provider.load_frame(SourceQuery(period=2024))

    assert isinstance(provider, SourceProvider)
    assert set(frame.tables) == {EntityType.HOUSEHOLD, EntityType.PERSON}
    assert frame.tables[EntityType.PERSON]["person_id"].tolist() == ["1:1", "1:2", "2:1"]
    assert frame.tables[EntityType.HOUSEHOLD]["year"].tolist() == [2024, 2024]
    assert frame.source.archetype is SourceArchetype.HOUSEHOLD_INCOME


def test_cps_parquet_source_provider_supports_household_sampling(tmp_path):
    households = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "state_fips": [6, 36, 48],
            "household_weight": [1.0, 2.0, 3.0],
        }
    )
    persons = pd.DataFrame(
        {
            "household_id": [1, 1, 2, 3],
            "person_number": [1, 2, 1, 1],
            "age": [34, 12, 52, 40],
            "weight": [1.0, 1.0, 2.0, 3.0],
        }
    )
    households.to_parquet(tmp_path / "cps_asec_households.parquet", index=False)
    persons.to_parquet(tmp_path / "cps_asec_persons.parquet", index=False)

    provider = CPSASECParquetSourceProvider(data_dir=tmp_path, year=2024)
    frame = provider.load_frame(
        SourceQuery(
            period=2024,
            provider_filters={"sample_n": 2, "random_seed": 0},
        )
    )

    assert len(frame.tables[EntityType.HOUSEHOLD]) == 2
    assert frame.tables[EntityType.PERSON]["household_id"].nunique() == 2


def test_cps_parquet_source_provider_sampling_respects_household_weights(tmp_path):
    households = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "state_fips": [6, 36, 48],
            "household_weight": [0.0, 0.0, 100.0],
        }
    )
    persons = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "person_number": [1, 1, 1],
            "age": [34, 52, 40],
            "weight": [0.0, 0.0, 100.0],
        }
    )
    households.to_parquet(tmp_path / "cps_asec_households.parquet", index=False)
    persons.to_parquet(tmp_path / "cps_asec_persons.parquet", index=False)

    provider = CPSASECParquetSourceProvider(data_dir=tmp_path, year=2024)
    frame = provider.load_frame(
        SourceQuery(
            period=2024,
            provider_filters={"sample_n": 1, "random_seed": 0},
        )
    )

    assert frame.tables[EntityType.HOUSEHOLD]["household_id"].tolist() == [3]
    assert frame.tables[EntityType.PERSON]["household_id"].tolist() == [3]


def test_cps_parquet_source_provider_applies_generic_atomic_variable_semantics(tmp_path):
    households = pd.DataFrame(
        {
            "household_id": [1],
            "state_fips": [6],
            "household_weight": [1.0],
        }
    )
    persons = pd.DataFrame(
        {
            "household_id": [1],
            "person_number": [1],
            "age": [34],
            "weight": [1.0],
            "qualified_dividend_income": [30.0],
            "non_qualified_dividend_income": [12.0],
            "dividend_income": [42.0],
            "ordinary_dividend_income": [42.0],
        }
    )
    households.to_parquet(tmp_path / "cps_asec_households.parquet", index=False)
    persons.to_parquet(tmp_path / "cps_asec_persons.parquet", index=False)

    provider = CPSASECParquetSourceProvider(data_dir=tmp_path, year=2024)
    frame = provider.load_frame(SourceQuery(period=2024))
    descriptor = frame.source

    assert not descriptor.is_authoritative_for("dividend_income")
    assert not descriptor.allows_conditioning_on("dividend_income")
    assert not descriptor.is_authoritative_for("ordinary_dividend_income")
    assert descriptor.is_authoritative_for("qualified_dividend_income")
    assert descriptor.allows_conditioning_on("qualified_dividend_income")


def test_load_cps_asec_rebuilds_stale_processed_cache_without_state_fips(tmp_path):
    stale_processed = pl.DataFrame(
        {
            "household_id": [1, 1, 2],
            "person_number": [1, 2, 1],
            "age": [34, 12, 52],
            "weight": [1.0, 1.0, 2.0],
            "year": [2023, 2023, 2023],
        }
    )
    stale_processed.write_parquet(tmp_path / "cps_asec_2023_processed.parquet")

    person_rows = pd.DataFrame(
        {
            "PH_SEQ": [1, 1, 2],
            "GESTFIPS": [6, 6, 36],
            "A_LINENO": [1, 2, 1],
            "A_AGE": [34, 12, 52],
            "A_FNLWGT": [100, 100, 200],
        }
    )
    with zipfile.ZipFile(tmp_path / "cps_asec_2023.zip", "w") as archive:
        archive.writestr("pppub23.csv", person_rows.to_csv(index=False))

    dataset = load_cps_asec(year=2023, cache_dir=tmp_path, download=False)

    assert "state_fips" in dataset.persons.columns
    assert sorted(dataset.households["state_fips"].to_list()) == [6, 36]


def test_load_cps_asec_caches_household_geography_on_persons(tmp_path):
    person_rows = pd.DataFrame(
        {
            "PH_SEQ": [1, 1, 2],
            "A_LINENO": [1, 2, 1],
            "A_AGE": [34, 12, 52],
            "A_FNLWGT": [100, 100, 200],
            "PRDTRACE": [4, 4, 1],
            "PRDTHSP": [0, 1, 0],
            "PEHSPNON": [2, 1, 2],
            "PEDISDRS": [0, 1, 0],
            "PEDISEAR": [0, 0, 0],
            "PEDISEYE": [0, 0, 0],
            "PEDISOUT": [0, 0, 0],
            "PEDISPHY": [0, 0, 0],
            "PEDISREM": [0, 0, 0],
            "NOW_MRK": [1, 0, 0],
            "NOW_GRP": [0, 1, 0],
        }
    )
    household_rows = pd.DataFrame(
        {
            "H_SEQ": [1, 2],
            "GESTFIPS": [6, 36],
            "GTCO": [1, 61],
            "HSUP_WGT": [100, 200],
        }
    )
    with zipfile.ZipFile(tmp_path / "cps_asec_2023.zip", "w") as archive:
        archive.writestr("pppub23.csv", person_rows.to_csv(index=False))
        archive.writestr("hhpub23.csv", household_rows.to_csv(index=False))

    first = load_cps_asec(year=2023, cache_dir=tmp_path, download=False)
    cached_persons = pl.read_parquet(
        processed_cps_asec_cache_path(year=2023, cache_dir=tmp_path)
    )
    second = load_cps_asec(year=2023, cache_dir=tmp_path, download=False)

    assert "state_fips" in first.persons.columns
    assert "county_fips" in first.persons.columns
    assert "cps_race" in first.persons.columns
    assert "is_hispanic" in first.persons.columns
    assert "is_disabled" in first.persons.columns
    assert "has_marketplace_health_coverage" in first.persons.columns
    assert "has_esi" in first.persons.columns
    assert cached_persons["state_fips"].to_list() == [6, 6, 36]
    assert cached_persons["county_fips"].to_list() == [1, 1, 61]
    assert cached_persons["cps_race"].to_list() == [4, 4, 1]
    assert cached_persons["is_hispanic"].to_list() == [False, True, False]
    assert cached_persons["is_disabled"].to_list() == [False, True, False]
    assert cached_persons["has_marketplace_health_coverage"].to_list() == [True, False, False]
    assert cached_persons["has_esi"].to_list() == [False, True, False]
    assert second.source.endswith("cps_asec_2023_processed_v20260330.parquet")
    assert sorted(second.households["state_fips"].to_list()) == [6, 36]
    assert sorted(second.households["county_fips"].to_list()) == [1, 61]


def test_load_cps_asec_derives_policyengine_value_inputs(tmp_path):
    person_rows = pd.DataFrame(
        {
            "PH_SEQ": [1, 1],
            "A_LINENO": [1, 2],
            "A_AGE": [34, 62],
            "A_FNLWGT": [100, 100],
            "OI_OFF": [20, 12],
            "OI_VAL": [1200, 800],
            "CSP_VAL": [300, -1],
            "DIS_VAL1": [500, 400],
            "DIS_SC1": [2, 1],
            "DIS_VAL2": [50, 25],
            "DIS_SC2": [3, 2],
            "PHIP_VAL": [900, -1],
            "POTC_VAL": [120, -1],
            "PMED_VAL": [450, -1],
            "PEMCPREM": [600, -1],
        }
    )
    with zipfile.ZipFile(tmp_path / "cps_asec_2023.zip", "w") as archive:
        archive.writestr("pppub23.csv", person_rows.to_csv(index=False))

    dataset = load_cps_asec(year=2023, cache_dir=tmp_path, download=False)
    persons = dataset.persons.to_pandas().sort_values("person_number").reset_index(drop=True)

    assert persons["alimony_income"].tolist() == [1200, 0]
    assert persons["child_support_received"].tolist() == [300, 0]
    assert persons["disability_benefits"].tolist() == [550, 25]
    assert persons["health_insurance_premiums_without_medicare_part_b"].tolist() == [900, 0]
    assert persons["over_the_counter_health_expenses"].tolist() == [120, 0]
    assert persons["other_medical_expenses"].tolist() == [450, 0]
    assert persons["medicare_part_b_premiums"].tolist() == [600, 0]


def test_cps_source_provider_repeat_loads_are_deterministic_for_cached_processed_data(
    tmp_path,
):
    cached_persons = pl.DataFrame(
        {
            "household_id": [2, 1, 2, 3, 1],
            "person_number": [1, 2, 2, 1, 1],
            "person_id": ["2:1", "1:2", "2:2", "3:1", "1:1"],
            "age": [52, 12, 49, 40, 34],
            "weight": [200.0, 100.0, 200.0, 300.0, 100.0],
            "state_fips": [36, 6, 36, 48, 6],
            "county_fips": [61, 1, 61, 201, 1],
            "cps_race": [1, 4, 1, 2, 4],
            "is_hispanic": [False, True, False, False, True],
            "is_disabled": [False, False, False, True, False],
            "has_esi": [True, False, True, False, False],
            "has_marketplace_health_coverage": [False, True, False, False, True],
            "alimony_income": [0.0, 0.0, 0.0, 0.0, 0.0],
            "child_support_received": [0.0, 0.0, 0.0, 0.0, 0.0],
            "disability_benefits": [0.0, 0.0, 0.0, 0.0, 0.0],
            "health_insurance_premiums_without_medicare_part_b": [0.0] * 5,
            "other_medical_expenses": [0.0] * 5,
            "over_the_counter_health_expenses": [0.0] * 5,
            "medicare_part_b_premiums": [0.0] * 5,
            "year": [2023, 2023, 2023, 2023, 2023],
        }
    )
    cached_persons.write_parquet(
        processed_cps_asec_cache_path(year=2023, cache_dir=tmp_path)
    )

    provider = CPSASECSourceProvider(year=2023, cache_dir=tmp_path, download=False)
    query = SourceQuery(provider_filters={"sample_n": 2, "random_seed": 42})

    first = provider.load_frame(query)
    second = provider.load_frame(query)

    first_households = first.tables[EntityType.HOUSEHOLD]
    second_households = second.tables[EntityType.HOUSEHOLD]
    first_persons = first.tables[EntityType.PERSON]
    second_persons = second.tables[EntityType.PERSON]

    assert first_households["household_id"].tolist() == second_households["household_id"].tolist()
    assert first_persons["person_id"].tolist() == second_persons["person_id"].tolist()
    assert first_households["household_weight"].tolist() == second_households["household_weight"].tolist()
    assert first_persons["weight"].tolist() == second_persons["weight"].tolist()


def test_load_cps_asec_rebuilds_stale_processed_cache_without_pe_presim_inputs(tmp_path):
    stale_processed = pl.DataFrame(
        {
            "household_id": [1, 1, 2],
            "person_number": [1, 2, 1],
            "age": [34, 12, 52],
            "weight": [1.0, 1.0, 2.0],
            "state_fips": [6, 6, 36],
            "year": [2023, 2023, 2023],
        }
    )
    stale_processed.write_parquet(tmp_path / "cps_asec_2023_processed.parquet")

    person_rows = pd.DataFrame(
        {
            "PH_SEQ": [1, 1, 2],
            "GESTFIPS": [6, 6, 36],
            "A_LINENO": [1, 2, 1],
            "A_AGE": [34, 12, 52],
            "A_FNLWGT": [100, 100, 200],
            "PRDTRACE": [4, 4, 1],
            "PRDTHSP": [0, 1, 0],
            "PEHSPNON": [2, 1, 2],
            "PEDISDRS": [0, 1, 0],
            "PEDISEAR": [0, 0, 0],
            "PEDISEYE": [0, 0, 0],
            "PEDISOUT": [0, 0, 0],
            "PEDISPHY": [0, 0, 0],
            "PEDISREM": [0, 0, 0],
            "NOW_MRK": [1, 0, 0],
            "NOW_GRP": [0, 1, 0],
            "OI_OFF": [20, 0, 0],
            "OI_VAL": [1200, 0, 0],
            "CSP_VAL": [300, 0, 0],
            "DIS_VAL1": [500, 0, 0],
            "DIS_SC1": [2, 0, 0],
            "DIS_VAL2": [50, 0, 0],
            "DIS_SC2": [3, 0, 0],
            "PHIP_VAL": [900, 0, 0],
            "POTC_VAL": [120, 0, 0],
            "PMED_VAL": [450, 0, 0],
            "PEMCPREM": [600, 0, 0],
        }
    )
    household_rows = pd.DataFrame(
        {
            "H_SEQ": [1, 2],
            "GESTFIPS": [6, 36],
            "GTCO": [1, 61],
            "HSUP_WGT": [100, 200],
        }
    )
    with zipfile.ZipFile(tmp_path / "cps_asec_2023.zip", "w") as archive:
        archive.writestr("pppub23.csv", person_rows.to_csv(index=False))
        archive.writestr("hhpub23.csv", household_rows.to_csv(index=False))

    dataset = load_cps_asec(year=2023, cache_dir=tmp_path, download=False)

    assert dataset.source.endswith("cps_asec_2023.zip")
    assert dataset.persons["county_fips"].to_list() == [1, 1, 61]
    assert dataset.persons["cps_race"].to_list() == [4, 4, 1]
    assert dataset.persons["is_hispanic"].to_list() == [False, True, False]
    assert dataset.persons["is_disabled"].to_list() == [False, True, False]
    assert dataset.persons["has_marketplace_health_coverage"].to_list() == [True, False, False]
    assert dataset.persons["has_esi"].to_list() == [False, True, False]
    assert dataset.persons["alimony_income"].to_list() == [1200, 0, 0]
    assert dataset.persons["child_support_received"].to_list() == [300, 0, 0]
    assert dataset.persons["disability_benefits"].to_list() == [550, 0, 0]
    assert (
        dataset.persons["health_insurance_premiums_without_medicare_part_b"].to_list()
        == [900, 0, 0]
    )
    assert dataset.persons["other_medical_expenses"].to_list() == [450, 0, 0]
    assert dataset.persons["over_the_counter_health_expenses"].to_list() == [120, 0, 0]
    assert dataset.persons["medicare_part_b_premiums"].to_list() == [600, 0, 0]
