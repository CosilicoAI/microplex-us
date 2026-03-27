"""Tests for CPS source-provider implementations."""

import zipfile

import pandas as pd
import polars as pl
from microplex.core import EntityType, SourceArchetype, SourceProvider, SourceQuery

from microplex_us.data_sources import CPSASECParquetSourceProvider
from microplex_us.data_sources.cps import load_cps_asec


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
        }
    )
    household_rows = pd.DataFrame(
        {
            "H_SEQ": [1, 2],
            "GESTFIPS": [6, 36],
            "HSUP_WGT": [100, 200],
        }
    )
    with zipfile.ZipFile(tmp_path / "cps_asec_2023.zip", "w") as archive:
        archive.writestr("pppub23.csv", person_rows.to_csv(index=False))
        archive.writestr("hhpub23.csv", household_rows.to_csv(index=False))

    first = load_cps_asec(year=2023, cache_dir=tmp_path, download=False)
    cached_persons = pl.read_parquet(tmp_path / "cps_asec_2023_processed.parquet")
    second = load_cps_asec(year=2023, cache_dir=tmp_path, download=False)

    assert "state_fips" in first.persons.columns
    assert cached_persons["state_fips"].to_list() == [6, 6, 36]
    assert second.source.endswith("cps_asec_2023_processed.parquet")
    assert sorted(second.households["state_fips"].to_list()) == [6, 36]
