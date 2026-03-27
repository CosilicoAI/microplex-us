"""US-specific block-geography tests for microplex-us."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from microplex.geography import GeographyProvider

from microplex_us.geography import (
    BLOCK_GEOID_LEN,
    BLOCK_LEN,
    COUNTY_GEOID_LEN,
    COUNTY_LEN,
    DEFAULT_BLOCK_PROBABILITIES_PATH,
    STATE_GEOID_LEN,
    STATE_LEN,
    TRACT_GEOID_LEN,
    TRACT_LEN,
    BlockGeography,
    derive_geographies,
    load_block_probabilities,
)


def _sample_block_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "geoid": ["060010001001001", "060010001001002", "360610001001001"],
            "state_fips": ["06", "06", "36"],
            "county": ["001", "001", "061"],
            "tract": ["000100", "000100", "000100"],
            "tract_geoid": ["06001000100", "06001000100", "36061000100"],
            "cd_id": ["CA-13", "CA-13", "NY-12"],
            "prob": [0.6, 0.4, 1.0],
            "national_prob": [0.3, 0.2, 0.5],
        }
    )


class TestGEOIDConstants:
    def test_state_len(self) -> None:
        assert STATE_LEN == 2

    def test_county_len(self) -> None:
        assert COUNTY_LEN == 3

    def test_tract_len(self) -> None:
        assert TRACT_LEN == 6

    def test_block_len(self) -> None:
        assert BLOCK_LEN == 4

    def test_cumulative_lengths(self) -> None:
        assert STATE_GEOID_LEN == 2
        assert COUNTY_GEOID_LEN == 5
        assert TRACT_GEOID_LEN == 11
        assert BLOCK_GEOID_LEN == 15


class TestStaticGeographyExtraction:
    SAMPLE_BLOCK = "060372073021001"

    def test_get_state_from_block_geoid(self) -> None:
        assert BlockGeography.get_state(self.SAMPLE_BLOCK) == "06"

    def test_get_county_from_block_geoid(self) -> None:
        assert BlockGeography.get_county(self.SAMPLE_BLOCK) == "06037"

    def test_get_tract_from_block_geoid(self) -> None:
        assert BlockGeography.get_tract(self.SAMPLE_BLOCK) == "06037207302"

    def test_geoid_length_validation(self) -> None:
        assert len(BlockGeography.get_state(self.SAMPLE_BLOCK)) == STATE_GEOID_LEN
        assert len(BlockGeography.get_county(self.SAMPLE_BLOCK)) == COUNTY_GEOID_LEN
        assert len(BlockGeography.get_tract(self.SAMPLE_BLOCK)) == TRACT_GEOID_LEN

    def test_multiple_blocks_different_states(self) -> None:
        blocks = {
            "010010201001000": ("01", "01001", "01001020100"),
            "060372073021001": ("06", "06037", "06037207302"),
            "481131234001234": ("48", "48113", "48113123400"),
        }
        for block, (state, county, tract) in blocks.items():
            assert BlockGeography.get_state(block) == state
            assert BlockGeography.get_county(block) == county
            assert BlockGeography.get_tract(block) == tract


class TestLoadBlockProbabilities:
    @pytest.fixture
    def data_path(self) -> Path:
        return DEFAULT_BLOCK_PROBABILITIES_PATH

    def test_load_block_probabilities_reads_parquet(self, tmp_path: Path) -> None:
        sample = _sample_block_table()
        path = tmp_path / "block_probabilities.parquet"
        sample.to_parquet(path)

        loaded = load_block_probabilities(path)

        pd.testing.assert_frame_equal(loaded, sample)

    def test_load_default_path(self, data_path: Path) -> None:
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        df = load_block_probabilities()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_explicit_path(self, data_path: Path) -> None:
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        df = load_block_probabilities(data_path)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self, data_path: Path) -> None:
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        df = load_block_probabilities(data_path)
        required_cols = ["geoid", "state_fips", "population", "prob"]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_block_probabilities("/nonexistent/path/file.parquet")


class TestBlockGeographyProvider:
    def test_block_geography_implements_provider_interface(self) -> None:
        geo = BlockGeography.from_data(
            pd.DataFrame(
                {
                    "geoid": ["060010201001000", "360590101001000"],
                    "state_fips": ["06", "36"],
                    "county": ["001", "059"],
                    "tract": ["020100", "010100"],
                    "prob": [1.0, 1.0],
                }
            )
        )

        assert isinstance(geo, GeographyProvider)
        crosswalk = geo.load_crosswalk()
        assigner = geo.load_assigner()
        result = assigner.assign(pd.DataFrame({"state_fips": ["06", "36"]}), random_state=0)

        assert crosswalk.atomic_id_column == "block_geoid"
        assert result["block_geoid"].tolist() == [
            "060010201001000",
            "360590101001000",
        ]

    def test_block_geography_materializes_and_samples_from_in_memory_data(self) -> None:
        geography = BlockGeography.from_data(_sample_block_table())
        assigned = geography.assign(pd.DataFrame({"state_fips": ["06", "36"]}), random_state=1)
        materialized = geography.materialize(assigned)

        assert "block_geoid" in assigned.columns
        assert set(["state_fips", "county_fips", "tract_geoid", "cd_id"]).issubset(
            materialized.columns
        )
        assert set(materialized["state_fips"]) == {"06", "36"}

    def test_derive_geographies_uses_block_string_structure(self) -> None:
        result = derive_geographies(["060010001001001", "360610001001001"])

        assert list(result["state_fips"]) == ["06", "36"]
        assert list(result["county_fips"]) == ["06001", "36061"]


class TestBlockGeography:
    @pytest.fixture
    def data_path(self) -> Path:
        return DEFAULT_BLOCK_PROBABILITIES_PATH

    @pytest.fixture
    def geo(self, data_path: Path) -> BlockGeography:
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        return BlockGeography(data_path, lazy_load=False)

    def test_lazy_load_default(self, data_path: Path) -> None:
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        geo = BlockGeography(data_path)
        assert geo._data is None

    def test_eager_load(self, data_path: Path) -> None:
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        geo = BlockGeography(data_path, lazy_load=False)
        assert geo._data is not None

    def test_data_property_loads(self, data_path: Path) -> None:
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        geo = BlockGeography(data_path)
        assert geo._data is None
        _ = geo.data
        assert geo._data is not None

    def test_get_cd_requires_lookup(self, geo: BlockGeography) -> None:
        sample_block = geo.data["geoid"].iloc[0]
        expected_cd = geo.data[geo.data["geoid"] == sample_block]["cd_id"].iloc[0]
        assert geo.get_cd(sample_block) == expected_cd

    def test_get_cd_unknown_block(self, geo: BlockGeography) -> None:
        assert geo.get_cd("000000000000000") is None

    def test_get_all_geographies(self, geo: BlockGeography) -> None:
        sample_block = geo.data["geoid"].iloc[0]
        result = geo.get_all_geographies(sample_block)

        assert isinstance(result, dict)
        assert "state_fips" in result
        assert "county_fips" in result
        assert "tract_geoid" in result
        assert "cd_id" in result

    def test_states_property(self, geo: BlockGeography) -> None:
        states = geo.states
        assert isinstance(states, list)
        assert len(states) > 0
        assert states == sorted(states)

    def test_n_blocks_property(self, geo: BlockGeography) -> None:
        assert isinstance(geo.n_blocks, int)
        assert geo.n_blocks > 0
        assert geo.n_blocks == len(geo.data)

    def test_from_data_supports_in_memory_crosswalks(self) -> None:
        data = pd.DataFrame(
            {
                "geoid": ["060010201001000", "060010201001001"],
                "state_fips": ["06", "06"],
                "county": ["001", "001"],
                "tract_geoid": ["06001020100", "06001020100"],
                "cd_id": ["CA-01", "CA-01"],
                "prob": [0.4, 0.6],
            }
        )

        geo = BlockGeography.from_data(data)

        assert geo.n_blocks == 2
        assert geo.get_cd("060010201001000") == "CA-01"

    def test_assign_and_materialize_round_trip(self) -> None:
        data = pd.DataFrame(
            {
                "geoid": [
                    "060010201001000",
                    "060010201001001",
                    "360590101001000",
                ],
                "state_fips": ["06", "06", "36"],
                "county": ["001", "001", "059"],
                "tract_geoid": ["06001020100", "06001020100", "36059010100"],
                "cd_id": ["CA-01", "CA-02", "NY-01"],
                "prob": [0.25, 0.75, 1.0],
            }
        )
        geo = BlockGeography.from_data(data)
        households = pd.DataFrame({"state_fips": [6.0, 47.9]})

        assigned = geo.assign(households, random_state=42)
        materialized = geo.materialize(assigned, columns=("tract_geoid", "cd_id"))

        assert "block_geoid" in assigned.columns
        assert assigned["block_geoid"].iloc[0].startswith("06")
        assert assigned["block_geoid"].iloc[1].startswith("36")
        assert materialized["cd_id"].tolist() == ["CA-02", "NY-01"]
        assert materialized["tract_geoid"].tolist() == [
            "06001020100",
            "36059010100",
        ]


class TestBlockSampling:
    @pytest.fixture
    def geo(self) -> BlockGeography:
        if not DEFAULT_BLOCK_PROBABILITIES_PATH.exists():
            pytest.skip("Block probabilities data not available")
        return BlockGeography(DEFAULT_BLOCK_PROBABILITIES_PATH, lazy_load=False)

    def test_sample_blocks_returns_array(self, geo: BlockGeography) -> None:
        blocks = geo.sample_blocks("06", n=10, random_state=42)
        assert isinstance(blocks, np.ndarray)
        assert len(blocks) == 10

    def test_sample_blocks_from_correct_state(self, geo: BlockGeography) -> None:
        blocks = geo.sample_blocks("06", n=100, random_state=42)
        for block in blocks:
            assert BlockGeography.get_state(block) == "06"

    def test_sample_blocks_reproducible(self, geo: BlockGeography) -> None:
        blocks1 = geo.sample_blocks("06", n=10, random_state=42)
        blocks2 = geo.sample_blocks("06", n=10, random_state=42)
        np.testing.assert_array_equal(blocks1, blocks2)

    def test_sample_blocks_invalid_state(self, geo: BlockGeography) -> None:
        with pytest.raises(ValueError, match="not found"):
            geo.sample_blocks("99", n=10)

    def test_sample_blocks_national(self, geo: BlockGeography) -> None:
        blocks = geo.sample_blocks_national(n=100, random_state=42)
        assert len(blocks) == 100
        states = set(BlockGeography.get_state(block) for block in blocks)
        assert len(states) > 1

    def test_sample_blocks_weighted(self, geo: BlockGeography) -> None:
        blocks = geo.sample_blocks("06", n=10000, random_state=42)
        counts = pd.Series(blocks).value_counts(normalize=True)
        state_df = geo.get_blocks_in_state("06")
        expected = state_df.set_index("geoid")["prob"] / state_df["prob"].sum()

        for geoid, expected_prob in expected.items():
            assert abs(counts.get(geoid, 0.0) - expected_prob) < 0.05
