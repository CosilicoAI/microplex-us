"""US-specific block-assignment tests for the core hierarchical synthesizer."""

from __future__ import annotations

import pandas as pd
import pytest
from microplex.geography import (
    AtomicGeographyCrosswalk,
    GeographyAssignmentPlan,
    StaticGeographyProvider,
)
from microplex.hierarchical import HierarchicalSynthesizer

from microplex_us.geography import derive_geographies


class TestBlockAssignment:
    """Tests for US-style block-level geographic assignment."""

    @pytest.fixture
    def sample_block_probs(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "geoid": [
                    "060010201001000",
                    "060010201001001",
                    "060010201001002",
                    "360590101001000",
                    "360590101001001",
                    "480010101001000",
                    "480010101001001",
                    "480010101001002",
                ],
                "state_fips": ["06", "06", "06", "36", "36", "48", "48", "48"],
                "county": ["001", "001", "001", "059", "059", "001", "001", "001"],
                "tract": ["020100", "020100", "020100", "010100", "010100", "010100", "010100", "010100"],
                "block": ["1000", "1001", "1002", "1000", "1001", "1000", "1001", "1002"],
                "population": [100, 200, 100, 300, 200, 150, 250, 100],
                "tract_geoid": [
                    "06001020100",
                    "06001020100",
                    "06001020100",
                    "36059010100",
                    "36059010100",
                    "48001010100",
                    "48001010100",
                    "48001010100",
                ],
                "cd_id": ["CA-01", "CA-01", "CA-01", "NY-01", "NY-01", "TX-01", "TX-01", "TX-01"],
                "prob": [0.25, 0.50, 0.25, 0.6, 0.4, 0.3, 0.5, 0.2],
            }
        )

    @pytest.fixture
    def sample_cd_probs(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "state_fips": [6, 6, 36, 36, 48, 48],
                "cd_id": ["CA-01", "CA-02", "NY-01", "NY-02", "TX-01", "TX-02"],
                "prob": [0.6, 0.4, 0.5, 0.5, 0.7, 0.3],
            }
        )

    def test_init_with_block_probabilities(self, sample_block_probs: pd.DataFrame) -> None:
        synthesizer = HierarchicalSynthesizer(block_probabilities=sample_block_probs)

        assert synthesizer.geography_assignment is not None
        assert synthesizer.geography_assignment.atomic_id_column == "block_geoid"
        assert synthesizer._geography_assigner is not None

    def test_init_with_cd_probabilities_backward_compat(
        self,
        sample_cd_probs: pd.DataFrame,
    ) -> None:
        synthesizer = HierarchicalSynthesizer(cd_probabilities=sample_cd_probs)

        assert synthesizer.geography_assignment is not None
        assert synthesizer.geography_assignment.atomic_id_column == "cd_id"
        assert synthesizer._geography_assigner is not None

    def test_cd_probabilities_allow_state_local_district_ids(self) -> None:
        cd_probs = pd.DataFrame(
            {
                "state_fips": [6, 6, 36, 36],
                "cd_id": [1, 2, 1, 2],
                "prob": [0.6, 0.4, 0.5, 0.5],
            }
        )
        households = pd.DataFrame({"state_fips": [6, 36]})
        synthesizer = HierarchicalSynthesizer(
            cd_probabilities=cd_probs,
            random_state=123,
        )

        result = synthesizer._apply_geography_assignment(households)

        assert "_microplex_cd_atomic_id" not in result.columns
        assert result["state_fips"].tolist() == [6, 36]
        assert result["cd_id"].isin([1, 2]).all()

    def test_block_probabilities_take_precedence(
        self,
        sample_block_probs: pd.DataFrame,
        sample_cd_probs: pd.DataFrame,
    ) -> None:
        synthesizer = HierarchicalSynthesizer(
            cd_probabilities=sample_cd_probs,
            block_probabilities=sample_block_probs,
        )

        assert synthesizer.geography_assignment is not None
        assert synthesizer.geography_assignment.atomic_id_column == "block_geoid"

    def test_init_with_geography_provider(self, sample_block_probs: pd.DataFrame) -> None:
        crosswalk = AtomicGeographyCrosswalk(
            data=sample_block_probs.rename(columns={"geoid": "block_geoid"}),
            atomic_id_column="block_geoid",
            geography_columns=tuple(
                column
                for column in ("state_fips", "cd_id", "tract_geoid")
                if column in sample_block_probs.columns
            ),
            probability_column="prob",
        )
        provider = StaticGeographyProvider(
            crosswalk=crosswalk,
            default_partition_columns=("state_fips",),
        )
        plan = GeographyAssignmentPlan(
            partition_columns=("state_fips",),
            atomic_id_column="block_geoid",
        )

        synthesizer = HierarchicalSynthesizer(
            geography_provider=provider,
            geography_assignment=plan,
        )

        assert synthesizer.geography_assignment == plan
        assert synthesizer._geography_assigner is not None

    def test_assign_blocks_adds_block_geoid_only(self, sample_block_probs: pd.DataFrame) -> None:
        synthesizer = HierarchicalSynthesizer(
            block_probabilities=sample_block_probs,
            random_state=42,
        )
        households = pd.DataFrame({"state_fips": [6, 36, 48], "n_persons": [3, 2, 4]})

        result = synthesizer._apply_geography_assignment(households)

        assert "block_geoid" in result.columns
        assert "tract_geoid" not in result.columns
        assert "county_fips" not in result.columns
        assert "cd_id" not in result.columns

    def test_block_geoid_structure(self, sample_block_probs: pd.DataFrame) -> None:
        synthesizer = HierarchicalSynthesizer(
            block_probabilities=sample_block_probs,
            random_state=42,
        )
        households = pd.DataFrame({"state_fips": [6, 36, 48], "n_persons": [3, 2, 4]})

        result = synthesizer._apply_geography_assignment(households)

        for _, row in result.iterrows():
            block_geoid = row["block_geoid"]
            assert len(block_geoid) == 15
            assert len(block_geoid[:11]) == 11
            assert len(block_geoid[:5]) == 5

    def test_derive_geographies_from_block(self, sample_block_probs: pd.DataFrame) -> None:
        synthesizer = HierarchicalSynthesizer(
            block_probabilities=sample_block_probs,
            random_state=42,
        )
        households = pd.DataFrame({"state_fips": [6, 36, 48], "n_persons": [3, 2, 4]})

        result = synthesizer._apply_geography_assignment(households)
        geographies = derive_geographies(
            result["block_geoid"],
            include_cd=True,
            block_data=sample_block_probs,
        )

        california_mask = geographies["state_fips"] == "06"
        assert all(cd.startswith("CA-") for cd in geographies.loc[california_mask, "cd_id"])

        new_york_mask = geographies["state_fips"] == "36"
        assert all(cd.startswith("NY-") for cd in geographies.loc[new_york_mask, "cd_id"])

    def test_state_fips_fixed_to_valid(self, sample_block_probs: pd.DataFrame) -> None:
        synthesizer = HierarchicalSynthesizer(
            block_probabilities=sample_block_probs,
            random_state=42,
        )
        households = pd.DataFrame({"state_fips": [6.3, 36.7, 47.9], "n_persons": [3, 2, 4]})

        result = synthesizer._apply_geography_assignment(households)

        assert result["state_fips"].tolist() == ["06", "36", "48"]
