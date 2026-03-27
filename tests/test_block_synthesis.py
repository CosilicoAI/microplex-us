"""US-specific block-geography integration around the core hierarchical synthesizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from microplex.hierarchical import HierarchicalSynthesizer, HouseholdSchema

from microplex_us.geography import BlockGeography, derive_geographies


@pytest.fixture
def block_probabilities() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[2] / "microplex" / "data" / "block_probabilities.parquet"
    if not data_path.exists():
        pytest.skip("Block probabilities data not available")
    return pd.read_parquet(data_path)


@pytest.fixture
def sample_cps_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n_households = 100

    households = pd.DataFrame(
        {
            "household_id": range(n_households),
            "n_persons": rng.choice([1, 2, 3, 4, 5], size=n_households, p=[0.3, 0.3, 0.2, 0.15, 0.05]),
            "state_fips": rng.choice([6, 48, 12, 36], size=n_households),
            "tenure": rng.choice([1, 2], size=n_households),
            "hh_weight": rng.uniform(100, 1000, size=n_households),
        }
    )
    households["n_adults"] = np.clip(
        households["n_persons"] - rng.integers(0, 2, size=n_households),
        1,
        None,
    )
    households["n_children"] = households["n_persons"] - households["n_adults"]

    people: list[dict[str, float | int]] = []
    for _, household in households.iterrows():
        for person_idx in range(int(household["n_persons"])):
            is_adult = person_idx < household["n_adults"]
            people.append(
                {
                    "household_id": household["household_id"],
                    "person_id": len(people),
                    "age": int(rng.integers(25, 65)) if is_adult else int(rng.integers(0, 18)),
                    "sex": int(rng.choice([1, 2])),
                    "income": float(rng.uniform(0, 100000)) if is_adult else 0.0,
                    "employment_status": int(rng.choice([1, 2, 3])) if is_adult else 0,
                    "education": int(rng.choice([1, 2, 3, 4])) if is_adult else 0,
                    "relationship_to_head": 0 if person_idx == 0 else int(rng.choice([1, 2, 3])),
                }
            )

    return households, pd.DataFrame(people)


def _schema() -> HouseholdSchema:
    return HouseholdSchema(
        hh_vars=["n_persons", "n_adults", "n_children", "state_fips", "tenure"],
        person_vars=["age", "sex", "income", "employment_status", "education", "relationship_to_head"],
    )


def test_generate_includes_block_geoid(sample_cps_data, block_probabilities: pd.DataFrame) -> None:
    households, persons = sample_cps_data
    synthesizer = HierarchicalSynthesizer(
        schema=_schema(),
        block_probabilities=block_probabilities,
        random_state=42,
    )

    synthesizer.fit(households, persons, hh_weight_col="hh_weight", epochs=5, verbose=False)
    synthetic_households, _ = synthesizer.generate(n_households=50, verbose=False)

    assert "block_geoid" in synthetic_households.columns
    assert all(synthetic_households["block_geoid"].str.len() == 15)


def test_derive_geographies_post_hoc(sample_cps_data, block_probabilities: pd.DataFrame) -> None:
    households, persons = sample_cps_data
    synthesizer = HierarchicalSynthesizer(
        schema=_schema(),
        block_probabilities=block_probabilities,
        random_state=42,
    )

    synthesizer.fit(households, persons, hh_weight_col="hh_weight", epochs=5, verbose=False)
    synthetic_households, _ = synthesizer.generate(n_households=50, verbose=False)

    geographies = derive_geographies(
        synthetic_households["block_geoid"],
        include_cd=True,
        include_sld=True,
        block_data=block_probabilities,
    )

    assert set(["tract_geoid", "county_fips", "cd_id", "sldu_id", "sldl_id"]).issubset(
        geographies.columns
    )


def test_block_geography_integration_uses_real_block_table(
    block_probabilities: pd.DataFrame,
) -> None:
    geography = BlockGeography(lazy_load=True)
    geography._data = block_probabilities

    sample_block = block_probabilities["geoid"].iloc[0]
    geos = geography.get_all_geographies(sample_block)

    assert geos["county_fips"][:2] == geos["state_fips"]
    assert geos["tract_geoid"].startswith(geos["county_fips"])
