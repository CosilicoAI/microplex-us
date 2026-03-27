"""Tests for PSID source-provider implementation."""

from __future__ import annotations

import pandas as pd
from microplex.core import EntityType, SourceProvider, SourceQuery

from microplex_us.data_sources import PSIDDataset, PSIDSourceProvider


def test_psid_source_provider_projects_single_year_frame(tmp_path):
    def loader(**_: object) -> PSIDDataset:
        persons = pd.DataFrame(
            {
                "person_id": ["a", "b", "a", "b"],
                "household_id": ["h1", "h1", "h1", "h1"],
                "year": [2019, 2019, 2021, 2021],
                "age": [30, 28, 32, 30],
                "is_male": [True, False, True, False],
                "education": [4, 4, 4, 4],
                "total_income": [50_000.0, 10_000.0, 55_000.0, 12_000.0],
            }
        )
        return PSIDDataset(persons=persons, source="mock")

    provider = PSIDSourceProvider(
        data_dir=tmp_path,
        survey_year=2021,
        loader=loader,
    )
    frame = provider.load_frame(
        SourceQuery(period=2021, provider_filters={"sample_n": 1, "random_seed": 0})
    )

    assert isinstance(provider, SourceProvider)
    assert set(frame.tables) == {EntityType.HOUSEHOLD, EntityType.PERSON}
    assert frame.tables[EntityType.HOUSEHOLD]["year"].tolist() == [2021]
    assert frame.tables[EntityType.PERSON]["year"].nunique() == 1
    assert frame.tables[EntityType.PERSON]["person_id"].str.startswith("2021:").all()
    assert "income" in frame.tables[EntityType.PERSON].columns
    assert provider.descriptor.name.startswith("psid_")
