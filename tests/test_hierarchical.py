"""US-specific hierarchical preprocessing helpers."""

from __future__ import annotations

import pandas as pd

from microplex_us.hierarchical import prepare_cps_for_hierarchical


def test_prepare_cps_for_hierarchical_builds_household_summary() -> None:
    cps_data = pd.DataFrame(
        {
            "household_id": [1, 1, 1, 2, 2, 3],
            "age": [45, 42, 12, 67, 65, 35],
            "state_fips": [6, 6, 6, 36, 36, 48],
            "tenure": [1, 1, 1, 2, 2, 1],
            "hh_weight": [1000, 1000, 1000, 800, 800, 1200],
        }
    )

    households, persons = prepare_cps_for_hierarchical(cps_data)

    assert len(households) == 3
    assert households.loc[households["household_id"] == 1, "n_persons"].iloc[0] == 3
    assert households.loc[households["household_id"] == 1, "n_adults"].iloc[0] == 2
    assert households.loc[households["household_id"] == 1, "n_children"].iloc[0] == 1
    assert len(persons) == 6
