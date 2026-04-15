from __future__ import annotations

import pandas as pd
from microplex.core import EntityType
from microplex.targets import FilterOperator, TargetFilter, TargetSpec

from microplex_us.pipelines import reweight_us_household_targets
from microplex_us.policyengine.us import PolicyEngineUSEntityTableBundle


def test_reweight_us_household_targets_updates_household_and_person_weights():
    tables = PolicyEngineUSEntityTableBundle(
        households=pd.DataFrame(
            {
                "household_id": [10, 20],
                "household_weight": [1.0, 1.0],
            }
        ),
        persons=pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 20],
                "weight": [1.0, 1.0, 1.0],
                "age": [5, 8, 30],
                "state_fips": ["06", "06", "36"],
            }
        ),
    )
    targets = [
        TargetSpec(
            name="state06_age0_10",
            entity=EntityType.PERSON,
            value=4.0,
            period=2024,
            aggregation="count",
            filters=(
                TargetFilter("state_fips", FilterOperator.EQ, "06"),
                TargetFilter("age", FilterOperator.GTE, 0),
                TargetFilter("age", FilterOperator.LT, 10),
            ),
        )
    ]

    result = reweight_us_household_targets(
        tables,
        targets=targets,
    )

    assert result.tables.households["household_weight"].tolist() == [2.0, 1.0]
    assert result.tables.persons["weight"].tolist() == [2.0, 2.0, 1.0]
    assert result.diagnostics.constraint_count == 1
    assert result.compilation.skipped_targets == ()
