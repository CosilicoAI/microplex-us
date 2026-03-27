"""Tests for the canonical-target calibration harness."""

import numpy as np
import pandas as pd
from microplex.core import EntityType
from microplex.targets import TargetAggregation, TargetFilter, TargetSpec

from microplex_us.calibration_harness import CalibrationHarness
from microplex_us.target_registry import (
    TargetCategory,
    TargetGroup,
    TargetRegistry,
)


def _make_registry() -> TargetRegistry:
    geography_target = TargetSpec(
        name="ca_people",
        entity=EntityType.PERSON,
        value=2.0,
        period=2024,
        aggregation=TargetAggregation.COUNT,
        filters=(TargetFilter(feature="state_fips", operator="==", value="06"),),
        metadata={
            "us_category": "geography",
            "us_level": "state",
            "us_group": "people",
            "available_in_cps": True,
            "requires_imputation": False,
        },
    )
    income_target = TargetSpec(
        name="ca_income",
        entity=EntityType.PERSON,
        value=30.0,
        period=2024,
        measure="employment_income",
        aggregation=TargetAggregation.SUM,
        filters=(TargetFilter(feature="state_fips", operator="==", value="06"),),
        metadata={
            "us_category": "income",
            "us_level": "state",
            "us_group": "people",
            "available_in_cps": True,
            "requires_imputation": False,
        },
    )
    return TargetRegistry(
        groups={
            "people": TargetGroup(
                name="people",
                category=TargetCategory.GEOGRAPHY,
                targets=[geography_target, income_target],
            ),
        },
        build_defaults=False,
    )


class TestCalibrationHarness:
    def test_get_target_vector_uses_canonical_target_spec(self):
        harness = CalibrationHarness(registry=_make_registry())
        df = pd.DataFrame(
            {
                "state_fips": ["06", "06", "08"],
                "employment_income": [10.0, 20.0, 5.0],
                "weight": [1.0, 1.0, 1.0],
            }
        )
        targets = harness.registry.get_all_targets()

        design_matrix, target_vector, target_names = harness.get_target_vector(
            df,
            targets,
            entity=EntityType.PERSON,
        )

        np.testing.assert_allclose(design_matrix[:, 0], np.array([1.0, 1.0, 0.0]))
        np.testing.assert_allclose(design_matrix[:, 1], np.array([10.0, 20.0, 0.0]))
        np.testing.assert_allclose(target_vector, np.array([2.0, 30.0]))
        assert target_names == ["ca_people", "ca_income"]

    def test_run_experiment_filters_to_selected_canonical_targets(self):
        harness = CalibrationHarness(registry=_make_registry())
        df = pd.DataFrame(
            {
                "state_fips": ["06", "06", "08"],
                "employment_income": [10.0, 20.0, 5.0],
                "weight": [1.0, 1.0, 1.0],
            }
        )

        result = harness.run_experiment(
            df,
            "people_only",
            groups=["people"],
            only_available=True,
            entity=EntityType.PERSON,
            verbose=False,
        )

        assert result.targets_used == ["ca_people", "ca_income"]
        np.testing.assert_allclose(result.weights, np.ones(3))
