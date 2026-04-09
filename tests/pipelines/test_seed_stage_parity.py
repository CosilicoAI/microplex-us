"""Tests for seed/source-impute parity auditing."""

from __future__ import annotations

import json

import h5py
import pandas as pd
import pytest

from microplex_us.pipelines.seed_stage_parity import (
    SeedStageBooleanLandingFeatureSpec,
    SeedStageCategoricalLandingFeatureSpec,
    SeedStageFocusVariableSpec,
    build_us_seed_stage_parity_audit,
    write_us_seed_stage_parity_audit,
)


def _write_period_dataset(path, data: dict[str, list | tuple], *, period: int = 2024) -> None:
    with h5py.File(path, "w") as handle:
        for variable, values in data.items():
            group = handle.create_group(variable)
            group.create_dataset(str(period), data=values)


def test_build_us_seed_stage_parity_audit_projects_reference_and_profiles_positive_rows(
    tmp_path,
) -> None:
    seed_path = tmp_path / "seed.parquet"
    reference_path = tmp_path / "reference.h5"

    pd.DataFrame(
        {
            "person_id": ["1", "2", "3"],
            "household_id": ["10", "10", "20"],
            "tax_unit_id": ["100", "100", "200"],
            "hh_weight": [10.0, 10.0, 20.0],
            "self_employment_income": [0.0, 50.0, 100.0],
            "employment_income": [10.0, 20.0, 30.0],
            "has_esi": [False, True, True],
            "has_marketplace_health_coverage": [False, True, False],
            "age": [5, 42, 67],
            "state_fips": [1, 1, 2],
            "employment_status": ["not_working", "self_employed", "self_employed"],
        }
    ).to_parquet(seed_path, index=False)

    _write_period_dataset(
        reference_path,
        {
            "household_id": [10, 20],
            "household_weight": [20.0, 40.0],
            "person_id": [101, 102, 103],
            "person_household_id": [10, 10, 20],
            "tax_unit_id": [100, 200],
            "person_tax_unit_id": [100, 100, 200],
            "self_employment_income_before_lsr": [0.0, 50.0, 100.0],
            "employment_income_before_lsr": [10.0, 20.0, 30.0],
            "has_esi": [False, True, True],
            "has_marketplace_health_coverage": [False, True, False],
            "age": [5, 42, 67],
            "state_fips": [1, 2],
        },
    )

    audit = build_us_seed_stage_parity_audit(
        seed_path,
        reference_path,
        focus_variables=(
            SeedStageFocusVariableSpec(
                "self_employment_income",
                "self_employment_income",
                "self_employment_income_before_lsr",
                value_kind="numeric",
            ),
            "missing_metric",
        ),
        boolean_landing_features=(
            SeedStageBooleanLandingFeatureSpec("has_esi", "has_esi"),
            SeedStageBooleanLandingFeatureSpec(
                "has_marketplace_health_coverage",
                "has_marketplace_health_coverage",
            ),
        ),
        categorical_landing_features=(
            SeedStageCategoricalLandingFeatureSpec(
                "age_bin",
                "age",
                "age",
                transform="age_bin",
            ),
            SeedStageCategoricalLandingFeatureSpec(
                "state_fips",
                "state_fips",
                "state_fips",
            ),
        ),
        candidate_only_landing_features=(
            SeedStageCategoricalLandingFeatureSpec(
                "employment_status",
                "employment_status",
            ),
        ),
    )

    assert audit["comparisonStage"] == "seed_source_impute"
    assert audit["weightScale"]["reference_to_seed_weight_scale"] == pytest.approx(2.0)
    assert audit["seedStructure"]["tax_unit_id_count"] == 2
    assert audit["seedStructure"]["weighted_mean_rows_per_household"] == pytest.approx(
        4.0 / 3.0
    )
    assert audit["referenceStructure"]["weighted_mean_rows_per_household"] == pytest.approx(
        4.0 / 3.0
    )

    self_employment = audit["focusVariables"]["self_employment_income"]
    assert self_employment["comparison"]["type"] == "numeric"
    assert self_employment["comparison"]["weighted_sum_ratio"] == pytest.approx(0.5)
    assert self_employment["comparison"]["reference_scaled_weighted_sum_ratio"] == pytest.approx(
        1.0
    )
    assert self_employment["positiveSupport"]["seed_positive_weight_share"] == pytest.approx(
        0.75
    )

    marketplace = self_employment["positiveBooleanProfiles"][
        "has_marketplace_health_coverage"
    ]
    assert marketplace["seed_positive_share"] == pytest.approx(1.0 / 3.0)
    assert marketplace["reference_positive_share"] == pytest.approx(1.0 / 3.0)
    assert marketplace["share_delta"] == pytest.approx(0.0)

    age_bin = self_employment["positiveCategoricalProfiles"]["age_bin"]
    assert age_bin["seed"]["top_values"][0]["value"] == "65-69"
    assert age_bin["seed"]["top_values"][0]["weighted_share"] == pytest.approx(2.0 / 3.0)

    state_fips = self_employment["positiveCategoricalProfiles"]["state_fips"]
    assert state_fips["reference_present"] is True
    assert state_fips["reference"]["top_values"][0]["value"] == "2"

    employment_status = self_employment["positiveCandidateOnlyProfiles"][
        "employment_status"
    ]
    assert employment_status["seed"]["top_values"][0]["value"] == "self_employed"
    assert employment_status["seed"]["top_values"][0]["weighted_share"] == pytest.approx(1.0)

    missing = audit["focusVariables"]["missing_metric"]
    assert missing["seed_present"] is False
    assert missing["reference_present"] is False


def test_write_us_seed_stage_parity_audit_persists_json(tmp_path) -> None:
    seed_path = tmp_path / "seed.parquet"
    reference_path = tmp_path / "reference.h5"
    output_path = tmp_path / "audit.json"

    pd.DataFrame(
        {
            "person_id": ["1"],
            "household_id": ["10"],
            "hh_weight": [10.0],
            "health_savings_account_ald": [25.0],
        }
    ).to_parquet(seed_path, index=False)
    _write_period_dataset(
        reference_path,
        {
            "household_id": [10],
            "household_weight": [10.0],
            "person_id": [101],
            "person_household_id": [10],
            "tax_unit_id": [100],
            "person_tax_unit_id": [100],
            "health_savings_account_ald": [25.0],
        },
    )

    written = write_us_seed_stage_parity_audit(
        seed_path,
        reference_path,
        output_path,
        focus_variables=("health_savings_account_ald",),
        boolean_landing_features=(),
        categorical_landing_features=(),
        candidate_only_landing_features=(),
    )

    payload = json.loads(written.read_text())
    assert written == output_path.resolve()
    assert payload["focusVariables"]["health_savings_account_ald"]["comparison"]["type"] == (
        "numeric"
    )
