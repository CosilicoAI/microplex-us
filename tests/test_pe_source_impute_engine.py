"""Tests for the PE source-impute block engine."""

from __future__ import annotations

import pandas as pd

from microplex_us.pe_source_impute_engine import PE_SOURCE_IMPUTE_BLOCK_ENGINE


def test_prepare_condition_surface_and_predictors_for_acs_block() -> None:
    donor_frame = pd.DataFrame(
        {
            "household_id": [1, 1, 2],
            "age": [45, 12, 70],
            "sex": [1, 2, 2],
            "is_head": [1.0, 0.0, 1.0],
            "tenure": [1, 1, 2],
            "employment_income": [50_000.0, 0.0, 12_000.0],
            "self_employment_income": [5_000.0, 0.0, 0.0],
            "gross_social_security": [0.0, 0.0, 20_000.0],
            "taxable_pension_income": [0.0, 0.0, 15_000.0],
            "state_fips": [6, 6, 36],
            "rent": [1_200.0, 0.0, 950.0],
        }
    )
    current_frame = donor_frame.copy()

    surface = PE_SOURCE_IMPUTE_BLOCK_ENGINE.prepare_condition_surface(
        donor_frame=donor_frame,
        current_frame=current_frame,
        donor_source_name="acs_2022",
        donor_block=("rent",),
    )

    assert surface is not None
    assert surface.spec.key == "acs"
    assert surface.donor_frame["is_household_head"].tolist() == [1.0, 0.0, 1.0]
    assert surface.donor_frame["pension_income"].tolist() == [0.0, 0.0, 15_000.0]
    assert surface.compatible_predictors(
        compatibility_fn=lambda donor, current: donor.notna().all() and current.notna().all(),
    ) == list(surface.spec.predictors)


def test_prepare_condition_surface_returns_none_for_unmapped_block() -> None:
    frame = pd.DataFrame({"tip_income": [1.0], "employment_income": [10.0]})

    surface = PE_SOURCE_IMPUTE_BLOCK_ENGINE.prepare_condition_surface(
        donor_frame=frame,
        current_frame=frame,
        donor_source_name="unknown_source",
        donor_block=("tip_income",),
    )

    assert surface is None
