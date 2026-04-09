"""Tests for PE pre-sim parity auditing."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from microplex_us.pipelines.pre_sim_parity import (
    PreSimParityVariableSpec,
    build_us_pre_sim_parity_audit,
)


def _write_period_dataset(path, data: dict[str, np.ndarray], *, period: int = 2024) -> None:
    with h5py.File(path, "w") as handle:
        for variable, values in data.items():
            group = handle.create_group(variable)
            group.create_dataset(str(period), data=values)


def test_build_us_pre_sim_parity_audit_reports_schema_and_support(tmp_path) -> None:
    reference_path = tmp_path / "reference.h5"
    candidate_path = tmp_path / "candidate.h5"

    _write_period_dataset(
        reference_path,
        {
            "household_id": np.array([1, 2], dtype=int),
            "household_weight": np.array([10.0, 20.0], dtype=float),
            "person_id": np.array([101, 102, 103], dtype=int),
            "person_household_id": np.array([1, 1, 2], dtype=int),
            "tax_unit_id": np.array([11, 12], dtype=int),
            "person_tax_unit_id": np.array([11, 11, 12], dtype=int),
            "age": np.array([4, 37, 42], dtype=int),
            "state_fips": np.array([1, 2], dtype=int),
            "county_fips": np.array([1, 3], dtype=int),
            "is_household_head": np.array([1, 0, 1], dtype=int),
            "has_esi": np.array([True, False, True], dtype=bool),
            "employment_income_before_lsr": np.array([0.0, 10.0, 100.0], dtype=float),
        },
    )
    _write_period_dataset(
        candidate_path,
        {
            "household_id": np.array([1, 2], dtype=int),
            "household_weight": np.array([1.0, 1.0], dtype=float),
            "person_id": np.array([201, 202], dtype=int),
            "person_household_id": np.array([1, 2], dtype=int),
            "tax_unit_id": np.array([21, 22], dtype=int),
            "person_tax_unit_id": np.array([21, 22], dtype=int),
            "age": np.array([4, 42], dtype=int),
            "state_fips": np.array([1, 2], dtype=int),
            "has_esi": np.array([1.0, 0.0], dtype=float),
            "employment_income_before_lsr": np.array([0.0, 100.0], dtype=float),
        },
    )

    audit = build_us_pre_sim_parity_audit(
        candidate_path,
        reference_path,
        focus_variables=(
            PreSimParityVariableSpec("age", "age", value_kind="numeric"),
            PreSimParityVariableSpec("state_fips", "state_fips", value_kind="categorical"),
            PreSimParityVariableSpec("has_esi", "has_esi", value_kind="categorical"),
            "county_fips",
        ),
        critical_reference_variables=("county_fips",),
    )

    assert audit["schema"]["reference_variable_count"] == 12
    assert audit["schema"]["candidate_variable_count"] == 10
    assert audit["schema"]["missing_in_candidate_count"] == 2
    assert audit["schema"]["missing_critical_reference_variables"] == ["county_fips"]

    reference_structure = audit["entity_structure"]["reference"]
    candidate_structure = audit["entity_structure"]["candidate"]
    assert reference_structure["share_multi_person_tax_units"] == 0.5
    assert candidate_structure["share_multi_person_tax_units"] == 0.0

    state_age = audit["state_age_support"]
    assert state_age["reference"]["nonempty_cell_count"] == 3
    assert state_age["candidate"]["nonempty_cell_count"] == 2
    assert state_age["support_recall"] == 2 / 3

    county = audit["focus_variables"]["county_fips"]
    assert county["reference_present"] is True
    assert county["candidate_present"] is False
    age = audit["focus_variables"]["age"]
    assert age["candidate"]["kind"] == "numeric"
    assert age["reference"]["kind"] == "numeric"
    assert age["comparison"]["type"] == "numeric"
    assert age["comparison"]["weighted_mean_ratio"] == pytest.approx(23.0 / 31.25)
    state = audit["focus_variables"]["state_fips"]
    assert state["candidate"]["kind"] == "categorical"
    assert state["reference"]["kind"] == "categorical"
    assert state["comparison"]["type"] == "categorical"
    has_esi = audit["focus_variables"]["has_esi"]
    assert has_esi["comparison"]["type"] == "categorical"
    assert has_esi["comparison"]["support_recall"] == 1.0
    assert has_esi["comparison"]["support_precision"] == 1.0
