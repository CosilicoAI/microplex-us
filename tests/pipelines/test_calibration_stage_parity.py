"""Tests for calibration-stage parity auditing."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from microplex_us.pipelines.calibration_stage_parity import (
    build_us_calibration_stage_parity_audit,
)
from microplex_us.pipelines.pre_sim_parity import PreSimParityVariableSpec


def _write_period_dataset(path, data: dict[str, np.ndarray], *, period: int = 2024) -> None:
    with h5py.File(path, "w") as handle:
        for variable, values in data.items():
            group = handle.create_group(variable)
            group.create_dataset(str(period), data=values)


def test_build_us_calibration_stage_parity_audit_reports_weight_lift_and_reference_support(
    tmp_path,
) -> None:
    synthetic_path = tmp_path / "synthetic.parquet"
    calibrated_path = tmp_path / "calibrated.parquet"
    reference_path = tmp_path / "reference.h5"

    pd.DataFrame(
        {
            "household_id": [1, 1, 2, 2],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "health_savings_account_ald": [100.0, 0.0, 0.0, 0.0],
            "has_esi": [1.0, 0.0, 0.0, 1.0],
        }
    ).to_parquet(synthetic_path, index=False)
    pd.DataFrame(
        {
            "household_id": [1, 1, 2, 2],
            "weight": [5.0, 5.0, 1.0, 1.0],
            "health_savings_account_ald": [100.0, 0.0, 0.0, 0.0],
            "has_esi": [True, False, False, True],
        }
    ).to_parquet(calibrated_path, index=False)

    _write_period_dataset(
        reference_path,
        {
            "household_id": np.array([1, 2], dtype=int),
            "household_weight": np.array([2.0, 1.0], dtype=float),
            "person_id": np.array([10, 11, 20, 21], dtype=int),
            "person_household_id": np.array([1, 1, 2, 2], dtype=int),
            "tax_unit_id": np.array([100, 200], dtype=int),
            "person_tax_unit_id": np.array([100, 100, 200, 200], dtype=int),
            "health_savings_account_ald": np.array([100.0, 0.0], dtype=float),
            "has_esi": np.array([True, False, False, True], dtype=bool),
        },
    )

    audit = build_us_calibration_stage_parity_audit(
        synthetic_path,
        calibrated_path,
        reference_dataset=reference_path,
        focus_variables=(
            PreSimParityVariableSpec(
                "health_savings_account_ald",
                "health_savings_account_ald",
                value_kind="numeric",
            ),
            PreSimParityVariableSpec("has_esi", "has_esi", value_kind="categorical"),
        ),
    )

    synthetic_weights = audit["weightDiagnostics"]["synthetic"]
    calibrated_weights = audit["weightDiagnostics"]["calibrated"]
    assert synthetic_weights["total_weight"] == pytest.approx(2.0)
    assert calibrated_weights["total_weight"] == pytest.approx(6.0)
    assert calibrated_weights["effective_sample_size"] < synthetic_weights["effective_sample_size"]

    hsa = audit["focusVariables"]["health_savings_account_ald"]
    assert hsa["calibrated_vs_synthetic"]["type"] == "numeric"
    assert hsa["calibrated_vs_synthetic"]["weighted_sum_ratio"] == pytest.approx(5.0)
    assert hsa["calibrated_vs_reference"]["weighted_sum_ratio"] == pytest.approx(2.5)

    has_esi = audit["focusVariables"]["has_esi"]
    assert has_esi["calibrated_vs_synthetic"]["type"] == "categorical"
    assert has_esi["calibrated_vs_synthetic"]["support_recall"] == 1.0
    assert has_esi["calibrated_vs_reference"]["support_precision"] == 1.0
