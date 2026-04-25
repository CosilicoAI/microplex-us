"""Adapter must not materialize the estimate matrix as float64 pandas.

At v7 scale (1.5M households x ~500 constraints) the adapter's pre-fix
behavior builds a float64 DataFrame (6 GB) *and* microcalibrate keeps
it alive in memory alongside a float32 torch copy. The combined footprint
pushes the workstation past macOS jetsam kill threshold.

These tests pin the adapter's memory contract: the estimate matrix passed
to microcalibrate.Calibration must be float32 from the start. Adapter
behavior on small inputs is unchanged; only the dtype is tightened.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
from microplex.calibration import LinearConstraint

from microplex_us.calibration import MicrocalibrateAdapter


def _toy_data(n_records: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=n_records),
            "income": rng.normal(40_000, 20_000, size=n_records).clip(0, None),
            "weight": np.ones(n_records),
        }
    )


def _age_band(
    data: pd.DataFrame, name: str, low: int, high: int, target: float
) -> LinearConstraint:
    mask = (data["age"] >= low) & (data["age"] < high)
    return LinearConstraint(
        name=name,
        coefficients=mask.astype(float).to_numpy(),
        target=target,
    )


class TestEstimateMatrixDtype:
    """The adapter must not pass a float64 estimate matrix to Calibration."""

    def test_estimate_matrix_passed_to_calibration_is_float32(self) -> None:
        """Intercept Calibration.__init__ and inspect the estimate_matrix arg."""
        captured: dict[str, Any] = {}

        from microcalibrate import Calibration as _RealCalibration

        original_init = _RealCalibration.__init__

        def spy_init(self: Any, *args: Any, **kwargs: Any) -> None:
            captured["estimate_matrix"] = kwargs.get("estimate_matrix")
            original_init(self, *args, **kwargs)

        data = _toy_data()
        constraints = (
            _age_band(data, "age_18_30", 18, 30, 40.0),
            _age_band(data, "age_30_45", 30, 45, 60.0),
            _age_band(data, "age_45_70", 45, 70, 100.0),
        )
        adapter = MicrocalibrateAdapter()
        with patch.object(_RealCalibration, "__init__", spy_init):
            adapter.fit_transform(data, linear_constraints=constraints)

        estimate_matrix = captured["estimate_matrix"]
        assert estimate_matrix is not None, "Calibration was not constructed"

        if isinstance(estimate_matrix, pd.DataFrame):
            for col, dtype in estimate_matrix.dtypes.items():
                assert dtype == np.float32, (
                    f"estimate_matrix column {col!r} is {dtype}, expected float32 "
                    "(float64 doubles adapter peak memory at v7 scale)"
                )
        else:
            arr = np.asarray(estimate_matrix)
            assert arr.dtype == np.float32, (
                f"estimate_matrix dtype is {arr.dtype}, expected float32"
            )

    def test_weights_still_converge_with_float32(self) -> None:
        """Dtype tightening must not break the convergence behavior."""
        from microplex_us.calibration import MicrocalibrateAdapterConfig

        data = _toy_data(n_records=300)
        constraints = (
            _age_band(data, "age_18_30", 18, 30, 60.0),
            _age_band(data, "age_30_45", 30, 45, 90.0),
            _age_band(data, "age_45_70", 45, 70, 150.0),
        )
        adapter = MicrocalibrateAdapter(
            MicrocalibrateAdapterConfig(
                epochs=400, learning_rate=0.05, noise_level=0.0
            )
        )
        result = adapter.fit_transform(data, linear_constraints=constraints)
        validation = adapter.validate(result)
        # Same tolerance the existing smoke tests in this package use.
        assert validation["max_error"] < 0.1, validation
