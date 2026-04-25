"""Small-scale smoke tests for the microcalibrate-backed calibration adapter.

These exercise the adapter's interface contract (matches the legacy
`Calibrator.fit_transform` shape) and verify that the underlying
gradient-descent chi-squared solver actually moves weights toward the
requested targets on a deliberately small problem.

Scale-up validation happens separately (see
`docs/synthesizer-benchmark-scale-up.md`). These tests are only expected
to run in seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from microplex.calibration import LinearConstraint

from microplex_us.calibration import (
    MicrocalibrateAdapter,
    MicrocalibrateAdapterConfig,
)


def _toy_data(n_records: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=n_records),
            "income": rng.normal(40_000, 20_000, size=n_records).clip(0, None),
            "weight": np.ones(n_records),
        }
    )


def _age_band_constraint(
    data: pd.DataFrame, name: str, low: int, high: int, target: float
) -> LinearConstraint:
    mask = (data["age"] >= low) & (data["age"] < high)
    return LinearConstraint(
        name=name,
        coefficients=mask.astype(float).to_numpy(),
        target=target,
    )


def _income_age_band_constraint(
    data: pd.DataFrame, name: str, low: int, high: int, target: float
) -> LinearConstraint:
    mask = (data["age"] >= low) & (data["age"] < high)
    coefs = (mask.astype(float) * data["income"]).to_numpy()
    return LinearConstraint(name=name, coefficients=coefs, target=target)


class TestInterfaceContract:
    """Adapter matches the legacy `Calibrator.fit_transform` signature."""

    def test_empty_constraints_returns_copy_unchanged(self) -> None:
        data = _toy_data()
        adapter = MicrocalibrateAdapter()
        result = adapter.fit_transform(data, marginal_targets={})
        pd.testing.assert_frame_equal(result, data)
        # Should not share storage with the input.
        assert result is not data

    def test_weight_column_validation(self) -> None:
        data = _toy_data().drop(columns=["weight"])
        adapter = MicrocalibrateAdapter()
        with pytest.raises(ValueError, match="weight column 'weight' not found"):
            adapter.fit_transform(
                data,
                marginal_targets={},
                linear_constraints=(
                    _age_band_constraint(_toy_data(), "age_18_30", 18, 30, 20.0),
                ),
            )

    def test_constraint_shape_validation(self) -> None:
        data = _toy_data()
        adapter = MicrocalibrateAdapter()
        bad_constraint = LinearConstraint(
            name="wrong_shape",
            coefficients=np.ones(len(data) + 5),
            target=10.0,
        )
        with pytest.raises(ValueError, match="constraint 'wrong_shape'"):
            adapter.fit_transform(
                data,
                marginal_targets={},
                linear_constraints=(bad_constraint,),
            )

    def test_preserves_all_records(self) -> None:
        data = _toy_data()
        adapter = MicrocalibrateAdapter(
            MicrocalibrateAdapterConfig(epochs=8, noise_level=0.0)
        )
        constraint = _age_band_constraint(data, "age_18_40", 18, 40, target=30.0)
        result = adapter.fit_transform(
            data,
            marginal_targets={},
            linear_constraints=(constraint,),
        )
        # Identity preservation: every record survives.
        assert len(result) == len(data)
        pd.testing.assert_index_equal(result.index, data.index)
        # No negative weights.
        assert (result["weight"] >= 0).all()


class TestCalibrationMovesWeights:
    """Adapter actually does the job — weights shift toward the targets."""

    def test_single_constraint_converges(self) -> None:
        """One age-band count constraint should be matched within tolerance."""
        data = _toy_data(n_records=200, seed=1)
        # Current weighted count in [25, 45) band.
        mask = (data["age"] >= 25) & (data["age"] < 45)
        current_count = float(mask.sum())
        # Ask for 2x the current weighted count.
        target = 2.0 * current_count

        constraint = _age_band_constraint(data, "age_25_45", 25, 45, target=target)
        adapter = MicrocalibrateAdapter(
            MicrocalibrateAdapterConfig(
                epochs=400,
                learning_rate=0.05,
                noise_level=0.0,
            )
        )
        result = adapter.fit_transform(
            data,
            marginal_targets={},
            linear_constraints=(constraint,),
        )

        validation = adapter.validate(result)
        errors = validation["linear_errors"]
        assert "age_25_45" in errors
        # 5 % relative tolerance is generous for 400 epochs on 1 constraint.
        assert errors["age_25_45"]["relative_error"] < 0.05
        # Weighted count actually moved.
        weighted_count = float(
            (result["age"] >= 25).values
            * (result["age"] < 45).values
            * result["weight"].to_numpy()
        ).sum() if False else float(result.loc[mask, "weight"].sum())
        # Should be close to target; at least 1.5x original (we asked for 2x).
        assert weighted_count > 1.5 * current_count

    def test_two_orthogonal_constraints_both_improve(self) -> None:
        """Separate age-band and income-age-band constraints should both reduce."""
        data = _toy_data(n_records=300, seed=2)

        # Current sums.
        band_mask = (data["age"] >= 30) & (data["age"] < 50)
        current_count = float(band_mask.sum())
        current_income_sum = float(data.loc[band_mask, "income"].sum())

        constraints = (
            _age_band_constraint(
                data, "count_30_50", 30, 50, target=1.4 * current_count
            ),
            _income_age_band_constraint(
                data, "income_30_50", 30, 50, target=1.4 * current_income_sum
            ),
        )

        adapter = MicrocalibrateAdapter(
            MicrocalibrateAdapterConfig(
                epochs=400,
                learning_rate=0.05,
                noise_level=0.0,
            )
        )
        result = adapter.fit_transform(
            data,
            marginal_targets={},
            linear_constraints=constraints,
        )

        validation = adapter.validate(result)
        # Both constraints should get meaningfully closer to target.
        # 10 % relative tolerance since there's inherent trade-off between
        # count and income-sum constraints on the same band.
        for name in ("count_30_50", "income_30_50"):
            rel = validation["linear_errors"][name]["relative_error"]
            assert rel < 0.10, f"constraint {name} still at rel_error={rel:.3f}"


class TestValidationShape:
    """Validation output has the keys the downstream pipeline expects."""

    def test_validation_keys(self) -> None:
        data = _toy_data()
        adapter = MicrocalibrateAdapter(
            MicrocalibrateAdapterConfig(epochs=4, noise_level=0.0)
        )
        constraint = _age_band_constraint(data, "a", 18, 40, target=30.0)
        _ = adapter.fit_transform(
            data,
            marginal_targets={},
            linear_constraints=(constraint,),
        )
        validation = adapter.validate()

        assert set(validation) == {
            "converged",
            "max_error",
            "sparsity",
            "linear_errors",
        }
        assert isinstance(validation["converged"], bool)
        assert isinstance(validation["max_error"], float)
        assert 0.0 <= validation["sparsity"] <= 1.0
        assert "a" in validation["linear_errors"]

        entry = validation["linear_errors"]["a"]
        assert set(entry) == {
            "target",
            "estimate",
            "relative_error",
            "absolute_error",
        }

    def test_validation_without_calibration_is_trivially_converged(self) -> None:
        adapter = MicrocalibrateAdapter()
        validation = adapter.validate()
        assert validation["converged"] is True
        assert validation["max_error"] == 0.0
        assert validation["sparsity"] == 0.0
        assert validation["linear_errors"] == {}
