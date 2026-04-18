"""Adapter that wraps `microcalibrate.Calibration` in the microplex-us interface.

Mainline production calibrator per `docs/calibrator-decision.md`.

`MicrocalibrateAdapter.fit_transform` has the same call signature as the
legacy `microplex.calibration.Calibrator.fit_transform` used by the current
`pe_us_data_rebuild` pipeline: take a DataFrame of records, a tuple of
`LinearConstraint` objects, and a `weight_col`; return a DataFrame with the
same rows and adjusted weights. Every input record survives to the output
with a non-negative weight — identity preservation is the contract.

This is a drop-in replacement for the calibration step that killed v6 with
`backend="entropy"`. Instead of materializing a dense Jacobian over
(n_records × n_constraints), `microcalibrate` does gradient descent over the
weight vector with an optional L0 regularizer that defaults off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from microcalibrate import Calibration
from microplex.calibration import LinearConstraint


@dataclass(frozen=True)
class MicrocalibrateAdapterConfig:
    """Hyperparameters for `MicrocalibrateAdapter`.

    Defaults come from `microcalibrate.Calibration`'s own defaults
    (epochs=32, learning_rate=1e-3, noise_level=10.0) except `device`,
    which microcalibrate picks automatically from CUDA > MPS > CPU but
    we pin to a single choice for reproducibility.
    """

    epochs: int = 32
    learning_rate: float = 1e-3
    noise_level: float = 10.0
    dropout_rate: float = 0.0
    device: str | None = None  # None = let microcalibrate auto-select
    seed: int = 42
    regularize_with_l0: bool = False
    l0_lambda: float = 5e-6
    init_mean: float = 0.999
    temperature: float = 0.5
    sparse_learning_rate: float = 0.2
    # Keep activation memory bounded at v7 scale. 100_000 rows per
    # backward step keeps the per-batch autograd activation under
    # ~200 MB at k = 500 constraints (100_000 * 500 * 4 B). None =
    # full-batch, which OOMs at 1.5M households.
    batch_size: int | None = 100_000


class MicrocalibrateAdapter:
    """Drop-in replacement for the `fit_transform` / `validate` surface.

    Usage:

        >>> adapter = MicrocalibrateAdapter()
        >>> result = adapter.fit_transform(
        ...     data=households_df,
        ...     marginal_targets={},  # unused; kept for signature parity
        ...     weight_col="household_weight",
        ...     linear_constraints=tuple_of_LinearConstraints,
        ... )
        >>> validation = adapter.validate(result)

    The returned DataFrame is a copy of `data` with `weight_col` updated.
    """

    def __init__(
        self,
        config: MicrocalibrateAdapterConfig | None = None,
    ) -> None:
        self.config = config or MicrocalibrateAdapterConfig()
        self._last_calibration: Calibration | None = None
        self._last_constraint_names: list[str] | None = None
        self._last_targets: np.ndarray | None = None
        self._last_performance: pd.DataFrame | None = None

    def fit_transform(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]] | None = None,
        continuous_targets: dict[str, float] | None = None,
        *,
        weight_col: str = "weight",
        linear_constraints: Sequence[LinearConstraint] = (),
    ) -> pd.DataFrame:
        """Calibrate weights via gradient-descent chi-squared.

        `marginal_targets` and `continuous_targets` are accepted for
        signature parity with the legacy `Calibrator`, but this adapter
        expects constraints to be expressed as `LinearConstraint` rows.
        Callers should compile their marginal / continuous targets into
        linear constraints before calling.
        """
        if weight_col not in data.columns:
            raise ValueError(
                f"MicrocalibrateAdapter: weight column {weight_col!r} "
                f"not found in data (columns: {list(data.columns)[:10]}...)"
            )

        n_records = len(data)
        initial_weights = data[weight_col].to_numpy(dtype=float)

        if not linear_constraints:
            # Nothing to calibrate — preserve caller expectations.
            self._last_calibration = None
            self._last_constraint_names = []
            self._last_targets = np.empty(0, dtype=float)
            self._last_performance = None
            return data.copy()

        target_names = [c.name for c in linear_constraints]
        targets = np.array([c.target for c in linear_constraints], dtype=float)

        for constraint in linear_constraints:
            if constraint.coefficients.shape != (n_records,):
                raise ValueError(
                    f"MicrocalibrateAdapter: constraint {constraint.name!r} has "
                    f"coefficients shape {constraint.coefficients.shape}, expected "
                    f"({n_records},) matching the data length."
                )

        # float32 keeps the adapter's peak allocation at half the
        # float64 default; downstream torch layer casts to float32
        # anyway, so this is a free precision-compatible win.
        estimate_matrix = pd.DataFrame(
            {
                c.name: np.asarray(c.coefficients, dtype=np.float32)
                for c in linear_constraints
            }
        )

        calibrator = Calibration(
            weights=initial_weights,
            targets=targets,
            target_names=np.array(target_names),
            estimate_matrix=estimate_matrix,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            noise_level=self.config.noise_level,
            dropout_rate=self.config.dropout_rate,
            device=self.config.device,
            seed=self.config.seed,
            regularize_with_l0=self.config.regularize_with_l0,
            l0_lambda=self.config.l0_lambda,
            init_mean=self.config.init_mean,
            temperature=self.config.temperature,
            sparse_learning_rate=self.config.sparse_learning_rate,
            batch_size=self.config.batch_size,
        )

        performance_df = calibrator.calibrate()
        self._last_calibration = calibrator
        self._last_constraint_names = target_names
        self._last_targets = targets
        self._last_performance = performance_df

        result = data.copy()
        result[weight_col] = calibrator.weights
        return result

    def validate(self, calibrated: pd.DataFrame | None = None) -> dict[str, Any]:
        """Return validation metrics in the shape the legacy pipeline expects.

        The legacy `Calibrator.validate` returns `{"converged", "max_error",
        "sparsity", "linear_errors"}`. We populate the same keys.

        `calibrated` is accepted for interface parity but not read; the
        authoritative values come from the last `calibrate()` call.
        """
        if self._last_calibration is None:
            return {
                "converged": True,
                "max_error": 0.0,
                "sparsity": 0.0,
                "linear_errors": {},
            }

        estimates = self._last_calibration.estimate().to_numpy(dtype=float)
        targets = self._last_targets
        names = self._last_constraint_names

        rel_errors = np.where(
            np.abs(targets) > 1e-12,
            np.abs(estimates - targets) / np.abs(targets),
            np.abs(estimates - targets),
        )
        linear_errors = {
            name: {
                "target": float(target_value),
                "estimate": float(estimate_value),
                "relative_error": float(rel_error),
                "absolute_error": float(abs(estimate_value - target_value)),
            }
            for name, target_value, estimate_value, rel_error in zip(
                names, targets, estimates, rel_errors, strict=True
            )
        }

        max_error = float(rel_errors.max()) if rel_errors.size else 0.0
        weights = self._last_calibration.weights
        sparsity = float((weights == 0).sum()) / max(len(weights), 1)

        return {
            "converged": bool(max_error < 0.05),  # 5 % relative error bar
            "max_error": max_error,
            "sparsity": sparsity,
            "linear_errors": linear_errors,
        }

    def performance_history(self) -> pd.DataFrame | None:
        """The per-epoch performance log from microcalibrate, if available."""
        return self._last_performance


__all__ = [
    "MicrocalibrateAdapter",
    "MicrocalibrateAdapterConfig",
]
