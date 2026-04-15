"""Tests for the PolicyEngine L0 calibrator adapter."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
from microplex.calibration import LinearConstraint

from microplex_us.pipelines.pe_l0 import PolicyEngineL0Calibrator


def _install_fake_policyengine_l0(monkeypatch, weights: np.ndarray) -> dict[str, object]:
    calls: dict[str, object] = {}

    def fake_fit_l0_weights(**kwargs):
        calls.update(kwargs)
        return np.asarray(weights, dtype=float)

    pe_pkg = types.ModuleType("policyengine_us_data")
    cal_pkg = types.ModuleType("policyengine_us_data.calibration")
    unified = types.ModuleType("policyengine_us_data.calibration.unified_calibration")
    unified.fit_l0_weights = fake_fit_l0_weights
    pe_pkg.calibration = cal_pkg
    cal_pkg.unified_calibration = unified
    monkeypatch.setitem(sys.modules, "policyengine_us_data", pe_pkg)
    monkeypatch.setitem(sys.modules, "policyengine_us_data.calibration", cal_pkg)
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.unified_calibration",
        unified,
    )
    return calls


def test_policyengine_l0_calibrator_supports_explicit_linear_constraints(monkeypatch):
    calls = _install_fake_policyengine_l0(monkeypatch, np.array([1.0, 2.0]))
    data = pd.DataFrame({"weight": [1.0, 1.0]})
    constraints = (
        LinearConstraint("row1", np.array([1.0, 0.0]), 1.0),
        LinearConstraint("row2", np.array([0.0, 1.0]), 2.0),
    )

    calibrator = PolicyEngineL0Calibrator(
        lambda_l0=1e-4,
        lambda_l2=1e-12,
        beta=0.35,
        learning_rate=0.15,
        epochs=25,
        tol=1e-6,
        device="cpu",
    )
    result = calibrator.fit_transform(
        data,
        {},
        weight_col="weight",
        linear_constraints=constraints,
    )
    validation = calibrator.validate(result)

    assert result["weight"].tolist() == [1.0, 2.0]
    assert calls["X_sparse"].shape == (2, 2)
    assert calls["target_names"] == ["row1", "row2"]
    assert calls["targets"].tolist() == [1.0, 2.0]
    assert calls["initial_weights"].tolist() == [1.0, 1.0]
    assert validation["converged"] is True
    assert validation["max_error"] < 1e-9
    assert validation["sparsity"] == 0.0


def test_policyengine_l0_calibrator_reports_sparsity(monkeypatch):
    _install_fake_policyengine_l0(monkeypatch, np.array([0.0, 3.0, 0.0]))
    data = pd.DataFrame({"weight": [1.0, 1.0, 1.0]})
    constraints = (
        LinearConstraint("row", np.array([0.0, 1.0, 0.0]), 3.0),
    )

    calibrator = PolicyEngineL0Calibrator(epochs=5, tol=1e-6)
    calibrator.fit(
        data,
        {},
        weight_col="weight",
        linear_constraints=constraints,
    )

    assert calibrator.get_sparsity() == 2 / 3
