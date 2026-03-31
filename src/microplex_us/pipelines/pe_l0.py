"""Adapters for PolicyEngine US calibration backends."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Any, Self

import numpy as np
import pandas as pd
from microplex.calibration import (
    LinearConstraint,
    _build_linear_constraint_system,
    _validate_calibration_inputs,
)
from scipy import sparse as sp

_PE_L0_SUBPROCESS_SCRIPT = """
import json
import sys

import numpy as np
from scipy import sparse as sp

REPO_ROOT = sys.argv[1]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from policyengine_us_data.calibration.unified_calibration import fit_l0_weights

X_sparse = sp.load_npz(sys.argv[2])
targets = np.load(sys.argv[3])
initial_weights = np.load(sys.argv[4])
with open(sys.argv[5]) as handle:
    target_names = json.load(handle)
output_path = sys.argv[6]
lambda_l0 = float(sys.argv[7])
epochs = int(sys.argv[8])
device = sys.argv[9]
verbose_freq = None if sys.argv[10] == "none" else int(sys.argv[10])
beta = float(sys.argv[11])
lambda_l2 = float(sys.argv[12])
learning_rate = float(sys.argv[13])
achievable = np.asarray(X_sparse.sum(axis=1)).reshape(-1) > 0

weights = fit_l0_weights(
    X_sparse=X_sparse,
    targets=targets,
    lambda_l0=lambda_l0,
    epochs=epochs,
    device=device,
    verbose_freq=verbose_freq,
    beta=beta,
    lambda_l2=lambda_l2,
    learning_rate=learning_rate,
    target_names=target_names,
    initial_weights=initial_weights,
    achievable=achievable,
)
np.save(output_path, np.asarray(weights, dtype=float))
""".strip()


class PolicyEngineL0Calibrator:
    """Wrap PolicyEngine US-data's L0 optimizer behind the Microplex interface."""

    def __init__(
        self,
        *,
        lambda_l0: float = 1e-4,
        lambda_l2: float = 1e-12,
        beta: float = 0.35,
        learning_rate: float = 0.15,
        epochs: int = 100,
        tol: float = 1e-6,
        device: str = "cpu",
        verbose_freq: int | None = None,
        policyengine_us_data_repo_root: str | os.PathLike[str] | None = None,
        policyengine_us_data_python: str | os.PathLike[str] | None = None,
    ) -> None:
        self.lambda_l0 = float(lambda_l0)
        self.lambda_l2 = float(lambda_l2)
        self.beta = float(beta)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.tol = float(tol)
        self.device = str(device)
        self.verbose_freq = verbose_freq
        self.policyengine_us_data_repo_root = policyengine_us_data_repo_root
        self.policyengine_us_data_python = policyengine_us_data_python

        self.weights_: np.ndarray | None = None
        self.is_fitted_: bool = False
        self.n_records_: int | None = None
        self.marginal_targets_: dict[str, dict[str, float]] | None = None
        self.continuous_targets_: dict[str, float] | None = None
        self.linear_constraints_: tuple[LinearConstraint, ...] = ()
        self.target_names_: list[str] = []
        self.calibration_error_: float = 0.0
        self.max_error_: float = 0.0
        self.converged_: bool = False
        self.n_iterations_: int = 0

    def fit(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None = None,
        weight_col: str = "weight",
        linear_constraints: tuple[LinearConstraint, ...] | list[LinearConstraint] | None = None,
    ) -> Self:
        self.n_records_ = len(data)
        self.marginal_targets_ = marginal_targets
        self.continuous_targets_ = continuous_targets or {}
        self.linear_constraints_ = tuple(linear_constraints or ())

        _validate_calibration_inputs(
            data,
            marginal_targets,
            continuous_targets,
            self.linear_constraints_,
        )

        A, b, names, _ = _build_linear_constraint_system(
            data,
            marginal_targets,
            continuous_targets,
            self.linear_constraints_,
        )
        self.target_names_ = names

        if A.shape[0] == 0:
            if weight_col in data.columns:
                self.weights_ = data[weight_col].to_numpy(dtype=float, copy=True)
            else:
                self.weights_ = np.ones(len(data), dtype=float)
            self.calibration_error_ = 0.0
            self.max_error_ = 0.0
            self.converged_ = True
            self.n_iterations_ = 0
            self.is_fitted_ = True
            return self

        if weight_col in data.columns:
            initial_weights = data[weight_col].to_numpy(dtype=float, copy=True)
        else:
            initial_weights = np.ones(len(data), dtype=float)
        initial_weights = np.maximum(initial_weights, 1e-12)

        X_sparse = sp.csr_matrix(A)
        weights = self._fit_weights(
            X_sparse=X_sparse,
            targets=b.astype(np.float64),
            initial_weights=initial_weights,
            target_names=names,
        )
        weights = np.maximum(np.asarray(weights, dtype=float), 0.0)

        residual = A @ weights - b
        rel_errors = np.abs(residual) / np.maximum(np.abs(b), 1e-10)
        self.weights_ = weights
        self.calibration_error_ = float(np.sqrt(np.mean(rel_errors**2)))
        self.max_error_ = float(rel_errors.max()) if len(rel_errors) else 0.0
        self.converged_ = bool(self.max_error_ < self.tol)
        self.n_iterations_ = self.epochs
        self.is_fitted_ = True
        return self

    def _fit_weights(
        self,
        *,
        X_sparse,
        targets: np.ndarray,
        initial_weights: np.ndarray,
        target_names: list[str],
    ) -> np.ndarray:
        try:
            from policyengine_us_data.calibration.unified_calibration import (
                fit_l0_weights,
            )

            achievable = np.asarray(X_sparse.sum(axis=1)).reshape(-1) > 0
            return fit_l0_weights(
                X_sparse=X_sparse,
                targets=targets,
                lambda_l0=self.lambda_l0,
                epochs=self.epochs,
                device=self.device,
                verbose_freq=self.verbose_freq,
                beta=self.beta,
                lambda_l2=self.lambda_l2,
                learning_rate=self.learning_rate,
                target_names=target_names,
                initial_weights=initial_weights,
                achievable=achievable,
            )
        except ImportError:
            return self._fit_weights_via_policyengine_python(
                X_sparse=X_sparse,
                targets=targets,
                initial_weights=initial_weights,
                target_names=target_names,
            )

    def _fit_weights_via_policyengine_python(
        self,
        *,
        X_sparse,
        targets: np.ndarray,
        initial_weights: np.ndarray,
        target_names: list[str],
    ) -> np.ndarray:
        from microplex_us.pipelines.pe_native_scores import (
            build_policyengine_us_data_pythonpath,
            resolve_policyengine_us_data_python,
            resolve_policyengine_us_data_repo_root,
        )

        repo_root = resolve_policyengine_us_data_repo_root(
            self.policyengine_us_data_repo_root
        )
        python_path = resolve_policyengine_us_data_python(
            self.policyengine_us_data_python,
            repo_root=repo_root,
        )
        env = {
            key: value
            for key, value in os.environ.items()
            if key in {"HOME", "PATH", "TMPDIR", "LANG", "LC_ALL", "TZ"}
        }
        env["PYTHONPATH"] = build_policyengine_us_data_pythonpath(
            repo_root,
            existing_pythonpath=os.environ.get("PYTHONPATH"),
        )

        with tempfile.TemporaryDirectory(prefix="microplex-pe-l0-") as tmpdir:
            matrix_path = os.path.join(tmpdir, "constraints.npz")
            targets_path = os.path.join(tmpdir, "targets.npy")
            initial_path = os.path.join(tmpdir, "initial_weights.npy")
            names_path = os.path.join(tmpdir, "target_names.json")
            output_path = os.path.join(tmpdir, "weights.npy")
            sp.save_npz(matrix_path, X_sparse)
            np.save(targets_path, targets)
            np.save(initial_path, initial_weights)
            with open(names_path, "w") as handle:
                json.dump(target_names, handle)

            verbose_freq = "none" if self.verbose_freq is None else str(self.verbose_freq)
            try:
                completed = subprocess.run(
                    [
                        str(python_path),
                        "-c",
                        _PE_L0_SUBPROCESS_SCRIPT,
                        str(repo_root),
                        matrix_path,
                        targets_path,
                        initial_path,
                        names_path,
                        output_path,
                        str(self.lambda_l0),
                        str(self.epochs),
                        self.device,
                        verbose_freq,
                        str(self.beta),
                        str(self.lambda_l2),
                        str(self.learning_rate),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                )
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                stdout = (exc.stdout or "").strip()
                detail = stderr or stdout or str(exc)
                raise RuntimeError(
                    "PolicyEngine L0 subprocess calibration failed. "
                    f"Detail: {detail}"
                ) from exc
            if completed.stderr:
                print(completed.stderr, end="")
            return np.load(output_path)

    def transform(
        self,
        data: pd.DataFrame,
        weight_col: str = "weight",
    ) -> pd.DataFrame:
        if not self.is_fitted_:
            raise ValueError("Not fitted.")
        if len(data) != self.n_records_:
            raise ValueError(
                f"Data length ({len(data)}) doesn't match fitted ({self.n_records_})"
            )
        result = data.copy()
        result[weight_col] = self.weights_
        return result

    def fit_transform(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None = None,
        weight_col: str = "weight",
        linear_constraints: tuple[LinearConstraint, ...] | list[LinearConstraint] | None = None,
    ) -> pd.DataFrame:
        self.fit(
            data,
            marginal_targets,
            continuous_targets,
            weight_col=weight_col,
            linear_constraints=linear_constraints,
        )
        return self.transform(data, weight_col=weight_col)

    def get_sparsity(self) -> float:
        if not self.is_fitted_:
            raise ValueError("Not fitted.")
        return float((self.weights_ < 1e-9).sum() / self.n_records_)

    def validate(self, data: pd.DataFrame) -> dict[str, Any]:
        if not self.is_fitted_:
            raise ValueError("Not fitted.")

        weights = self.weights_
        results = {
            "targets": {},
            "marginal_errors": {},
            "continuous_errors": {},
            "linear_errors": {},
            "sparsity": self.get_sparsity(),
            "converged": self.converged_,
        }

        if self.marginal_targets_:
            for var, var_targets in self.marginal_targets_.items():
                results["marginal_errors"][var] = {}
                for category, target in var_targets.items():
                    mask = data[var] == category
                    actual = weights[mask].sum()
                    rel_error = abs(actual - target) / target if target > 0 else 0.0
                    info = {
                        "actual": actual,
                        "target": target,
                        "relative_error": rel_error,
                    }
                    results["marginal_errors"][var][category] = info
                    results["targets"][f"{var}={category}"] = {
                        **info,
                        "error": rel_error,
                    }

        if self.continuous_targets_:
            for var, target in self.continuous_targets_.items():
                actual = float((weights * data[var].to_numpy(dtype=float)).sum())
                rel_error = abs(actual - target) / abs(target) if target != 0 else 0.0
                info = {
                    "actual": actual,
                    "target": target,
                    "relative_error": rel_error,
                }
                results["continuous_errors"][var] = info
                results["targets"][var] = {
                    **info,
                    "error": rel_error,
                }

        for constraint in self.linear_constraints_:
            actual = float(weights @ constraint.coefficients)
            target = float(constraint.target)
            rel_error = abs(actual - target) / abs(target) if target != 0 else 0.0
            results["linear_errors"][constraint.name] = {
                "actual": actual,
                "target": target,
                "relative_error": rel_error,
            }

        errors = [t["error"] for t in results["targets"].values()]
        errors.extend(
            item["relative_error"] for item in results["linear_errors"].values()
        )
        results["max_error"] = max(errors) if errors else 0.0
        results["mean_error"] = float(np.mean(errors)) if errors else 0.0
        results["rmse"] = self.calibration_error_
        return results
