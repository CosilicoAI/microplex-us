"""CPS-specific summary-stat synthetic data helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.interpolate import interp1d


@dataclass
class CPSSummaryStats:
    """Summary statistics needed to generate CPS-shaped synthetic records."""

    variables: list[str]
    means: dict[str, float]
    stds: dict[str, float]
    quantiles: dict[str, np.ndarray]
    zero_fractions: dict[str, float]
    discrete_vars: list[str]
    discrete_distributions: dict[str, dict[int, float]]
    correlation_matrix: np.ndarray
    continuous_vars: list[str] = field(default_factory=list)
    quantile_values: dict[str, np.ndarray] = field(default_factory=dict)
    min_values: dict[str, float] = field(default_factory=dict)
    max_values: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        weight_col: str | None = None,
        discrete_threshold: int = 10,
    ) -> Self:
        variables = [column for column in data.columns if column != weight_col]
        discrete_vars: list[str] = []
        continuous_vars: list[str] = []
        for variable in variables:
            if (
                data[variable].nunique() <= discrete_threshold
                and data[variable].dtype in [np.int64, np.int32, int, "int64", "int32"]
            ):
                discrete_vars.append(variable)
            else:
                continuous_vars.append(variable)

        if weight_col and weight_col in data.columns:
            weights = data[weight_col].to_numpy(dtype=float)
        else:
            weights = np.ones(len(data), dtype=float)
        weights = weights / weights.sum()

        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        quantiles: dict[str, np.ndarray] = {}
        quantile_values: dict[str, np.ndarray] = {}
        zero_fractions: dict[str, float] = {}
        min_values: dict[str, float] = {}
        max_values: dict[str, float] = {}

        for variable in continuous_vars:
            values = data[variable].to_numpy(dtype=float)
            means[variable] = float(np.sum(weights * values))
            variance = np.sum(weights * (values - means[variable]) ** 2)
            stds[variable] = float(np.sqrt(variance))
            zero_fractions[variable] = float(np.mean(values == 0))

            positive_values = values[values > 0]
            if len(positive_values) > 0:
                min_values[variable] = float(np.min(positive_values))
                max_values[variable] = float(np.max(positive_values))
                q_probs = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
                q_values = np.quantile(positive_values, q_probs)
            else:
                min_values[variable] = 0.0
                max_values[variable] = 1.0
                q_probs = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
                q_values = np.zeros_like(q_probs)
            quantiles[variable] = q_probs
            quantile_values[variable] = q_values

        discrete_distributions: dict[str, dict[int, float]] = {}
        for variable in discrete_vars:
            values = data[variable].to_numpy()
            distribution: dict[int, float] = {}
            for category in np.unique(values):
                distribution[int(category)] = float(np.sum(weights[values == category]))
            discrete_distributions[variable] = distribution

        numeric_data = data[variables].apply(pd.to_numeric, errors="coerce")
        correlation_matrix = numeric_data.corr(method="spearman").fillna(0).to_numpy()

        return cls(
            variables=variables,
            means=means,
            stds=stds,
            quantiles=quantiles,
            zero_fractions=zero_fractions,
            discrete_vars=discrete_vars,
            discrete_distributions=discrete_distributions,
            correlation_matrix=correlation_matrix,
            continuous_vars=continuous_vars,
            quantile_values=quantile_values,
            min_values=min_values,
            max_values=max_values,
        )


class CPSSyntheticGenerator:
    """Gaussian-copula generator over CPS summary statistics."""

    def __init__(self, stats: CPSSummaryStats):
        self.stats = stats
        correlation = stats.correlation_matrix.copy()
        np.fill_diagonal(correlation, 1.0)
        min_eigenvalue = np.min(np.linalg.eigvalsh(correlation))
        if min_eigenvalue < 1e-6:
            correlation = correlation + (1e-6 - min_eigenvalue) * np.eye(correlation.shape[0])
        self.cholesky = np.linalg.cholesky(correlation)
        self._build_marginal_transforms()

    def _build_marginal_transforms(self) -> None:
        self.marginal_transforms: dict[str, interp1d] = {}
        for variable in self.stats.continuous_vars:
            probabilities = np.concatenate(
                [[0.0], self.stats.quantiles[variable], [1.0]]
            )
            values = np.concatenate(
                [
                    [self.stats.min_values[variable]],
                    self.stats.quantile_values[variable],
                    [self.stats.max_values[variable]],
                ]
            )
            unique_mask = np.concatenate([[True], np.diff(probabilities) > 1e-10])
            self.marginal_transforms[variable] = interp1d(
                probabilities[unique_mask],
                values[unique_mask],
                kind="linear",
                bounds_error=False,
                fill_value=(values[unique_mask][0], values[unique_mask][-1]),
            )

    def generate(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if seed is not None:
            np.random.seed(seed)
        z = np.random.standard_normal((n, len(self.stats.variables)))
        z_correlated = z @ self.cholesky.T
        uniforms = scipy_stats.norm.cdf(z_correlated)

        result: dict[str, np.ndarray] = {}
        for index, variable in enumerate(self.stats.variables):
            if variable in self.stats.discrete_vars:
                result[variable] = self._sample_discrete(variable, uniforms[:, index])
            else:
                result[variable] = self._transform_continuous(variable, uniforms[:, index])
        return pd.DataFrame(result)

    def _sample_discrete(self, variable: str, uniforms: np.ndarray) -> np.ndarray:
        distribution = self.stats.discrete_distributions[variable]
        categories = sorted(distribution)
        probabilities = np.array([distribution[category] for category in categories], dtype=float)
        probabilities = probabilities / probabilities.sum()
        cdf = np.cumsum(probabilities)

        result = np.zeros(len(uniforms), dtype=int)
        for index, category in enumerate(categories):
            lower = cdf[index - 1] if index > 0 else 0.0
            mask = (uniforms > lower) & (uniforms <= cdf[index])
            result[mask] = category
        result[uniforms > cdf[-1]] = categories[-1]
        return result

    def _transform_continuous(self, variable: str, uniforms: np.ndarray) -> np.ndarray:
        zero_fraction = self.stats.zero_fractions.get(variable, 0.0)
        if zero_fraction <= 0:
            return self.marginal_transforms[variable](uniforms)

        result = np.zeros(len(uniforms), dtype=float)
        positive_mask = uniforms >= zero_fraction
        if positive_mask.any():
            positive_uniforms = (uniforms[positive_mask] - zero_fraction) / (1 - zero_fraction)
            positive_uniforms = np.clip(positive_uniforms, 0, 1)
            result[positive_mask] = self.marginal_transforms[variable](positive_uniforms)
        return result


def validate_synthetic(
    reference: pd.DataFrame,
    synthetic: pd.DataFrame,
    variables: list[str] | None = None,
) -> dict[str, dict[str, float] | float]:
    """Compare synthetic data to a CPS-like reference table."""
    if variables is None:
        variables = [column for column in reference.columns if column in synthetic.columns]

    metrics: dict[str, dict[str, float] | float] = {
        "ks_statistics": {},
        "mean_errors": {},
        "std_errors": {},
        "correlation_errors": {},
    }

    for variable in variables:
        reference_values = reference[variable].dropna().to_numpy()
        synthetic_values = synthetic[variable].dropna().to_numpy()
        if len(reference_values) == 0 or len(synthetic_values) == 0:
            continue

        ks_statistic, _ = scipy_stats.ks_2samp(reference_values, synthetic_values)
        metrics["ks_statistics"][variable] = float(ks_statistic)

        reference_mean = float(np.mean(reference_values))
        synthetic_mean = float(np.mean(synthetic_values))
        metrics["mean_errors"][variable] = (
            abs(synthetic_mean - reference_mean) / abs(reference_mean)
            if reference_mean != 0
            else abs(synthetic_mean)
        )

        reference_std = float(np.std(reference_values))
        synthetic_std = float(np.std(synthetic_values))
        metrics["std_errors"][variable] = (
            abs(synthetic_std - reference_std) / reference_std
            if reference_std != 0
            else abs(synthetic_std)
        )

    numeric_vars = [variable for variable in variables if variable in reference.columns and variable in synthetic.columns]
    if len(numeric_vars) >= 2:
        reference_corr = reference[numeric_vars].corr(method="spearman").fillna(0)
        synthetic_corr = synthetic[numeric_vars].corr(method="spearman").fillna(0)
        for index, left in enumerate(numeric_vars):
            for right in numeric_vars[index + 1:]:
                pair = f"{left}_vs_{right}"
                metrics["correlation_errors"][pair] = abs(
                    float(synthetic_corr.loc[left, right]) - float(reference_corr.loc[left, right])
                )

    metrics["mean_ks"] = (
        float(np.mean(list(metrics["ks_statistics"].values())))
        if metrics["ks_statistics"]
        else 0.0
    )
    metrics["mean_corr_error"] = (
        float(np.mean(list(metrics["correlation_errors"].values())))
        if metrics["correlation_errors"]
        else 0.0
    )
    return metrics


__all__ = [
    "CPSSummaryStats",
    "CPSSyntheticGenerator",
    "validate_synthetic",
]
