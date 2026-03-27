"""Focused tests for CPS summary-stat synthesis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from microplex_us.cps_synthetic import (
    CPSSummaryStats,
    CPSSyntheticGenerator,
    validate_synthetic,
)


def _sample_reference_data() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 85, size=512),
            "sex": rng.integers(1, 3, size=512),
            "income": np.where(
                rng.random(512) < 0.2,
                0.0,
                rng.lognormal(mean=10.5, sigma=0.6, size=512),
            ),
            "education": rng.integers(1, 5, size=512),
        }
    )


def test_summary_stats_and_generator_round_trip() -> None:
    reference = _sample_reference_data()

    stats = CPSSummaryStats.from_dataframe(reference)
    synthetic = CPSSyntheticGenerator(stats).generate(n=256, seed=77)

    assert set(synthetic.columns) == set(reference.columns)
    assert len(synthetic) == 256
    assert (synthetic["income"] >= 0).all()


def test_validate_synthetic_returns_aggregate_metrics() -> None:
    reference = _sample_reference_data()
    synthetic = reference.sample(n=256, replace=True, random_state=42).reset_index(drop=True)

    metrics = validate_synthetic(reference, synthetic)

    assert "ks_statistics" in metrics
    assert "mean_ks" in metrics
    assert "mean_corr_error" in metrics
    assert metrics["mean_ks"] >= 0
