from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from microplex_us.data_sources.family_imputation_benchmark import (
    DecomposableFamilyBenchmarkSpec,
    benchmark_decomposable_family_imputers,
    reconcile_component_predictions_to_total,
)


def _toy_family_frame() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for _ in range(20):
        rows.append(
            {
                "age_bucket": "child",
                "age": 12.0,
                "is_male": 0.0,
                "weight": 1.0,
                "social_security": 100.0,
                "social_security_retirement": 0.0,
                "social_security_disability": 0.0,
                "social_security_survivors": 0.0,
                "social_security_dependents": 100.0,
            }
        )
        rows.append(
            {
                "age_bucket": "working",
                "age": 45.0,
                "is_male": 1.0,
                "weight": 1.0,
                "social_security": 100.0,
                "social_security_retirement": 0.0,
                "social_security_disability": 100.0,
                "social_security_survivors": 0.0,
                "social_security_dependents": 0.0,
            }
        )
        rows.append(
            {
                "age_bucket": "senior",
                "age": 72.0,
                "is_male": 0.0,
                "weight": 1.0,
                "social_security": 100.0,
                "social_security_retirement": 100.0,
                "social_security_disability": 0.0,
                "social_security_survivors": 0.0,
                "social_security_dependents": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_reconcile_component_predictions_to_total_respects_total_and_fallback():
    predicted = pd.DataFrame(
        {
            "ret": [1.0, 0.0],
            "dis": [3.0, 0.0],
            "surv": [0.0, 0.0],
            "dep": [0.0, 0.0],
        }
    )
    total = pd.Series([20.0, 10.0], dtype=float)
    reconciled = reconcile_component_predictions_to_total(
        predicted,
        family_total=total,
        component_columns=("ret", "dis", "surv", "dep"),
        fallback_shares={"ret": 0.25, "dis": 0.25, "surv": 0.25, "dep": 0.25},
    )

    assert reconciled.sum(axis=1).tolist() == [20.0, 10.0]
    assert reconciled.iloc[0]["ret"] == 5.0
    assert reconciled.iloc[0]["dis"] == 15.0
    assert reconciled.iloc[1]["dep"] == 2.5


def test_grouped_share_benchmark_is_exact_on_group_determined_family():
    frame = _toy_family_frame()
    report = benchmark_decomposable_family_imputers(
        frame,
        spec=DecomposableFamilyBenchmarkSpec(
            total_column="social_security",
            component_columns=(
                "social_security_retirement",
                "social_security_disability",
                "social_security_survivors",
                "social_security_dependents",
            ),
            grouped_feature_sets=(("age_bucket",),),
            qrf_condition_vars=("age", "is_male", "social_security"),
            implicit_component_column="social_security_dependents",
            group_eval_columns=("age_bucket",),
            reweight_feature_sets=(("age_bucket",),),
            reweight_initial_weight_mode="uniform",
            qrf_n_estimators=20,
        ),
        train_frac=0.75,
        target_frac=0.1,
        random_seed=42,
    )

    grouped = report.methods["grouped_share"]
    forest = report.methods["forest_share"]
    assert report.train_row_count + report.eval_row_count + report.target_row_count == report.row_count
    assert grouped.component_group_sum_mare["social_security_retirement"] == 0.0
    assert grouped.component_group_sum_mare["social_security_disability"] == 0.0
    assert grouped.component_group_sum_mare["social_security_dependents"] == 0.0
    assert grouped.post_reweight_mean_component_total_relative_error == pytest.approx(0.0)
    assert grouped.post_reweight_mean_component_group_sum_mare == pytest.approx(0.0)
    assert grouped.post_reweight_mean_component_total_error_degradation == pytest.approx(0.0)
    assert grouped.oracle_post_reweight_mean_component_total_relative_error == pytest.approx(0.0)
    assert grouped.post_reweight_mean_component_total_error_excess_over_oracle == pytest.approx(0.0)
    assert grouped.reweighting_summary is not None
    assert grouped.reweighting_summary["initial_weight_mode"] == "uniform"
    assert grouped.reweighting_summary["target_row_count"] == report.target_row_count
    assert grouped.reweighting_summary["eval_row_count"] == report.eval_row_count
    assert forest.component_group_sum_mare["social_security_retirement"] < 1.0
    assert forest.component_group_sum_mare["social_security_disability"] < 1.0
    assert forest.component_group_sum_mare["social_security_dependents"] < 1.0
    assert forest.post_reweight_mean_component_total_relative_error is not None
    assert forest.oracle_post_reweight_mean_component_total_relative_error is not None
    assert forest.post_reweight_mean_component_total_error_excess_over_oracle is not None


@pytest.mark.skipif(
    importlib.util.find_spec("quantile_forest") is None,
    reason="quantile_forest not installed",
)
def test_qrf_benchmark_returns_expected_metric_surface():
    frame = _toy_family_frame()
    report = benchmark_decomposable_family_imputers(
        frame,
        spec=DecomposableFamilyBenchmarkSpec(
            total_column="social_security",
            component_columns=(
                "social_security_retirement",
                "social_security_disability",
                "social_security_survivors",
                "social_security_dependents",
            ),
            grouped_feature_sets=(("age_bucket",),),
            qrf_condition_vars=("age", "is_male", "social_security"),
            implicit_component_column="social_security_dependents",
            group_eval_columns=("age_bucket",),
            reweight_feature_sets=(("age_bucket",),),
            qrf_n_estimators=20,
        ),
        train_frac=0.75,
        target_frac=0.1,
        random_seed=1,
    )

    qrf = report.methods["qrf"]
    forest = report.methods["forest_share"]
    assert set(qrf.component_total_relative_error) == {
        "social_security_retirement",
        "social_security_disability",
        "social_security_survivors",
        "social_security_dependents",
    }
    assert qrf.mean_component_total_relative_error >= 0.0
    assert qrf.post_reweight_mean_component_total_relative_error is not None
    assert qrf.oracle_post_reweight_mean_component_total_relative_error is not None
    assert qrf.post_reweight_mean_component_total_error_degradation is not None
    assert set(forest.component_total_relative_error) == {
        "social_security_retirement",
        "social_security_disability",
        "social_security_survivors",
        "social_security_dependents",
    }
    assert forest.mean_component_total_relative_error >= 0.0
    assert forest.post_reweight_mean_component_total_relative_error is not None
    assert forest.oracle_post_reweight_mean_component_total_relative_error is not None
