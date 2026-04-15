from __future__ import annotations

import pandas as pd

from microplex_us.data_sources.share_imputation import (
    fit_grouped_share_model,
    predict_grouped_component_shares,
)


def test_grouped_share_model_predicts_specific_group_shares():
    reference = pd.DataFrame(
        {
            "age_bucket": ["young", "young", "old", "old"],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "ret": [0.0, 0.0, 1.0, 1.0],
            "dis": [1.0, 1.0, 0.0, 0.0],
            "surv": [0.0, 0.0, 0.0, 0.0],
            "dep": [0.0, 0.0, 0.0, 0.0],
        }
    )
    model = fit_grouped_share_model(
        reference,
        explicit_component_columns=("ret", "dis", "surv"),
        implicit_component_column="dep",
        feature_sets=(("age_bucket",),),
        weight_column="weight",
    )

    target = pd.DataFrame({"age_bucket": ["young", "old"]})
    shares = predict_grouped_component_shares(target, model)

    assert shares["ret"].tolist() == [0.0, 1.0]
    assert shares["dis"].tolist() == [1.0, 0.0]
    assert shares["surv"].tolist() == [0.0, 0.0]
    assert shares["dep"].tolist() == [0.0, 0.0]


def test_grouped_share_model_falls_back_to_overall_and_keeps_mece():
    reference = pd.DataFrame(
        {
            "age_bucket": ["young", "old"],
            "weight": [3.0, 1.0],
            "ret": [0.0, 1.0],
            "dis": [1.0, 0.0],
            "surv": [0.0, 0.0],
            "dep": [0.0, 0.0],
        }
    )
    model = fit_grouped_share_model(
        reference,
        explicit_component_columns=("ret", "dis", "surv"),
        implicit_component_column="dep",
        feature_sets=(("age_bucket",),),
        weight_column="weight",
    )

    target = pd.DataFrame({"age_bucket": ["missing"]})
    shares = predict_grouped_component_shares(target, model)

    assert shares["ret"].iloc[0] == 0.25
    assert shares["dis"].iloc[0] == 0.75
    assert shares["surv"].iloc[0] == 0.0
    assert shares["dep"].iloc[0] == 0.0
    assert shares.sum(axis=1).iloc[0] == 1.0
