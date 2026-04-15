"""Tests for imputation ablation scorecards."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from microplex_us.pipelines.imputation_ablation import (
    ImputationAblationSliceSpec,
    ImputationAblationVariant,
    default_imputation_ablation_variants,
    score_imputation_ablation_variants,
)


def test_score_imputation_ablation_variants_ranks_structured_candidate() -> None:
    observed = pd.DataFrame(
        {
            "age_band": ["adult", "adult", "senior", "senior"],
            "tax_unit_is_joint": [0, 1, 0, 1],
            "weight": [1.0, 2.0, 1.0, 2.0],
            "employment_income": [0.0, 100.0, 0.0, 200.0],
        }
    )
    broad = observed.copy()
    broad["employment_income"] = [50.0, 50.0, 50.0, 50.0]
    structured = observed.copy()
    structured["employment_income"] = [0.0, 90.0, 0.0, 210.0]

    report = score_imputation_ablation_variants(
        observed_frame=observed,
        imputed_frames={
            "broad_common_qrf": broad,
            "structured_pe_conditioning": structured,
        },
        target_variables=("employment_income",),
        slice_specs=(
            ImputationAblationSliceSpec(
                name="age_by_joint",
                columns=("age_band", "tax_unit_is_joint"),
            ),
        ),
        weight_column="weight",
    )

    broad_metrics = report.variants["broad_common_qrf"].aggregate_metrics
    structured_metrics = report.variants["structured_pe_conditioning"].aggregate_metrics
    assert structured_metrics["mean_weighted_mae"] < broad_metrics["mean_weighted_mae"]
    assert structured_metrics["mean_support_f1"] > broad_metrics["mean_support_f1"]
    assert (
        structured_metrics["mean_slice_positive_rate_delta"]
        < broad_metrics["mean_slice_positive_rate_delta"]
    )
    assert report.variants["structured_pe_conditioning"].variant.semantic_guards is True


def test_score_imputation_ablation_report_is_json_serializable() -> None:
    observed = pd.DataFrame(
        {
            "slice": ["a", "a", "b"],
            "weight": [1.0, 1.0, 1.0],
            "rent": [0.0, 100.0, 200.0],
        }
    )
    imputed = observed.copy()
    imputed["rent"] = [0.0, 120.0, 180.0]

    report = score_imputation_ablation_variants(
        observed_frame=observed,
        imputed_frames={"custom": imputed},
        variants=(
            ImputationAblationVariant(
                name="custom",
                description="custom test",
                condition_selection="fixture",
            ),
        ),
        target_variables=("rent",),
        slice_specs=(ImputationAblationSliceSpec(name="slice", columns=("slice",)),),
        weight_column="weight",
        post_calibration_metrics={
            "custom": {
                "post_calibration_native_loss": 0.4,
                "household_effective_sample_size": 2.5,
            }
        },
    )

    payload = report.to_dict()
    assert payload["variants"]["custom"]["post_calibration_metrics"] == {
        "post_calibration_native_loss": 0.4,
        "household_effective_sample_size": 2.5,
    }
    json.dumps(payload)


def test_score_imputation_ablation_scores_matching_non_default_indexes() -> None:
    observed = pd.DataFrame(
        {
            "slice": ["a", "b"],
            "weight": [1.0, 1.0],
            "rent": [100.0, 200.0],
        },
        index=[10, 11],
    )
    imputed = pd.DataFrame(
        {
            "rent": [100.0, 200.0],
        },
        index=[10, 11],
    )

    report = score_imputation_ablation_variants(
        observed_frame=observed,
        imputed_frames={"candidate": imputed},
        target_variables=("rent",),
        slice_specs=(ImputationAblationSliceSpec(name="slice", columns=("slice",)),),
        weight_column="weight",
    )

    score = report.variants["candidate"].target_scores["rent"]
    assert score.mean_absolute_error == 0.0
    assert score.weighted_mean_absolute_error == 0.0
    assert score.weighted_total_relative_error == 0.0
    slice_score = report.variants["candidate"].slice_scores[0]
    assert slice_score.total_js_divergence == 0.0


def test_score_imputation_ablation_rejects_mismatched_indexes() -> None:
    observed = pd.DataFrame({"weight": [1.0, 1.0], "rent": [100.0, 200.0]})
    imputed = pd.DataFrame(
        {
            "rent": [100.0, 200.0],
        },
        index=[10, 11],
    )

    with pytest.raises(ValueError, match="matching indexes"):
        score_imputation_ablation_variants(
            observed_frame=observed,
            imputed_frames={"candidate": imputed},
            target_variables=("rent",),
            weight_column="weight",
        )


def test_default_imputation_ablation_variants_encode_hypothesis() -> None:
    variants = {
        variant.name: variant for variant in default_imputation_ablation_variants()
    }

    assert set(variants) == {
        "broad_common_qrf",
        "structured_pe_conditioning",
        "broad_common_with_guards",
        "rich_predictor_stress",
    }
    assert variants["broad_common_qrf"].condition_selection == "all_shared"
    structured = variants["structured_pe_conditioning"]
    assert structured.condition_selection == "pe_prespecified"
    assert "age" in structured.primary_predictors
    assert "tax_unit_is_joint" in structured.hard_gate_columns
    assert structured.support_mapping == "zero_inflated_positive"
