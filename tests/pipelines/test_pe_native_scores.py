"""Tests for PE-native scoring helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

from microplex_us.pipelines.pe_native_scores import (
    PolicyEngineUSEnhancedCPSNativeScores,
    compute_batch_us_pe_native_scores,
    compute_us_pe_native_scores,
    write_us_pe_native_scores,
)


def test_compute_us_pe_native_scores_wraps_broad_loss(monkeypatch, tmp_path) -> None:
    candidate = tmp_path / "candidate.h5"
    baseline = tmp_path / "baseline.h5"
    candidate.write_text("candidate")
    baseline.write_text("baseline")

    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.compute_policyengine_us_enhanced_cps_native_scores",
        lambda *_args, **_kwargs: PolicyEngineUSEnhancedCPSNativeScores(
            metric="enhanced_cps_native_loss",
            period=2024,
            candidate_dataset=str(candidate),
            baseline_dataset=str(baseline),
            candidate_enhanced_cps_native_loss=0.25,
            baseline_enhanced_cps_native_loss=0.5,
            enhanced_cps_native_loss_delta=-0.25,
            candidate_unweighted_msre=0.3,
            baseline_unweighted_msre=0.6,
            unweighted_msre_delta=-0.3,
            n_targets_total=2863,
            n_targets_kept=2853,
            n_targets_zero_dropped=10,
            n_targets_bad_dropped=10,
            n_national_targets=2000,
            n_state_targets=853,
            candidate_weight_sum=100.0,
            baseline_weight_sum=200.0,
            family_breakdown=(
                {
                    "family": "state_age_distribution",
                    "n_targets": 900,
                    "candidate_loss_contribution": 0.1,
                    "baseline_loss_contribution": 0.05,
                    "loss_contribution_delta": 0.05,
                    "candidate_mean_weighted_loss": 0.2,
                    "baseline_mean_weighted_loss": 0.1,
                    "candidate_mean_unweighted_msre": 0.3,
                    "baseline_mean_unweighted_msre": 0.2,
                    "unweighted_msre_delta": 0.1,
                },
            ),
        ),
    )

    payload = compute_us_pe_native_scores(
        candidate_dataset_path=candidate,
        baseline_dataset_path=baseline,
        period=2024,
    )

    assert payload["metric"] == "enhanced_cps_native_loss"
    assert payload["summary"]["candidate_enhanced_cps_native_loss"] == 0.25
    assert payload["summary"]["baseline_enhanced_cps_native_loss"] == 0.5
    assert payload["summary"]["enhanced_cps_native_loss_delta"] == -0.25
    assert payload["summary"]["candidate_beats_baseline"] is True
    assert payload["summary"]["candidate_unweighted_msre"] == 0.3
    assert payload["summary"]["n_targets_kept"] == 2853
    assert payload["family_breakdown"][0]["family"] == "state_age_distribution"
    assert payload["broad_loss"]["family_breakdown"][0]["n_targets"] == 900


def test_write_us_pe_native_scores_persists_payload(monkeypatch, tmp_path) -> None:
    candidate = tmp_path / "candidate.h5"
    baseline = tmp_path / "baseline.h5"
    output_path = tmp_path / "native.json"
    candidate.write_text("candidate")
    baseline.write_text("baseline")

    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.compute_us_pe_native_scores",
        lambda **_kwargs: {
            "metric": "enhanced_cps_native_loss",
            "summary": {
                "candidate_enhanced_cps_native_loss": 0.2,
                "baseline_enhanced_cps_native_loss": 0.4,
                "enhanced_cps_native_loss_delta": -0.2,
            },
        },
    )

    written = write_us_pe_native_scores(
        output_path,
        candidate_dataset_path=candidate,
        baseline_dataset_path=baseline,
    )

    assert written == output_path
    assert output_path.exists()


def test_compute_batch_us_pe_native_scores_wraps_multiple_candidates(
    monkeypatch,
    tmp_path,
) -> None:
    candidate_a = tmp_path / "candidate-a.h5"
    candidate_b = tmp_path / "candidate-b.h5"
    baseline = tmp_path / "baseline.h5"
    for path in (candidate_a, candidate_b, baseline):
        path.write_text(path.stem)

    payload = [
        {
            "metric": "enhanced_cps_native_loss",
            "period": 2024,
            "candidate_dataset": str(candidate_a),
            "baseline_dataset": str(baseline),
            "candidate_enhanced_cps_native_loss": 0.25,
            "baseline_enhanced_cps_native_loss": 0.5,
            "enhanced_cps_native_loss_delta": -0.25,
            "candidate_beats_baseline": True,
            "candidate_unweighted_msre": 0.3,
            "baseline_unweighted_msre": 0.6,
            "unweighted_msre_delta": -0.3,
            "n_targets_total": 2865,
            "n_targets_kept": 2853,
            "n_targets_zero_dropped": 10,
            "n_targets_bad_dropped": 10,
            "n_national_targets": 677,
            "n_state_targets": 2176,
            "candidate_weight_sum": 100.0,
            "baseline_weight_sum": 200.0,
            "family_breakdown": [
                {
                    "family": "state_age_distribution",
                    "n_targets": 900,
                    "candidate_loss_contribution": 0.1,
                    "baseline_loss_contribution": 0.05,
                    "loss_contribution_delta": 0.05,
                    "candidate_mean_weighted_loss": 0.2,
                    "baseline_mean_weighted_loss": 0.1,
                    "candidate_mean_unweighted_msre": 0.3,
                    "baseline_mean_unweighted_msre": 0.2,
                    "unweighted_msre_delta": 0.1,
                }
            ],
        },
        {
            "metric": "enhanced_cps_native_loss",
            "period": 2024,
            "candidate_dataset": str(candidate_b),
            "baseline_dataset": str(baseline),
            "candidate_enhanced_cps_native_loss": 0.75,
            "baseline_enhanced_cps_native_loss": 0.5,
            "enhanced_cps_native_loss_delta": 0.25,
            "candidate_beats_baseline": False,
            "candidate_unweighted_msre": 0.8,
            "baseline_unweighted_msre": 0.6,
            "unweighted_msre_delta": 0.2,
            "n_targets_total": 2865,
            "n_targets_kept": 2853,
            "n_targets_zero_dropped": 10,
            "n_targets_bad_dropped": 10,
            "n_national_targets": 677,
            "n_state_targets": 2176,
            "candidate_weight_sum": 120.0,
            "baseline_weight_sum": 200.0,
            "family_breakdown": [
                {
                    "family": "state_agi_distribution",
                    "n_targets": 918,
                    "candidate_loss_contribution": 0.2,
                    "baseline_loss_contribution": 0.05,
                    "loss_contribution_delta": 0.15,
                    "candidate_mean_weighted_loss": 0.4,
                    "baseline_mean_weighted_loss": 0.1,
                    "candidate_mean_unweighted_msre": 0.5,
                    "baseline_mean_unweighted_msre": 0.2,
                    "unweighted_msre_delta": 0.3,
                }
            ],
        },
    ]

    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.resolve_policyengine_us_data_repo_root",
        lambda _repo=None: tmp_path,
    )

    results = compute_batch_us_pe_native_scores(
        candidate_dataset_paths=[candidate_a, candidate_b],
        baseline_dataset_path=baseline,
        period=2024,
        policyengine_us_data_repo=tmp_path,
        policyengine_us_data_python=tmp_path / "python",
    )

    assert len(results) == 2
    assert results[0]["summary"]["candidate_beats_baseline"] is True
    assert results[1]["summary"]["candidate_beats_baseline"] is False
    assert results[1]["broad_loss"]["enhanced_cps_native_loss_delta"] == 0.25
    assert results[0]["family_breakdown"][0]["family"] == "state_age_distribution"
    assert results[1]["broad_loss"]["family_breakdown"][0]["family"] == "state_agi_distribution"
