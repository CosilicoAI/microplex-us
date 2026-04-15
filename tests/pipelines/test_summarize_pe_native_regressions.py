"""Tests for PE-native regression summary helpers."""

from __future__ import annotations

import json
from pathlib import Path

from microplex_us.pipelines.summarize_pe_native_regressions import (
    summarize_us_pe_native_regressions,
)


def test_summarize_us_pe_native_regressions_aggregates_family_counts(tmp_path) -> None:
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    _write_scored_bundle(
        root_a / "run-1",
        loss_delta=2.5,
        families=[
            ("state_agi_distribution", 1.2),
            ("national_irs_other", 0.8),
            ("state_aca_spending", 0.2),
        ],
        audit_family="state_agi_distribution",
        audit_target="state/WI/adjusted_gross_income/count/500000_inf",
    )
    _write_scored_bundle(
        root_a / "run-2",
        loss_delta=1.5,
        families=[
            ("national_irs_other", 1.1),
            ("state_agi_distribution", 0.6),
        ],
    )
    _write_scored_bundle(
        root_b / "run-3",
        loss_delta=0.5,
        families=[
            ("state_aca_spending", 0.4),
            ("national_irs_other", 0.3),
            ("state_agi_distribution", 0.1),
        ],
        audit_family="state_aca_spending",
        audit_target="nation/irs/aca_spending/mi",
        missing_inputs=["has_esi"],
    )

    summary = summarize_us_pe_native_regressions([root_a, root_b], top_k=2)

    assert summary["totalScoredRuns"] == 3
    assert summary["totalAuditedRuns"] == 2
    assert summary["largestFamilyCounts"] == [
        {"family": "national_irs_other", "count": 1},
        {"family": "state_aca_spending", "count": 1},
        {"family": "state_agi_distribution", "count": 1},
    ]
    assert summary["top3FamilyCounts"][0] == {
        "family": "national_irs_other",
        "top3Count": 3,
        "rank1Count": 1,
        "rank2Count": 2,
        "rank3Count": 0,
    }
    assert summary["familyCountsByRoot"]["root-a"] == [
        {"family": "national_irs_other", "count": 2},
        {"family": "state_agi_distribution", "count": 2},
        {"family": "state_aca_spending", "count": 1},
    ]
    assert summary["targetCountsFromAudits"] == [
        {"target": "nation/irs/aca_spending/mi", "count": 1},
        {"target": "state/WI/adjusted_gross_income/count/500000_inf", "count": 1},
    ]
    assert summary["missingCriticalInputsCounts"] == [
        {"variable": "has_esi", "count": 1}
    ]
    assert summary["worstRuns"][0]["artifactPath"] == "run-1"
    assert summary["bestRuns"][0]["artifactPath"] == "run-3"


def test_summarize_us_pe_native_regressions_ignores_non_bundle_scores(tmp_path) -> None:
    root = tmp_path / "root"
    stray = root / "stray"
    stray.mkdir(parents=True)
    (stray / "policyengine_native_scores.json").write_text(
        json.dumps({"summary": {"enhanced_cps_native_loss_delta": 99.0}})
    )

    _write_scored_bundle(
        root / "run-1",
        loss_delta=1.0,
        families=[("national_irs_other", 0.9)],
    )

    summary = summarize_us_pe_native_regressions([root])

    assert summary["totalScoredRuns"] == 1
    assert summary["worstRuns"][0]["artifactPath"] == "run-1"


def _write_scored_bundle(
    bundle_dir: Path,
    *,
    loss_delta: float,
    families: list[tuple[str, float]],
    audit_family: str | None = None,
    audit_target: str | None = None,
    missing_inputs: list[str] | None = None,
) -> None:
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "policyengine_us.h5").write_text("dataset")
    (bundle_dir / "policyengine_native_scores.json").write_text(
        json.dumps(
            {
                "summary": {
                    "enhanced_cps_native_loss_delta": loss_delta,
                    "candidate_beats_baseline": False,
                },
                "family_breakdown": [
                    {
                        "family": family,
                        "loss_contribution_delta": delta,
                    }
                    for family, delta in families
                ],
            }
        )
    )
    if audit_family is None and audit_target is None and not missing_inputs:
        return
    (bundle_dir / "pe_us_data_rebuild_native_audit.json").write_text(
        json.dumps(
            {
                "verdictHints": {
                    "largestRegressingFamily": audit_family,
                    "largestRegressingTarget": audit_target,
                },
                "supportAuditSummary": {
                    "missingStoredCriticalInputs": list(missing_inputs or []),
                },
            }
        )
    )
