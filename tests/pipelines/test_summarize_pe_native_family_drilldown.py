"""Tests for PE-native family drilldown summaries."""

from __future__ import annotations

import json
from pathlib import Path

from microplex_us.pipelines.summarize_pe_native_family_drilldown import (
    classify_pe_native_target_family,
    summarize_us_pe_native_family_drilldown,
)


def test_classify_pe_native_target_family_covers_national_irs_bucket() -> None:
    assert (
        classify_pe_native_target_family(
            "nation/irs/count/count/AGI in 500k-1m/taxable/Single"
        )
        == "national_irs_other"
    )


def test_summarize_us_pe_native_family_drilldown_aggregates_matching_targets(
    tmp_path,
) -> None:
    root = tmp_path / "root"
    _write_audit_bundle(
        root / "run-a",
        largest_regressing_family="national_irs_other",
        top_targets=[
            (
                "nation/irs/ordinary dividends/total/AGI in 500k-1m/taxable/All",
                100.0,
            ),
            (
                "state/WI/adjusted_gross_income/count/500000_inf",
                250.0,
            ),
        ],
        filing_status_gaps=[("JOINT", 1000.0), ("SEPARATE", -250.0)],
        mfs_agi_gaps=[("100k_to_200k", -50.0), ("500k_plus", -10.0)],
    )
    _write_audit_bundle(
        root / "run-b",
        largest_regressing_family="state_agi_distribution",
        top_targets=[
            (
                "nation/irs/ordinary dividends/total/AGI in 500k-1m/taxable/All",
                75.0,
            ),
            (
                "nation/irs/count/count/AGI in 500k-1m/taxable/Single",
                60.0,
            ),
        ],
        filing_status_gaps=[("HEAD_OF_HOUSEHOLD", 500.0)],
        mfs_agi_gaps=[("75k_to_100k", -20.0)],
    )

    summary = summarize_us_pe_native_family_drilldown(
        [root],
        family="national_irs_other",
        top_k=5,
    )

    assert summary["totalAudits"] == 2
    assert summary["auditsWithMatchingTargets"] == 2
    assert summary["auditsWhereFamilyLeads"] == 1
    assert summary["matchingTargetCounts"][0] == {
        "target": "nation/irs/ordinary dividends/total/AGI in 500k-1m/taxable/All",
        "count": 2,
        "weightedTermDeltaSum": 175.0,
        "weightedTermDeltaMean": 87.5,
    }
    assert summary["leadTargetCounts"] == [
        {
            "target": "nation/irs/ordinary dividends/total/AGI in 500k-1m/taxable/All",
            "count": 1,
            "weightedTermDeltaSum": 100.0,
            "weightedTermDeltaMean": 100.0,
        }
    ]
    assert summary["leadFilingStatusGapSummary"] == [
        {
            "filingStatus": "JOINT",
            "count": 1,
            "positiveCount": 1,
            "negativeCount": 0,
            "weightedCountDeltaSum": 1000.0,
            "meanAbsWeightedCountDelta": 1000.0,
        },
        {
            "filingStatus": "SEPARATE",
            "count": 1,
            "positiveCount": 0,
            "negativeCount": 1,
            "weightedCountDeltaSum": -250.0,
            "meanAbsWeightedCountDelta": 250.0,
        },
    ]
    assert summary["leadMFSAgiGapSummary"][0] == {
        "agiBin": "100k_to_200k",
        "count": 1,
        "positiveCount": 0,
        "negativeCount": 1,
        "weightedCountDeltaSum": -50.0,
        "meanAbsWeightedCountDelta": 50.0,
    }
    assert summary["matchingAudits"][0]["artifactPath"] == "run-a"
    assert summary["leadAudits"][0]["artifactPath"] == "run-a"


def test_summarize_us_pe_native_family_drilldown_ignores_non_bundle_audits(tmp_path) -> None:
    root = tmp_path / "root"
    stray = root / "stray"
    stray.mkdir(parents=True)
    (stray / "pe_us_data_rebuild_native_audit.json").write_text(
        json.dumps({"verdictHints": {"largestRegressingFamily": "national_irs_other"}})
    )
    _write_audit_bundle(
        root / "run-a",
        largest_regressing_family="national_irs_other",
        top_targets=[("nation/irs/aca_spending/mi", 10.0)],
    )

    summary = summarize_us_pe_native_family_drilldown(
        [root],
        family="state_aca_spending",
    )

    assert summary["totalAudits"] == 1
    assert summary["auditsWithMatchingTargets"] == 1
    assert summary["matchingAudits"][0]["artifactPath"] == "run-a"


def _write_audit_bundle(
    bundle_dir: Path,
    *,
    largest_regressing_family: str,
    top_targets: list[tuple[str, float]],
    filing_status_gaps: list[tuple[str, float]] | None = None,
    mfs_agi_gaps: list[tuple[str, float]] | None = None,
) -> None:
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "policyengine_us.h5").write_text("dataset")
    (bundle_dir / "pe_us_data_rebuild_native_audit.json").write_text(
        json.dumps(
            {
                "verdictHints": {
                    "largestRegressingFamily": largest_regressing_family,
                    "largestRegressingTarget": top_targets[0][0],
                },
                "topTargetRegressions": [
                    {
                        "target_name": target_name,
                        "weighted_term_delta": weighted_term_delta,
                    }
                    for target_name, weighted_term_delta in top_targets
                ],
                "supportAuditSummary": {
                    "topFilingStatusGaps": [
                        {
                            "filing_status": filing_status,
                            "weighted_count_delta": weighted_count_delta,
                        }
                        for filing_status, weighted_count_delta in list(
                            filing_status_gaps or []
                        )
                    ],
                    "topMFSAgiGaps": [
                        {
                            "agi_bin": agi_bin,
                            "weighted_count_delta": weighted_count_delta,
                        }
                        for agi_bin, weighted_count_delta in list(mfs_agi_gaps or [])
                    ],
                },
            }
        )
    )
