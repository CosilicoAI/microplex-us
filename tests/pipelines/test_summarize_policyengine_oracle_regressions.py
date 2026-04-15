"""Tests for calibration-oracle regression summary helpers."""

from __future__ import annotations

import json
from pathlib import Path

from microplex_us.pipelines.summarize_policyengine_oracle_regressions import (
    summarize_us_policyengine_oracle_regressions,
)


def test_summarize_us_policyengine_oracle_regressions_aggregates_groups(tmp_path) -> None:
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    _write_oracle_bundle(
        root_a / "run-1",
        scope_capped_loss=2.5,
        families=[
            ("person_count|domain=age", 0.4),
            ("tax_unit_count|domain=salt", 0.3),
            ("aca_ptc|domain=aca_ptc", 0.1),
        ],
        geographies=[
            ("state:OR", 0.5),
            ("state:GA", 0.2),
            ("nation", 0.1),
        ],
    )
    _write_oracle_bundle(
        root_a / "run-2",
        scope_capped_loss=1.5,
        families=[
            ("tax_unit_count|domain=salt", 0.6),
            ("person_count|domain=age", 0.2),
        ],
        geographies=[
            ("state:GA", 0.4),
            ("state:OR", 0.1),
        ],
    )
    _write_oracle_bundle(
        root_b / "run-3",
        scope_capped_loss=0.5,
        families=[
            ("aca_ptc|domain=aca_ptc", 0.5),
            ("person_count|domain=age", 0.4),
            ("tax_unit_count|domain=salt", 0.3),
        ],
        geographies=[
            ("state:OR", 0.6),
            ("state:CA", 0.2),
            ("state:GA", 0.1),
        ],
    )

    summary = summarize_us_policyengine_oracle_regressions([root_a, root_b], top_k=2)

    assert summary["lossScope"] == "full_oracle"
    assert summary["totalScoredRuns"] == 3
    assert summary["largestFamilyCounts"] == [
        {"group": "aca_ptc|domain=aca_ptc", "count": 1},
        {"group": "person_count|domain=age", "count": 1},
        {"group": "tax_unit_count|domain=salt", "count": 1},
    ]
    assert summary["largestGeographyCounts"] == [
        {"group": "state:OR", "count": 2},
        {"group": "state:GA", "count": 1},
    ]
    assert summary["top3FamilyCounts"][0] == {
        "group": "person_count|domain=age",
        "top3Count": 3,
        "rank1Count": 1,
        "rank2Count": 2,
        "rank3Count": 0,
    }
    assert summary["top3GeographyCounts"][0] == {
        "group": "state:GA",
        "top3Count": 3,
        "rank1Count": 1,
        "rank2Count": 1,
        "rank3Count": 1,
    }
    assert summary["familyCountsByRoot"]["root-a"] == [
        {"group": "person_count|domain=age", "count": 2},
        {"group": "tax_unit_count|domain=salt", "count": 2},
        {"group": "aca_ptc|domain=aca_ptc", "count": 1},
    ]
    assert summary["geographyCountsByRoot"]["root-a"] == [
        {"group": "state:GA", "count": 2},
        {"group": "state:OR", "count": 2},
        {"group": "nation", "count": 1},
    ]
    assert summary["worstRuns"][0]["artifactPath"] == "run-1"
    assert summary["bestRuns"][0]["artifactPath"] == "run-3"


def test_summarize_us_policyengine_oracle_regressions_ignores_non_bundle_manifests(
    tmp_path,
) -> None:
    root = tmp_path / "root"
    stray = root / "stray"
    stray.mkdir(parents=True)
    (stray / "manifest.json").write_text(
        json.dumps(
            {
                "calibration": {
                    "oracle_loss": {
                        "full_oracle": {
                            "capped_mean_abs_relative_error": 99.0,
                            "family_ranking": [],
                            "geography_ranking": [],
                        }
                    }
                }
            }
        )
    )

    _write_oracle_bundle(
        root / "run-1",
        scope_capped_loss=1.0,
        families=[("person_count|domain=age", 0.9)],
        geographies=[("state:OR", 0.9)],
    )

    summary = summarize_us_policyengine_oracle_regressions([root])

    assert summary["totalScoredRuns"] == 1
    assert summary["worstRuns"][0]["artifactPath"] == "run-1"


def _write_oracle_bundle(
    bundle_dir: Path,
    *,
    scope_capped_loss: float,
    families: list[tuple[str, float]],
    geographies: list[tuple[str, float]],
) -> None:
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "policyengine_us.h5").write_text("dataset")
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "calibration": {
                    "active_solve_capped_mean_abs_relative_error": scope_capped_loss
                    / 2.0,
                    "n_constraints": 10,
                    "n_supported_targets": 100,
                    "n_unsupported_targets": 0,
                    "n_calibration_stages_applied": 1,
                    "oracle_loss": {
                        "full_oracle": {
                            "capped_mean_abs_relative_error": scope_capped_loss,
                            "mean_abs_relative_error": scope_capped_loss * 2.0,
                            "family_ranking": [
                                {
                                    "group": group,
                                    "capped_loss_share": share,
                                    "capped_sum_abs_relative_error": share * 100.0,
                                }
                                for group, share in families
                            ],
                            "geography_ranking": [
                                {
                                    "group": group,
                                    "capped_loss_share": share,
                                    "capped_sum_abs_relative_error": share * 100.0,
                                }
                                for group, share in geographies
                            ],
                        }
                    },
                }
            }
        )
    )
