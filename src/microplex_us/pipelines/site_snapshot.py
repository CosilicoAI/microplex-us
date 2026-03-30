"""Canonical site-snapshot helpers for saved US microplex artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from microplex.targets import assert_valid_benchmark_artifact_manifest

FOCUS_TAG_PRIORITY: tuple[str, ...] = (
    "state",
    "local",
    "parity",
    "all_targets",
    "national",
    "tax",
    "benchmark",
)


def build_us_microplex_site_snapshot(
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Build one site-facing snapshot from a versioned US artifact bundle."""
    artifact_root = Path(artifact_dir)
    manifest = json.loads((artifact_root / "manifest.json").read_text())
    assert_valid_benchmark_artifact_manifest(
        manifest,
        artifact_dir=artifact_root,
        manifest_path=artifact_root / "manifest.json",
        summary_section="policyengine_harness",
        required_artifact_keys=(
            "seed_data",
            "synthetic_data",
            "calibrated_data",
            "targets",
            "policyengine_harness",
        ),
        required_summary_keys=(
            "candidate_mean_abs_relative_error",
            "baseline_mean_abs_relative_error",
            "mean_abs_relative_error_delta",
        ),
    )
    harness = json.loads((artifact_root / "policyengine_harness.json").read_text())
    summary = dict(harness.get("summary", {}))
    tag_summaries = {
        key: dict(value)
        for key, value in dict(summary.get("tag_summaries", {})).items()
    }
    focus_tag = _select_focus_tag(tag_summaries)
    focus_summary = tag_summaries.get(focus_tag, summary)
    synthesis = dict(manifest.get("synthesis", {}))
    calibration = dict(manifest.get("calibration", {}))
    config = dict(manifest.get("config", {}))

    return {
        "generatedAt": manifest.get("created_at"),
        "sourceArtifact": {
            "artifactDir": str(artifact_root.resolve()),
            "manifestPath": str((artifact_root / "manifest.json").resolve()),
            "harnessPath": str((artifact_root / "policyengine_harness.json").resolve()),
            "versionId": artifact_root.name,
        },
        "currentRun": {
            "id": artifact_root.name,
            "benchmarkTag": focus_tag,
            "nSynthetic": config.get("n_synthetic"),
            "scaffoldSource": synthesis.get("scaffold_source"),
            "candidateMeanAbsRelativeError": focus_summary.get(
                "candidate_mean_abs_relative_error"
            ),
            "baselineMeanAbsRelativeError": focus_summary.get(
                "baseline_mean_abs_relative_error"
            ),
            "meanAbsRelativeErrorDelta": focus_summary.get(
                "mean_abs_relative_error_delta"
            ),
            "candidateCompositeParityLoss": focus_summary.get(
                "candidate_composite_parity_loss"
            ),
            "baselineCompositeParityLoss": focus_summary.get(
                "baseline_composite_parity_loss"
            ),
            "targetWinRate": focus_summary.get("target_win_rate"),
            "sliceWinRate": focus_summary.get("slice_win_rate"),
            "supportedTargetRate": focus_summary.get("supported_target_rate"),
            "calibration": {
                "loadedTargets": calibration.get("n_loaded_targets"),
                "supportedTargets": calibration.get("n_supported_targets"),
                "converged": calibration.get("converged"),
                "weightCollapseSuspected": calibration.get(
                    "weight_collapse_suspected"
                ),
                "householdEffectiveSampleSize": _nested_metric(
                    calibration,
                    "household_weight_diagnostics",
                    "effective_sample_size",
                ),
                "personEffectiveSampleSize": _nested_metric(
                    calibration,
                    "person_weight_diagnostics",
                    "effective_sample_size",
                ),
                "householdTinyWeightShare": _nested_metric(
                    calibration,
                    "household_weight_diagnostics",
                    "tiny_share",
                ),
                "personTinyWeightShare": _nested_metric(
                    calibration,
                    "person_weight_diagnostics",
                    "tiny_share",
                ),
            },
            "supportProxies": dict(
                synthesis.get(
                    "state_program_support_proxies",
                    {"available": [], "missing": []},
                )
            ),
            "availableTags": list(tag_summaries.keys()),
        },
        "summary": summary,
        "tagSummaries": tag_summaries,
        "parityScorecard": {
            key: dict(value)
            for key, value in dict(summary.get("parity_scorecard", {})).items()
        },
        "attributeCellSummaries": {
            key: dict(value)
            for key, value in dict(summary.get("attribute_cell_summaries", {})).items()
        },
    }


def write_us_microplex_site_snapshot(
    artifact_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Write the canonical US site snapshot JSON for one saved artifact bundle."""
    snapshot = build_us_microplex_site_snapshot(artifact_dir)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    return destination


def _nested_metric(
    payload: dict[str, Any],
    section: str,
    key: str,
) -> float | int | None:
    section_payload = payload.get(section)
    if not isinstance(section_payload, dict):
        return None
    return section_payload.get(key)


def _select_focus_tag(tag_summaries: dict[str, dict[str, Any]]) -> str:
    for candidate in FOCUS_TAG_PRIORITY:
        if candidate in tag_summaries:
            return candidate
    if tag_summaries:
        return next(iter(tag_summaries))
    return "summary"
