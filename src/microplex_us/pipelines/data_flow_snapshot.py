"""Artifact-backed data-flow snapshot helpers for the US microplex pipeline."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from microplex.core import SourceDescriptor
from microplex.fusion import FusionPlan

from microplex_us.data_sources.cps import (
    CPSASECParquetSourceProvider,
    CPSASECSourceProvider,
)
from microplex_us.data_sources.psid import PSIDSourceProvider
from microplex_us.data_sources.puf import PUFSourceProvider
from microplex_us.source_manifests import load_us_source_manifest
from microplex_us.variables import (
    donor_imputation_block_specs,
    variable_semantic_spec_for,
)

DATA_FLOW_SNAPSHOT_SCHEMA_VERSION = 1


def build_us_microplex_data_flow_snapshot(
    artifact_dir: str | Path,
    *,
    manifest_payload: dict[str, Any] | None = None,
    prefer_saved: bool = True,
) -> dict[str, Any]:
    """Build one site-facing US data-flow snapshot from a saved artifact bundle."""
    artifact_root = Path(artifact_dir)
    if prefer_saved and manifest_payload is None:
        saved_snapshot = _load_saved_data_flow_snapshot(artifact_root)
        if saved_snapshot is not None:
            return saved_snapshot

    return _materialize_us_microplex_data_flow_snapshot(
        artifact_root,
        manifest_payload=manifest_payload,
    )


def require_saved_us_microplex_data_flow_snapshot(
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Load the saved canonical US data-flow snapshot or raise."""
    artifact_root = Path(artifact_dir)
    snapshot_path = artifact_root / "data_flow_snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"US artifact bundle is missing data_flow_snapshot.json: {snapshot_path}"
        )
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get("schemaVersion") != DATA_FLOW_SNAPSHOT_SCHEMA_VERSION:
        raise RuntimeError(
            "US artifact bundle has a stale or unsupported data_flow_snapshot.json "
            f"schema: {snapshot.get('schemaVersion')!r}"
        )
    return snapshot


def write_us_microplex_data_flow_snapshot(
    artifact_dir: str | Path,
    output_path: str | Path,
    *,
    manifest_payload: dict[str, Any] | None = None,
) -> Path:
    """Write the canonical US data-flow snapshot JSON for one saved artifact bundle."""
    snapshot = _materialize_us_microplex_data_flow_snapshot(
        artifact_dir,
        manifest_payload=manifest_payload,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(destination, snapshot)
    return destination


def _materialize_us_microplex_data_flow_snapshot(
    artifact_dir: str | Path,
    *,
    manifest_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_root = Path(artifact_dir)
    manifest = (
        dict(manifest_payload)
        if manifest_payload is not None
        else json.loads((artifact_root / "manifest.json").read_text())
    )
    synthesis = dict(manifest.get("synthesis", {}))
    calibration = dict(manifest.get("calibration", {}))
    artifacts = dict(manifest.get("artifacts", {}))
    config = dict(manifest.get("config", {}))

    source_names = tuple(
        dict.fromkeys(
            value
            for value in (
                *list(synthesis.get("source_names", ())),
                synthesis.get("scaffold_source"),
            )
            if isinstance(value, str) and value
        )
    )
    source_entries = [
        _source_snapshot_entry(source_name)
        for source_name in source_names
    ]
    resolved_descriptors = [
        entry["descriptor"]
        for entry in source_entries
        if entry["descriptor"] is not None
    ]
    fusion_plan = (
        FusionPlan.from_sources(resolved_descriptors)
        if resolved_descriptors
        else None
    )

    donor_integrated_variables = tuple(
        variable
        for variable in synthesis.get("donor_integrated_variables", ())
        if isinstance(variable, str)
    )
    semantic_variables = tuple(
        dict.fromkeys(
            [
                *donor_integrated_variables,
                *[
                    variable
                    for variable in synthesis.get("condition_vars", ())
                    if isinstance(variable, str)
                ],
                *[
                    variable
                    for variable in synthesis.get("target_vars", ())
                    if isinstance(variable, str)
                ],
                *[
                    variable
                    for variable in synthesis.get(
                        "donor_authoritative_override_variables",
                        (),
                    )
                    if isinstance(variable, str)
                ],
            ]
        )
    )

    data_flow_snapshot = {
        "schemaVersion": DATA_FLOW_SNAPSHOT_SCHEMA_VERSION,
        "generatedAt": manifest.get("created_at"),
        "coverageMode": "artifact_frozen",
        "runtime": {
            "sourceNames": list(source_names),
            "scaffoldSource": synthesis.get("scaffold_source"),
            "nSynthetic": config.get("n_synthetic"),
            "rows": {
                key: manifest.get("rows", {}).get(key)
                for key in ("seed", "synthetic", "calibrated")
            },
            "synthesisBackend": synthesis.get("backend"),
            "conditionVars": list(synthesis.get("condition_vars", ())),
            "targetVars": list(synthesis.get("target_vars", ())),
            "donorIntegratedVariables": list(donor_integrated_variables),
            "donorExcludedVariables": list(
                synthesis.get("donor_excluded_variables", ())
            ),
            "donorAuthoritativeOverrideVariables": list(
                synthesis.get("donor_authoritative_override_variables", ())
            ),
            "supportProxies": dict(
                synthesis.get(
                    "state_program_support_proxies",
                    {"available": [], "missing": []},
                )
            ),
        },
        "sources": [
            _serialize_source_snapshot_entry(entry)
            for entry in source_entries
        ],
        "sharedCoverage": _build_shared_coverage_summary(fusion_plan),
        "donorBlocks": _build_donor_block_summary(donor_integrated_variables),
        "semanticHighlights": _build_semantic_highlights(semantic_variables),
        "stages": _build_pipeline_stage_summary(
            synthesis=synthesis,
            calibration=calibration,
            artifacts=artifacts,
            config=config,
            donor_integrated_variables=donor_integrated_variables,
            source_names=source_names,
            manifest=manifest,
        ),
    }
    return data_flow_snapshot


def _load_saved_data_flow_snapshot(artifact_root: Path) -> dict[str, Any] | None:
    snapshot_path = artifact_root / "data_flow_snapshot.json"
    if not snapshot_path.exists():
        return None
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get("schemaVersion") != DATA_FLOW_SNAPSHOT_SCHEMA_VERSION:
        return None
    return snapshot


def _source_snapshot_entry(source_name: str) -> dict[str, Any]:
    descriptor: SourceDescriptor | None = None
    manifest_name: str | None = None
    notes: list[str] = []

    if source_name == "cps_asec_parquet":
        descriptor = replace(
            CPSASECParquetSourceProvider(data_dir=".").descriptor,
            name=source_name,
        )
        notes.append(
            "This source was loaded from split household/person parquet files rather than "
            "the Census download path."
        )
    elif source_name == "cps_asec" or source_name.startswith("cps_asec_"):
        descriptor = replace(CPSASECSourceProvider(download=False).descriptor, name=source_name)
        notes.append(
            "CPS coverage expands at load time from processed household and person tables; "
            "the static provider descriptor intentionally stays minimal until a frame is materialized."
        )
    elif source_name.startswith("irs_soi_puf"):
        descriptor = replace(PUFSourceProvider().descriptor, name=source_name)
        manifest_name = "puf"
        notes.append(
            "PUF is manifest-backed, so raw-to-canonical tax mappings are available even "
            "without loading the microdata file."
        )
    elif source_name.startswith("psid"):
        descriptor = replace(PSIDSourceProvider(data_dir=".").descriptor, name=source_name)
        notes.append(
            "PSID is panel-backed and enters the US build as an optional donor family."
        )

    return {
        "name": source_name,
        "descriptor": descriptor,
        "manifestName": manifest_name,
        "notes": notes,
    }


def _serialize_source_snapshot_entry(entry: dict[str, Any]) -> dict[str, Any]:
    descriptor = entry["descriptor"]
    manifest_name = entry["manifestName"]
    manifest = load_us_source_manifest(manifest_name) if manifest_name is not None else None

    if descriptor is None:
        return {
            "name": entry["name"],
            "resolved": False,
            "notes": list(entry["notes"]),
        }

    variable_names = sorted(descriptor.all_variable_names)
    authoritative_only = [
        variable
        for variable in variable_names
        if descriptor.is_authoritative_for(variable)
        and not descriptor.allows_conditioning_on(variable)
    ]
    non_conditionable = [
        variable
        for variable in variable_names
        if not descriptor.allows_conditioning_on(variable)
    ]

    manifest_mappings = None
    if manifest is not None:
        sample_mappings: list[dict[str, str]] = []
        mapped_column_count = 0
        for observation in manifest.observations:
            mapped_column_count += len(observation.columns)
            for column in observation.columns:
                if len(sample_mappings) >= 8:
                    break
                sample_mappings.append(
                    {
                        "entity": observation.entity.value,
                        "rawColumn": column.raw_column,
                        "canonicalName": column.canonical_name,
                    }
                )
        manifest_mappings = {
            "observationCount": len(manifest.observations),
            "mappedColumnCount": mapped_column_count,
            "sampleMappings": sample_mappings,
        }

    return {
        "name": descriptor.name,
        "resolved": True,
        "shareability": descriptor.shareability.value,
        "timeStructure": descriptor.time_structure.value,
        "archetype": descriptor.archetype.value if descriptor.archetype is not None else None,
        "population": descriptor.population,
        "description": descriptor.description,
        "manifestName": manifest_name,
        "manifestBacked": manifest is not None,
        "observationCount": len(descriptor.observations),
        "observations": [
            {
                "entity": observation.entity.value,
                "keyColumn": observation.key_column,
                "weightColumn": observation.weight_column,
                "periodColumn": observation.period_column,
                "variableCount": len(observation.variable_names),
                "sampleVariables": list(observation.variable_names[:8]),
            }
            for observation in descriptor.observations
        ],
        "capabilitySummary": {
            "authoritativeVariableCount": sum(
                1 for variable in variable_names if descriptor.is_authoritative_for(variable)
            ),
            "conditionableVariableCount": sum(
                1 for variable in variable_names if descriptor.allows_conditioning_on(variable)
            ),
            "authoritativeOnlyVariables": authoritative_only[:8],
            "nonConditionableVariables": non_conditionable[:8],
        },
        "manifestMappings": manifest_mappings,
        "notes": list(entry["notes"]),
    }


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp_path.replace(path)


def _build_shared_coverage_summary(
    fusion_plan: FusionPlan | None,
) -> dict[str, Any]:
    if fusion_plan is None:
        return {
            "sourceNames": [],
            "entities": [],
        }

    entity_summaries = []
    for entity in fusion_plan.output_entities:
        entity_coverage = fusion_plan.coverage.get(entity, {})
        source_counts = {
            source_name: sum(
                1
                for coverage in entity_coverage.values()
                if source_name in coverage.sources
            )
            for source_name in fusion_plan.source_names
        }
        entity_summaries.append(
            {
                "entity": entity.value,
                "variableCount": len(entity_coverage),
                "publicVariableCount": sum(
                    1 for coverage in entity_coverage.values() if coverage.publicly_observed
                ),
                "syntheticReleaseVariableCount": sum(
                    1
                    for coverage in entity_coverage.values()
                    if coverage.requires_synthetic_release
                ),
                "sampleVariables": list(entity_coverage.keys())[:10],
                "sourceCounts": [
                    {"source": source_name, "variableCount": source_counts[source_name]}
                    for source_name in fusion_plan.source_names
                ],
            }
        )

    return {
        "sourceNames": list(fusion_plan.source_names),
        "entities": entity_summaries,
    }


def _build_donor_block_summary(
    donor_integrated_variables: tuple[str, ...],
) -> list[dict[str, Any]]:
    block_specs = donor_imputation_block_specs(donor_integrated_variables)
    return [
        {
            "id": f"block-{index + 1}",
            "nativeEntity": block_spec.native_entity.value,
            "conditionEntities": [
                entity.value for entity in block_spec.condition_entities
            ],
            "modelVariables": list(block_spec.model_variables),
            "restoredVariables": list(block_spec.restored_variables),
            "matchStrategies": {
                variable_name: strategy.value
                for variable_name, strategy in block_spec.match_strategies.items()
            },
            "prepareFrame": (
                block_spec.prepare_frame.__name__
                if block_spec.prepare_frame is not None
                else None
            ),
            "restoreFrame": (
                block_spec.restore_frame.__name__
                if block_spec.restore_frame is not None
                else None
            ),
        }
        for index, block_spec in enumerate(block_specs)
    ]


def _build_semantic_highlights(
    variable_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    highlights: list[dict[str, Any]] = []
    for variable_name in variable_names:
        spec = variable_semantic_spec_for(variable_name)
        if (
            not spec.derived_from
            and spec.support_family.value == "continuous"
            and spec.donor_transform is None
            and spec.donor_check is None
            and spec.notes is None
        ):
            continue
        highlights.append(
            {
                "variableName": variable_name,
                "nativeEntity": spec.native_entity.value,
                "conditionEntities": [
                    entity.value for entity in spec.condition_entities
                ],
                "supportFamily": spec.support_family.value,
                "derivedFrom": list(spec.derived_from),
                "donorMatchStrategy": spec.donor_match_strategy.value,
                "hasDonorTransform": spec.donor_transform is not None,
                "hasDonorCheck": spec.donor_check is not None,
                "notes": spec.notes,
            }
        )
    return highlights


def _build_pipeline_stage_summary(
    *,
    synthesis: dict[str, Any],
    calibration: dict[str, Any],
    artifacts: dict[str, Any],
    config: dict[str, Any],
    donor_integrated_variables: tuple[str, ...],
    source_names: tuple[str, ...],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    harness_summary = dict(manifest.get("policyengine_harness", {}))
    native_scores_summary = dict(manifest.get("policyengine_native_scores", {}))

    return [
        {
            "id": "source-mix",
            "step": "01",
            "title": "Source mix",
            "summary": "Descriptor-backed source families declared for the saved run.",
            "status": "ready" if source_names else "missing",
            "metrics": [
                {"label": "Sources", "value": len(source_names)},
                {"label": "Scaffold", "value": synthesis.get("scaffold_source")},
            ],
            "outputs": list(source_names),
        },
        {
            "id": "donor-integration",
            "step": "02",
            "title": "Donor integration",
            "summary": "Authoritative donor variables projected onto the scaffold before synthesis.",
            "status": "ready" if donor_integrated_variables else "inactive",
            "metrics": [
                {
                    "label": "Integrated vars",
                    "value": len(donor_integrated_variables),
                },
                {
                    "label": "Overrides",
                    "value": len(
                        synthesis.get("donor_authoritative_override_variables", ())
                    ),
                },
            ],
            "outputs": list(donor_integrated_variables[:12]),
        },
        {
            "id": "synthesis",
            "step": "03",
            "title": "Synthesis",
            "summary": "Seed rows become the candidate population under the configured backend.",
            "status": "ready",
            "metrics": [
                {"label": "Backend", "value": synthesis.get("backend")},
                {"label": "Conditions", "value": len(synthesis.get("condition_vars", ()))},
                {"label": "Targets", "value": len(synthesis.get("target_vars", ()))},
                {"label": "nSynthetic", "value": config.get("n_synthetic")},
            ],
            "outputs": [
                f"seed={manifest.get('rows', {}).get('seed')}",
                f"synthetic={manifest.get('rows', {}).get('synthetic')}",
            ],
        },
        {
            "id": "calibration",
            "step": "04",
            "title": "Calibration",
            "summary": "Target support and convergence remain attached to the saved run.",
            "status": "ready" if calibration else "missing",
            "metrics": [
                {"label": "Backend", "value": calibration.get("backend")},
                {"label": "Loaded", "value": calibration.get("n_loaded_targets")},
                {"label": "Supported", "value": calibration.get("n_supported_targets")},
                {"label": "Converged", "value": calibration.get("converged")},
            ],
            "outputs": [
                f"calibrated={manifest.get('rows', {}).get('calibrated')}",
            ],
        },
        {
            "id": "pe-export",
            "step": "05",
            "title": "PolicyEngine export",
            "summary": "The runtime narrows to the PE-facing artifact contract before scoring.",
            "status": "ready" if artifacts.get("policyengine_dataset") else "missing",
            "metrics": [
                {
                    "label": "Dataset artifact",
                    "value": artifacts.get("policyengine_dataset"),
                },
                {
                    "label": "Direct overrides",
                    "value": len(config.get("policyengine_direct_override_variables", ())),
                },
            ],
            "outputs": [
                value
                for value in (
                    artifacts.get("policyengine_dataset"),
                    artifacts.get("manifest"),
                )
                if value
            ],
        },
        {
            "id": "benchmark",
            "step": "06",
            "title": "PolicyEngine benchmark",
            "summary": "Harness and native-loss diagnostics stay attached to the same artifact bundle.",
            "status": "ready" if harness_summary or native_scores_summary else "missing",
            "metrics": [
                {
                    "label": "Harness delta",
                    "value": harness_summary.get("mean_abs_relative_error_delta"),
                },
                {
                    "label": "Native delta",
                    "value": native_scores_summary.get("enhanced_cps_native_loss_delta"),
                },
                {
                    "label": "Win rate",
                    "value": harness_summary.get("target_win_rate"),
                },
            ],
            "outputs": [
                value
                for value in (
                    artifacts.get("policyengine_harness"),
                    artifacts.get("policyengine_native_scores"),
                )
                if value
            ],
        },
    ]
