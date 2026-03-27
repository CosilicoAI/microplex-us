"""Persistent run-registry helpers for saved US microplex artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

FrontierMetric = Literal[
    "candidate_composite_parity_loss",
    "candidate_mean_abs_relative_error",
    "mean_abs_relative_error_delta",
]


@dataclass(frozen=True)
class USMicroplexRunRegistryEntry:
    """Compact cross-build summary for one saved artifact bundle."""

    created_at: str
    artifact_id: str
    artifact_dir: str
    manifest_path: str
    policyengine_harness_path: str | None = None
    config_hash: str | None = None
    synthesis_backend: str | None = None
    calibration_backend: str | None = None
    source_names: tuple[str, ...] = ()
    rows: dict[str, int] = field(default_factory=dict)
    weights: dict[str, float | int] = field(default_factory=dict)
    candidate_mean_abs_relative_error: float | None = None
    baseline_mean_abs_relative_error: float | None = None
    mean_abs_relative_error_delta: float | None = None
    candidate_composite_parity_loss: float | None = None
    baseline_composite_parity_loss: float | None = None
    composite_parity_loss_delta: float | None = None
    slice_win_rate: float | None = None
    target_win_rate: float | None = None
    supported_target_rate: float | None = None
    tag_summaries: dict[str, dict[str, float | None]] = field(default_factory=dict)
    parity_scorecard: dict[str, dict[str, float | bool | None]] = field(
        default_factory=dict
    )
    baseline_dataset: str | None = None
    targets_db: str | None = None
    target_period: int | None = None
    target_variables: tuple[str, ...] = ()
    target_domains: tuple[str, ...] = ()
    target_geo_levels: tuple[str, ...] = ()
    target_reform_id: int | None = None
    policyengine_us_runtime_version: str | None = None
    improved_candidate_frontier: bool | None = None
    improved_delta_frontier: bool | None = None
    improved_composite_frontier: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the registry entry to a JSON-compatible dict."""
        return {
            "created_at": self.created_at,
            "artifact_id": self.artifact_id,
            "artifact_dir": self.artifact_dir,
            "manifest_path": self.manifest_path,
            "policyengine_harness_path": self.policyengine_harness_path,
            "config_hash": self.config_hash,
            "synthesis_backend": self.synthesis_backend,
            "calibration_backend": self.calibration_backend,
            "source_names": list(self.source_names),
            "rows": dict(self.rows),
            "weights": dict(self.weights),
            "candidate_mean_abs_relative_error": self.candidate_mean_abs_relative_error,
            "baseline_mean_abs_relative_error": self.baseline_mean_abs_relative_error,
            "mean_abs_relative_error_delta": self.mean_abs_relative_error_delta,
            "candidate_composite_parity_loss": self.candidate_composite_parity_loss,
            "baseline_composite_parity_loss": self.baseline_composite_parity_loss,
            "composite_parity_loss_delta": self.composite_parity_loss_delta,
            "slice_win_rate": self.slice_win_rate,
            "target_win_rate": self.target_win_rate,
            "supported_target_rate": self.supported_target_rate,
            "tag_summaries": dict(self.tag_summaries),
            "parity_scorecard": dict(self.parity_scorecard),
            "baseline_dataset": self.baseline_dataset,
            "targets_db": self.targets_db,
            "target_period": self.target_period,
            "target_variables": list(self.target_variables),
            "target_domains": list(self.target_domains),
            "target_geo_levels": list(self.target_geo_levels),
            "target_reform_id": self.target_reform_id,
            "policyengine_us_runtime_version": self.policyengine_us_runtime_version,
            "improved_candidate_frontier": self.improved_candidate_frontier,
            "improved_delta_frontier": self.improved_delta_frontier,
            "improved_composite_frontier": self.improved_composite_frontier,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> USMicroplexRunRegistryEntry:
        """Restore a registry entry from JSON payload."""
        return cls(
            created_at=payload["created_at"],
            artifact_id=payload["artifact_id"],
            artifact_dir=payload["artifact_dir"],
            manifest_path=payload["manifest_path"],
            policyengine_harness_path=payload.get("policyengine_harness_path"),
            config_hash=payload.get("config_hash"),
            synthesis_backend=payload.get("synthesis_backend"),
            calibration_backend=payload.get("calibration_backend"),
            source_names=tuple(payload.get("source_names", [])),
            rows=dict(payload.get("rows", {})),
            weights=dict(payload.get("weights", {})),
            candidate_mean_abs_relative_error=payload.get(
                "candidate_mean_abs_relative_error"
            ),
            baseline_mean_abs_relative_error=payload.get(
                "baseline_mean_abs_relative_error"
            ),
            mean_abs_relative_error_delta=payload.get("mean_abs_relative_error_delta"),
            candidate_composite_parity_loss=payload.get(
                "candidate_composite_parity_loss"
            ),
            baseline_composite_parity_loss=payload.get(
                "baseline_composite_parity_loss"
            ),
            composite_parity_loss_delta=payload.get("composite_parity_loss_delta"),
            slice_win_rate=payload.get("slice_win_rate"),
            target_win_rate=payload.get("target_win_rate"),
            supported_target_rate=payload.get("supported_target_rate"),
            tag_summaries={
                key: dict(value)
                for key, value in dict(payload.get("tag_summaries", {})).items()
            },
            parity_scorecard={
                key: dict(value)
                for key, value in dict(payload.get("parity_scorecard", {})).items()
            },
            baseline_dataset=payload.get("baseline_dataset"),
            targets_db=payload.get("targets_db"),
            target_period=payload.get("target_period"),
            target_variables=tuple(payload.get("target_variables", [])),
            target_domains=tuple(payload.get("target_domains", [])),
            target_geo_levels=tuple(payload.get("target_geo_levels", [])),
            target_reform_id=payload.get("target_reform_id"),
            policyengine_us_runtime_version=payload.get(
                "policyengine_us_runtime_version"
            ),
            improved_candidate_frontier=payload.get("improved_candidate_frontier"),
            improved_delta_frontier=payload.get("improved_delta_frontier"),
            improved_composite_frontier=payload.get("improved_composite_frontier"),
            metadata=dict(payload.get("metadata", {})),
        )


def load_us_microplex_run_registry(
    path: str | Path,
) -> list[USMicroplexRunRegistryEntry]:
    """Load a JSONL run registry from disk."""
    registry_path = Path(path)
    if not registry_path.exists():
        return []
    entries: list[USMicroplexRunRegistryEntry] = []
    for line in registry_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entries.append(USMicroplexRunRegistryEntry.from_dict(json.loads(line)))
    return entries


def select_us_microplex_frontier_entry(
    path: str | Path,
    *,
    metric: FrontierMetric = "candidate_composite_parity_loss",
) -> USMicroplexRunRegistryEntry | None:
    """Select the current best run from the registry using one summary metric."""
    entries = load_us_microplex_run_registry(_resolve_run_registry_path(path))
    metric_values = {
        "candidate_composite_parity_loss": lambda entry: entry.candidate_composite_parity_loss,
        "candidate_mean_abs_relative_error": lambda entry: entry.candidate_mean_abs_relative_error,
        "mean_abs_relative_error_delta": lambda entry: entry.mean_abs_relative_error_delta,
    }
    value_fn = metric_values[metric]
    comparable_entries = [
        entry for entry in entries if value_fn(entry) is not None
    ]
    if not comparable_entries:
        return None
    return min(comparable_entries, key=value_fn)


def resolve_us_microplex_frontier_artifact_dir(
    path: str | Path,
    *,
    metric: FrontierMetric = "candidate_composite_parity_loss",
) -> Path | None:
    """Return the artifact directory for the current frontier run, if any."""
    frontier = select_us_microplex_frontier_entry(path, metric=metric)
    if frontier is None:
        return None
    return Path(frontier.artifact_dir)


def append_us_microplex_run_registry_entry(
    path: str | Path,
    entry: USMicroplexRunRegistryEntry,
) -> USMicroplexRunRegistryEntry:
    """Append one registry entry, computing frontier flags from prior history."""
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    existing_entries = load_us_microplex_run_registry(registry_path)
    entry_to_write = USMicroplexRunRegistryEntry(
        **{
            **entry.to_dict(),
            "improved_candidate_frontier": _improves_candidate_frontier(
                existing_entries,
                entry,
            ),
            "improved_delta_frontier": _improves_delta_frontier(existing_entries, entry),
            "improved_composite_frontier": _improves_composite_frontier(
                existing_entries,
                entry,
            ),
        }
    )
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry_to_write.to_dict(), sort_keys=True) + "\n")
    return entry_to_write


def build_us_microplex_run_registry_entry(
    *,
    artifact_dir: str | Path,
    manifest_path: str | Path,
    manifest: dict[str, Any],
    policyengine_harness_path: str | Path | None = None,
    policyengine_harness_payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> USMicroplexRunRegistryEntry:
    """Build a compact registry entry from a saved artifact manifest."""
    harness_payload = dict(policyengine_harness_payload or {})
    harness_summary = dict(manifest.get("policyengine_harness", {}))
    harness_metadata = dict(harness_payload.get("metadata", {}))
    created_at = manifest.get("created_at") or datetime.now(UTC).isoformat()
    config = dict(manifest.get("config", {}))
    synthesis = dict(manifest.get("synthesis", {}))

    return USMicroplexRunRegistryEntry(
        created_at=created_at,
        artifact_id=Path(artifact_dir).name,
        artifact_dir=str(Path(artifact_dir).resolve()),
        manifest_path=str(Path(manifest_path).resolve()),
        policyengine_harness_path=(
            str(Path(policyengine_harness_path).resolve())
            if policyengine_harness_path is not None
            else None
        ),
        config_hash=_stable_config_hash(config),
        synthesis_backend=config.get("synthesis_backend"),
        calibration_backend=config.get("calibration_backend"),
        source_names=tuple(synthesis.get("source_names", [])),
        rows={key: int(value) for key, value in dict(manifest.get("rows", {})).items()},
        weights=dict(manifest.get("weights", {})),
        candidate_mean_abs_relative_error=harness_summary.get(
            "candidate_mean_abs_relative_error"
        ),
        baseline_mean_abs_relative_error=harness_summary.get(
            "baseline_mean_abs_relative_error"
        ),
        mean_abs_relative_error_delta=harness_summary.get(
            "mean_abs_relative_error_delta"
        ),
        candidate_composite_parity_loss=harness_summary.get(
            "candidate_composite_parity_loss"
        ),
        baseline_composite_parity_loss=harness_summary.get(
            "baseline_composite_parity_loss"
        ),
        composite_parity_loss_delta=harness_summary.get(
            "composite_parity_loss_delta"
        ),
        slice_win_rate=harness_summary.get("slice_win_rate"),
        target_win_rate=harness_summary.get("target_win_rate"),
        supported_target_rate=harness_summary.get("supported_target_rate"),
        tag_summaries={
            key: dict(value)
            for key, value in dict(harness_summary.get("tag_summaries", {})).items()
        },
        parity_scorecard={
            key: dict(value)
            for key, value in dict(harness_summary.get("parity_scorecard", {})).items()
        },
        baseline_dataset=harness_metadata.get("baseline_dataset"),
        targets_db=harness_metadata.get("targets_db"),
        target_period=harness_metadata.get("target_period"),
        target_variables=tuple(harness_metadata.get("target_variables", [])),
        target_domains=tuple(harness_metadata.get("target_domains", [])),
        target_geo_levels=tuple(harness_metadata.get("target_geo_levels", [])),
        target_reform_id=harness_metadata.get("target_reform_id"),
        policyengine_us_runtime_version=harness_metadata.get(
            "policyengine_us_runtime_version"
        ),
        metadata=dict(metadata or {}),
    )


def _stable_config_hash(config: dict[str, Any]) -> str | None:
    if not config:
        return None
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_run_registry_path(path: str | Path) -> Path:
    candidate_path = Path(path)
    if candidate_path.suffix == ".jsonl":
        return candidate_path
    return candidate_path / "run_registry.jsonl"


def _improves_candidate_frontier(
    entries: list[USMicroplexRunRegistryEntry],
    entry: USMicroplexRunRegistryEntry,
) -> bool | None:
    candidate_error = entry.candidate_mean_abs_relative_error
    if candidate_error is None:
        return None
    prior_errors = [
        item.candidate_mean_abs_relative_error
        for item in entries
        if item.candidate_mean_abs_relative_error is not None
    ]
    if not prior_errors:
        return True
    return candidate_error < min(prior_errors)


def _improves_delta_frontier(
    entries: list[USMicroplexRunRegistryEntry],
    entry: USMicroplexRunRegistryEntry,
) -> bool | None:
    error_delta = entry.mean_abs_relative_error_delta
    if error_delta is None:
        return None
    prior_deltas = [
        item.mean_abs_relative_error_delta
        for item in entries
        if item.mean_abs_relative_error_delta is not None
    ]
    if not prior_deltas:
        return True
    return error_delta < min(prior_deltas)


def _improves_composite_frontier(
    entries: list[USMicroplexRunRegistryEntry],
    entry: USMicroplexRunRegistryEntry,
) -> bool | None:
    composite_loss = entry.candidate_composite_parity_loss
    if composite_loss is None:
        return None
    prior_losses = [
        item.candidate_composite_parity_loss
        for item in entries
        if item.candidate_composite_parity_loss is not None
    ]
    if not prior_losses:
        return True
    return composite_loss < min(prior_losses)
