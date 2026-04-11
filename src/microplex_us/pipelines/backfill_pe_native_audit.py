"""Backfill PE rebuild native-audit sidecars for historical US artifact bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microplex.targets import assert_valid_benchmark_artifact_manifest

from microplex_us.pipelines.backfill_pe_native_scores import (
    discover_us_candidate_artifact_dirs,
)
from microplex_us.pipelines.pe_us_data_rebuild_audit import (
    build_policyengine_us_data_rebuild_native_audit,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint import (
    _refresh_checkpoint_data_flow_snapshot,
)


def backfill_us_pe_native_audit_bundle(
    artifact_dir: str | Path,
    *,
    force: bool = False,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> Path:
    """Backfill PE rebuild native-audit sidecar + manifest summary for one bundle."""

    bundle_dir = Path(artifact_dir)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = dict(manifest.get("artifacts", {}))
    dataset_name = artifacts.get("policyengine_dataset")
    if not dataset_name:
        raise ValueError(f"{bundle_dir} does not declare a policyengine_dataset artifact")

    native_scores_path = _resolve_required_native_scores_path(bundle_dir, artifacts)
    native_scores_payload = json.loads(native_scores_path.read_text())
    native_audit_path = bundle_dir / "pe_us_data_rebuild_native_audit.json"
    if native_audit_path.exists() and not force:
        payload = json.loads(native_audit_path.read_text())
    else:
        payload = build_policyengine_us_data_rebuild_native_audit(
            bundle_dir,
            manifest_payload=manifest,
            native_scores_payload=native_scores_payload,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
        )
    return _write_native_audit_payload_to_bundle(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        payload=payload,
    )


def backfill_us_pe_native_audit_bundles(
    artifact_dirs: list[str | Path] | tuple[str | Path, ...],
    *,
    force: bool = False,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> list[Path]:
    """Backfill PE rebuild native audits for a batch of saved bundles."""

    return [
        backfill_us_pe_native_audit_bundle(
            artifact_dir,
            force=force,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
        )
        for artifact_dir in artifact_dirs
    ]


def backfill_us_pe_native_audit_root(
    artifact_root: str | Path,
    *,
    force: bool = False,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> list[Path]:
    """Backfill every eligible artifact bundle under one saved-output root."""

    return backfill_us_pe_native_audit_bundles(
        discover_us_candidate_artifact_dirs(artifact_root),
        force=force,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
    )


def _write_native_audit_payload_to_bundle(
    *,
    bundle_dir: Path,
    manifest_path: Path,
    manifest: dict,
    payload: dict,
) -> Path:
    native_audit_path = bundle_dir / "pe_us_data_rebuild_native_audit.json"
    native_audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    artifacts = dict(manifest.get("artifacts", {}))
    artifacts["policyengine_native_audit"] = native_audit_path.name
    manifest["artifacts"] = artifacts
    manifest["policyengine_native_audit"] = dict(payload.get("verdictHints", {}))

    _refresh_checkpoint_data_flow_snapshot(bundle_dir, manifest)
    assert_valid_benchmark_artifact_manifest(
        manifest,
        artifact_dir=bundle_dir,
        manifest_path=manifest_path,
        summary_section=(
            "policyengine_harness"
            if manifest.get("policyengine_harness") is not None
            else None
        ),
        required_artifact_keys=(
            "policyengine_dataset",
            "policyengine_native_scores",
            "policyengine_native_audit",
        ),
        required_summary_keys=(
            (
                "candidate_mean_abs_relative_error",
                "baseline_mean_abs_relative_error",
                "mean_abs_relative_error_delta",
            )
            if manifest.get("policyengine_harness") is not None
            else ()
        ),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


def _resolve_required_native_scores_path(
    bundle_dir: Path,
    artifacts: dict,
) -> Path:
    artifact_name = artifacts.get("policyengine_native_scores") or "policyengine_native_scores.json"
    path = bundle_dir / str(artifact_name)
    if path.exists():
        return path
    raise ValueError(
        f"{bundle_dir} is missing policyengine_native_scores.json; backfill native scores first"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for historical PE rebuild native-audit backfill."""

    parser = argparse.ArgumentParser(
        description="Backfill PE rebuild native-audit sidecars for US artifact bundles.",
    )
    parser.add_argument(
        "artifact_root",
        nargs="?",
        default="artifacts",
        help="Artifact output root to scan (defaults to ./artifacts).",
    )
    parser.add_argument("--policyengine-us-data-repo")
    parser.add_argument("--policyengine-us-data-python")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute native audits even if a sidecar already exists.",
    )
    args = parser.parse_args(argv)

    manifest_paths = backfill_us_pe_native_audit_root(
        args.artifact_root,
        force=args.force,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        policyengine_us_data_python=args.policyengine_us_data_python,
    )
    print(
        json.dumps(
            {
                "artifact_root": str(Path(args.artifact_root).resolve()),
                "backfilled_count": len(manifest_paths),
                "manifest_paths": [str(path) for path in manifest_paths],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
