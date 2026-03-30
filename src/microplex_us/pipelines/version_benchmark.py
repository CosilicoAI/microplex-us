"""Canonical US version-bump benchmark entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from microplex_us.data_sources.cps import CPSASECParquetSourceProvider
from microplex_us.data_sources.psid import PSIDSourceProvider
from microplex_us.data_sources.puf import PUFSourceProvider
from microplex_us.pipelines.artifacts import (
    build_and_save_versioned_us_microplex_from_source_providers,
)
from microplex_us.pipelines.site_snapshot import write_us_microplex_site_snapshot
from microplex_us.pipelines.us import USMicroplexBuildConfig


def _resolve_site_snapshot_path(
    *,
    output_root: str | Path,
    site_snapshot_path: str | Path | None,
) -> Path:
    if site_snapshot_path is not None:
        return Path(site_snapshot_path)
    output_root_path = Path(output_root).resolve()
    if output_root_path.name == "artifacts":
        return output_root_path / "site_snapshot_us.json"
    return output_root_path.parent / "site_snapshot_us.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the canonical US version-bump benchmark build."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cps-parquet-dir", required=True)
    parser.add_argument("--baseline-dataset", required=True)
    parser.add_argument("--targets-db", required=True)
    parser.add_argument("--policyengine-us-data-repo")
    parser.add_argument("--version-id")
    parser.add_argument("--site-snapshot-path")
    parser.add_argument("--target-period", type=int, default=2024)
    parser.add_argument("--n-synthetic", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--target-profile", default="pe_native_broad")
    parser.add_argument("--calibration-target-profile")
    parser.add_argument("--puf-path")
    parser.add_argument("--puf-demographics-path")
    parser.add_argument("--psid-data-dir")
    parser.add_argument("--target-variable", action="append", default=[])
    parser.add_argument("--target-domain", action="append", default=[])
    parser.add_argument("--target-geo-level", action="append", default=[])
    parser.add_argument(
        "--require-beat-pe-native-loss",
        action="store_true",
        help="Fail if the saved run does not beat the PE baseline on PE's native enhanced-CPS loss.",
    )
    args = parser.parse_args(argv)

    providers = [CPSASECParquetSourceProvider(data_dir=args.cps_parquet_dir)]
    if args.puf_path is not None:
        providers.append(
            PUFSourceProvider(
                puf_path=args.puf_path,
                demographics_path=args.puf_demographics_path,
                target_year=args.target_period,
            )
        )
    if args.psid_data_dir is not None:
        providers.append(PSIDSourceProvider(data_dir=args.psid_data_dir))

    config = USMicroplexBuildConfig(
        n_synthetic=args.n_synthetic,
        random_seed=args.random_seed,
        policyengine_baseline_dataset=args.baseline_dataset,
        policyengine_targets_db=args.targets_db,
        policyengine_dataset_year=args.target_period,
        policyengine_target_period=args.target_period,
        policyengine_target_variables=tuple(args.target_variable),
        policyengine_target_domains=tuple(args.target_domain),
        policyengine_target_geo_levels=tuple(args.target_geo_level),
        policyengine_target_profile=args.target_profile,
        policyengine_calibration_target_profile=(
            args.calibration_target_profile or args.target_profile
        ),
    )

    artifacts = build_and_save_versioned_us_microplex_from_source_providers(
        providers=providers,
        output_root=args.output_root,
        config=config,
        version_id=args.version_id,
        frontier_metric="enhanced_cps_native_loss_delta",
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        require_policyengine_native_score=True,
    )

    native_delta = (
        artifacts.current_entry.enhanced_cps_native_loss_delta
        if artifacts.current_entry is not None
        else None
    )
    candidate_native_loss = (
        artifacts.current_entry.candidate_enhanced_cps_native_loss
        if artifacts.current_entry is not None
        else None
    )
    baseline_native_loss = (
        artifacts.current_entry.baseline_enhanced_cps_native_loss
        if artifacts.current_entry is not None
        else None
    )
    if native_delta is None:
        raise SystemExit(
            "Saved US benchmark artifact is missing PE-native enhanced-CPS loss delta."
        )
    if args.require_beat_pe_native_loss and native_delta >= 0.0:
        raise SystemExit(
            "US version-bump benchmark did not beat PE on PE-native enhanced-CPS loss: "
            f"candidate={candidate_native_loss:.6f}, "
            f"baseline={baseline_native_loss:.6f}, "
            f"delta={native_delta:.6f}"
        )

    write_us_microplex_site_snapshot(
        artifacts.artifact_paths.output_dir,
        _resolve_site_snapshot_path(
            output_root=args.output_root,
            site_snapshot_path=args.site_snapshot_path,
        ),
    )

    print(artifacts.artifact_paths.output_dir)
    print(
        "PE native enhanced-CPS loss: "
        f"candidate={candidate_native_loss:.6f} "
        f"baseline={baseline_native_loss:.6f} "
        f"delta={native_delta:.6f}"
    )
    if artifacts.artifact_paths.run_registry is not None:
        print(artifacts.artifact_paths.run_registry)


if __name__ == "__main__":
    main()
