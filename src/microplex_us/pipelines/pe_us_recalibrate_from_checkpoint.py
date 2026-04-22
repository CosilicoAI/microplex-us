"""Recalibrate a saved US microplex checkpoint with a new calibration config.

Load a ``post_imputation`` pipeline checkpoint previously saved via
``pe_us_data_rebuild_checkpoint --pipeline-checkpoint-save-post-imputation-path``
and rerun the calibration stage without repeating the ~11 hours of
synthesis + donor imputation.

Intended for rapid iteration on calibration backends / target sets /
sparsity schedules: change one flag, run for ~30 min instead of half a
day.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    recalibrate_policyengine_us_from_checkpoint,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun US microplex calibration from a saved post-imputation "
            "checkpoint (skips the ~11 h synthesis stage)."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help=(
            "Path to a directory written by the main pipeline with "
            "--pipeline-checkpoint-save-post-imputation-path."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory for the recalibrated bundle and summary.",
    )
    parser.add_argument(
        "--targets-db",
        type=Path,
        required=True,
        help="Path to the PolicyEngine US targets SQLite database.",
    )
    parser.add_argument(
        "--target-period",
        type=int,
        default=None,
        help="Calendar year for calibration targets (default: config default).",
    )
    parser.add_argument(
        "--calibration-backend",
        type=str,
        default="pe_l0",
        help="Calibration backend (pe_l0, microcalibrate, hardconcrete, etc.).",
    )
    parser.add_argument(
        "--calibration-max-iter",
        type=int,
        default=None,
        help="Max iterations / epochs for the calibration solver.",
    )
    parser.add_argument(
        "--policyengine-materialize-batch-size",
        type=int,
        default=100_000,
        help=(
            "Batch size for PE variable materialization (default 100_000; "
            "keeps a single Microsimulation under a few GB at 1.5M-household scale)."
        ),
    )
    parser.add_argument(
        "--pipeline-checkpoint-save-post-microsim-path",
        type=Path,
        default=None,
        help=(
            "If set, also save a post-microsim checkpoint during this "
            "recalibration so the next iteration can skip microsim too."
        ),
    )
    args = parser.parse_args(argv)

    config_kwargs: dict[str, object] = {
        "calibration_backend": args.calibration_backend,
        "policyengine_targets_db": args.targets_db,
        "policyengine_materialize_batch_size": int(
            args.policyengine_materialize_batch_size
        ),
    }
    if args.target_period is not None:
        config_kwargs["policyengine_target_period"] = int(args.target_period)
    if args.calibration_max_iter is not None:
        config_kwargs["calibration_max_iter"] = int(args.calibration_max_iter)
    if args.pipeline_checkpoint_save_post_microsim_path is not None:
        config_kwargs["pipeline_checkpoint_save_post_microsim_path"] = (
            args.pipeline_checkpoint_save_post_microsim_path
        )

    config = USMicroplexBuildConfig(**config_kwargs)
    result = recalibrate_policyengine_us_from_checkpoint(config, args.checkpoint_path)

    args.output_root.mkdir(parents=True, exist_ok=True)
    result.calibrated_data.to_parquet(args.output_root / "calibrated_data.parquet")
    result.policyengine_tables.households.to_parquet(
        args.output_root / "households.parquet"
    )
    if result.policyengine_tables.persons is not None:
        result.policyengine_tables.persons.to_parquet(
            args.output_root / "persons.parquet"
        )
    (args.output_root / "calibration_summary.json").write_text(
        json.dumps(result.calibration_summary, indent=2, default=str)
    )
    print(
        f"Recalibrated from {args.checkpoint_path} → {args.output_root} "
        f"(stage={result.loaded_stage}, "
        f"rows={len(result.calibrated_data)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
