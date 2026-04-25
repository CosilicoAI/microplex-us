"""Measure whether `microcalibrate` on top of a synthesizer rescues weak synthesis.

Stage-1 PRDC coverage compared synthesizers with uniform unit weights. The
actual production pipeline is synthesize → calibrate. If calibration can
pull a weak synthesizer's weighted aggregates onto the real targets, the
choice of synthesizer matters less than PRDC alone would suggest.

Procedure:

1. Load enhanced_cps_2024 (`ScaleUpRunner.load_frame`), split 80/20.
2. For each method (ZI-QRF / ZI-MAF / ZI-QDNN):
   a. Fit method, generate synthetic records with uniform weights.
   b. Compute holdout aggregates for each target column
      (total, count-of-nonzero).
   c. Build `LinearConstraint`s that require the weighted synthetic
      aggregates to match the holdout aggregates.
   d. Run `MicrocalibrateAdapter.fit_transform`.
   e. Report per-target relative error pre- and post-calibration.

Usage:
    uv run python scripts/calibrate_on_synthesizer.py --n-rows 20000

~10 minutes on a 48 GB M3 for 20k × 50 × 3 methods.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from microplex.calibration import LinearConstraint
from microplex.eval.benchmark import ZIMAFMethod, ZIQDNNMethod, ZIQRFMethod

from microplex_us.bakeoff import (
    DEFAULT_CONDITION_COLS,
    DEFAULT_TARGET_COLS,
    ScaleUpRunner,
    ScaleUpStageConfig,
    stage1_config,
)
from microplex_us.calibration import (
    MicrocalibrateAdapter,
    MicrocalibrateAdapterConfig,
)

LOGGER = logging.getLogger(__name__)

METHOD_REGISTRY = {
    "ZI-QRF": ZIQRFMethod,
    "ZI-MAF": ZIMAFMethod,
    "ZI-QDNN": ZIQDNNMethod,
}


def build_target_constraints(
    holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_cols: tuple[str, ...],
) -> tuple[LinearConstraint, ...]:
    """One total-sum constraint per target column.

    Target = sum of `holdout[col]`; coefficients = `synthetic[col].values`.
    After calibration, `(weights * coefficients).sum()` should match target.
    """
    constraints: list[LinearConstraint] = []
    for col in target_cols:
        if col not in synthetic.columns or col not in holdout.columns:
            continue
        target = float(holdout[col].sum())
        coefs = synthetic[col].to_numpy(dtype=float)
        constraints.append(
            LinearConstraint(
                name=f"sum_{col}",
                coefficients=coefs,
                target=target,
            )
        )
    return tuple(constraints)


def evaluate_aggregates(
    holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    weights: np.ndarray,
    target_cols: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Per-target: real total, weighted-synth total, relative error."""
    out: dict[str, dict[str, float]] = {}
    for col in target_cols:
        if col not in synthetic.columns or col not in holdout.columns:
            continue
        real_total = float(holdout[col].sum())
        synth_weighted = float((synthetic[col].to_numpy(dtype=float) * weights).sum())
        rel_err = abs(synth_weighted - real_total) / max(abs(real_total), 1.0)
        out[col] = {
            "real_total": real_total,
            "weighted_synth_total": synth_weighted,
            "relative_error": rel_err,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=20_000)
    parser.add_argument(
        "--methods", nargs="+", default=["ZI-QRF", "ZI-MAF", "ZI-QDNN"]
    )
    parser.add_argument("--calibration-epochs", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/calibrate_on_synthesizer.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    base = stage1_config()
    cfg = ScaleUpStageConfig(
        stage="calibrate_on_synth",
        n_rows=args.n_rows,
        methods=tuple(args.methods),
        condition_cols=DEFAULT_CONDITION_COLS,
        target_cols=DEFAULT_TARGET_COLS,
        holdout_frac=0.2,
        seed=args.seed,
        k=5,
        data_path=base.data_path,
        year=base.year,
        rare_cell_checks=(),
        prdc_max_samples=15_000,
    )
    runner = ScaleUpRunner(cfg)
    df = runner.load_frame()
    train, holdout = runner.split(df)
    LOGGER.info(
        "loaded %d rows; train=%d holdout=%d", len(df), len(train), len(holdout)
    )

    results = []
    for method_name in args.methods:
        LOGGER.info("== %s ==", method_name)
        if method_name not in METHOD_REGISTRY:
            LOGGER.warning("unknown method %r, skipping", method_name)
            continue
        method = METHOD_REGISTRY[method_name]()
        t0 = time.time()
        method.fit(sources={"ecps": train.copy()}, shared_cols=list(DEFAULT_CONDITION_COLS))
        fit_s = time.time() - t0

        t0 = time.time()
        synthetic = method.generate(len(train), seed=args.seed)
        gen_s = time.time() - t0
        LOGGER.info("  fit=%.1fs gen=%.1fs n_synth=%d", fit_s, gen_s, len(synthetic))

        constraints = build_target_constraints(
            holdout, synthetic, DEFAULT_TARGET_COLS
        )
        LOGGER.info("  %d calibration constraints", len(constraints))

        synthetic = synthetic.copy()
        synthetic["weight"] = 1.0

        # Rescale initial weights so synth totals sum to holdout-scale before
        # calibration. Otherwise gradient descent has to travel a long way.
        for col in DEFAULT_TARGET_COLS:
            if col not in holdout.columns or col not in synthetic.columns:
                continue
            r_sum = float(holdout[col].sum())
            s_sum = float(synthetic[col].sum())
            if r_sum > 0 and s_sum > 0:
                synthetic["weight"] = synthetic["weight"] * (r_sum / s_sum)
                break

        pre_weights = synthetic["weight"].to_numpy(dtype=float)
        pre = evaluate_aggregates(holdout, synthetic, pre_weights, DEFAULT_TARGET_COLS)

        adapter = MicrocalibrateAdapter(
            MicrocalibrateAdapterConfig(
                epochs=args.calibration_epochs,
                learning_rate=1e-3,
                noise_level=0.0,
                seed=args.seed,
            )
        )
        t0 = time.time()
        calibrated = adapter.fit_transform(
            synthetic,
            marginal_targets={},
            weight_col="weight",
            linear_constraints=constraints,
        )
        cal_s = time.time() - t0

        post_weights = calibrated["weight"].to_numpy(dtype=float)
        post = evaluate_aggregates(
            holdout, calibrated, post_weights, DEFAULT_TARGET_COLS
        )
        validation = adapter.validate()

        pre_mean_err = float(
            np.mean([v["relative_error"] for v in pre.values()])
        )
        post_mean_err = float(
            np.mean([v["relative_error"] for v in post.values()])
        )
        LOGGER.info(
            "  pre-cal mean rel err = %.4f; post-cal mean rel err = %.4f; cal=%.1fs",
            pre_mean_err,
            post_mean_err,
            cal_s,
        )

        results.append(
            {
                "method": method_name,
                "n_train": int(len(train)),
                "n_holdout": int(len(holdout)),
                "n_synthetic": int(len(synthetic)),
                "n_constraints": int(len(constraints)),
                "fit_wall_seconds": fit_s,
                "generate_wall_seconds": gen_s,
                "calibration_wall_seconds": cal_s,
                "pre_cal_mean_rel_err": pre_mean_err,
                "post_cal_mean_rel_err": post_mean_err,
                "calibration_max_error": validation["max_error"],
                "calibration_converged": validation["converged"],
                "pre_cal_per_target": pre,
                "post_cal_per_target": post,
                "calibrated_weights_summary": {
                    "min": float(post_weights.min()),
                    "max": float(post_weights.max()),
                    "mean": float(post_weights.mean()),
                    "std": float(post_weights.std()),
                    "zero_fraction": float((post_weights == 0).mean()),
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str))

    print()
    print("== Pre / post mean-relative-error per method ==")
    for r in sorted(results, key=lambda x: x["post_cal_mean_rel_err"]):
        print(
            f"  {r['method']:8s}: pre={r['pre_cal_mean_rel_err']:.4f}  "
            f"post={r['post_cal_mean_rel_err']:.4f}  "
            f"max={r['calibration_max_error']:.4f}  "
            f"cal={r['calibration_wall_seconds']:.1f}s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
