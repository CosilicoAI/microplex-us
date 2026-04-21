"""Isolate the calibration stage and profile its peak memory.

The v7 (microcalibrate) and v8 (pe_l0) pipelines both OOM'd at the
calibration step with ~172–197 GB of compressed memory on a 48 GB
workstation. PE-US-data's production setup runs the same L0 fit on a
T4 GPU (16 GB VRAM) successfully, which strongly suggests our
pipeline has a leak or duplication an order of magnitude larger than
the legitimate workload.

This script runs ``fit_l0_weights`` on a synthetic sparse matrix that
matches the v7 shape (1.5M records × 4k constraints, ~5% density)
*without* the surrounding pipeline. If it OOMs in isolation, the
problem is inside the L0 fit itself. If it completes at a reasonable
memory footprint, the leak is upstream (PE-table construction,
intermediate frame retained in memory, adapter build, etc.) and we
should bisect further.

Usage:

    uv run python scripts/isolate_calibration_memory.py \
        --n-records 1500000 --n-constraints 4000 --density 0.05 \
        --epochs 5

Smaller smoke:

    uv run python scripts/isolate_calibration_memory.py \
        --n-records 100000 --n-constraints 500 --density 0.05 --epochs 2
"""

from __future__ import annotations

import argparse
import gc
import os
import resource
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse as sp


def _peak_rss_gb() -> float:
    """Return current process peak RSS in GB (platform-aware)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        # macOS reports bytes.
        return r / (1024**3)
    # Linux / most BSDs: kilobytes.
    return r * 1024 / (1024**3)


@dataclass
class Stage:
    name: str
    elapsed_s: float
    peak_rss_gb: float


def _timestamp_stage(name: str, t0: float) -> Stage:
    elapsed = time.perf_counter() - t0
    peak = _peak_rss_gb()
    print(
        f"[{elapsed:>7.1f}s | peak RSS {peak:>6.2f} GB] {name}",
        flush=True,
    )
    return Stage(name=name, elapsed_s=elapsed, peak_rss_gb=peak)


def build_synthetic_problem(
    n_records: int,
    n_constraints: int,
    density: float,
    seed: int = 42,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, list[str]]:
    """Synthetic calibration fixture matching the v7/v8 shape.

    Builds a ``(n_constraints, n_records)`` CSR matrix at the given
    density with binary-indicator-ish entries (uniform in [0, 1] for
    the nonzero entries — enough to exercise torch.sparse.mm paths
    without the realism of a PE constraint system).
    """
    rng = np.random.default_rng(seed)
    total = n_constraints * n_records
    nnz = int(total * density)
    rows = rng.integers(0, n_constraints, size=nnz)
    cols = rng.integers(0, n_records, size=nnz)
    data = rng.uniform(0.5, 1.5, size=nnz).astype(np.float64)
    X = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_constraints, n_records),
        dtype=np.float64,
    )
    weights = rng.uniform(0.5, 2.0, size=n_records).astype(np.float64)
    estimated = X @ weights
    # Perturb each target by ±20% so the calibration has real work to do.
    targets = estimated * rng.uniform(0.8, 1.2, size=n_constraints)
    target_names = [f"t{i}" for i in range(n_constraints)]
    return X, targets, weights, target_names


def fit_l0(
    X_sparse: sp.csr_matrix,
    targets: np.ndarray,
    initial_weights: np.ndarray,
    target_names: list[str],
    epochs: int,
    device: str,
    lambda_l0: float,
) -> np.ndarray:
    """Delegate to PE-US-data's fit_l0_weights (same path pe_l0.py calls)."""
    try:
        from policyengine_us_data.calibration.unified_calibration import (
            fit_l0_weights,
        )
    except ImportError as exc:
        raise SystemExit(
            f"policyengine-us-data not importable: {exc}. Install it or "
            "run this script from the microplex-us venv."
        ) from exc

    achievable = np.asarray(X_sparse.sum(axis=1)).reshape(-1) > 0
    return fit_l0_weights(
        X_sparse=X_sparse,
        targets=targets,
        lambda_l0=lambda_l0,
        epochs=epochs,
        device=device,
        verbose_freq=max(1, epochs // 5),
        target_names=target_names,
        initial_weights=initial_weights,
        achievable=achievable,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--n-records", type=int, default=100_000)
    parser.add_argument("--n-constraints", type=int, default=500)
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lambda-l0", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    print(
        f"Configuration: n_records={args.n_records:,} "
        f"n_constraints={args.n_constraints:,} density={args.density} "
        f"epochs={args.epochs} device={args.device}",
        flush=True,
    )

    stages: list[Stage] = []

    t0 = time.perf_counter()
    X, targets, weights, names = build_synthetic_problem(
        n_records=args.n_records,
        n_constraints=args.n_constraints,
        density=args.density,
        seed=args.seed,
    )
    stages.append(_timestamp_stage("build CSR + targets + weights", t0))
    print(
        f"  CSR shape {X.shape}, nnz={X.nnz:,} "
        f"({X.nnz * 12 / 1024**3:.2f} GB raw storage estimate)",
        flush=True,
    )

    t0 = time.perf_counter()
    fit_l0(
        X_sparse=X,
        targets=targets,
        initial_weights=weights,
        target_names=names,
        epochs=args.epochs,
        device=args.device,
        lambda_l0=args.lambda_l0,
    )
    stages.append(_timestamp_stage("fit_l0_weights complete", t0))

    gc.collect()
    stages.append(_timestamp_stage("after gc.collect", time.perf_counter()))

    print("\n--- summary ---")
    for s in stages:
        print(f"  {s.name:<40} {s.elapsed_s:>8.1f}s   peak={s.peak_rss_gb:>6.2f} GB")
    print(f"\nFinal peak RSS: {_peak_rss_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
