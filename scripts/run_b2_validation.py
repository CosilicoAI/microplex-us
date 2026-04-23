"""Run B2 downstream validation on a calibrated PE-US h5.

One variable at a time, flushing progress and intermediate output to
disk so a partial run leaves usable state. Uses the
``microplex_us.validation.downstream`` module for the benchmark set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from microplex_us.validation.downstream import (
    DOWNSTREAM_BENCHMARKS_2024,
    compute_downstream_comparison,
    compute_downstream_weighted_aggregate,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--period", default=2024, type=int)
    args = parser.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] loading Microsimulation from {args.dataset}", flush=True)
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=str(args.dataset))
    print(f"[{time.strftime('%H:%M:%S')}] loaded", flush=True)

    variables = [spec.name for spec in DOWNSTREAM_BENCHMARKS_2024]
    aggregates: dict[str, float] = {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_path = args.output.with_suffix(".partial.json")

    for variable in variables:
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] computing {variable} ...", flush=True)
        try:
            total = compute_downstream_weighted_aggregate(sim, variable, args.period)
        except Exception as exc:
            print(f"  {variable}: FAILED ({exc})", flush=True)
            aggregates[variable] = float("nan")
        else:
            aggregates[variable] = total
            elapsed = time.time() - t0
            print(
                f"  {variable}: ${total/1e9:,.2f}B (in {elapsed:.1f}s)",
                flush=True,
            )
        # Flush partial state to disk after each variable so an OOM
        # kill after N variables still leaves N results on disk.
        intermediate_path.write_text(json.dumps(aggregates, indent=2))

    comparison = compute_downstream_comparison(aggregates, DOWNSTREAM_BENCHMARKS_2024)
    report = {name: rec.to_dict() for name, rec in comparison.items()}
    args.output.write_text(json.dumps(report, indent=2))
    intermediate_path.unlink(missing_ok=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] B2 validation complete", flush=True)
    print(f"Wrote {args.output}", flush=True)

    print(f"\n{'variable':<12s} {'computed':>12s} {'benchmark':>12s} {'rel_error':>10s}")
    for name, rec in sorted(comparison.items()):
        rel = rec.rel_error
        rel_str = f"{rel*100:+.1f}%" if rel is not None else "N/A"
        print(
            f"{name:<12s} ${rec.computed/1e9:>9.2f}B "
            f"${rec.benchmark/1e9:>9.2f}B  {rel_str:>10s}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
