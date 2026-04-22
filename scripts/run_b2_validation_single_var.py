"""Compute one B2 downstream aggregate in a fresh process.

Fresh-per-variable keeps the peak memory of each variable independent
so one heavy variable (e.g. income_tax) OOM-killing doesn't wipe out
progress on the others. Append-writes to the output JSON.
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
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variable", required=True, type=str)
    parser.add_argument("--period", default=2024, type=int)
    args = parser.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] loading Microsimulation", flush=True)
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=str(args.dataset))
    print(f"[{time.strftime('%H:%M:%S')}] loaded — computing {args.variable}", flush=True)
    t0 = time.time()
    total = float(sim.calculate(args.variable, args.period).sum())
    elapsed = time.time() - t0
    print(
        f"[{time.strftime('%H:%M:%S')}] {args.variable} = ${total/1e9:.2f}B "
        f"(in {elapsed:.1f}s)",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = json.loads(args.output.read_text())
    else:
        existing = {}

    # Re-read intermediate file if present (accumulates across runs).
    raw_agg_path = args.output.with_suffix(".raw.json")
    raw_aggs = (
        json.loads(raw_agg_path.read_text()) if raw_agg_path.exists() else {}
    )
    raw_aggs[args.variable] = total
    raw_agg_path.write_text(json.dumps(raw_aggs, indent=2))

    comparison = compute_downstream_comparison(raw_aggs, DOWNSTREAM_BENCHMARKS_2024)
    report = {name: rec.to_dict() for name, rec in comparison.items()}
    args.output.write_text(json.dumps(report, indent=2))
    print(f"[{time.strftime('%H:%M:%S')}] updated {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
