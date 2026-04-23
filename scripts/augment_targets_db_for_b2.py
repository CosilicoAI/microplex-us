"""Copy the calibration targets DB and add direct targets on SSI / CTC / ACA PTC.

The v11 downstream validation showed those three aggregates drifting
+64% / +32% / -76% from their benchmark totals. They weren't in the
original calibration target set (which focuses on AGI / income
marginals, not downstream-disbursed amounts). Adding them as direct
national targets should drive their calibrated aggregates toward the
benchmark values.

Stratum 1 is "United States" (from the existing DB). Period 2024 and
reform_id=0 (baseline) match the rest of the 2024 target set.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

from microplex_us.validation.downstream import DOWNSTREAM_BENCHMARKS_2024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["ssi", "ctc", "aca_ptc"],
    )
    parser.add_argument("--period", default=2024, type=int)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.source, args.output)

    benchmarks_by_name = {spec.name: spec for spec in DOWNSTREAM_BENCHMARKS_2024}

    con = sqlite3.connect(args.output)
    cur = con.cursor()
    for variable in args.variables:
        spec = benchmarks_by_name.get(variable)
        if spec is None:
            raise KeyError(f"No 2024 benchmark spec for {variable}")
        cur.execute(
            "SELECT COUNT(*) FROM targets WHERE variable=? AND period=? "
            "AND stratum_id=1 AND reform_id=0",
            (variable, args.period),
        )
        if cur.fetchone()[0] > 0:
            print(f"[skip] {variable} already has a national 2024 target")
            continue
        cur.execute(
            "INSERT INTO targets "
            "(variable, period, stratum_id, reform_id, value, active, source, notes) "
            "VALUES (?, ?, 1, 0, ?, 1, ?, ?)",
            (
                variable,
                args.period,
                float(spec.benchmark),
                spec.source,
                f"B2 follow-up direct target for {variable}",
            ),
        )
        print(
            f"[add ] {variable} @ 2024 national: ${spec.benchmark/1e9:.1f}B ({spec.source})"
        )
    con.commit()
    con.close()
    print(f"\nWrote augmented DB to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
