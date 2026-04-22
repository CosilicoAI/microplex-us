"""Batched Microsimulation aggregate for one variable.

The naive one-shot ``Microsimulation.calculate(income_tax, 2024).sum()``
OOMs on 1.5M households because the dependency chain materializes
~100+ intermediate arrays (each 3.4M floats = 27 MB) in memory
simultaneously. This runner subsets the h5 into household-size chunks,
runs a fresh Microsimulation per chunk, and accumulates the weighted
sum.

Entity-level subsetting is done by index, matching
``policyengine_us_data``'s h5 layout: household-level arrays index by
position in ``household_id``; person-level arrays index by position in
``person_household_id``; same for tax_unit, spm_unit, family,
marital_unit.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np


HOUSEHOLD_ID = "household_id"

ENTITY_ID_COLUMNS = {
    "household": "household_id",
    "person": "person_id",
    "tax_unit": "tax_unit_id",
    "spm_unit": "spm_unit_id",
    "family": "family_id",
    "marital_unit": "marital_unit_id",
}
# Person → group-entity foreign keys.
PERSON_TO_GROUP_LINK = {
    "tax_unit": "person_tax_unit_id",
    "spm_unit": "person_spm_unit_id",
    "family": "person_family_id",
    "marital_unit": "person_marital_unit_id",
}


def _load_all_arrays(h5_path: Path, period_key: str) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        out = {}
        for key in f.keys():
            if period_key in f[key]:
                out[key] = np.asarray(f[key][period_key])
        return out


def _entity_of(variable: str, arrays: dict[str, np.ndarray]) -> str:
    """Classify a variable by matching its array length to an entity's id column."""
    n = len(arrays[variable])
    entity_lengths = {
        entity: len(arrays[id_col])
        for entity, id_col in ENTITY_ID_COLUMNS.items()
        if id_col in arrays
    }
    for entity, length in entity_lengths.items():
        if length == n:
            return entity
    return "unknown"


def _build_entity_masks(
    arrays: dict[str, np.ndarray], chunk_hh_ids: np.ndarray
) -> dict[str, np.ndarray]:
    """Produce boolean masks into each entity array for the households in ``chunk_hh_ids``."""
    hh_id = arrays["household_id"]
    chunk_set = set(chunk_hh_ids.tolist())
    masks: dict[str, np.ndarray] = {}
    masks["household"] = np.isin(hh_id, chunk_hh_ids)
    person_hh = arrays["person_household_id"]
    person_mask = np.isin(person_hh, chunk_hh_ids)
    masks["person"] = person_mask
    for entity, link_col in PERSON_TO_GROUP_LINK.items():
        id_col = ENTITY_ID_COLUMNS[entity]
        if link_col not in arrays or id_col not in arrays:
            continue
        group_ids_in_chunk = np.unique(arrays[link_col][person_mask])
        masks[entity] = np.isin(arrays[id_col], group_ids_in_chunk)
    return masks


def _write_chunk_h5(
    arrays: dict[str, np.ndarray],
    entity_masks: dict[str, np.ndarray],
    period_key: str,
    tmp_path: Path,
) -> None:
    """Write a subset h5 keeping only rows matching each variable's entity mask."""
    with h5py.File(tmp_path, "w") as f:
        for variable, values in arrays.items():
            entity = _entity_of(variable, arrays)
            mask = entity_masks.get(entity)
            if mask is None or len(values) != len(mask):
                continue
            group = f.create_group(variable)
            group.create_dataset(period_key, data=values[mask])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--variable", required=True, type=str)
    parser.add_argument("--period", default=2024, type=int)
    parser.add_argument("--batch-size", default=50_000, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    period_key = str(args.period)
    print(f"[{time.strftime('%H:%M:%S')}] loading all arrays from {args.dataset}", flush=True)
    arrays = _load_all_arrays(args.dataset, period_key)
    print(
        f"[{time.strftime('%H:%M:%S')}] loaded {len(arrays)} variables",
        flush=True,
    )

    hh_ids = arrays[HOUSEHOLD_ID]
    n_hh = len(hh_ids)
    print(f"[{time.strftime('%H:%M:%S')}] {n_hh} households; batch_size={args.batch_size}", flush=True)

    total = 0.0
    n_batches = (n_hh + args.batch_size - 1) // args.batch_size

    from policyengine_us import Microsimulation  # noqa: PLC0415

    for batch_idx in range(n_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, n_hh)
        chunk_hh_ids = hh_ids[start:end]

        entity_masks = _build_entity_masks(arrays, chunk_hh_ids)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "chunk.h5"
            _write_chunk_h5(arrays, entity_masks, period_key, tmp_path)

            t0 = time.time()
            sim = Microsimulation(dataset=str(tmp_path))
            values = sim.calculate(args.variable, args.period)
            chunk_sum = float(values.sum())
            total += chunk_sum
            elapsed = time.time() - t0

        print(
            f"[{time.strftime('%H:%M:%S')}] batch {batch_idx+1}/{n_batches} "
            f"(households {start}-{end}): ${chunk_sum/1e9:.3f}B "
            f"cumulative=${total/1e9:.3f}B ({elapsed:.1f}s)",
            flush=True,
        )

    print(
        f"\n[{time.strftime('%H:%M:%S')}] {args.variable} total = ${total/1e9:.2f}B",
        flush=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_agg_path = args.output.with_suffix(".raw.json")
    raw_aggs = (
        json.loads(raw_agg_path.read_text()) if raw_agg_path.exists() else {}
    )
    raw_aggs[args.variable] = total
    raw_agg_path.write_text(json.dumps(raw_aggs, indent=2))

    from microplex_us.validation.downstream import (  # noqa: PLC0415
        DOWNSTREAM_BENCHMARKS_2024,
        compute_downstream_comparison,
    )

    comparison = compute_downstream_comparison(raw_aggs, DOWNSTREAM_BENCHMARKS_2024)
    report = {name: rec.to_dict() for name, rec in comparison.items()}
    args.output.write_text(json.dumps(report, indent=2))
    print(f"[{time.strftime('%H:%M:%S')}] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
