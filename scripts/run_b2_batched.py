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
STRUCTURAL_VARIABLE_ENTITIES = {
    "household_id": "household",
    "household_weight": "household",
    "person_id": "person",
    "person_household_id": "person",
    "person_weight": "person",
    "tax_unit_id": "tax_unit",
    "person_tax_unit_id": "person",
    "tax_unit_weight": "tax_unit",
    "spm_unit_id": "spm_unit",
    "person_spm_unit_id": "person",
    "spm_unit_weight": "spm_unit",
    "family_id": "family",
    "person_family_id": "person",
    "family_weight": "family",
    "marital_unit_id": "marital_unit",
    "person_marital_unit_id": "person",
    "marital_unit_weight": "marital_unit",
}


def _load_all_arrays(h5_path: Path, period_key: str) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        out = {}
        for key in f.keys():
            if period_key in f[key]:
                out[key] = np.asarray(f[key][period_key])
        return out


def _load_policyengine_variable_entities() -> dict[str, str]:
    try:
        from policyengine_us import (
            system as policyengine_system_module,  # noqa: PLC0415
        )
    except ImportError:
        return {}

    tax_benefit_system = getattr(policyengine_system_module, "system", None)
    if tax_benefit_system is None:
        return {}
    variables = getattr(tax_benefit_system, "variables", {})
    entity_map: dict[str, str] = {}
    for name, metadata in variables.items():
        entity_key = getattr(getattr(metadata, "entity", None), "key", None)
        if entity_key is not None:
            entity_map[str(name)] = str(entity_key)
    return entity_map


def _entity_of(
    variable: str,
    arrays: dict[str, np.ndarray],
    *,
    variable_entities: dict[str, str] | None = None,
) -> str:
    """Classify a variable, preferring PE metadata over fragile length matching."""
    explicit_entity = STRUCTURAL_VARIABLE_ENTITIES.get(variable)
    if explicit_entity is not None:
        return explicit_entity
    if variable_entities is not None and variable in variable_entities:
        return variable_entities[variable]
    n = len(arrays[variable])
    entity_lengths = {
        entity: len(arrays[id_col])
        for entity, id_col in ENTITY_ID_COLUMNS.items()
        if id_col in arrays
    }
    matches = [entity for entity, length in entity_lengths.items() if length == n]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous entity for variable {variable!r}: matched {matches} by length"
        )
    return "unknown"


def _build_entity_masks(
    arrays: dict[str, np.ndarray], chunk_hh_ids: np.ndarray
) -> dict[str, np.ndarray]:
    """Produce boolean masks into each entity array for the households in ``chunk_hh_ids``."""
    hh_id = arrays["household_id"]
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
    *,
    variable_entities: dict[str, str] | None = None,
) -> None:
    """Write a subset h5 keeping only rows matching each variable's entity mask."""
    with h5py.File(tmp_path, "w") as f:
        for variable, values in arrays.items():
            entity = _entity_of(
                variable,
                arrays,
                variable_entities=variable_entities,
            )
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
    variable_entities = _load_policyengine_variable_entities()
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

    from microplex_us.validation.downstream import (  # noqa: PLC0415
        compute_downstream_weighted_aggregate,
    )

    for batch_idx in range(n_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, n_hh)
        chunk_hh_ids = hh_ids[start:end]

        entity_masks = _build_entity_masks(arrays, chunk_hh_ids)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "chunk.h5"
            _write_chunk_h5(
                arrays,
                entity_masks,
                period_key,
                tmp_path,
                variable_entities=variable_entities,
            )

            t0 = time.time()
            sim = Microsimulation(dataset=str(tmp_path))
            chunk_sum = compute_downstream_weighted_aggregate(
                sim,
                args.variable,
                args.period,
            )
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
