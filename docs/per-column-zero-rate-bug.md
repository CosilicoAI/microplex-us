# Per-column zero-rate breakdown reveals upstream bug

*Analysis of `artifacts/per_col_zero_rate_20k.json` at 20k × 50, all three methods. The top-10 "most broken" columns across every method are **conditioning** variables, which the synthesizer is supposed to preserve — not target them.*

## The pattern

Top-diff columns per method include, identically across ZI-QRF / ZI-MAF / ZI-QDNN:

| Column | Real zero-rate | Synth zero-rate | Diff |
|---|---:|---:|---:|
| `is_military` | 0.998 | 0.000 | 0.998 |
| `is_separated` | 0.991 | 0.000 | 0.991 |
| `is_blind` | 0.984 | 0.000 | 0.984 |
| `has_marketplace_health_coverage` | 0.958 | 0.000 | 0.958 |
| `is_full_time_college_student` | 0.955 | 0.000 | 0.955 |
| `is_disabled` | 0.900 | 0.000 | 0.900 |
| `is_hispanic` | 0.783 | 0.000 | 0.783 |
| `own_children_in_household` | 0.707 | 0.000 | 0.707 |
| `pre_tax_contributions` | 0.557 | 0.000 | 0.557 |
| `is_female` | 0.494 | 0.000 | 0.494 |

Every one of these is in `DEFAULT_CONDITION_COLS`, not in the target column set. Stage-1's synthesizer framework treats conditioning variables as shared input, sampled from the training pool without generation. In real data these are binary (`0.0` or `1.0`). In synthetic output they are continuous floats with values like `-0.34`, `0.75`, `1.14`.

## Root cause (upstream bug)

In `microplex/src/microplex/eval/benchmark.py::_MultiSourceBase.generate` (lines 260–262):

```python
sample_idx = rng.choice(len(self.shared_data_), size=n, replace=True)
shared_values = self.shared_data_.iloc[sample_idx].values.copy()
shared_values += rng.normal(0, 0.1, shared_values.shape)  # <-- bug
```

A constant Gaussian noise of σ=0.1 is added to **every** shared-column value, including binary-valued categoricals (`is_female`, `is_military`, etc.). This is presumably there to prevent memorization of training records, but it has two destructive effects:

1. **Binary variables become continuous.** `is_military=1` becomes `1.04` or `0.87`; `is_military=0` becomes `-0.05` or `0.08`. No synthetic record has exactly 0 or exactly 1.
2. **Categorical integers become continuous.** `cps_race=3` becomes `3.02` or `2.93`. State FIPS codes, occupation codes, etc. all get noise-perturbed into non-integer values.

## How this affects stage-1

1. **Per-column zero-rate breakdown is dominated by the bug.** The "most-broken" columns are conditioning variables that were never the synthesizer's job to produce; the large `abs_diff` entries are the noise knocking binary values off the integer grid. Downstream consumers reading the zero-rate per-column need to filter out conditioning columns to see the real target-column story.

2. **PRDC coverage numbers are roughly preserved in their ordering.** All three methods receive the same noise on the same shared columns, so the 10× gap between ZI-QRF and ZI-MAF isn't an artifact of the bug. Noise reduces coverage uniformly across methods; it doesn't flip ordering. But the *absolute* coverage numbers would be higher if the bug were fixed — likely by 5–15 %.

3. **Calibrate-on-synth is affected.** The initial-weight rescale in the calibration script uses `synthetic[col].sum()` for target-column proxies; those target columns don't have the shared-col noise bug, so that part is unaffected. But if any categorical target was in the shared-cols set (it isn't with current defaults), its noise-polluted values would distort weighted aggregates.

## What to fix

In `microplex/src/microplex/eval/benchmark.py::_MultiSourceBase.generate`, replace the unconditional noise injection with a type-aware version:

```python
shared_values = self.shared_data_.iloc[sample_idx].values.copy()
# Only add noise to continuous shared columns, not categoricals.
for j, col in enumerate(self.shared_cols_):
    dtype = self.shared_data_[col].dtype
    n_unique = self.shared_data_[col].nunique()
    if dtype.kind == "f" and n_unique > 10:  # heuristic: continuous float
        shared_values[:, j] += rng.normal(0, 0.1, size=n)
```

Or, cleaner: pass explicit `continuous_shared_cols` / `categorical_shared_cols` lists into the method class, so the noise logic is explicit rather than heuristic.

## Local mitigation in microplex-us

Until the upstream fix lands, microplex-us can:

- Post-process synthetic output in the harness to round/snap binary conditioning columns to their nearest value (0 or 1) before PRDC and before calibration. One-liner per column.
- Filter the per-column zero-rate report to only show target columns, so the signal from the bug doesn't drown the actual synthesis quality signal.

Both are good follow-ups; not blocking for G1.

## What to publish in the scale-up doc

The stage-1 method ordering is still valid — noise is uniform across methods and doesn't reorder them. But the absolute coverage numbers should be annotated: "measured with the upstream `_MultiSourceBase.generate` noise-injection bug in place; corrected numbers pending fix."

## Artifact

`artifacts/per_col_zero_rate_20k.json` — full per-method zero-rate breakdown including all columns.
