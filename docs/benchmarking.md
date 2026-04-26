# Benchmarking

The benchmark question is:

> Is Microplex closer to the real target DB than `policyengine-us-data` is?

## What is truth

Truth is the active target set loaded from the PE-US targets DB.

Main provider:

- `microplex_us.policyengine.PolicyEngineUSDBTargetProvider`

The baseline dataset is not truth. It is only the incumbent comparator.

## What PolicyEngine does

`policyengine-us` is the shared measurement operator.

Both:

- the Microplex candidate dataset
- the `policyengine-us-data` baseline dataset

are run through the same PE-US variable materialization and the same target
compiler before being compared to the same targets.

So the benchmark shape is:

`dataset -> policyengine-us -> implied aggregates -> compare to target DB`

## Current default harness

Default saved-build evaluation now uses:

- the full active PE-US target estate
- one `all_targets` slice

Main files:

- `src/microplex_us/policyengine/harness.py`
- `src/microplex_us/policyengine/comparison.py`

## Main metrics

Per run:

- `candidate_composite_parity_loss`
- `baseline_composite_parity_loss`
- `candidate_mean_abs_relative_error`
- `baseline_mean_abs_relative_error`
- `target_win_rate`
- `supported_target_rate`

The frontier metric is currently:

- `candidate_composite_parity_loss`

This is a diversity-aware outer loss over the target set rather than a raw
target-count-weighted mean alone.

## Saved outputs

Every serious saved run can write:

- artifact bundle directory
- `policyengine_harness.json`
- `run_registry.jsonl`
- `run_index.duckdb`
- `pe_native_target_diagnostics_current.json`

These live under the selected artifact root.

## Diagnostics dashboard

The repo includes a static dashboard at `dashboard/` for inspecting the full
PE-native target diagnostic dataset. It expects the JSON payload written by:

```bash
microplex-us-pe-native-target-diagnostics \
  --from-dataset /path/to/enhanced_cps_2024.h5 \
  --to-dataset /path/to/policyengine_us.h5 \
  --policyengine-targets-db /path/to/policy_data.db \
  --output-path artifacts/pe_native_target_diagnostics_current.json
```

The JSON includes full per-target rows, family summaries, scope summaries, top
improvements, top regressions, and target DB match metadata when a structured
PolicyEngine target DB is available. The dashboard loads that default artifact
when served from the repo root, and can also load an arbitrary diagnostic JSON
from disk.

## Inspecting runs

Useful Python APIs:

- `select_us_microplex_frontier_entry(...)`
- `select_us_microplex_frontier_index_row(...)`
- `list_us_microplex_target_delta_rows(...)`
- `compare_us_microplex_target_delta_rows(...)`

The last helper is meant for questions like:

- what changed between two broad runs?
- which targets improved under a source-policy change?
- which target families regressed even when overall loss improved?

## Current broad reference point

As of March 27, 2026, the best recorded broad `national + state` `CPS+PUF`
frontier in the main artifact root was:

- artifact id: `cps_puf_500_native_wages`
- candidate composite parity loss: `0.8906`
- baseline composite parity loss: `4.5412`
- candidate mean absolute relative error: `0.9928`
- baseline mean absolute relative error: `1.1920`

That does **not** mean Microplex is already better on most targets. The same run
had a low `target_win_rate`, meaning the gain comes from improving the overall
loss surface rather than beating the incumbent on a majority of individual
targets.

## Important caveats

- This is parity evaluation, not held-out evaluation.
- Calibration and evaluation still overlap unless explicitly separated in build
  config.
- A broad win on the composite loss is not the same thing as a majority-target
  win.
- Local-area production parity is not finished yet.

## Repro pattern

Broad versioned builds use:

- `build_and_save_versioned_us_microplex(...)`
- `build_and_save_versioned_us_microplex_from_source_provider(...)`
- `build_and_save_versioned_us_microplex_from_source_providers(...)`

The resulting run can then be inspected through the JSON artifacts or via the
DuckDB index.
