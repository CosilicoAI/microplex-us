# Superseding `policyengine-us-data`

This document is the current working roadmap for fully superseding
`policyengine-us-data` on the US path.

It is not a paper claim. It is the operational plan that ties together:

- the `microplex-us` runtime
- the `microplex-evals` benchmark stack
- the current PE-US measurement contract
- the remaining gates between "diagnostic replacement path" and "real
  supersession"

## Core principle

`policyengine-us-data` is the incumbent dataset, not truth.

Truth is the active PE-US target database, measured through the shared
`policyengine-us` runtime. So the supersession question is:

> Can `microplex-us` produce a PE-ingestable dataset that is more useful than
> `policyengine-us-data` on the real PE-US target estate, under the same
> measurement operator, with stable runtime and artifact discipline?

That means "supersede" is not one thing. It has layers.

## Three distinct success claims

We should keep three different claims separate:

1. Architectural supersession
   - `microplex-us` is a cleaner, more modular, more spec-driven US runtime
     than `policyengine-us-data`, with better provenance, eval discipline, and
     portability.
2. PE-construction parity
   - for the important mapping and rule layers, `microplex-us` either matches
     PE's construction logic or differs intentionally with the difference
     documented.
3. PE-benchmark superiority
   - the resulting Microplex build beats matched-size PE baselines on the
     canonical PE-native benchmark frontier.

These are ordered. (1) is already valuable on its own. (2) is the main bridge
between a cleaner architecture and a trustworthy replacement claim. (3) remains
the ultimate performance goal, but it should not be the only lens used to judge
progress.

## What superseded means

There are four increasingly strong meanings of supersession:

1. Benchmark supersession
   - On the canonical PE-US broad frontier metric, a Microplex build beats the
     matched-size PE baseline.
2. Runtime supersession
   - The Microplex build/export/evaluate path is stable enough to replace the
     PE-US-data build path for regular US experiments.
3. Local-area supersession
   - The Microplex path is good enough to replace PE-US-data for the
     local-area and subnational production use cases we actually care about.
4. Architectural supersession
   - The Microplex approach is no longer "PE-US-data plus wrappers"; it is the
     canonical US runtime, with only shared measurement still delegated to
     `policyengine-us`.

Today we are somewhere between (1) as an active benchmark mission and a partial
version of (2) for research runs. We are not yet at (3) or (4).

In practice, the current program should read as:

1. architecturally supersede the PE-US-data build path
2. prove parity or intentional difference at the construction layer
3. then push for stable benchmark superiority

## Non-goals

This roadmap is **not**:

- a plan to replace `policyengine-us`
- a plan to declare victory from one narrow target slice
- a plan to port unstable US abstractions into `microplex` core early
- a plan to freeze a final architecture before the benchmark frontier settles

## Current replacement contract

The intended replacement path already has these pieces:

1. Canonical source loading
   - CPS, PUF, and other source providers load into observation frames with
     source metadata and variable capability metadata.
2. Canonical fusion and donor semantics
   - source registry
   - variable semantic registry
   - donor block specs
   - native-entity-aware donor execution where IDs exist
3. PE-US-compatible entity build/export
   - a final H5 that `policyengine-us` can ingest directly
4. Real-target evaluation
   - candidate and PE baseline are both scored through the same PE-US
     materialization/target compiler stack
5. Durable run registry
   - artifact bundle
   - `policyengine_harness.json`
   - `run_registry.jsonl`
   - `run_index.duckdb`
6. Separate eval workspace
   - method bakeoffs, family benchmarks, and paper-facing evidence live in
     `microplex-evals`

## Canonical mission metric

The current US mission is:

- beat PE on the PE-native broad loss frontier

The canonical comparison should be:

- `Microplex@N` vs `PE@N`

where:

- `N` is matched household/sample scale
- PE is allowed to be reweighted/recalibrated after sampling where the
  comparison contract requires that

Important caveat:

- the full `enhanced_cps_2024` PE dataset remains the stretch reference, but it
  is not the only pass/fail bar

## Full roadmap

### Phase 0: Measurement and artifact discipline

Goal:

- make every serious claim reproducible and comparable

Required capabilities:

- common-target PE-US harness
- active targets DB as truth
- durable artifact bundles
- run registry and DuckDB frontier index
- explicit baseline comparison against `policyengine-us-data`

Status:

- mostly done

Exit criteria:

- every headline run writes the standard artifact bundle
- frontier selection is reproducible from the registry/index alone
- candidate vs baseline comparisons are apples-to-apples on common targets

### Phase 1: PE-compatible US runtime replacement

Goal:

- replace ad hoc dataset construction with a library-first US build/export path

Required capabilities:

- source providers with canonical metadata
- fusion planning
- donor integration from declarative source/variable semantics
- PE-style entity table build
- PE-ingestable final H5 export

Status:

- done enough for research use, not frozen

Exit criteria:

- the standard US build path no longer depends on one-off scripts
- the H5 export is stable enough to be the default input for PE-native scoring
- major semantic guards live in declarative specs, not pipeline hacks

### Phase 2: Record-construction superiority

Goal:

- build records that are structurally more believable than the incumbent path

Why this phase matters:

- current evidence says record construction/support is still a larger bottleneck
  than the final weight objective
- small calibration tricks do not rescue structurally weak records

Required capabilities:

- better source-backed semantics
- decomposable-family modeling where relevant
- support-aware imputation benchmarks
- explicit support diagnostics for important policy variables and family
  decompositions

Primary evidence:

- method benchmark
- family benchmark
- family portfolio screens

Status:

- active

Exit criteria:

- the chosen runtime imputation stack clears the eval gates on the current
  family portfolio
- no major support family remains known-broken on core US variables needed by
  PE-US targets
- record realism improvements survive beyond one family or one source

### Phase 3: Full-support candidate construction and selection

Goal:

- generate a candidate population with enough real support that PE-native
  selection can operate on a strong search space

Why this phase matters:

- current broad-read evidence says full-support candidate construction plus
  budgeted household selection is a stronger lever than source subsampling or
  post-export weight tuning alone

Required capabilities:

- full-support candidate generation
- household-budgeted selection backends
- PE-native-loss-based selector path
- diagnostics for feasibility drop and weight collapse

Status:

- active, not solved

Exit criteria:

- the full-support selector path beats the current simpler broad baselines
  consistently
- selector gains survive through export and post-selection calibration
- the selected population does not rely on extreme weight collapse to win

### Phase 4: Broad frontier superiority

Goal:

- beat PE on the canonical US broad frontier, not just on a narrow diagnostic
  slice

Canonical score:

- PE-native broad loss on common targets

Secondary diagnostics:

- target win rate
- supported target rate
- common-target MARE
- family and target-delta analysis from the run index

Status:

- not done

Required evidence:

- repeated broad runs, not one lucky artifact
- matched-size PE baseline comparisons
- no hidden narrowing of the target estate without explicit benchmark contract
  changes

Exit criteria:

- Microplex wins on the canonical broad mission score against matched-size PE
  baselines
- the win survives repeated runs and nearby configuration changes
- the win is not solely explained by an overly favorable or infeasible target
  slice

### Phase 5: Held-out and local-area replacement

Goal:

- move from parity-style broad wins to replacement-quality downstream behavior
  where local-area and subnational use cases matter

Why this phase exists:

- broad parity alone is not enough for real replacement
- current docs explicitly say held-out target evaluation is not the default yet
- local-area production replacement is still future work

Required capabilities:

- held-out or shifted-target evaluation loops
- better subnational/local-area target coverage
- production-relevant calibration scopes
- explicit replacement checks for the downstream use cases that still favor
  PE-US-data today

Status:

- future work

Exit criteria:

- local-area/subnational replacement claims are backed by explicit benchmark
  contracts
- the path is no longer winning only on broad national/state composites while
  failing production-critical local slices

### Phase 6: Supersession and extraction

Goal:

- make Microplex the default US data path and extract only the stable generic
  pieces upward

Required capabilities:

- stable benchmark win
- stable runtime/export path
- clear country-pack/core boundary
- reusable abstractions promoted to `microplex` only after surviving a second
  adapter

Status:

- future work

Exit criteria:

- `microplex-us` is the canonical US dataset-generation path
- the PE-US-data path is treated as incumbent baseline/reference, not the
  default runtime dependency
- only stable benchmark/runtime semantics have moved to `microplex`

## Current blockers

As of April 5, 2026, the highest-leverage blockers are:

1. Record construction/support is still incomplete.
   - the eval stack still rejects current aggregate challengers because support
     realism is not good enough across families
2. Construction parity is still only partially audited.
   - some high-value families now have explicit parity evidence, but the
     construction/mapping contract is not yet written down broadly enough to
     call the system PE-equivalent at the rules layer
3. Full-support candidate selection gains are still being damaged downstream.
   - current build-log evidence points to post-selection entropy calibration
     undoing the strongest selector path
4. Broad superiority is not stable yet.
   - broad wins exist on some metrics/slices, but they do not yet amount to a
     stable "Microplex broadly beats PE" claim
5. Held-out and local-area replacement are not yet in the default loop.

## Current operating sequence

This is the current working order of operations:

1. Use `microplex-us` to replace PE-US-data construction logic with clearer
   source specs, variable semantics, and export contracts.
2. Keep an explicit PE construction parity matrix so "match", "close",
   and "intentionally different" are written down instead of implied.
3. Use `microplex-evals` to choose and prune runtime methods where method-level
   variation matters.
4. Re-run broad PE-native frontier experiments on matched-size baselines at
   regular checkpoints rather than after every local change.
5. Only when broad evidence stabilizes, expand held-out/local-area replacement
   work.
6. Only when a pattern survives US and appears likely to generalize, lift the
   abstraction into `microplex` and then port the shape to UK.

## Concrete gates for saying "we superseded it"

For practical purposes, we should not say "Microplex supersedes
`policyengine-us-data`" until all of these are true:

1. Runtime gate
   - the library-first US build/export path is the normal path for serious US
     runs
2. Broad benchmark gate
   - Microplex beats matched-size PE baselines on the canonical PE-native broad
     loss frontier
3. Stability gate
   - the win survives repeated runs and nearby build/config perturbations
4. Support gate
   - the chosen runtime method stack clears the support-realism bar on the eval
     portfolio
5. Local-area gate
   - production-relevant local/subnational replacement work is no longer a
     known missing layer

If only the first three are true, we can say:

- Microplex has a credible broad replacement path

If the runtime gate and construction parity are true, but the broad benchmark
gate is not yet true, we can still say:

- Microplex has architecturally superseded the PE-US-data build path

If all five are true, we can say:

- Microplex has fully superseded `policyengine-us-data` for the intended US
  use cases

## Where this plan lives

- runtime architecture and implementation:
  - `microplex-us`
- benchmark contracts and paper-facing evidence:
  - `microplex-evals`
- stable generic abstractions that survive multiple country packs:
  - `microplex`

That split is intentional. The plan should stay written here, but the evidence
for each phase should live in the repo that actually owns it.
