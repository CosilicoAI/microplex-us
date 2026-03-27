# Architecture

`microplex-us` is the US-specific country package built on top of the generic
`microplex` engine.

## Package split

- `microplex`: generic engine pieces
  - source descriptors and observation frames
  - fusion planning
  - synthesis and calibration
  - canonical target spec and provider protocol
  - generic geography and entity abstractions
- `microplex-us`: US-specific implementations
  - CPS, PUF, and other source providers
  - PE-US target import and compilation
  - PE-US export and evaluation
  - US experiment, registry, and artifact layers

## Current build flow

Main entrypoint:

- `microplex_us.pipelines.USMicroplexPipeline`

Current broad flow:

1. Load one or more `SourceProvider`s into `ObservationFrame`s.
2. Build a `FusionPlan` from the source descriptors.
3. Choose a public structured scaffold source.
4. Prepare canonical seed data from the scaffold.
5. Integrate donor-only variables from other sources using source and variable
   capability metadata, with donor-block-specific automatic condition selection,
   declared condition-entity policy, and native-entity projection when entity
   IDs are available.
6. Synthesize a new population.
7. Build PolicyEngine-style entity tables.
8. Materialize PE-derived features needed by targets.
9. Calibrate against PE-US DB targets.
10. Export a PE-ingestable H5 and evaluate against the full active target set.

Important files:

- `src/microplex_us/pipelines/us.py`
- `src/microplex_us/policyengine/us.py`
- `src/microplex_us/policyengine/comparison.py`
- `src/microplex_us/pipelines/artifacts.py`
- `src/microplex_us/pipelines/index_db.py`

## What is already true

- The package is library-first. The core build, artifact saving, experiment
  running, and frontier tracking all live in importable APIs.
- PolicyEngine evaluation uses the real `policyengine-us-data` targets DB as
  truth targets.
- Saved runs persist:
  - artifact bundle
  - `policyengine_harness.json`
  - `run_registry.jsonl`
  - `run_index.duckdb`

## What is not final yet

- Broad PE-US parity is not stable yet.
- The current US path is still scaffold-plus-donors rather than a fully
  symmetric multientity latent-population model.
- Held-out target evaluation is not the default loop yet.
- Local-area production replacement is still future work.

## Design direction

The intended long-run shape is:

- canonical source metadata
- canonical variable semantics
- multientity fusion
- derived-variable materialization after atomic modeling
- target compilation as a generic feature/filter/aggregation problem

The current implementation is already moving in that direction:

- canonical target spec
- source capability registry
- variable semantic registry
- donor block specs with declared match strategies
- donor block specs with declared condition-entity policy
- variable semantics with declared projection aggregation for group-level donor fits
- automatic donor condition selection from source overlap plus data signal
- native-entity donor execution for tax-unit-native blocks when IDs are present
- full-target PE-US harness

But it is still an actively evolving system, not a finished paper architecture.
