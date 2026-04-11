# Methodology Ledger

This document is the living methods record for `microplex-us`.

It is not the paper. It is the shortest place in the repo that should answer:

- what the current canonical pipeline is
- what PolicyEngine is doing in that pipeline
- which methodological choices are considered canonical today
- which choices are explicitly provisional or challenger-only
- where the evidence for those choices is stored

## Core framing

`microplex-us` is not trying to literally recreate `policyengine-us-data`.

Current framing:

- `policyengine-us` is the shared measurement operator
- the active PE-US targets DB is the truth surface we score against
- `policyengine-us-data` is the incumbent comparator and interface reference
- `microplex-us` is an independent US data-construction runtime

That means incumbent-compatibility work exists to improve attribution and
interface confidence, not to define the project as a wrapper around PE-US-data.

## Claim separation

We keep four claims separate:

1. Architecture claim
   - `microplex-us` is a cleaner, more modular, more auditable runtime.
2. Oracle-compatibility claim
   - where important, Microplex matches or intentionally departs from incumbent
     PE-US-data construction behavior.
3. Benchmark claim
   - Microplex produces a better PE-ingestable dataset than the incumbent on
     the active target estate.
4. Paper claim
   - a stable narrative about methodology, evidence, and novelty that can be
     defended externally.

The first three live in code and artifacts now. The fourth should be written
from them later, not invented separately.

## Canonical pipeline

The current broad US pipeline is:

1. Load raw survey/tax sources into canonical observation frames.
2. Apply source semantics and variable semantics.
3. Build donor blocks and donor-condition surfaces.
4. Impute donor-only variables into the scaffold population.
5. Synthesize a candidate population.
6. Build PolicyEngine-ingestable entity tables.
7. Export final H5.
8. Run PolicyEngine materialization and compare implied aggregates to the active
   target DB.
9. Save artifact bundles, sidecars, and registry/index records.

This is a fresh Microplex pipeline with a PolicyEngine evaluation boundary, not
an attempt to make PE-US-data the runtime architecture.

## What is currently canonical

- Source and variable semantics are declared in Microplex-owned registries and
  manifests.
- Final evaluation uses the shared PE-US runtime and active targets DB.
- Artifact discipline is required for serious runs:
  - `manifest.json`
  - `data_flow_snapshot.json`
  - `policyengine_harness.json` when harness evaluation runs
  - `policyengine_native_scores.json` when PE-native broad loss runs
  - `pe_us_data_rebuild_parity.json` for incumbent-compatibility checkpoints
  - `pe_us_data_rebuild_native_audit.json` for target/family/support audit
  - `run_registry.jsonl`
  - `run_index.duckdb`
- Incumbent-compatibility modes are allowed when they improve attribution.
- Materially different model choices should be explicit challenger variants.

## What is still provisional

- The default imputation stack is still under active evaluation.
- Support realism vs MAE tradeoffs are still live methodological questions.
- Full-support candidate construction and selector design are not settled.
- Calibration is still operationally important, but it is not the only or even
  always the dominant methodological lever.
- Held-out evaluation is not yet the default outer loop.

These should not be written up later as if they were settled all along.

## Current methodological evidence surfaces

Use these surfaces when writing claims down later:

- [benchmarking.md](/Users/maxghenis/CosilicoAI/microplex-us/docs/benchmarking.md)
  for the truth/comparator/operator contract
- [policyengine-oracle-compatibility.md](/Users/maxghenis/CosilicoAI/microplex-us/docs/policyengine-oracle-compatibility.md)
  for incumbent-compatibility rules
- [pe-construction-parity.md](/Users/maxghenis/CosilicoAI/microplex-us/docs/pe-construction-parity.md)
  for audited construction-layer matching vs intentional difference
- saved artifact bundles for actual run-level evidence
- tests for the code-enforced contract behind those claims

## Update rule

Update this document when any of the following changes:

- the canonical measurement contract
- the default runtime pipeline shape
- the default imputation or selection method family
- the meaning of the parity/audit sidecars
- the set of artifacts required for a headline claim
- the boundary between incumbent-compatibility work and challenger work

## Naming note

Some internal module names still say `pe_us_data_rebuild`.

Treat that as historical naming, not as the canonical project description. The
canonical description is:

- Microplex is the runtime
- PolicyEngine is the oracle/evaluator
- PE-US-data is the incumbent comparator
