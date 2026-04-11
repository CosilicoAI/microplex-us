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

## Methods snapshot

### Snapshot as of 2026-04-10

This is the current working methods snapshot, not a claim of finality.

| Area | Current reading | Status | Main evidence |
| --- | --- | --- | --- |
| Measurement contract | `policyengine-us` plus the active targets DB are the oracle. `policyengine-us-data` is the incumbent comparator. | `Canonical` | [benchmarking.md](/Users/maxghenis/CosilicoAI/microplex-us/docs/benchmarking.md) |
| Runtime boundary | Microplex owns source loading, donor integration, synthesis, entity build, export, artifacts, and experiment tracking. PolicyEngine owns measurement/materialization at eval time. | `Canonical` | [architecture.md](/Users/maxghenis/CosilicoAI/microplex-us/docs/architecture.md) |
| Incumbent-compatibility work | PE-style modes are used where they improve attribution or interface confidence, but they do not define the whole project. | `Canonical` | [policyengine-oracle-compatibility.md](/Users/maxghenis/CosilicoAI/microplex-us/docs/policyengine-oracle-compatibility.md) |
| Construction parity claim | Some construction layers are close or compatible, but general PE-construction parity is not yet established. | `Canonical` | [pe-construction-parity.md](/Users/maxghenis/CosilicoAI/microplex-us/docs/pe-construction-parity.md) |
| Imputation evaluation | We currently track both support realism and MAE. Neither should be collapsed into a single unqualified "best" method. | `Canonical` | [pe_us_data_rebuild_parity.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_parity.json), [pe_us_data_rebuild_native_audit.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_native_audit.json) |
| Current production imputation reading | `structured_pe_conditioning` is the support winner on the current checkpoint ablation; `top_correlated_qrf` is the MAE winner. | `Provisional` | [pe_us_data_rebuild_parity.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_parity.json), [pe_us_data_rebuild_native_audit.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_native_audit.json) |
| Broad mission metric | The mission metric is PE-native broad loss frontier, but pre-calibration support evidence is retained so unrealistic imputations do not hide behind later weighting. | `Canonical` | [superseding-policyengine-us-data.md](/Users/maxghenis/CosilicoAI/microplex-us/docs/superseding-policyengine-us-data.md), [pe_us_data_rebuild_native_audit.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_native_audit.json) |
| Current benchmark reading | On the current checkpoint artifact, harness metrics improved versus the incumbent comparator, but native broad loss is still much worse than `enhanced_cps_2024`. | `Canonical` | [pe_us_data_rebuild_parity.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_parity.json), [pe_us_data_rebuild_native_audit.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_native_audit.json) |
| Current cross-run regression reading | Across 66 scored modelpass checkpoint runs, `national_irs_other` appears in the top 3 every time, `state_agi_distribution` in 63/66, and `state_aca_spending` in 54/66. Near-term model work should target those recurring families directly rather than broad tuning. | `Provisional` | [live_pe_us_data_rebuild_checkpoint_modelpass_regression_summary_20260410.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_modelpass_regression_summary_20260410.json) |
| Current `national_irs_other` drilldown reading | The audited `national_irs_other` failures are concentrated in filing-status-sensitive IRS cells and coincide with large `SINGLE` and `JOINT` overcounts plus `SEPARATE` undercounts. The first remediation step is to preserve source-authoritative filing-status inputs into the PE construction path. | `Provisional` | [live_pe_us_data_rebuild_checkpoint_national_irs_other_drilldown_20260410.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_national_irs_other_drilldown_20260410.json) |

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

## Current open questions

- Should runtime imputation selection prioritize support realism, weighted MAE,
  or a gated combination of the two?
- How much conditioning structure should be imposed before flexible donor/QRF
  prediction begins?
- How much of the remaining broad-loss gap is record construction versus
  selection/calibration?
- Which incumbent-compatible modes are worth keeping as long-run options, and
  which should remain diagnostic-only?
- When should held-out evaluation become a required gate rather than an optional
  extra?

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

For the current checkpoint-style evidence bundle, the most useful files are:

- [manifest.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/manifest.json)
- [data_flow_snapshot.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/data_flow_snapshot.json)
- [policyengine_harness.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/policyengine_harness.json)
- [policyengine_native_scores.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/policyengine_native_scores.json)
- [pe_us_data_rebuild_parity.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_parity.json)
- [pe_us_data_rebuild_native_audit.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_native_audit.json)
- [imputation_ablation.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/imputation_ablation.json)
- [live_pe_us_data_rebuild_checkpoint_modelpass_regression_summary_20260410.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_modelpass_regression_summary_20260410.json)
- [live_pe_us_data_rebuild_checkpoint_national_irs_other_drilldown_20260410.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_national_irs_other_drilldown_20260410.json)

## Decision log

### 2026-04-10: Project framing

- Decision:
  - describe `policyengine-us` as the oracle/evaluator and
    `policyengine-us-data` as the incumbent comparator
- Why:
  - this matches how the system is actually being used
  - it avoids understating the novelty of the Microplex runtime
  - it keeps incumbent-compatibility work from swallowing the whole project
- Evidence:
  - [benchmarking.md](/Users/maxghenis/CosilicoAI/microplex-us/docs/benchmarking.md)
  - [policyengine-oracle-compatibility.md](/Users/maxghenis/CosilicoAI/microplex-us/docs/policyengine-oracle-compatibility.md)

### 2026-04-10: Imputation evaluation contract

- Decision:
  - keep support realism and MAE as separate evidence channels
  - do not summarize imputation quality using post-calibration loss alone
- Why:
  - the current checkpoint artifact shows a real tradeoff
  - `structured_pe_conditioning` wins support
  - `top_correlated_qrf` wins weighted MAE
  - collapsing the two too early would hide methodology risk
- Evidence:
  - [pe_us_data_rebuild_parity.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_parity.json)
  - [pe_us_data_rebuild_native_audit.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/pe_us_data_rebuild_native_audit.json)
  - [imputation_ablation.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/imputation_ablation.json)

### 2026-04-10: Artifact contract for headline claims

- Decision:
  - treat sidecars and registry metadata as part of the methodology, not just
    engineering exhaust
- Why:
  - paper-facing claims will need reproducible evidence with exact configs,
    metrics, and comparison slices
  - the artifact bundle is now the canonical storage layer for that evidence
- Evidence:
  - [manifest.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/manifest.json)
  - [data_flow_snapshot.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/checkpoints/checkpoint-ablation-real-20260410a/data_flow_snapshot.json)

### 2026-04-10: Cross-run regression priority

- Decision:
  - prioritize targeted fixes for `national_irs_other`,
    `state_agi_distribution`, and then `state_aca_spending`
- Why:
  - across recent modelpass checkpoint families, the same regressions recur even
    when total loss improves substantially
  - `national_irs_other` appears in the top 3 for all 66 scored runs
  - `state_agi_distribution` appears in the top 3 for 63/66 runs and is the
    largest regressing family in 34 runs
  - `state_aca_spending` appears in the top 3 for 54/66 runs but is more often
    a secondary or tertiary regression
- Evidence:
  - [live_pe_us_data_rebuild_checkpoint_modelpass_regression_summary_20260410.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_modelpass_regression_summary_20260410.json)

### 2026-04-10: First `national_irs_other` remediation target

- Decision:
  - first fix the preservation of source-authoritative filing-status inputs in
    the PE-oracle rebuild path before attempting more downstream status tuning
- Why:
  - audited `national_irs_other` lead runs show repeated IRS target failures in
    filing-status-sensitive cells, especially `Single`, `Joint`, and high-AGI
    bins
  - those same audited runs show large `SINGLE` and `JOINT` count surpluses,
    large `SEPARATE` deficits, and missing or distorted MFS support bins
  - the saved candidate seed/synthetic/calibrated rows for leading runs retain
    `marital_status` but not `filing_status_code`, so the authoritative PUF tax
    filing code is disappearing before tax-unit construction
- Evidence:
  - [live_pe_us_data_rebuild_checkpoint_national_irs_other_drilldown_20260410.json](/Users/maxghenis/CosilicoAI/microplex-us/artifacts/live_pe_us_data_rebuild_checkpoint_national_irs_other_drilldown_20260410.json)

## Update rule

Update this document when any of the following changes:

- the canonical measurement contract
- the default runtime pipeline shape
- the default imputation or selection method family
- the meaning of the parity/audit sidecars
- the set of artifacts required for a headline claim
- the boundary between incumbent-compatibility work and challenger work

## Paper extraction rule

When writing the eventual paper:

1. Start from this ledger, not from memory.
2. Pull claims only from code-backed docs and artifact-backed evidence.
3. Preserve the distinction between canonical, provisional, and open items.
4. Cite the exact artifact family that supported each headline claim.
5. Avoid rewriting temporary engineering names like `pe_us_data_rebuild` into
   misleading methodological claims.

## Naming note

Some internal module names still say `pe_us_data_rebuild`.

Treat that as historical naming, not as the canonical project description. The
canonical description is:

- Microplex is the runtime
- PolicyEngine is the oracle/evaluator
- PE-US-data is the incumbent comparator
