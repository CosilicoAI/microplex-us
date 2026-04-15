# Imputation Conditioning Contract

This document states the current execution rule for donor conditioning in
`microplex-us`.

It is meant to answer three questions:

1. Which parts of donor conditioning are conceptually required?
2. Which parts are still experimental tuning choices?
3. Which artifact files should we read to evaluate those choices?

## Core rule

Keep three layers separate:

1. Structural contract
   - what the donor block is trying to represent
   - which entity the block lives on
   - which variables are allowed to define support
2. Predictor-surface choice
   - which compatible conditioning variables are actually used for one block
3. Downstream evaluation
   - how the imputation choice propagates through synthesis, calibration, and
     the PolicyEngine oracle

Those layers interact, but they are not the same decision.

## Conceptually required structure

These are not optional shortcuts. They are the current conceptual contract.

- Donor integration is block-based, not one flat shared-variable imputer.
- Each block has a native entity and an allowed conditioning-entity policy.
- Zero-inflated positive variables should preserve support, not just totals.
- Structural tax-unit roles matter.
  - `is_tax_unit_head`
  - `is_tax_unit_spouse`
  - `is_tax_unit_dependent`
  - `tax_unit_is_joint`
  - `tax_unit_count_dependents`
- Variable semantics decide whether a quantity is atomic, derived, signed,
  zero-inflated, or share-like.

This is the layer where we should encode ideas like "dependents are a distinct
role in the tax-unit support process" or "dividend components should not be
treated as unrelated continuous totals."

## Current production modes

The current donor-conditioning modes are:

- `all_shared`
  - use every compatible shared predictor
- `top_correlated`
  - score compatible shared predictors and keep the strongest subset
- `pe_prespecified`
  - use a PE-style structural predictor backbone declared in variable semantics
  - optionally admit a narrow supplemental shared set from the *actual*
    compatible overlap

For the current PUF IRS tax-leaf family, PE alignment means the structural-only
path. The local `policyengine-us-data`
`policyengine_us_data/calibration/puf_impute.py` implementation trains the PUF
clone QRF on demographic / tax-unit-role predictors only, and the PUF source
capability policy intentionally blocks derived convenience columns like
`income`, `employment_status`, and synthetic `state_fips` from entering donor
conditioning.

The important practical point is that `pe_prespecified` is not "use some hard
coded list no matter what." It still depends on what survives source
capabilities, semantic compatibility, entity projection, and prepared condition
surface construction.

## What is structural vs experimental

Structural:

- donor block boundaries
- support family and donor match strategy
- native entity
- role-aware PE structural predictors
- semantic transforms/checks that prevent category errors

Experimental:

- whether `all_shared`, `top_correlated`, or `pe_prespecified` wins for a given
  block family
- whether a particular variable should admit a
  `supplemental_shared_condition_vars` set
- which compatible shared predictors should be let back into a PE-structured
  block
- whether a condition surface should be widened upstream or left narrow

Usually the failure mode has been treating an experimental choice as if it were
a structural truth, or vice versa.

## What is not a real fix

These can still be useful probes, but they should not be confused for upstream
imputation repairs:

- late export-layer patches
- post-donor clipping/zeroing guards
- calibration-only improvements that hide unrealistic pre-calibration support

The current working rule is:

- if a patch improves only after calibration but worsens the pre-calibration
  imputation evidence or the mission metric, it is not a clean imputation win

## Evidence contract

We read four artifact layers for imputation questions.

### 1. Block-level conditioning evidence

- `manifest.json`
  - `synthesis.donor_conditioning_diagnostics`
- `python -m microplex_us.pipelines.summarize_donor_conditioning <artifact>`

Use this first when the question is:

- Which predictors did this donor block actually use?
- Which shared predictors were available but dropped?
- Did the block use a prepared PE-style condition surface?
- Did a requested predictor fail at raw overlap, projection, or prepared
  compatibility?

### 2. Pre-calibration imputation evidence

- `imputation_ablation.json`

Use this when the question is:

- Which variant wins support realism?
- Which variant wins weighted MAE?
- Are we trading support realism against MAE?

### 3. Full checkpoint parity evidence

- `pe_us_data_rebuild_parity.json`
- `pe_us_data_rebuild_native_audit.json`

Use these when the question is:

- Did the candidate beat the incumbent on harness slices?
- Did it beat the incumbent on the native broad loss?
- Which target families regressed?

### 4. Calibration trajectory evidence

- `manifest.json`
  - `calibration.full_oracle_capped_mean_abs_relative_error`
  - `calibration.active_solve_capped_mean_abs_relative_error`
  - deferred-stage summaries

Use this when the question is:

- Did calibration rescue the candidate?
- Did the change make the solve harder before any rescue happened?

## Current read as of 2026-04-14

- Post-hoc dependent tax-leaf guards are not a satisfactory repair.
  - they regress the mission metric
- A narrow PE-structured supplemental shared patch also failed as a real fix.
  - the raw-gate diagnostics now show why: the PUF source policy blocks
    `income`, `employment_status`, and synthetic `state_fips` from donor
    conditioning before they ever reach live overlap for these tax-leaf blocks
- The local `policyengine-us-data` read resolves the PE-alignment question.
  - PE's PUF clone QRF uses the structural demographic / tax-unit-role
    predictors only for this family
- That means the next question is not "why did compatible overlap lose these
  vars?"
  - the real question is whether we want a challenger path with source-native
    PUF predictors that survive source policy, or whether we keep the current
    structural-only PE-aligned contract

This is a better next question than "which post-hoc guard should we try next,"
because it targets the actual modeling choice instead of clipping the output
after the fact or chasing a nonexistent overlap bug.
