# Source semantics

`microplex-us` is moving away from pipeline-level source special cases and
toward declarative source and variable semantics.

## Core idea

There are two different questions for any variable in any source:

1. Is this source authoritative for that variable?
2. Is that variable safe to use as a donor conditioning feature?

Those are not the same question.

Examples:

- A source can be authoritative for a variable while still being a bad shared
  conditioning feature.
- A variable can exist in two sources but be semantically incompatible as a
  conditioning feature because one side is derived or placeholder-filled.

## Source-level capabilities

Core type:

- `microplex.core.SourceVariableCapability`

Attached through:

- `microplex.core.SourceDescriptor.variable_capabilities`

Current public helpers on `SourceDescriptor`:

- `capability_for(variable_name)`
- `is_authoritative_for(variable_name)`
- `allows_conditioning_on(variable_name)`

## US registry layer

Country-specific policy lives in:

- `src/microplex_us/source_registry.py`

Main types:

- `SourceVariablePolicy`
- `SourceVariablePolicySpec`
- `resolve_source_variable_capabilities(...)`

This lets a source provider declare policy without embedding donor logic in the
pipeline itself.

## Variable semantics

Generic atomic-vs-derived semantics live in:

- `src/microplex_us/variables.py`

Main types and helpers:

- `VariableSemanticSpec`
- `DonorImputationBlockSpec`
- `DonorMatchStrategy`
- `VariableSupportFamily`
- `VARIABLE_SEMANTIC_SPECS`
- `resolve_variable_semantic_capabilities(...)`
- `prune_redundant_variables(...)`
- `donor_imputation_blocks(...)`
- `donor_imputation_block_specs(...)`

The donor path also now selects conditioning features per donor block rather
than using the same shared-variable set for every imputation target. There are
two distinct selection modes:

- generic selection:
  - start from variables allowed by source and variable capability metadata
  - score shared variables against the donor block
  - keep the strongest conditioning features instead of every available overlap
- `pe_prespecified` selection:
  - build a PE-style structural predictor surface when the variable semantics
    ask for it
  - use the variable's declared `preferred_condition_vars` as the structural
    backbone
  - optionally admit a narrow `supplemental_shared_condition_vars` set from the
    actual shared overlap, instead of reopening the full common-predictor pool

For the problematic PUF tax-leaf family, the PE-aligned default is still the
structural backbone only. The local `policyengine-us-data`
`calibration/puf_impute.py` path trains the PUF clone QRF on demographic /
tax-unit-role predictors only, and the PUF source policy intentionally marks
derived convenience columns like `income`, `employment_status`, and synthetic
`state_fips` as not usable for donor conditioning.

That keeps the donor path closer to the intended Microplex shape:

- declarative semantics define what is valid
- the pipeline chooses what is useful from the data
- source-specific predictor policy lives in semantics metadata rather than
  expanding ad hoc pipeline branches

The donor blocks themselves are also now declarative:

- native entity
- allowed condition entities
- projection aggregation for person-native controls when projected to a group
- block model variables
- restored output variables
- match strategy per modeled variable
- preferred conditioning variables
- supplemental shared conditioning variables
- optional frame preparation / restoration hooks

That means `us.py` now executes donor block specs rather than deciding inline
which blocks need special handling.

Artifacts now also record `synthesis.donor_conditioning_diagnostics` for each
executed donor block, including:

- donor source
- modeled/restored variables
- raw shared overlap before block preparation
- block-level shared overlap after model-variable exclusion
- whether entity projection ran, and which shared vars survived projection
- selected condition vars
- shared vars that were available but dropped
- requested supplemental shared vars
- raw-stage supplemental rejection reasons
- prepared-stage supplemental rejection reasons
- whether the block used a prepared condition surface

Use `python -m microplex_us.pipelines.summarize_donor_conditioning <artifact>`
to inspect those diagnostics from a finished artifact.

When a donor block declares a non-person native entity and those IDs are
available in the working frame, the pipeline now:

- projects scaffold and donor rows to that entity
- filters donor conditioning features through the block's declared
  `condition_entities` policy
- projects person-native conditioning variables using their semantic aggregation
  rule instead of blindly taking the first row
- fits the donor block once per native entity
- broadcasts imputed values back to person rows after matching

Current example:

- `dividend_income` and `ordinary_dividend_income` are treated as derived when
  `qualified_dividend_income` and `non_qualified_dividend_income` are present.

That means the system can automatically:

- avoid learning redundant totals as donor targets
- avoid using redundant totals as donor conditioning features
- keep the atomic basis as the source of truth
- distinguish variable families that should use household-only controls from
  those that should use person + household + native-entity controls
- respect tax-unit-native donor blocks without forcing all tax-unit variables
  through the same condition policy

## Why this matters

This is the beginning of the general rule we want:

- source-specific policy should be declarative
- variable-level atomic/derived semantics should be generic
- the donor integration pipeline should consume metadata, not source names

That is what makes the approach portable to future country packs.

## Current examples

### PUF

Current PUF policy expresses that:

- `state_fips` is not a real usable donor geography in the current build
- `tenure` is scaffold filler
- `income` is a derived convenience field, not an atomic donor target
- `employment_status` is derived, not directly observed
- `employment_income` is source-native but should not be used as a shared donor
  condition

### CPS

CPS now resolves capabilities through the same registry path, so it also picks
up generic variable semantics such as redundant dividend totals.

## Extension rule

When adding a new source, prefer:

1. Declare source-specific overrides in the source registry.
2. Declare atomic-vs-derived relationships in variable semantics.
3. Let the pipeline consume those capabilities generically.

Avoid:

- source-name `if/else` branches in the donor path
- learning overlapping derived variables independently
- using placeholder or derived variables as donor conditions just because they
  are numeric and present in both tables
- forcing tax-unit-native donor variables through person-native conditioning
  only because the seed frame happens to be person-indexed
- assuming every tax-unit-native variable should share the same household-only
  or person-level conditioning policy
