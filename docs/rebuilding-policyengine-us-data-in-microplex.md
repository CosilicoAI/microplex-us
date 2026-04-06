# Rebuilding `policyengine-us-data` in Microplex

This document states the current execution rule for the rebuild track:

> Rebuild the incumbent `policyengine-us-data` pipeline inside the cleaner
> `microplex-us` structure first. Improve architecture immediately. Improve
> results only on the margin during the rebuild pass. Save materially different
> modeling choices for a later, explicit challenger phase.

That is a stricter rule than "make it better however we can."

## Current runtime entry points

The rebuild track now has explicit code-owned runtime entry points in
[`pe_us_data_rebuild.py`](/Users/maxghenis/CosilicoAI/microplex-us/src/microplex_us/pipelines/pe_us_data_rebuild.py):

- `default_policyengine_us_data_rebuild_config(...)`
- `default_policyengine_us_data_rebuild_source_providers(...)`
- `build_policyengine_us_data_rebuild_pipeline(...)`

These are meant to make the incumbent-parity path callable as a first-class
Microplex profile rather than a loose collection of remembered settings.

That profile now also includes:

- the PE-style PUF Social Security QRF split mode
- the PE-style prespecified donor-predictor mode for source imputations
- opt-in ACS/SIPP/SCF donor-survey source providers for the rebuild path

## Why this rule exists

If we mix:

- source and mapping parity
- model-family changes
- predictor-surface changes
- weighting-backend changes
- calibration-objective changes

in the same pass, then we lose attribution.

We may still end up with a better system, but we will not know whether it is:

- a faithful rebuild of the incumbent path
- a materially different model stack
- or both

The rebuild track is meant to answer a simpler question first:

> Can Microplex reproduce the incumbent PE-US-data build path in a more
> sustainable, modular, spec-driven form?

## What is allowed in the rebuild pass

Allowed changes are architectural improvements that should move outputs only on
the margin:

- replacing implicit scripts with explicit source/provider contracts
- turning inline special cases into declarative stage specs
- wrapping incumbent PE weighting backends behind Microplex interfaces
- making parity assumptions explicit in docs, tests, and artifacts
- adding provenance, parity audits, and stage-level artifacts
- reorganizing code so country-pack boundaries and pipeline ownership are clear

These are improvements in:

- maintainability
- provenance
- portability
- reproducibility
- evaluation discipline

without trying to win by changing the underlying statistical contract.

## What is not allowed by default in the rebuild pass

These should be treated as explicit departures, not silent cleanup:

- changing model class
  - e.g. replacing incumbent QRF stages with grouped-share or forest-share
- materially changing predictor surfaces
  - e.g. replacing a PE-style prespecified predictor set with a broader
    data-driven feature search
- changing fallback heuristics in ways likely to move support or totals
- changing weighting/calibration objectives or optimization backends
- introducing new target surfaces as if they were still measuring the same
  incumbent pipeline

Any such change can still be good. It just belongs in the challenger phase,
where it is measured as an intentional departure.

## Practical decision rule

When we face a design choice during the rebuild:

1. Ask whether the incumbent PE-US-data behavior is clear enough to reproduce.
2. If yes, reproduce it in cleaner Microplex structure.
3. Only deviate if the incumbent choice would create an obvious architectural
   problem.
4. If we deviate, choose the smallest alternative that should change outputs
   only on the margin.
5. Write the deviation down as `intentional` rather than letting it masquerade
   as parity.

## Examples

### Good rebuild-pass changes

- Keep PE's QRF family, but call it through a Microplex method spec instead of
  an inline script.
- Keep the incumbent predictor set, but declare it in one stage config rather
  than scattering it across files.
- Keep the PE weighting backend, but call it through a Microplex-owned adapter.
- Keep the same CPS reason-code logic, but express it in a source adapter with
  explicit parity tests.

### Too far for the rebuild pass

- switching from PE's prespecified QRF predictors to a broader automatic
  feature search because it seems statistically cleaner
- replacing the incumbent SS split model with a new forest-share family before
  we have a parity implementation
- changing the calibration objective because the incumbent one is inconvenient

## Relationship to the parity matrix

This rule is what makes the parity matrix meaningful.

If we follow it, then the matrix statuses:

- `Exact`
- `Close`
- `Compatible, not equivalent`
- `Different`

actually describe the build path.

If we ignore it, the matrix becomes unstable because every "cleanup" quietly
changes the underlying model contract.

## Relationship to later outperformance work

This rule does **not** say we should stop improving the pipeline.

It says:

1. rebuild the incumbent path cleanly
2. prove parity or intentional difference
3. then run challenger methods against that rebuilt baseline

That sequence is what gives later benchmark wins credibility.
