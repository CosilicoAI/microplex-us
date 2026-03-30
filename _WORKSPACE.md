# _WORKSPACE.md

This file is the durable local context for `microplex-us`.

## Repo role

`microplex-us` is the US country pack. It should specialize core `microplex`, not fork it conceptually.

Core repo:

- [`/Users/maxghenis/CosilicoAI/microplex`](/Users/maxghenis/CosilicoAI/microplex)

Sibling country pack:

- [`/Users/maxghenis/CosilicoAI/microplex-uk`](/Users/maxghenis/CosilicoAI/microplex-uk)

## Current high-value modules

### PolicyEngine-US

- `src/microplex_us/policyengine/us.py`
- `src/microplex_us/policyengine/harness.py`
- `src/microplex_us/policyengine/comparison.py`

### Local reweighting

- `src/microplex_us/pipelines/local_reweighting.py`

### Source semantics / manifests

- `src/microplex_us/variables.py`
- `src/microplex_us/data_sources/`
- `src/microplex_us/manifests/`

## Current architectural boundary

US should keep local:

- PE-US microsimulation/materialization details
- US target database/provider specifics
- raw CPS/PUF and other US source mappings

US should not keep local if it generalizes:

- benchmark math
- benchmark suite/result types
- reweighting math
- generic target querying/filtering

## Important current caveat

US tax filing units may eventually be policy-endogenous. Avoid hard-baking tax-unit structure too deeply into shared abstractions.

## Current benchmark guidance

- Use common-target comparisons when claiming candidate vs baseline wins.
- Composite parity loss remains useful for US frontier work, but it is not interchangeable with suite-level mean absolute relative error.

## High-signal tests

- `tests/policyengine/test_comparison.py`
- `tests/policyengine/test_harness.py`
- `tests/policyengine/test_us.py`
- `tests/pipelines/test_local_reweighting.py`

## Working rule

If the same helper starts appearing in both PE-US and PE-UK benchmark/reweighting flows, promote it to core.

## Review handoff

- Current durable Claude request:
  - `/Users/maxghenis/CosilicoAI/microplex-us/reviews/PENDING_CLAUDE_REVIEW.md`
- Full saved reviews belong under:
  - `/Users/maxghenis/CosilicoAI/microplex-us/reviews/`
- `_BUILD_LOG.md` should only keep a concise review summary, not the full review body.
