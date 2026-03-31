# AGENTS.md

This repo is the US country pack for `microplex`. Keep it thin where possible and push shared abstractions upstream into core.

## Default posture

- Prefer spec-driven behavior over ad hoc logic in large pipeline files.
- If a seam is useful for both UK and US, move it to `microplex` instead of polishing a US-only local helper.
- Keep PolicyEngine-US execution details local unless there is a clean shared protocol.

## Current architectural intent

- `microplex-us` owns:
  - US source manifests and raw source adapters
  - PolicyEngine-US execution/materialization
  - US-specific target providers and benchmark harnesses
  - US-local pipeline orchestration
- `microplex` core owns:
  - targets specs/providers/protocols
  - reweighting bundles and solver
  - benchmark metrics/comparisons/suites
  - shared result-based benchmark builders

## Current mission notes

- For US, the canonical mission metric is the PE-native broad loss frontier, not composite parity.
- When evaluating progress, prefer:
  - matched-size `Microplex@N` vs `PE@N`
  - full `enhanced_cps_2024` only as a stretch reference
- Recent direct-objective testing showed that changing only the post-export weight objective moves loss very little on the same fixed candidate.
- Bias effort toward:
  - better candidate records
  - fuller support coverage
  - budgeted selection on larger candidates
- Bias away from:
  - repeated small-candidate donor-backend A/Bs
  - more entropy tuning without evidence that the candidate population itself improved

## Review checklist

When reviewing recent changes here, check:

1. Is this still duplicating something that should now live in core?
2. Is the US harness using shared core benchmarking helpers instead of rebuilding them inline?
3. Are any benchmark claims relying on non-common-target comparisons?
4. Is the work using PE-native broad loss when it claims mission progress?
5. Does PE-US materialization handle dependency chains and partial failures safely?
6. Is this baking in fixed tax-unit structure more deeply than necessary?

## Be careful around

- `src/microplex_us/policyengine/us.py`
  - Large file with execution/materialization logic and remaining monolith risk.
- `src/microplex_us/policyengine/harness.py`
  - Should keep delegating more suite/result logic to core.
- `src/microplex_us/pipelines/local_reweighting.py`
  - Should remain a thin adapter over core bundle/reweighting surfaces.

## Standard commands

- Ruff: `uv run ruff check src tests`
- Focused comparison/harness tests: `uv run pytest -q tests/policyengine/test_comparison.py tests/policyengine/test_harness.py`
- Local reweighting tests: `uv run pytest -q tests/pipelines/test_local_reweighting.py`

## Claude/Codex review shortcut

For a quick review, read:

1. [`/Users/maxghenis/CosilicoAI/microplex-us/AGENTS.md`](/Users/maxghenis/CosilicoAI/microplex-us/AGENTS.md)
2. [`/Users/maxghenis/CosilicoAI/microplex-us/_WORKSPACE.md`](/Users/maxghenis/CosilicoAI/microplex-us/_WORKSPACE.md)
3. [`/Users/maxghenis/CosilicoAI/microplex-us/_BUILD_LOG.md`](/Users/maxghenis/CosilicoAI/microplex-us/_BUILD_LOG.md)

Then inspect changed files and return findings first.

## Review handoff

To avoid rebuilding long prompts in chat:

1. Treat [`/Users/maxghenis/CosilicoAI/microplex-us/reviews/PENDING_CLAUDE_REVIEW.md`](/Users/maxghenis/CosilicoAI/microplex-us/reviews/PENDING_CLAUDE_REVIEW.md) as the current review request.
2. Read that file after the standard repo context files above.
3. Write the full review to a dated file under [`/Users/maxghenis/CosilicoAI/microplex-us/reviews/`](/Users/maxghenis/CosilicoAI/microplex-us/reviews/).
4. Append only a concise summary to [`/Users/maxghenis/CosilicoAI/microplex-us/_BUILD_LOG.md`](/Users/maxghenis/CosilicoAI/microplex-us/_BUILD_LOG.md).
