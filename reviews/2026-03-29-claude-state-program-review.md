# Claude Review — 2026-03-29

Synthesized code review: US state-program accuracy work

## Combined findings

### Critical

1. Calibration does not converge; the earlier "beating PE" claim was on unconverged weights.
   - Artifact: `microplex-us/artifacts/tmp_state_programs_feasible_bootstrap_rerun_20260329.json`
   - The review’s core point was that the entropy solver still had `converged: false`, with large remaining error, so the headline PE win was not yet a credible solved result.

### High

2. `min_active_households=1` lets degenerate constraints through.
   - File: `microplex-us/src/microplex_us/pipelines/us.py`
   - Recommendation: raise the floor to `5-10`.

3. `has_medicaid` used the wrong support family.
   - File: `microplex-us/src/microplex_us/variables.py`
   - Recommendation: treat it as binary / zero-inflated rather than `BOUNDED_SHARE`.

4. `ensure_target_support()` is a band-aid, not a structural support fix.
   - File: `microplex-us/src/microplex_us/pipelines/us.py`
   - Recommendation: do not mistake dtype/exemplar fixes for real calibration support.

5. SNAP entity mismatch risk: `SPM_UNIT` spec vs household calibration path.
   - Files: `microplex-us/src/microplex_us/variables.py`, `microplex-us/src/microplex_us/policyengine/harness.py`
   - Recommendation: explicitly verify the SPM-unit-to-household projection path.

### Medium

6. All-zero transform fix had no warning.
   - File: `microplex/src/microplex/transforms.py`
   - Recommendation: emit a warning when the identity fallback is used.

7. Condition-var auto-promotion was unconditional.
   - File: `microplex-us/src/microplex_us/pipelines/us.py`
   - Recommendation: avoid blindly promoting sparse or continuous proxies into the conditioning space.

8. The review claimed there were effectively no project-level tests around these seams.
   - This was directionally aimed at regression protection, but factually overstated; `microplex-us` does have substantial test coverage.

9. Core transform fix only had Synthesizer-level coverage.
   - File: `microplex/tests/test_synthesizer.py`
   - Recommendation: add more direct transform-path coverage.

### Low

10. No warning when a scaffold is missing all support proxies.
11. `ZeroInflatedTransform.combine()` length mismatch guard is still absent.
12. Artifact manifest paths are local-filesystem-coupled.

## Architectural risks

1. Sparse state coverage is structural; calibration tuning alone does not create small-state support.
2. The corrected 102-constraint state-only path and the broader 3,611-constraint path can tell very different stories.
3. Without regression gates, condition-var and feasibility changes can silently flip the diagnosis again.

## Top 3 next fixes

1. Add a small-state oversampling floor in bootstrap/synthesis.
2. Raise `min_active_households` and warn when many constraints are dropped.
3. Add regression coverage for feasibility filtering, support filling, condition-var promotion, and harness slice stability.

## Direct answers

- Is the new diagnosis actually right?
  - Partially. Calibration infeasibility was real, but the deeper issue is still sparse small-state support.

- Is the corrected feasible state-only benchmark path sound and comparable to PE?
  - It is a sound diagnostic slice, but not a full replacement for the broader canonical benchmark.

- Do the new `n=2000` results support the claim that Microplex now beats PE on the corrected US state-program slice?
  - Not convincingly at the time of review, because the solve was still unconverged.

- What is the next highest-leverage fix?
  - Improve small-state support, especially via sampling / support strategy, rather than only filtering constraints.
