# 2026-03-30 broad PE-native loss checkpoint review

Reviewer: Claude Opus 4.6
Scope: v2 clean broad PE-native result after deterministic CPS + rebuilt cache fixes

## Artifacts reviewed

- Clean v2 result: `artifacts/tmp_parity_inputs_broad_pe_native_20260330_v2.json`
- Earlier bad/stale result: `artifacts/tmp_parity_inputs_broad_pe_native_20260330.json`
- Provider repeatability: `artifacts/tmp_provider_repeatability_20260330.json`
- Pre-calibration repeatability: `artifacts/tmp_qrf_repeatability_precal_20260330.json`

## Code reviewed

- `src/microplex_us/data_sources/cps.py`
- `tests/test_cps_source_provider.py`
- `src/microplex_us/pipelines/pe_native_scores.py`
- `src/microplex_us/pipelines/us.py`
- `src/microplex_us/pipelines/performance.py`
- `src/microplex_us/policyengine/us.py`

## Key numbers confirmed

| Metric | Value |
|---|---|
| v2 candidate broad loss | 0.8754 |
| v1 (stale cache) candidate broad loss | 7.4331 |
| PE baseline broad loss | 0.0202 |
| v2 calibration converged | false |
| v2 constraints before feasibility | 3,611 |
| v2 constraints after feasibility | 1,255 |
| v2 constraints dropped | 2,356 (65.2%) |
| v2 kept scoring targets | 2,817 |

---

## Finding 1 — SEVERITY: HIGH — Calibration-vs-scoring target mismatch dominates the loss

The candidate is calibrated against 1,255 constraints but scored against 2,817 targets. The 1,562 unsupported targets are scored as if the candidate has zero mass for them. This is the structural reason the candidate's unweighted MSRE is ~0.887 across the board.

The top three family deltas confirm this:

| Family | Loss delta | n_targets | Candidate mean unweighted MSRE |
|---|---|---|---|
| `national_irs_other` | +0.255 | 401 | 0.841 |
| `state_agi_distribution` | +0.182 | 917 | 0.885 |
| `state_age_distribution` | +0.180 | 900 | 0.889 |

These three account for 72% of the total loss delta. The near-uniform ~0.88 MSRE across all three families is diagnostic: the problem is not family-specific accuracy but blanket thin support/zero mass.

**Impact**: Even if the calibration solver were perfect on its 1,255 constraints, the scored loss would still be dominated by the ~1,562 unsupported targets. This is the main engineering bottleneck.

**Recommendation**: Increase source sample size to widen the support surface before tuning anything else. The build log shows `sample_n=1000` improved state-age support recall from 0.464 to 0.630 vs `sample_n=500`.

## Finding 2 — SEVERITY: HIGH — Calibration never converges

All saved artifacts across the entire build history show `converged=false` on the broad path. The v2 result has `mean_error=0.789` and `max_error=1.670`.

Unconverged entropy weights are sensitive to solver internals (iteration count, step size, regularization). This means:
- The exact loss of 0.875 is not reproducible to better than ~0.02-0.03 even with fully deterministic inputs
- A/B comparisons between runs on the broad path are unreliable unless the delta exceeds the solver noise floor
- The `converged=false` flag makes it impossible to distinguish "the support surface is too thin" from "the solver ran out of iterations on a solvable problem"

**Recommendation**: Diagnose whether the solver *can* converge on the 1,255 post-filter constraints by running with 10x iterations. If it still doesn't converge, the constraint set itself may be infeasible, and the feasibility filter needs tightening.

## Finding 3 — SEVERITY: MEDIUM — Cache invalidation checks column presence, not derivation correctness

`_processed_persons_have_household_geography` (`cps.py:786-791`) validates the processed cache by checking whether required columns exist and have non-null values. It does not check whether the *derivation logic* that produced those columns matches current code.

This is the same class of bug that caused the 7.43 blow-up. The specific instance (missing columns) is fixed, but the pattern (stale derivation passing validation) is still latent.

Example future trigger: if `is_disabled` derivation changes from any-of-6-flags to 3-of-6-flags, the stale cache will pass validation because the column exists, but contain incorrect values.

**Recommendation**: Add a schema-version constant or derivation-hash to the processed cache filename (e.g., `cps_asec_2023_processed_v3.parquet`). Any loader logic change bumps the version, automatically invalidating stale caches.

## Finding 4 — SEVERITY: LOW — `national_irs_other` is a heterogeneous bucket

The single largest family delta (+0.255) is `national_irs_other`, which contains 401 targets across many distinct IRS dimensions:
- AGI bins by filing status (HOH, MFJ, Single)
- Income type totals (capital gains, partnership/S-corp, pension, qualified dividends)
- Count targets by AGI bracket

The build log drilldown (`_BUILD_LOG.md:840-870`) confirms these are different failure modes:
- Hard-zero candidate mass on capital gains, partnership/S-corp, pension
- Missing high-AGI filer mass (no tax units above $1M AGI)
- HOH filing status bins with zero mass

Treating this as one family obscures which sub-problems are fixable with current tools vs which require new source data.

**Recommendation**: Split `national_irs_other` into sub-families (e.g., `national_irs_agi_by_filing`, `national_irs_income_type`, `national_irs_count`) in the family classifier to make diagnosis actionable.

## Finding 5 — SEVERITY: LOW — Effective sample ratio is moderate but not alarming

The v2 result shows:
- Household effective sample ratio: 0.404 (808 effective households from 2,000 rows)
- Person effective sample ratio: 0.431 (2,067 effective persons from 4,791 rows)
- No weight collapse suspected
- No tiny weights

This is healthy enough for the current diagnostic phase. But as constraints increase (from fixing the support gap), the effective sample ratio will likely drop further.

---

## Answers to the five review questions

### Q1: Is the v2 result trustworthy enough for the next diagnosis step?

**Yes, conditionally.** Determinism is confirmed at the provider and pre-calibration levels. The result is safe for identifying which families dominate the gap. It is not safe for claiming precision better than ~0.02-0.03 on the loss value itself due to unconverged calibration.

### Q2: Was the 7.43 blow-up adequately explained?

**Yes, fully.** Root cause chain: new CPS-derived PE inputs added -> stale processed cache served old data missing those columns -> zeros carried through to exported H5 -> PE targets saw zero mass -> `national_census_other` blew up by +6.58. Fix: extended `PERSON_CACHE_REQUIRED_COLUMNS`, rebuilt cache. No evidence of deeper bugs.

### Q3: Are the top-3 attack surfaces correctly identified?

**Yes.** `national_irs_other` (+0.255), `state_agi_distribution` (+0.182), and `state_age_distribution` (+0.180) account for 72% of the total delta. The nuance is that `national_irs_other` is heterogeneous and should be split for actionable diagnosis.

### Q4: Is the code correct on determinism/cache?

**The two specific bugs are fixed.** Latent seam: cache invalidation checks column presence but not derivation correctness (same bug class as the 7.43 blow-up, different trigger).

### Q5: What should the next concrete engineering step be?

**Priority-ordered:**

1. **Increase source sample size** to 2000-3000 households. This is the steepest part of the support-recall curve and directly attacks the calibration-vs-scoring mismatch.

2. **Diagnose calibration convergence** by running the solver with 10x iterations on the current 1,255 post-filter constraints. If it converges, the support gap is the bottleneck. If not, the constraint set is infeasible and the filter needs tightening.

3. **Add a cache derivation version** to prevent the stale-cache class of bugs.

4. **Split `national_irs_other`** in the family classifier for actionable sub-family diagnosis.

**Do NOT pursue:**
- Sparse/L0 calibration on the broad path (634.0 loss, 3 orders of magnitude worse)
- Donor-imputer backend changes as a broad-loss lever (~0.035 total effect)
- State-floor oversampling in CPS subsampling (worsened loss in the smoke test)
- `n_synthetic > 3000` at current support levels (weight collapse documented at n=5000)
