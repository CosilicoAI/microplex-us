# PSID coverage = 0 in `benchmark_multi_seed.json`: diagnosed

*Closes the open question raised in `docs/synthesizer-benchmark-scale-up.md`.*

## Summary

PSID coverage is 0.0 across all 6 methods (QRF, ZI-QRF, QDNN, ZI-QDNN, MAF, ZI-MAF) for all 10 seeds **not because PSID is unsynthesizable, but because the benchmark harness collapses PSID conditioning to 2 variables** (`is_male` and `age`) when it computes the shared-column pool.

This is a benchmark-architecture bug, not a data limitation. PSID is still a viable backbone for the SS-model longitudinal extension, conditional on fixing or bypassing this specific benchmark setup.

## Reproduction

Input: `microplex/data/stacked_comprehensive.parquet` (630,216 rows, 38 cols, stacks sipp + cps + psid).

Benchmark setup (`microplex/scripts/run_benchmark.py` + `microplex/src/microplex/eval/benchmark.py`):

1. For each source, keep only numeric columns with <5 % NaN, then `dropna()`.
2. Compute `shared_cols` = columns present in ALL sources with <5 % NaN each.
3. Each synthesizer is trained as a multi-source fusion: pool `shared_cols` across sources, fit a per-column model for each non-shared column on only the source that has it.
4. At generation: sample a shared-column record, then predict each non-shared column from its per-source model conditioned on the shared columns.
5. Per-source PRDC coverage: holdout = that source's full column set; synthetic = generated records' intersecting column set; `prdc` library computes coverage with k=5.

Diagnostic script (runs in a few seconds):

```python
import pandas as pd
import numpy as np

df = pd.read_parquet("data/stacked_comprehensive.parquet")
numeric_dtypes = [np.float64, np.int64, np.float32, np.int32]
exclude = {"weight", "person_id", "household_id", "interview_number"}

survey_dfs = {}
for src in ["sipp", "cps", "psid"]:
    sub = df[df["_survey"] == src].drop(columns=["_survey"]).copy()
    num = [c for c in sub.columns
           if sub[c].dtype in numeric_dtypes and sub[c].isna().mean() < 0.05]
    survey_dfs[src] = sub[num].dropna().reset_index(drop=True)
    print(src, len(survey_dfs[src]), num)

first = next(iter(survey_dfs.values()))
shared = [c for c in first.columns
          if c not in exclude and all(c in d.columns for d in survey_dfs.values())]
print("shared_cols:", shared)
```

Output:

| Source | Rows after dropna | Low-NaN numeric columns |
|---|---:|---|
| SIPP | 476,744 | hispanic, race, is_male, wave, job_gain, age, job_loss, weight, month |
| CPS | 144,265 | state_fips, is_male, dividend_income, farm_income, age, self_employment_income, weight, rental_income, wage_income, interest_income |
| PSID | 9,207 | state_fips, food_stamps, total_family_income, is_male, marital_status, year, dividend_income, taxable_income, age, weight, rental_income, wage_income, interview_number, social_security, interest_income |

**Intersection after excluding `{weight, person_id, household_id, interview_number}`: `['is_male', 'age']` — 2 columns.**

## Why this gives PSID coverage 0

- PSID has the **most** unique non-shared columns (13 of its 15 are non-shared), all trained per-column on only 9,207 rows conditioned on 2 shared variables.
- PRDC for PSID is computed on PSID's full 15-column feature space. The synthesizer's predicted values for the 13 non-shared columns are drawn from a model that's severely under-conditioned (2D conditioning on 13 target dimensions, each with a per-column RF or flow trained on 9,207 rows).
- k-NN coverage with k=5 in 15D looks for any synthetic record within the k-th nearest-neighbor distance of each real holdout record. With under-conditioned predictions the synthetic records cluster around model means and rarely fall within the real holdout's neighborhood ball. Coverage → 0.
- CPS has 10 total columns with 8 non-shared and 144,265 rows → coverage ~0.34–0.50 (mediocre but non-zero). SIPP has 9 total columns with 7 non-shared and 476,744 rows → coverage ~0.72–0.95 (highest). **The pattern tracks column-uniqueness ratio and row count.** PSID is worst because its non-shared ratio is highest and its row count is lowest.

## Why this is a benchmark bug, not a PSID limitation

The benchmark implicitly assumes sources share rich conditioning information. Here the `<5 % NaN` filter removes many latently-shared columns from individual sources. For example, `wage_income` appears in both CPS (144,265 non-null) and PSID (9,207 non-null) but NOT in SIPP — so it's excluded from `shared_cols`. If the benchmark harmonized the column schema across sources before applying the NaN filter (either by imputing cross-source or by using an intersection-of-non-null-across-sources strategy), `shared_cols` would be much richer and all sources would benefit.

PSID itself has 15 low-NaN columns — more than either SIPP (9) or CPS (10). On a **PSID-only** benchmark (train on PSID, test on PSID holdout), coverage would likely be competitive with SIPP's.

## Implications for the architecture work

### For synthesizer selection (G1 cross-section)

- **The benchmark's PSID=0 verdict should not influence cross-section synthesizer choice.** G1 works with CPS-core scaffold, not PSID, so the issue doesn't propagate. My earlier recommendation of ZI-MAF for cross-section and ZI-QRF for panel stands.

### For SS-model longitudinal extension (G3)

- **PSID can still be the trajectory-training backbone.** The SS-model methodology doc's plan to use PSID (1968–present) for lifetime earnings trajectories is not invalidated by this benchmark.
- However, before committing compute, run a **PSID-only synthesizer benchmark**: train ZI-MAF / ZI-QRF / ZI-QDNN on PSID alone, test on PSID holdout. That is the relevant evaluation for the SS-model use case. The existing multi-source benchmark result for PSID is not the relevant number.
- If PSID-only benchmarks still show low coverage, the real issue may be the attrition-induced sparsity in PSID's joint feature space (real data limitation). That is a separate investigation.

### For the benchmark harness itself (deprioritized)

- The benchmark's `find_shared_cols` policy is brittle at the intersection: any source with a different NaN rate on a column knocks that column out of the shared pool for every source. For future benchmark work, consider:
  - Lift the NaN filter or pre-impute cross-source.
  - Report results **per-source** on same-source train/test splits, not cross-source.
  - Report `shared_cols` and per-source `non_shared_cols` counts alongside coverage so reviewers can see the conditioning bottleneck.

## Action items

1. **Update `docs/synthesizer-benchmark-scale-up.md`** to note this finding — the PSID=0 line in the initial summary should be annotated, not taken as evidence that PSID is unusable.
2. **Before any SS-model work commits compute to PSID-based trajectory training**, run a PSID-only synthesizer benchmark. That is a ~day of work on `experiments/` with existing method classes.
3. **No change to G1 plan.** Cross-section proceeds with CPS-scaffold as planned; PSID is not on the G1 critical path.

## What was reliable in the original PSID=0 signal

- It is genuine that the specific multi-source fusion benchmark here cannot cover PSID well. Consumers who use that benchmark output (e.g., paper draft in `microplex/paper/paper_results.py`) need to adjust claims accordingly — it is not valid to say "all methods fail on PSID." The valid claim is "cross-source fusion with 2 shared variables fails on PSID, in a way that tracks non-shared column ratio."
