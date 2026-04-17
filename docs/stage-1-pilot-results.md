# Stage 1 pilot results — synthesizer scale-up on real ECPS

*First execution of `docs/synthesizer-benchmark-scale-up.md`'s stage-1 protocol on real enhanced_cps_2024 data. This doc captures the pilot (5,000-row subsample, 1 method) and the first full stage-1 run (77,006 rows, 3 methods) as they complete.*

## Data

- Source: `~/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/enhanced_cps_2024.h5`
- Full row count: **77,006** (PE's national-scale 2024 ECPS)
- Columns: 50 (14 demographics conditioning + 36 income / wealth / benefit targets)
- Stage-1 split: 61,604 train / 15,402 holdout (80/20, seed=42)

Note: ECPS has 77k rows in its national-scale build; the 100k-row stage-1 target from the protocol doc isn't achievable from this file alone. The harness uses `n_rows=None` to take all 77k and reports actual row counts in each result.

## Pilot — ZI-QRF at 5,000 rows × 50 columns

First validation that the harness runs end-to-end on real data with the curated default columns. Sanity-check result, not a benchmark claim.

| Method | Train rows | Holdout rows | Cols | Coverage | Precision | Density | Fit (s) | Gen (s) | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ZI-QRF | 4,000 | 1,000 | 50 | **0.641** | 0.617 | 0.233 | 5.0 | 1.0 | 0.87 GB |

Interpretation: PRDC coverage of 0.641 on 5k × 50 is a sensible baseline — better than the existing benchmark's 10k × 7 synthetic ZI-QRF CPS coverage of 0.347 (per `benchmark_multi_seed.json`). Two possible explanations, both worth noting:

1. **Data realism:** real ECPS has structure that multi-source-fusion-from-synthetic doesn't. Single-source QRF can fit the real marginals and correlations directly.
2. **Column set:** the new 50-column default includes richer conditioning signal than the prior 7-column setup.

### Rare-cell preservation (pilot)

| Check | Synthetic / Real ratio |
|---|---:|
| elderly_self_employed | 2.00 |
| young_dividend | 4.38 |
| disabled_ssdi | 0.00 |
| top_1pct_employment | 3.91 |

Pattern: ZI-QRF *over-samples* rare non-zero cells (elderly SE, young dividend, top-1 % employment) — the zero-inflation classifier predicts non-zero slightly too aggressively for these categories. The `disabled_ssdi` check returning 0 is concerning: the model is predicting zero SSDI for disabled persons, which is the opposite of what the underlying data structure says. Likely because SSDI receipt conditional on disability is lower in ECPS than intuition suggests, and the model learned the unconditional zero-rate. Needs follow-up at full scale.

### Zero-rate MAE (pilot)

0.180 — mean absolute error in per-column zero-rate between real and synthetic is ~18 percentage points. That's substantial. Most likely driven by target columns where the zero-inflation classifier diverges from real; worth breaking down per column at stage 1.

## Stage 1 — ZI-QRF + ZI-MAF + ZI-QDNN at 77,006 rows × 50 columns

**Status: running at 2026-04-16 23:50 ET.** Results will be appended here when the job completes.

Expected completion based on ballpark from `docs/synthesizer-benchmark-scale-up.md`:

- ZI-QRF fit: ~15 minutes (36 target cols × ~25s each on 61k rows × 100 trees)
- ZI-MAF fit: probably 45 min – 2 hours on CPU (no MPS integration in the benchmark class; one flow per column × 50 epochs × 256 batch size)
- ZI-QDNN fit: ~20 min (smaller network, CPU-friendly)
- Generation: 5–15 min per method

Total stage 1 wall time: 1–3 hours.

Output: `artifacts/scale_up_stage1.json`, `artifacts/scale_up_stage1.log`.

### Results (TO BE POPULATED)

Template table — update in place once the job completes:

| Method | Coverage | Precision | Density | Fit (s) | Gen (s) | Peak RSS | Zero-rate MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ZI-QRF | — | — | — | — | — | — | — |
| ZI-MAF | — | — | — | — | — | — | — |
| ZI-QDNN | — | — | — | — | — | — | — |

### Rare-cell preservation ratios (TO BE POPULATED)

| Method | elderly_SE | young_div | disabled_SSDI | top_1% |
|---|---:|---:|---:|---:|
| ZI-QRF | — | — | — | — |
| ZI-MAF | — | — | — | — |
| ZI-QDNN | — | — | — | — |

## Interpretation guide (for when results land)

Key comparisons to watch for:

1. **Does the small-benchmark ordering (ZI-MAF > ZI-QDNN > ZI-QRF on CPS) hold on real 77k × 50?**
   - Previously on 10k × 7 synthetic CPS-schema: ZI-MAF 0.499 > ZI-QDNN 0.406 > ZI-QRF 0.347.
   - If preserved → supports the preliminary G1 synthesizer default of ZI-MAF.
   - If inverted → the small-scale ordering was an artifact of the synthetic generator's simplicity and needs revisiting.

2. **Is ZI-QRF competitive at real 77k × 50?**
   - Pilot gave 0.641 at 5k. If stage 1 sustains > 0.55 on 77k, ZI-QRF is a viable fallback for environments without PyTorch.

3. **Rare-cell preservation at scale**:
   - Does every method preserve `disabled_ssdi` at non-zero ratio, unlike the pilot? Failure at scale would confirm a systematic zero-inflation bug.

4. **Runtime vs coverage frontier**:
   - ZI-QRF fit in minutes, ZI-MAF in hours. If ZI-MAF gets 0.65 and ZI-QRF gets 0.60 but with 30× the compute, the effective production choice is ZI-QRF until ZI-MAF's lead grows or GPU acceleration lands.

5. **Does PRDC in 50D give interpretable numbers?**
   - The scale-up doc predicted PRDC may degenerate in high dimensions. If all three methods cluster between 0.60 and 0.75 (noise range) on stage 1, raw-feature PRDC has hit its ceiling and we need to add an embedding-based PRDC for stage 2+.

## Known limitations of this stage

- **Single-source only.** The harness runs each synthesizer on ECPS alone; the multi-source fusion aspect of the v6 pipeline is out of scope for stage 1. Fusion is exercised earlier in the microplex-us pipeline (donor integration) upstream of calibration.
- **No calibration.** These are synthesis-only results. Calibration via `MicrocalibrateAdapter` happens downstream and is not part of this benchmark.
- **CPU-only torch.** The benchmark method classes don't expose a `device` argument. ZI-MAF and ZI-QDNN fit on CPU, which is a conservative upper bound on training time. Adding MPS or CUDA support to the benchmark classes is a discrete follow-up that could shrink stage-1 wall time by 3–5×.
- **No seed replication.** Stage 1 runs at seed=42 only. Confidence intervals across seeds are in the protocol but deferred.

## Follow-up work flagged by this stage

1. **Incremental result persistence.** Current harness writes all results atomically at the end. If ZI-QDNN fails, ZI-QRF and ZI-MAF numbers are lost. Patch the runner to save each method's ScaleUpResult as soon as it completes.
2. **Embedding-based PRDC.** Fit a 16-dim autoencoder on `holdout` and compute PRDC in that space. Compare to raw-feature PRDC to diagnose dimensionality effects.
3. **Per-column zero-rate breakdown.** Expose `zero_rate_per_column` alongside the scalar MAE so the doc can pinpoint which columns drive the error.
4. **GPU support in benchmark methods.** Pass `device` through to torch-based methods.
