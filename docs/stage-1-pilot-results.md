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

## Stage 1 — ZI-QRF + ZI-MAF + ZI-QDNN at 40k and 77k rows × 50 columns

Ran both scales. **Ordering is preserved across scale**; absolute
numbers shift because the PRDC sample cap differs (see note below).

### Why the 40k intermediate run

The first 77k attempt OOM-killed during PRDC computation, not during
synthesizer fitting. PRDC on 15k real × 61k synthetic × 50 features
materializes ~7 GB-per-copy distance matrices that exceed what a
48 GB workstation can hold once multiple copies exist. Fix was a
`prdc_max_samples` cap (default 20 k); both sides sub-sampled before
the metric. With the cap in place, 77k × 50 runs cleanly.

40 k result is kept because it ran earlier without the cap (8 k real
vs 32 k synth) and is useful for the same-method-different-scale
comparison.

### Results (real ECPS, 40k × 50) — uncapped PRDC (8k × 32k)

| Method | Coverage | Precision | Density | Fit (s) | Gen (s) | Peak RSS (GB) | Zero-rate MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ZI-QRF** | **0.465** | **0.230** | **0.120** | 20.5 | 2.0 | **3.5** | **0.179** |
| ZI-MAF | 0.054 | 0.009 | 0.004 | 115.6 | 0.6 | 23.6 | 0.246 |
| ZI-QDNN | 0.306 | 0.155 | 0.063 | 52.3 | 0.6 | 32.5 | 0.299 |

### Results (real ECPS, 77k × 50) — capped PRDC at 15k × 15k

| Method | Coverage | Precision | Density | Fit (s) | Gen (s) | Peak RSS (GB) | Zero-rate MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ZI-QRF** | **0.256** | **0.233** | **0.121** | 36.0 | 3.0 | 6.0 | **0.177** |
| ZI-MAF | 0.014 | 0.008 | 0.003 | 216.2 | 1.0 | 11.0 | 0.246 |
| ZI-QDNN | 0.147 | 0.171 | 0.065 | 95.0 | 0.9 | 11.0 | 0.300 |

Total 77k wall time: 362 s (6:02). ZI-MAF's 216 s fit and ZI-QDNN's
95 s fit are the compute-bottleneck stages. ZI-QRF finishes in 36 s.

### Apples-to-apples 40k vs 77k (both PRDC-capped at 15k × 15k)

Reran 40k with the same PRDC cap as 77k so the cross-scale comparison
is directly interpretable:

| Method | 40k coverage | 77k coverage | Δ |
|---|---:|---:|---:|
| ZI-QRF | 0.352 | 0.256 | −27 % |
| ZI-QDNN | 0.222 | 0.147 | −34 % |
| ZI-MAF | 0.029 | 0.014 | −52 % |

**Coverage drops with training scale, not with data quality.** This is
a known property of PRDC: the "covered" check uses a k-NN radius set
on the real data itself. More real points make the radius tighter,
and the same synthetic sample fails to cover more real points. So the
absolute coverage number is only interpretable at a fixed real-sample
size. The *ordering*, however, is invariant — and ZI-QRF wins at both
scales. That's the production-relevant fact.

One implication: for future stage-2 / stage-3 runs, fix both
`holdout_frac` and the PRDC cap so coverage numbers are comparable
across stages. Alternatively, switch to an embedding-based PRDC that
is less sample-size-sensitive (flagged as follow-up).

### Summary across both scales

Ordering: **ZI-QRF > ZI-QDNN > ZI-MAF** on both 40k and 77k
runs. ZI-MAF coverage < 0.1 at both scales, effectively
near-collapsed. ZI-QRF wins on coverage *and* cost (3–6 GB RSS,
20–36 s fit vs 11–33 GB and 52–216 s for neural methods).

### Rare-cell preservation ratios (synthetic count / holdout count)

| Method | elderly_SE | young_dividend | disabled_SSDI | top_1% |
|---|---:|---:|---:|---:|
| ZI-QRF | 2.4 | 3.8 | **0.0** | 3.95 |
| ZI-MAF | 103.6 | 3.8 | **0.0** | 3.95 |
| ZI-QDNN | 116.7 | 3.4 | **0.0** | 3.95 |

Neural methods severely over-produce `elderly_self_employed` (100×+) —
suggests their zero-inflation classifiers are fundamentally
miscalibrated for this cell on real data. Every method drives
`disabled_ssdi` to 0.0, consistent with the pilot finding. Every method
over-produces top-1% employment at ~4×.

## Major finding: the small-benchmark ordering inverts at production scale

| Method | 10k × 7 synthetic (benchmark_multi_seed, CPS column) | 40k × 50 real ECPS |
|---|---:|---:|
| ZI-MAF | 0.499 ← winner | **0.054** |
| ZI-QDNN | 0.406 | 0.306 |
| ZI-QRF | 0.347 | **0.465** ← winner |

**Read from this result before trusting any small-scale benchmark.** The
published ranking that named ZI-MAF (and by implication ZI-QDNN as the
near-term production direction in the SS-model doc) best reversed
completely as soon as we moved to:

1. Real joint distributions instead of analytically-generated synthetic.
2. 50 columns instead of 7 (~7× feature dimensionality).
3. 40 k rows instead of 10 k (4× data).

## Interpretation

1. **ZI-MAF at 0.054 is near-collapsed.** Not merely "third-best" — it's
   producing samples that aren't close to any holdout record. Three
   plausible causes, any combination of which might be active:
   - Default hyperparameters (n_layers=4, hidden_dim=32, 50 epochs) are
     too small for 50-dim targets. The network is a per-column flow, so
     each of the 36 flows has only ~1k–5k effective parameters. May be
     fundamentally under-capacity.
   - Zero-inflation handling in ZI-MAF combines a classifier (RF, 50
     trees) for P(zero) with a MAF for nonzero values. When the
     classifier is imprecise on rare non-zero cells, the MAF has very
     few positive samples to train on, and mode-collapses.
   - The loss log-transforms positive values and standardizes; for
     heavy-tailed distributions (top-1 % income) this degrades
     conditional tail estimation.
2. **ZI-QDNN at 0.306 is mid-pack.** Better than ZI-MAF but materially
   worse than ZI-QRF. Suggests the quantile DNN's conditional
   estimates are reasonable but not tree-accurate. Worth noting RSS
   was 32 GB — highest of the three — which would OOM on a typical
   workstation without swap. Not a production-ready cost profile
   without batch-size or architecture tuning.
3. **ZI-QRF at 0.465 is the clear winner.** 3.5 GB RSS, 20-second fit,
   and nearly 2× ZI-QDNN's coverage. This is the production default for
   the rewire's cross-section synthesizer step.

## Implications for the SS-model methodology doc

The SS-model methodology doc's "production direction: ZI-QDNN" claim
does not survive this benchmark. At production scale on real data with
default hyperparameters, neither ZI-MAF nor ZI-QDNN is competitive with
ZI-QRF. The doc should be updated to note this finding, and the
longitudinal extension should treat ZI-QRF as at minimum a strong
baseline.

Two caveats that keep the SS-model direction alive:

1. Hyperparameter-tuned ZI-MAF / ZI-QDNN *might* beat ZI-QRF. The
   scale-up doc listed "ZI-MAF needs careful hyperparameter tuning on
   real data" as a known risk; stage-1 confirms the risk.
2. Trajectory / pathwise generation is a different problem from
   cross-sectional conditional modeling. A sequence-model win at
   longitudinal need not follow from cross-sectional results.
3. Both neural methods used 32-GB-class memory to train; at the 3.4 M
   row v6 scale the naive extrapolation is ~1.6 TB. Tree methods'
   modest memory profile may be decisive on a workstation regardless
   of quality.

## Follow-up work flagged by this run

1. **61k ZI-QRF OOM diagnosis.** Scaling is clean up to 40 k (3.5 GB
   RSS). 61 k fails silently in < 2 min with SIGKILL. Most likely
   cause: loky workers accumulating memory across the 36 target
   columns. Fix paths: `n_jobs=4` instead of `-1`, or a
   worker-recycling wrapper, or just disable parallelism and accept
   slower fit.
2. **ZI-MAF hyperparameter search.** Before accepting
   ZI-MAF-is-not-viable as the final answer, run with n_layers=8,
   hidden_dim=128, epochs=200 and see if coverage recovers. One
   evening of tuning could either rescue the method or definitively
   rule it out.
3. **Embedding-based PRDC.** Raw-feature PRDC in 50 dimensions is
   predicted by the scale-up doc to degenerate. Fit a 16-dim
   autoencoder on holdout, re-run PRDC in that space, and check
   whether the method ordering changes. If it does, the 50 k result
   is a metric artifact, not a method verdict.
4. **Per-column zero-rate breakdown.** All three methods drive
   `disabled_ssdi` to 0.0 synthetic count. Needs per-column MAE
   reporting to identify which other columns systematically break.
5. **`microcalibrate` applied on top.** The synthesizer results above
   are uncalibrated. The mainline pipeline runs synthesis then
   calibration. Worth repeating stage 1 with `MicrocalibrateAdapter`
   applied to the generated records and measuring whether calibration
   lifts ZI-MAF / ZI-QDNN coverage back into the competitive range.

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
