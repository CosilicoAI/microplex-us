# Stage-1 results after fixing the shared-col noise bug

*Corrected stage-1 numbers after the categorical-snap mitigation landed. The raw numbers in `docs/stage-1-pilot-results.md` are preserved for historical reference but should not be cited; the post-snap numbers here are the real measurement.*

## The fix in one line

`microplex.eval.benchmark._MultiSourceBase.generate` adds σ=0.1 Gaussian noise to *every* shared-column value, including binary / categorical ones. The harness now snaps those values back to their training-pool grid after generation. See `docs/per-column-zero-rate-bug.md`.

## Corrected stage-1 at 40k × 50 (PRDC capped 15k/15k)

| Method | Coverage | Precision | Density | Fit (s) | Peak RSS (GB) | Zero-rate MAE |
|---|---:|---:|---:|---:|---:|---:|
| **ZI-QRF** | **0.979** | 0.913 | 0.902 | 20.0 | 3.5 | 0.016 |
| ZI-QDNN | 0.796 | 0.848 | 0.766 | 52.5 | 11.8 | 0.136 |
| ZI-MAF | 0.168 | 0.030 | 0.022 | 114.6 | 11.8 | 0.084 |

## Corrected stage-1 at 77k × 50 (full ECPS)

| Method | Coverage | Precision | Density | Fit (s) | Peak RSS (GB) | Zero-rate MAE |
|---|---:|---:|---:|---:|---:|---:|
| **ZI-QRF** | **0.928** | 0.910 | 0.885 | 37.0 | 6.0 | 0.013 |
| ZI-QDNN | 0.707 | 0.835 | 0.664 | 105.5 | 11.0 | 0.136 |
| ZI-MAF | 0.106 | 0.036 | 0.025 | 227.0 | 11.0 | 0.083 |

Total 77k wall time: 386 s.

## Before vs after the snap fix (coverage at 77k × 50)

| Method | Pre-snap (original stage-1) | Post-snap (this doc) | Uplift |
|---|---:|---:|---:|
| ZI-QRF | 0.256 | 0.928 | +0.672 (3.6×) |
| ZI-QDNN | 0.147 | 0.707 | +0.560 (4.8×) |
| ZI-MAF | 0.014 | 0.106 | +0.092 (7.6×) |

Neural methods get a bigger absolute uplift because their per-column models received the noise-polluted conditioning directly; QRF's tree splits are somewhat robust to small perturbations, which reduces the pre-snap damage to it.

## What changed in the headline story

### Findings that STILL hold

1. **Ordering preserved**: ZI-QRF > ZI-QDNN > ZI-MAF at every scale, every config.
2. **ZI-MAF is still the worst** method tested. Even with the bug fix, ZI-MAF at 0.106 is 9× worse than ZI-QRF at 0.928.
3. **ZI-QRF is the G1 production synthesizer** default. No change.
4. **Calibration-on-synth** result holds (ZI-MAF too far off to rescue via weights).
5. **Embedding-PRDC** validation holds.
6. **ZI-MAF hyperparameter tuning** result holds (wider/longer doesn't rescue it).

### Findings that need revision

1. **ZI-QRF quality is much higher than the pilot suggested.** Stage-1 coverage is 0.928 at 77k, not 0.256. The G1 cross-section is in way better shape than the pre-snap numbers implied.
2. **ZI-QDNN is legitimately competitive.** Pre-snap 0.147 looked mediocre; post-snap 0.707 is respectable. In production if compute budget allows, ZI-QDNN is a reasonable fallback.
3. **The "ZI-MAF is broken" claim is softer than the pre-snap numbers.** At 0.106 it's still worst, but it's not "1% coverage is so bad no amount of calibration rescues it." 10.6% is bad but measurable; the calibrate-on-synth result (mean rel err 15) still says the structure is too far off to rescue via weights, but the PRDC gap is not orders-of-magnitude.

### How confident to be

Four independent robustness checks still agree (raw 50-d PRDC at 40k, raw 50-d PRDC at 77k, embedding 16-d PRDC at 40k, calibrate-on-synth at 20k). Adding the snap fix to stage-1 gives a fifth confirmation. Ordering is robust; absolute numbers finally match the fix.

## What this means for G1

The headline is now cleaner: **ZI-QRF produces 92.8% PRDC coverage on a held-out 15k-record slice of enhanced_cps_2024 at 77k × 50 scale in 37 seconds.** That's a production-credible starting point. Downstream calibration via MicrocalibrateAdapter will pull weighted aggregates to target. We have a working cross-section synthesizer.

The next-action playbook (launch v7 with `--calibration-backend microcalibrate`, see `docs/quickstart-rewire.md`) stays the same. This snap fix is a measurement improvement, not a direction change.

## Artifacts

- `artifacts/stage1_40k_snap.json`
- `artifacts/stage1_40k_snap.jsonl`
- `artifacts/stage1_77k_snap.json`
- `artifacts/stage1_77k_snap.jsonl`

Reproduction:

```bash
uv run python -m microplex_us.bakeoff --stage stage1 --methods ZI-QRF ZI-MAF ZI-QDNN
```

(Uses the snap by default in the harness.)
