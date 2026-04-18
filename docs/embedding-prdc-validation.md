# Embedding-PRDC validation — is the stage-1 ordering real?

*Settles the open question flagged in `docs/synthesizer-benchmark-scale-up.md`: is PRDC in 50-dim raw feature space too noisy to trust? Answer: the ordering is preserved.*

## Setup

40,000 rows × 50 columns of real enhanced_cps_2024. Same setup as stage-1.

Autoencoder: 50 → 64 → 64 → **16** → 64 → 64 → 50 (2 hidden layers encoder + decoder, ReLU activations). Fit on holdout only (not on synthetic) for 200 epochs, batch 256, lr 1e-3. Final reconstruction MSE loss: 0.054.

For each method (ZI-QRF / ZI-MAF / ZI-QDNN) at default hyperparameters: fit on 32k train, generate 32k synthetic, compute PRDC on 15k/15k samples (capped) in both the raw 50-dim feature space and the 16-dim latent space.

## Results (post-snap-fix rerun 2026-04-17 21:12)

| Method | Raw-50 coverage | Raw-50 precision | Raw-50 density | Emb-16 coverage | Emb-16 precision | Emb-16 density |
|---|---:|---:|---:|---:|---:|---:|
| ZI-QRF | **0.982** | 0.914 | 0.908 | **0.984** | 0.943 | 0.935 |
| ZI-QDNN | 0.791 | 0.847 | 0.763 | 0.819 | 0.905 | 0.802 |
| ZI-MAF | 0.183 | 0.033 | 0.026 | 0.201 | 0.070 | 0.042 |

**Ordering preserved in both spaces: ZI-QRF > ZI-QDNN > ZI-MAF.**

### Pre-snap numbers (archived)

The original run was executed before the shared-col categorical-noise
fix landed upstream. Those artifacts are preserved as
`artifacts/embedding_prdc_compare.pre-snap.json` and showed much lower
absolute PRDC coverages (ZI-QRF 0.348 raw / 0.309 embed), because
noise-injected integer conditioning variables reduced PRDC scores
uniformly across all methods. Ordering was preserved in both
pre-snap and post-snap regimes; only the absolute values shift.

## Observations

1. **The stage-1 verdict is not a metric artifact.** The concern in the scale-up protocol doc was that raw-feature PRDC in 50 dimensions concentrates distances and becomes noise-dominated. The embedding variant has 16 dimensions with more informative axes (learned from the data), which is where PRDC is known to behave best. The ordering is the same. So the 10× gap between ZI-QRF and ZI-MAF is a real quality gap, not a measurement artifact.

2. **Precision rises in embedding space for all three methods.** The AE compresses noise: random synthetic variation that looked far from real records in 50-dim now falls near them in 16-dim. This improves precision and, in the post-snap regime, slightly raises coverage too (likely because the smaller latent dimension is easier to cover).

3. **ZI-QRF's edge is close to the ceiling.** 0.982 raw → 0.984 embed — already near-perfect on holdout. ZI-QDNN rises modestly (0.791 → 0.819). ZI-MAF rises from 0.183 → 0.201. The gap narrows in absolute terms (ZI-QRF / ZI-MAF ratio 5.4× raw, 4.9× embed) but the ordering is invariant.

4. **ZI-MAF is still structurally behind.** Even in the embedding space, ZI-MAF coverage is 0.201 — about a quarter of ZI-QDNN and a fifth of ZI-QRF. Hyperparameter tuning (see `docs/zi-maf-hyperparameter-search.md`) does not close this at the architectural level.

## Interpretation

The ZI-QRF / ZI-QDNN / ZI-MAF ranking is robust across:

- **Scale**: small synthetic (10 k × 7) → 5 k × 50 real → 40 k × 50 real → 77 k × 50 real.
- **PRDC sample cap**: uncapped (8 k × 32 k) and capped (15 k × 15 k).
- **Feature space**: 50 raw features and 16 learned latent dimensions.

That's four independent robustness checks. The production default for G1 cross-section synthesis is **ZI-QRF**.

## One thing this does not settle

Neither raw-50 nor embed-16 PRDC weighs rare cells more than bulk cells. The `sparse_coverage.csv` finding — sparse L0 selection drives rare-cell ratios to 0 — is a different failure mode that neither PRDC variant measures. That finding still drives the calibrator decision (microcalibrate as mainline, not sparse reweighting). Both findings hold independently.

## Artifact

`artifacts/embedding_prdc_compare.json` — full per-method raw and embed PRDC dicts.

Reproduction:

```bash
uv run python scripts/embedding_prdc_compare.py --n-rows 40000 --output artifacts/embedding_prdc_compare.json
```

~5 minutes on a 48 GB M3.
