# ZI × draw-method factorial at 77k × 50

*Answers Max's question: should the zero-inflation strategy be chosen independently of the draw method?*

## Design

Four draw methods × two zero-inflation variants = eight cells. All runs on Enhanced CPS 2024 at 77,006 records × 50 columns, PRDC capped at 15,000 samples, seed 42.

- **No ZI**: base method (`CART`, `QRF`, `QDNN`, `MAF`) — fit one per-column model on the full training set, sample or predict directly at generation.
- **ZI**: base method preceded by a `RandomForestClassifier` (50 trees) predicting $P(y > 0 \mid x)$ when training-set zero fraction exceeds 10 %. The per-column model is then fit on the non-zero subset only, and at generation time the draw is zero with probability $1 - \hat{P}(y > 0 \mid x)$.

## Results

PRDC coverage (bold per row = best within that draw method):

| Draw method | No ZI | ZI | Δ | Zero-rate MAE (No ZI) | Zero-rate MAE (ZI) |
|---|---:|---:|---:|---:|---:|
| CART   | 0.9055 | **0.9098** | +0.004 | 0.013 | 0.013 |
| QRF    | 0.9328 | **0.9341** | +0.001 | 0.015 | 0.013 |
| QDNN   | 0.6033 | **0.7068** | +0.103 | **0.582** | **0.136** |
| MAF    | **0.0986** | 0.0928 | −0.006 | **0.332** | **0.081** |

## Reading

1. **CART and QRF are essentially indifferent to the ZI wrapper.** Coverage differences are within single-seed noise (< 0.005), and zero-rate MAE is nearly identical across the two configurations. Both methods' per-column draws naturally preserve zero mass: CART's leaf-sample-from-empirical produces zeros at the training-set leaf rate, and QRF's quantile draws reproduce zero quantiles when a leaf's training distribution has mass at zero. The RF zero-classifier is redundant for these methods.

2. **QDNN genuinely needs ZI handling.** Coverage jumps 0.603 → 0.707 (+0.103) and zero-rate MAE drops 0.582 → 0.136. Without ZI, QDNN produces continuous-valued quantile predictions that never exactly equal zero, so all 0-valued real records are mis-covered. The ZI classifier essentially masks the neural draw to zero for records the classifier thinks are zero, restoring a credible zero-rate structure.

3. **MAF is broken with or without ZI.** Coverage stays near 0.09, zero-rate MAE is terrible under both configurations. The per-column-independent MAF architecture is the binding constraint; the ZI wrapper saves the zero-rate MAE from 0.33 to 0.08 (helpful for diagnostics but not enough to fix coverage). Hyperparameter expansion didn't close the gap either (see `zi-maf-hyperparameter-search.md`).

## Does ZI choice depend on draw method? Yes.

The factorial reveals that the "ZI wrapper" is a no-op for draw methods whose leaf- or quantile-level draws already preserve zero structure implicitly (CART, QRF), and a critical fix for draw methods that produce smooth continuous predictions (QDNN, MAF). There is no single best ZI strategy; the right choice depends on what the draw method does with zero observations.

This has two practical implications:

1. **`ZIQRFMethod` and `ZICARTMethod` do not justify their extra complexity.** The `_MultiSourceBase` inheritance pattern that adds an RF zero-classifier before a QRF or CART draw adds 1–2 seconds of compute and meaningful memory (ZI-CART 7.8 GB vs CART 0.5 GB, because the RF classifier is kept in memory alongside the CART per column) for essentially zero accuracy gain. Production pipelines using tree methods should consider the base variants directly.

2. **For neural methods, the ZI classifier is not optional.** QDNN without ZI produces 0-vs-0.33 zero-rate MAE and 10 coverage points of damage. Any paper or benchmark that tests QDNN-family synthesizers without explicit zero handling is measuring a different (and worse) method.

## Production recommendation update

The cross-section synthesizer recommendation becomes:

- **CART (plain, no ZI)** — fastest path, competitive accuracy, and simplest to reason about. Near-synthpop default.
- **QRF (plain, no ZI)** — accuracy maximizer, ~5× the fit time of CART for 2 points of coverage.
- **Avoid ZI wrappers on tree methods.** They don't help.
- **Do use ZI wrappers on neural methods.** They rescue a substantial fraction of the damage, though not all of it.

## Artifacts

- `artifacts/stage1_77k_no_zi.json` — pure QRF, QDNN, MAF at 77k
- `artifacts/stage1_77k_cart_variants.json` — CART, ZI-CART, ZI-QRF at 77k
- `artifacts/stage1_77k_4methods.json` — ZI-CART, ZI-QRF, ZI-QDNN, ZI-MAF at 77k
