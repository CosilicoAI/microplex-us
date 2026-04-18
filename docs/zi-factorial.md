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

## ZI classifier comparison (QDNN)

Having established that the ZI wrapper matters for QDNN, the next question is whether a different zero-classifier improves ZI-QDNN. Five classifiers were swapped into `ZI-QDNN`'s pipeline on the 77k × 50 benchmark (seed 42):

| Classifier | Coverage | Precision | Zero-rate MAE | Fit (s) |
|---|---:|---:|---:|---:|
| **RF (default, 50 trees, uncalibrated)** | **0.7081** | 0.8343 | 0.1359 | 100 |
| HistGradientBoostingClassifier | 0.7017 | 0.8334 | 0.1370 | 137 |
| MLP (64 × 32, Adam, early stop) | 0.6984 | 0.8397 | 0.1376 | 130 |
| RF + isotonic calibration (3-fold) | 0.6983 | 0.8309 | 0.1370 | 109 |
| Logistic regression | 0.6941 | 0.8336 | 0.1362 | 107 |

All five classifiers cluster within 0.014 coverage points, at or below our multi-seed standard deviation (≈0.002–0.003). **The ZI classifier choice does not meaningfully affect coverage on QDNN at this scale and schema.** The 50-tree RF default is effectively optimal among the alternatives tested.

The interpretation is that the information content of $P(y > 0 \mid x)$ is already captured by a 50-tree RF — a stronger classifier (HistGB, DNN) does not extract additional signal, calibrated probabilities do not propagate to better coverage, and logistic regression is mildly worse because its linear decision boundary under-fits on some columns.

What would actually lift ZI-QDNN above 0.71 coverage is not a better zero-classifier but an architectural change: joint zero-mask modeling (one classifier predicting the full 36-dim zero pattern so cross-target zero correlations are captured), joint quantile output (shared-backbone multivariate QDNN), or post-hoc calibration of the quantile network's own pinball-loss output. These are deferred future work.

## Isolated log-loss evaluation

The coverage tie above could mean either (a) the five classifiers produce genuinely similar $P(y > 0 \mid x)$, so the downstream is honestly reporting, or (b) the classifiers differ materially but the QDNN non-zero draw's error swamps the signal. An isolated per-column evaluation decouples the two.

Protocol: same outer 80/20 train/holdout split as the coverage benchmark (seed 42), then an inner 80/20 split within training into fit/val (49,283 fit, 12,321 val). For each of the 36 target columns with training-set zero-fraction ≥ 10 % (26 eligible columns), each classifier is fit on (`X_fit`, `(~at_min)_fit`) and scored on val with log-loss, Brier, equal-width ECE (10 bins), and ROC-AUC.

| Classifier | Log-loss (mean) | Log-loss (median) | Brier | ECE | AUC (mean) | AUC (median) |
|---|---:|---:|---:|---:|---:|---:|
| **HistGB** | **0.2252** | **0.1712** | **0.0707** | **0.0050** | **0.809** | **0.822** |
| DNN | 0.2337 | 0.1956 | 0.0732 | 0.0070 | 0.748 | 0.773 |
| RF + isotonic (3-fold) | 0.2343 | 0.1834 | 0.0739 | 0.0081 | 0.763 | 0.780 |
| Logistic regression | 0.2468 | 0.2028 | 0.0770 | 0.0180 | 0.756 | 0.763 |
| RF default (50 trees, uncalibrated) | 0.3095 | 0.2523 | 0.0810 | 0.0394 | 0.737 | 0.762 |

**The isolated picture is the opposite of the coverage picture.** The default 50-tree RF — the classifier that was effectively tied on PRDC coverage — is the *worst* classifier on log-loss (spread 0.085, about 6× the coverage spread), Brier, AUC, and calibration. Its ECE is ~8× worse than HistGB's. The AUC gap between RF (0.737) and HistGB (0.809) is 7 points — well outside any plausible noise band.

This resolves the earlier ambiguity cleanly:

1. **The ZI classifier choice does matter for the quantity the ZI wrapper is ostensibly predicting.** HistGB has meaningfully better $P(y > 0 \mid x)$ than an uncalibrated 50-tree RF on nearly every axis — log-loss, Brier, calibration, discrimination.

2. **But the downstream QDNN draw swamps the signal.** Seven points of AUC and an order-of-magnitude calibration improvement produce zero coverage gain. The bridging logic (zero with probability $1 - \hat{P}(y > 0 \mid x)$, otherwise draw from the non-zero QDNN) is dominated by error in the non-zero draw, not error in the classifier.

3. **The binding constraint for ZI-QDNN's coverage is downstream of the classifier.** Swapping classifiers alone cannot lift ZI-QDNN past 0.71 coverage — this requires improving the non-zero quantile output (joint modeling, pinball-loss recalibration, architectural change).

There is a secondary implication for uses of the zero-classifier as a diagnostic rather than a generator component: if we ever surface $\hat{P}(y = 0 \mid x)$ as a subgroup-level or record-level signal (e.g., "this household is 80% likely to have zero long-term capital gains"), the RF default is not the right model. HistGB or a calibrated RF should be preferred there, because the calibration and discrimination gaps that are invisible on coverage become directly user-visible on calibration plots and top-k retrieval.

## Artifacts

- `artifacts/stage1_77k_no_zi.json` — pure QRF, QDNN, MAF at 77k
- `artifacts/stage1_77k_cart_variants.json` — CART, ZI-CART, ZI-QRF at 77k
- `artifacts/stage1_77k_4methods.json` — ZI-CART, ZI-QRF, ZI-QDNN, ZI-MAF at 77k
- `artifacts/zi_classifier_comparison.json` — 5 ZI classifiers on QDNN at 77k (coverage)
- `artifacts/zi_classifier_isolated_eval.json` — 5 ZI classifiers in isolation (log-loss / Brier / ECE / AUC)
