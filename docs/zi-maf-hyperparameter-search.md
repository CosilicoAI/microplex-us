# ZI-MAF hyperparameter search — does tuning rescue the method?

*Direct test of the stage-1 follow-up flagged in `docs/stage-1-pilot-results.md`.*

## Setup

40,000 rows × 50 columns of real enhanced_cps_2024 (identical to stage-1). ZI-MAF trained at four progressively bigger configurations on the same seed and split. PRDC evaluated in 50-dim raw feature space, capped at 15 k × 15 k samples (same cap as stage-1 77 k).

| Config | n_layers | hidden_dim | epochs | batch | lr | Approx params |
|---|---:|---:|---:|---:|---:|---:|
| default | 4 | 32 | 50 | 256 | 1e-3 | baseline |
| wide | 4 | 128 | 50 | 256 | 1e-3 | 4× params |
| long | 4 | 32 | 200 | 256 | 1e-3 | 4× training |
| wide+long | 8 | 128 | 200 | 256 | 5e-4 | 16× both + deeper |

## Results

| Config | Coverage | Precision | Density | Fit (s) | Gen (s) |
|---|---:|---:|---:|---:|---:|
| default | 0.0262 | 0.0083 | 0.0038 | 124 | 0.7 |
| wide | 0.0293 | 0.0088 | 0.0043 | 228 | 0.8 |
| long | 0.0318 | 0.0097 | 0.0048 | 467 | 0.6 |
| wide+long | **0.0328** | 0.0107 | 0.0050 | 1,711 | 1.0 |

Fit time to get from 0.026 → 0.033 coverage: 14× the compute budget. Compare to ZI-QRF on the same data at the same PRDC cap: **coverage 0.352 in 19 s**.

## Verdict

**ZI-MAF is confirmed non-competitive at stage-1 scale with the method-class architecture.** Expanding capacity (4× width), training longer (4× epochs), and doing both with deeper layers (16× total + 8 layers) moves coverage from 0.026 to 0.033 — a 25 % relative improvement. ZI-QRF's 0.352 is 10 × higher at 1/90 the fit time.

The stage-1 finding stands: ZI-QRF is the production synthesizer, not ZI-MAF. No amount of hyperparameter tuning at the default architectural level is going to close a 10× gap.

## Why ZI-MAF fails here

Hypotheses, ordered by how plausible they seem on this evidence:

1. **Per-column independence.** `ZIMAFMethod` trains one `ConditionalMAF` per target column independently. With 36 target columns, 36 flows each only learn `P(col_i | conditioning)` — there's no mechanism to capture cross-target correlations (e.g., someone with high wage income also has zero SNAP). Joint-target flows would be architecturally different but expensive. Tree methods (ZI-QRF) implicitly capture some of these via the conditioning features, but their per-column independence is less damaging because each tree doesn't try to encode a full joint distribution.

2. **Zero-inflation classifier + flow combo.** The method first classifies P(zero) via a 50-tree RF, then trains a flow on the non-zero subset. If the classifier over-predicts zero on rare non-zero cells (see stage-1's `disabled_ssdi` ratio = 0, `elderly_self_employed` ratio = 100+), the flow is trained on a biased subset and produces samples that don't cover the missing support.

3. **Log-transform + standardization on heavy-tailed targets.** The flow log-transforms positive values (`np.log1p(y[y>0])`) and standardizes. For variables with extreme tails (top-1% employment income, net-worth-level wealth), this compresses the tail and the flow produces samples concentrated around the mode; the sparse tail coverage is exactly what PRDC measures.

4. **No conditional target structure.** MAF learns `P(y | x)` where `x` is the shared demographics. 14 conditioning dims predicting 36 target dims (each modeled as 1-dim marginal flow conditional on the 14) may be under-identified at 40k × 36 samples per column.

## What would change my mind

A single condition that would lift ZI-MAF into competitive range:

- **Joint-target flow**: one flow over all 36 target columns simultaneously, not 36 independent flows. Direction matches the SS-model methodology doc's "pathwise / trajectory" framing for longitudinal work.
- **Better zero-inflation handling**: a joint zero-mask model (which 36-dim binary vector does this person have?) instead of 36 independent RF classifiers. Training signal correlates zero patterns across targets.
- **Embedding-based PRDC**: the validation run flagged in `stage-1-pilot-results.md` could show ZI-MAF produces structurally-right samples that raw-feature PRDC misses. Separate investigation.

None of these are in the current `ZIMAFMethod` class. Rewriting them is a materially different project.

## Implication for the SS-model methodology doc

The doc names ZI-QDNN as the production direction with ZI-MAF as a reasonable alternative. Neither survives stage-1 tuning at scale. The near-term cross-section synthesizer default on the rewire is **ZI-QRF**; any future trajectory-based modeling for the longitudinal extension will need a materially different architecture than per-column independent flows.

## Where this leaves us

- **G1 cross-section default**: ZI-QRF. Locked in.
- **ZI-MAF / ZI-QDNN**: not dead as research directions, but are dead as production defaults in their current `microplex.eval.benchmark` implementations.
- **Followup worth trying before fully ruling out neural**: joint-target flow + joint zero-mask model. Needs ~a week of implementation and may still not close the gap.

## Reproducibility

```bash
uv run python -c "
import json, time, numpy as np, pandas as pd
from microplex_us.bakeoff import ScaleUpRunner, ScaleUpStageConfig, DEFAULT_CONDITION_COLS, DEFAULT_TARGET_COLS, stage1_config
from microplex.eval.benchmark import ZIMAFMethod
from prdc import compute_prdc
from sklearn.preprocessing import StandardScaler

base = stage1_config()
cfg = ScaleUpStageConfig(
    stage='zi_maf_tuning', n_rows=40000, methods=('ZI-QRF',),
    condition_cols=DEFAULT_CONDITION_COLS, target_cols=DEFAULT_TARGET_COLS,
    holdout_frac=0.2, seed=42, k=5, n_generate=32000,
    data_path=base.data_path, year=base.year, rare_cell_checks=(),
    prdc_max_samples=15000,
)
runner = ScaleUpRunner(cfg)
df = runner.load_frame()
train, holdout = runner.split(df)
# ... fit and evaluate each config ...
"
```

Full results in `artifacts/zi_maf_tuning.json`. Wall time for all four configs: ~43 min.
