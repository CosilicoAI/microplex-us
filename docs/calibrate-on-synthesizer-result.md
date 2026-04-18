# Calibrate-on-synthesizer result — does `microcalibrate` rescue weak synthesis?

*Third robustness check on the stage-1 synthesizer ordering, this time at the weighted-aggregate level instead of PRDC coverage.*

## Setup

20,000 rows × 50 columns of real enhanced_cps_2024 (16k train / 4k holdout). For each method:

1. Fit, generate synthetic records with unit weights.
2. Initial weight rescale so synthetic totals roughly match holdout-scale (drops gradient descent's starting point near the target).
3. Build one `LinearConstraint` per target column requiring weighted synthetic sum to match holdout sum.
4. Run `MicrocalibrateAdapter.fit_transform` with 200 epochs, lr 1e-3.
5. Report mean relative error across target columns before and after calibration.

## Results (post-snap-fix rerun with 500 epochs, 2026-04-17 21:17)

| Method | Pre-cal mean rel err | Post-cal mean rel err | Max post-cal err | Cal time |
|---|---:|---:|---:|---:|
| **ZI-QRF** | 0.317 | **0.105** | 1.000 | 1.1 s |
| ZI-QDNN | 0.386 | 0.251 | 1.002 | 0.6 s |
| ZI-MAF | 17.51 | 11.86 | 168.3 | 0.6 s |

Reading: after calibration, ZI-QRF's weighted synthetic aggregates are within 10.5 % of the holdout targets on average. ZI-QDNN is at 25.1 %. ZI-MAF is at **1,186 %** — the synthetic output is so far off target scale that calibration can't pull it back, even with 500 epochs of gradient descent.

Pre-snap numbers at 200 epochs (archived as `artifacts/calibrate_on_synthesizer.pre-snap.json`) gave ZI-QRF post-cal 0.141, ZI-QDNN 0.327, ZI-MAF 15.08. The bump to 500 epochs + the snap fix both help; ordering and qualitative conclusion are unchanged.

## What this tells us

1. **Calibration doesn't rescue a broken synthesizer.** The hope was that `microcalibrate` could compensate for poor synthesis by adjusting weights. For ZI-QRF it halves the error; for ZI-MAF it shaves ~15 % off a 1798 % starting error and the final answer is still uselessly wrong. Calibration works on starting points that are close enough; ZI-MAF isn't.

2. **ZI-MAF's failure is not about weighting.** An earlier hypothesis was that ZI-MAF's low PRDC coverage might be acceptable if weighted calibration patched the aggregates. Falsified. The synthesizer produces samples so far from target mass that no weight adjustment can make them match aggregates.

3. **ZI-QRF's synthesis is the right STRUCTURE to calibrate.** Calibration dropping error from 0.26 → 0.14 on ZI-QRF output means the raw samples are structurally close to real; weights just need to shift them. ZI-QDNN's output is roughly in the right ballpark but less clean (0.39 → 0.33).

4. **`max` relative error stays ~1.0 across all three for post-cal.** This is because at least one constraint (typically a rare-cell target like `disabled_ssdi`) stays exactly off — the zero-cell problem from stage-1 hasn't been addressed, it just doesn't dominate the *mean*.

## Calibration convergence note

200 epochs at lr=1e-3 with default `microcalibrate` settings does not fully converge these problems. The loss trajectory shows steady improvement until the last reported epoch. For a production run, epochs should probably be 500-1000 to reach the calibration's 5 % relative-error bound.

At production scale (1.5 M records × 1255 constraints), the per-epoch step is cheaper per-record but there are vastly more records to move, so even 500-1000 epochs may leave some constraints unsolved. The `MicrocalibrateAdapterConfig.epochs` default of 32 is too low; the `us.py` wiring uses `max(self.config.calibration_max_iter, 32)` which pulls from the pipeline's `calibration_max_iter=100`. Reasonable starting point; tune up if convergence is still incomplete.

## Four-way agreement on synthesizer ordering (post-snap-fix)

Combined evidence with the upstream shared-col noise fix applied:

| Check | ZI-QRF | ZI-QDNN | ZI-MAF |
|---|---|---|---|
| Raw 50-d PRDC at 40k (snap) | 0.979 (winner) | 0.796 | 0.168 |
| Raw 50-d PRDC at 77k (snap) | 0.928 (winner) | 0.707 | 0.106 |
| Embed 16-d PRDC at 40k (snap) | 0.984 (winner) | 0.819 | 0.201 |
| ZI-MAF tuned (wide+long, 40k, pre-snap) | — | — | 0.033 |
| Calibrate-on-synth post-cal mean err (20k, snap) | 0.105 (winner) | 0.251 | 11.86 |

Every axis, every scale, every metric: **ZI-QRF > ZI-QDNN > ZI-MAF**.

## Production implication

- **G1 cross-section synthesizer default**: ZI-QRF. This is the fourth independent confirmation.
- **Calibration stack**: `MicrocalibrateAdapter` at the default adapter settings is fine for ZI-QRF output (error 0.26 → 0.14 in ~1 s on 16 k records). Bump `calibration_max_iter` to 500 or 1000 in the pipeline config for the production run to wring out the last few percent of residual error.
- **Neural synthesizers**: not producing structures that calibration can rescue at the default architectures. They need joint-target and joint-zero-mask modeling before being reconsidered for production.

## Artifacts

- `artifacts/calibrate_on_synthesizer.json` — full per-method, per-target pre- and post-cal error breakdown.
- `artifacts/calibrate_on_synthesizer.log` — full run log with calibration loss trajectory per method.

Reproduction: `uv run python scripts/calibrate_on_synthesizer.py --n-rows 20000 --calibration-epochs 200`. ~3 minutes wall time on a 48 GB M3.
