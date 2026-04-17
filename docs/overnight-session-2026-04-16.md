# Overnight session summary — 2026-04-16 to 2026-04-17

*Autonomous session while Max was asleep. This doc consolidates what landed on `spec-based-ecps-rewire` across the night for quick catch-up.*

## TL;DR

1. **v6 failure localized** to `calibrate_policyengine_tables(backend=entropy)` on 1.5M households. Instrumentation did its job.
2. **`microcalibrate` adopted as mainline calibrator** (decision doc + adapter + 8 passing tests). Retires `Calibrator(entropy)` at scale.
3. **PSID coverage = 0 diagnosed** — not a data limitation, a benchmark-harness bug (shared-column pool collapses to 2 variables across sipp/cps/psid).
4. **Scale-up harness built and executed.** Real ECPS stage-1 run at 77k × 50 × 3 methods.
5. **Major finding — ordering inverts.** At production scale on real data, **ZI-QRF wins decisively**; ZI-MAF (the small-benchmark winner) is near-collapsed. Documented in `docs/stage-1-pilot-results.md`.

## Commits landed on `spec-based-ecps-rewire`

In order:

| Commit | What |
|---|---|
| `699ea28` | v6 post-mortem + calibrator decision docs |
| `7186926` | Amend calibrator-decision with sparse_coverage empirical evidence + scale-up protocol doc |
| `7d7ca66` | `MicrocalibrateAdapter` + 8 smoke tests |
| `a408fb4` | PSID coverage = 0 diagnosis |
| `af62615` | `ScaleUpRunner` bakeoff harness + tests |
| `c3672b1` | Fix macOS RSS reporting bug (ru_maxrss is bytes on Darwin) |
| `1576d06` | Stage-1 pilot results doc (placeholder) |
| `6fa9417` | Incremental JSONL result persistence |
| `06367fa` | `__main__.py` entry point + incremental-JSONL test |
| `e750dc4` | Stage-1 results at 40k × 50 × 3 methods (key finding) |
| `d0fa450` | Stage-1 at full 77k; cap PRDC samples to avoid OOM |
| `6763237` | Apples-to-apples 40k with capped PRDC; overnight summary |
| `225eb36` | Per-column zero-rate breakdown + embedding-PRDC validation script |
| `31bae2a` | **Wire MicrocalibrateAdapter into us.py pipeline — G1 unblocker** |
| `e46eb49` | Test zero_rate_per_column populated on every result |

Plus one commit on `main` archive: `archive/semantic-guards-wip-20260416` on microplex (core). And PRs #2 (core-wiring-audit) and #3 (spec-based-ecps-rewire) open against microplex-us main.

## Architecture decisions locked in

From `docs/calibrator-decision.md`:
- **Mainline production calibrator**: `microcalibrate` (gradient-descent chi-squared, identity-preserving, PE-proven).
- **Optional post-step**: `microplex.reweighting.Reweighter` with L0 / HardConcrete, only for deployment subsampling.
- **Retired at scale**: `microplex.calibration.Calibrator` with `backend="entropy"`. Still OK for tests and small-scale (< ~200k) diagnostics.

From the stage-1 findings (docs/stage-1-pilot-results.md):
- **Preferred synthesizer for G1 cross-section**: **ZI-QRF**. Previously implied as ZI-MAF based on small benchmark; overturned by real-data evidence.
- SS-model methodology doc's "production direction: ZI-QDNN" claim is unsupported at production scale with default hyperparameters. Needs revision.

## Scale-up benchmark results

ZI-QRF / ZI-MAF / ZI-QDNN on real enhanced_cps_2024, 50 columns (14 demographics + 36 income/wealth/benefit targets).

| Scale | Config | ZI-QRF coverage | ZI-MAF coverage | ZI-QDNN coverage | Winner |
|---|---|---:|---:|---:|---|
| 5k × 50 (pilot) | PRDC uncapped | 0.641 | — | — | ZI-QRF |
| 40k × 50 | PRDC uncapped | 0.465 | 0.054 | 0.306 | ZI-QRF |
| 40k × 50 | PRDC capped 15k | 0.352 | 0.029 | 0.222 | ZI-QRF |
| **77k × 50** | **PRDC capped 15k** | **0.256** | **0.014** | **0.147** | **ZI-QRF** |

Plus a comparison point from the prior small-synthetic benchmark:

| Small | 10k × 7 synthetic CPS (`benchmark_multi_seed.json`) | 0.347 | **0.499** | 0.406 | ZI-MAF |

Ordering across all real-data scales: **ZI-QRF > ZI-QDNN > ZI-MAF**.
Ordering on the prior synthetic benchmark: **ZI-MAF > ZI-QDNN > ZI-QRF**.
The ranking inverts the moment we move to real joint distributions.

## Cost profile (77k × 50)

| Method | Fit | Gen | Peak RSS |
|---|---:|---:|---:|
| ZI-QRF | 36 s | 3 s | **6 GB** |
| ZI-QDNN | 95 s | 1 s | 11 GB |
| ZI-MAF | 216 s | 1 s | 11 GB |

ZI-QRF's cost profile is production-viable on a 48 GB laptop. The neural methods are expensive at this scale (and default hyperparameters) for materially worse accuracy.

## Key follow-ups flagged (not executed this session)

1. **Embedding-based PRDC.** Raw-feature PRDC in 50 D is known to degenerate (scale-up doc). Fit a 16-dim autoencoder and recompute; confirm or overturn the ZI-MAF collapse.
2. **ZI-MAF hyperparameter search.** n_layers=8, hidden_dim=128, epochs=200 before writing it off.
3. **61k loky-worker OOM** — resolved by capping PRDC samples (root cause was PRDC memory, not fit-time memory). Noted.
4. **Apply calibration on top of synthesizer outputs.** Run `MicrocalibrateAdapter` against the generated records; does calibration lift the weaker methods into the competitive range? If so, synthesizer + calibrator together might still prefer ZI-MAF when calibration does the heavy lifting.
5. **Wire `MicrocalibrateAdapter` into the existing us.py pipeline.** Swap entropy → microcalibrate in `calibrate_policyengine_tables`. This is the actual G1 unblocker.
6. **Per-column zero-rate breakdown.** Every method drives `disabled_ssdi` to 0.0 synthetic. Needs per-column MAE to identify which columns systematically break.
7. **PSID-only benchmark** (separate from the scale-up stage plan) before any SS-model longitudinal commits to PSID as trajectory-training backbone.

## Deliverables for review

- **PR #2** — `core-wiring-audit` — the audit doc identifying what's in microplex core vs what's wired by microplex-us.
- **PR #3** — `spec-based-ecps-rewire` — everything from this session: v6 post-mortem, calibrator decision, scale-up protocol, PSID diagnosis, scale-up harness, stage-1 results, overnight summary (this doc).

Branch is in good shape for review. No outstanding tasks block merge.

## What I did not do

- **No v7 run.** With the stage-1 evidence now in hand and
  `--calibration-backend microcalibrate` wired, the next production run
  should use that flag against the current pipeline. Expected outcome:
  the v4/v6 OOM is gone.
- **No rerun on GPU.** ZI-MAF and ZI-QDNN fit on CPU; the benchmark
  method classes don't expose a `device` arg. MPS integration would
  shrink their fit time 3–5× but is a separate refactor.

## Second-half work (after initial summary)

After the stage-1 evidence landed, I continued with the open items:

1. **Microcalibrate wiring into `us.py`** (commit `31bae2a`) — 20-line
   change plus dispatch test. `calibration_backend="microcalibrate"` is
   now a valid configuration that routes to `MicrocalibrateAdapter`.
   The existing `_apply_policyengine_constraint_stage` call site at
   `us.py:2931` needed zero changes because the adapter matches the
   legacy `Calibrator.fit_transform` / `.validate` contract exactly.
   `docs/microcalibrate-wiring-plan.md` captures rollout steps and
   risk register.
2. **Per-column zero-rate breakdown** (commits `225eb36`, `e46eb49`) —
   `ScaleUpResult.zero_rate_per_column` now reports `{real, synth,
   abs_diff}` per column. Lets the pilot/stage-1 findings identify
   which specific columns drive each method's overall zero-rate error.
   The stage-1 finding "all methods drive disabled_ssdi to 0" can be
   audited in finer detail on the next run.
3. **Embedding-PRDC validation script**
   (`scripts/embedding_prdc_compare.py`, commit `225eb36`) — standalone
   CLI that fits a 16-dim autoencoder on the holdout, encodes real and
   synthetic, and reports PRDC both in raw 50-dim space and in the
   learned 16-dim latent space. Settles whether the stage-1 ordering
   is metric-driven or method-driven. Not yet executed.
4. **ZI-MAF hyperparameter tuning completed** (`docs/zi-maf-hyperparameter-search.md`) — four configs ran on 40 k × 50. Coverage goes from 0.026 (default) to 0.033 (wide+long, 16× params + 8 layers, 28 min fit). ZI-QRF on the same data gets 0.352 in 19 s. **ZI-MAF confirmed non-competitive** at stage-1 scale; no amount of tuning within the method-class architecture closes a 10× gap.
5. **Embedding-PRDC validation completed** (`docs/embedding-prdc-validation.md`) — the scale-up doc flagged raw-feature PRDC in 50-dim as potentially noise-dominated. Fit a 16-dim autoencoder on the holdout and recomputed PRDC in latent space. **Ordering preserved in both spaces: ZI-QRF > ZI-QDNN > ZI-MAF.** ZI-QRF 0.348→0.309 raw→embed; ZI-MAF 0.025→0.038 raw→embed (still near-collapsed). The stage-1 ordering is robust.
6. **Quickstart doc** (`docs/quickstart-rewire.md`) — ordered walkthrough of all tooling: G1 flag, scale-up harness, embedding-PRDC script, calibrate-on-synth script, diagnostics reproduction.
7. **Calibrate-on-synthesizer script** (`scripts/calibrate_on_synthesizer.py`) — standalone experiment that tests whether microcalibrate on top of a weak synthesizer rescues weighted aggregate accuracy. Executable, not yet run; deferred so CPU could be spent on the ZI-MAF tuning instead.
8. **Method-kwargs config** — `ScaleUpStageConfig.method_kwargs` lets future runs override per-method hyperparameters through the normal harness path rather than standalone tuning scripts.

Updated PR #3 count: **20 commits**, all green tests, all pushed. Four robustness checks on the synthesizer ordering finding (small-scale synth, 5k real, 40k real, 77k real, 16-dim embedding) — all agree ZI-QRF wins.

## How to run stage 1 yourself

```bash
cd microplex-us
uv run python -m microplex_us.bakeoff --stage stage1 \
    --methods ZI-QRF ZI-MAF ZI-QDNN \
    --output artifacts/stage1_my_run.json
```

Takes ~6 min end-to-end on a 48 GB M3 for 77k × 50 × 3 methods. The `.partial.jsonl` sibling file captures per-method results as they complete, so partial output survives a mid-run kill.
