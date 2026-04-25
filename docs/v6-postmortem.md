# v6 post-mortem — 2026-04-16

Record of the `broader-donors-puf-native-challenger-v6` run (launched 2026-04-16 10:20:10 ET, died 22:56:05 ET).

## Outcome

**RUN_EXIT status=1** after 12h 36m of wall time. Killed by the kernel during entropy calibration. No artifact directory created; no final dataset persisted.

## Timeline of the post-donor window

The post-donor stage instrumentation (commit `960ac2f`) was the single highest-value diagnostic change of the session. It let us localize the OOM to a specific named stage for the first time.

| Time (ET) | Stage marker |
|---|---|
| 10:20:10 | RUN_START |
| ~19:29 (9h 9m in) | last donor block complete (`scf_2022/social_security_pension_income`) |
| 21:04:03 | `seed ready` → `targets start`/`complete` → `synthesis variables ready` → `synthesis start`/`complete` → `support enforcement start`/`complete` → `policyengine tables start` (all in one burst; synthesis backend = seed-copy so the burst is dominated by the strip+cap pass between donor integration and tables) |
| ~22:25 | `policyengine tables complete` [households=1,505,108, persons=3,373,378] |
| ~22:25 | `policyengine calibration start [backend=entropy]` |
| 22:56:05 | RUN_EXIT status=1, kernel signal (macOS `time -l` reported "signal: Invalid argument" on the wrapper) |

## Memory signature

From macOS `time -l` rusage at exit:

| Metric | v6 | v4 (previous run) |
|---|---|---|
| Wall time | 45,355 s (12h 36m) | 39,476 s (10h 58m) |
| Max RSS | 22.0 GB | 20.5 GB |
| Peak phys_footprint | 293 GB | 287 GB |
| Instructions retired | 614 T | 612 T |
| Involuntary context switches | 317 K | 264 K |

v6's signature is nearly identical to v4's — same killer, same point.

## Diagnosis

**`calibrate_policyengine_tables` with `backend=entropy` on 1.5M households is the OOM killer.**

Proximate cause: a 48 GB machine cannot hold the working set the entropy solver needs for that scale. Peak phys_footprint of 293 GB on 48 GB RAM implies heavy compression and swap pressure; eventually the kernel kills the process.

Likely underlying structural cost (not measured, but fits the profile):

- Entropy calibration materializes a dense Jacobian-like matrix roughly `(n_households × n_constraints)` in float64.
- With 1,505,108 households and ~1,255 constraints post-feasibility-filter (from the 2026-03-30 review), that's 15 GB for a single copy. Multiple working copies (gradient, Hessian approximation, line-search scratch) easily exceed RAM.
- `_evaluate_policyengine_target_fit_context` then runs a full PolicyEngine simulation on the calibrated frame, which adds its own memory cost on top.

## What survived

v6 demonstrated that the **tables-build phase works at scale**: `build_policyengine_entity_tables` successfully produced a 1.5M-household × 3.4M-person entity bundle. This was an open question after v4. The stage isn't free (roughly 1h 25m at 180–210% CPU, RSS oscillating 0.2–16%), but it doesn't OOM.

The donor integration also ran clean. All 129 donor blocks across CPS ASEC, IRS SOI PUF, SIPP tips, SIPP assets, and SCF completed without failure. The tax-unit entity-bundle construction took ~89 min (one-time cost per run). Multi-source donor imputation is not the bottleneck.

## What v6 ruled out as the killer

The initial v4 diagnosis hypothesized the silent post-donor window might be in synthesis, support enforcement, or tables-build. v6's instrumentation showed those all complete instantly or within ~1.5 hours. The killer is specifically **entropy calibration**, not an earlier stage.

## What this means for the architecture direction

v6 is an evidence point *for* the `spec-based-ecps-rewire` direction rather than against it:

1. **Entropy calibration on a 1.5M-household monolithic solve is a dead end on a 48 GB machine.** The rearchitecture's hierarchical / identity-preserving calibration pattern (national → state → stratum, `microcalibrate`-style chi-squared) avoids the dense-matrix blow-up by chunking over strata.
2. **Scaffold scale is the real lever.** The 3.4M-row ACS scaffold drives both tables-build size and calibration-matrix size. CPS-core at ~430k persons cuts this at the source.
3. **The instrumentation pattern is reusable.** Keeping named stage markers at every pipeline boundary in the new pipeline will make any future OOM localizable in a single run rather than requiring multiple exploratory runs.

## What v6 does NOT tell us

- Whether the imputation quality would have beaten `enhanced_cps_2024` on PE-native broad loss had it finished. No parity artifact was produced.
- Whether the `pe_plus_puf_native_challenger` condition selection is an improvement. Moot now that the pipeline direction is changing.
- The actual numerical Calibrator's behavior on 1.5M households. The failure was upstream of any Calibrator numerical work — the process died while setting up the constraint matrices.

## Status of v6 artifacts

- Log file: `artifacts/live_pe_us_data_rebuild_checkpoint_20260414_pe_plus_puf_native_challenger_broader/broader-donors-puf-native-challenger-v6.log` (~2,224 lines)
- No output artifact directory (build never completed persistence step)
- tmux session: cleaned up
- No action required on artifacts — they stay on disk as part of the experiment trail.
