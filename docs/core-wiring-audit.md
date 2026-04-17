# Core wiring audit

*Snapshot: 2026-04-16. Audit of `microplex` core against the H+ rearchitecture proposal for `microplex-us`.*

## TL;DR

The architectural thinking already happened. `microplex` core has ~80% of the primitives the rearchitecture needs — most of them unused. `microplex-us` has grown a parallel set of donor-block, calibration, and entity-table machinery that duplicates what core already provides.

The project is **wire + complete + deprecate**, not **design + build**:

1. Wire `microplex-us` pipelines to use existing core primitives.
2. Complete half-baked primitives where "thought went in" but production-readiness did not.
3. Deprecate `microplex-us` duplicates as each replacement lands.

**Blocker:** `microplex` core is on a stale `codex/core-semantic-guards` branch (last commit 2026-04-02) with ~200 uncommitted/deleted files. Nothing destructive should land in core until that state resolves.

## What exists in core (status by category)

Legend:

- **WIRED** — used by microplex-us today
- **READY** — implemented, untested in production, no obvious gaps
- **PARTIAL** — implemented with gaps or known rough edges
- **PROTOTYPE** — substantial design but probably needs finishing for production
- **UNKNOWN** — needs hands-on testing to classify

### Spec primitives (`microplex.core`)

| Primitive | File | Status | Notes |
|---|---|---|---|
| `Period`, `PeriodType` | `core/periods.py` | READY | Pydantic. DAY/MONTH/QUARTER/YEAR arithmetic and containment. microplex-us does not use. |
| `EntityType` | `core/entities.py` | WIRED | — |
| `SourceArchetype`, `TimeStructure`, `Shareability` | `core/sources.py` | WIRED | Country-agnostic source taxonomy. `LONGITUDINAL_SOCIOECONOMIC`, `PANEL`, `EVENT_HISTORY` values already defined. |
| `SourceProvider`, `SourceQuery`, `ObservationFrame` | `core/sources.py` | WIRED | — |
| `SourceManifest` | `core/source_manifests.py` | WIRED | — |
| `FrameSemanticTransform` | `core/semantics.py` | PARTIAL | Declarative frame transforms with POST_SYNTHESIS / POST_IMPUTATION / POST_DONOR_INTEGRATION / POST_CALIBRATION / POST_EXPORT stages. microplex-us imports the module but coverage is unclear. |
| `SourceVariableCapability` | `core/variables.py` | WIRED | — |

### Transitions (`microplex.transitions`)

| Primitive | File | Status | Notes |
|---|---|---|---|
| `Mortality` | `transitions/mortality.py` | PROTOTYPE | Hardcoded SSA 2021 period life tables (male + female qx arrays for ages 0–119). US-specific data in a "generic" module — likely belongs in microplex-us or a country-pack seam. |
| `MarriageTransition`, `DivorceTransition` | `transitions/demographic.py` | PROTOTYPE | Hardcoded rate tables from CPS/ACS. Same US-specificity concern. |
| `DisabilityOnset`, `DisabilityRecovery`, `DisabilityTransitionModel` | `transitions/disability.py` | PROTOTYPE | Hardcoded SSA DI rates. |

**Decision point:** the hardcoded US rates in `microplex.transitions` violate the core/country split. Either (a) move these to microplex-us and leave core as pure interface, or (b) make the rate tables pluggable with country-specific providers.

### Neural trajectory models (`microplex.models`)

| Primitive | File | Status | Notes |
|---|---|---|---|
| `TrajectoryTransformer` | `models/trajectory_transformer.py` | PROTOTYPE | Autoregressive Transformer for panel synthesis. ZI-QDNN candidate per SS-model docs. |
| `TrajectoryVAE` | `models/trajectory_vae.py` | PROTOTYPE | — |
| `SequenceSynthesizer` | `models/sequence_synthesizer.py` | PROTOTYPE | Variable-length sequence synthesizer. |
| `PanelEvolutionModel` | `models/panel_evolution.py` | PROTOTYPE | **Unified autoregressive replacement** for separate `transitions/*` classes. Docstring explicitly frames it as the replacement: `state[t+1] ~ state[t], state[t-1], ..., X`. |
| `BaseSynthesisModel`, `BaseTrajectoryModel`, `BaseGraphModel` | `models/base.py` | PROTOTYPE | Abstract bases. |

**Decision point:** `transitions/*` classes and `PanelEvolutionModel` overlap. If `PanelEvolutionModel` is the intended canonical form, the separate transitions either become (a) feature-engineering helpers for it, or (b) deleted. Right now both coexist, neither is wired, and microplex-us uses neither.

### Fusion (`microplex.fusion`)

| Primitive | File | Status | Notes |
|---|---|---|---|
| `FusionPlan`, `VariableCoverage` | `fusion/planning.py` | WIRED | microplex-us already uses for planning. Good design: tracks source-by-variable coverage, shareability, time structure. |
| `MaskedMAF` | `fusion/masked_maf.py` | PROTOTYPE | Masked normalizing flow over stacked multi-survey data with per-record observed masks. Country-agnostic. |
| `MultiSourceFusion` | `fusion/multi_source_fusion.py` | PROTOTYPE | Per-source + cross-source + unified three-model pipeline. Direct alternative to microplex-us's donor-block system. |
| `harmonize_surveys`, `stack_surveys`, `COMMON_SCHEMA` | `fusion/harmonize.py` | PROTOTYPE | CPS/PUF-specific mappings baked in — needs generalization before being called "core". |
| `FusionSynthesizer`, `FusionConfig`, `FusionResult` | `fusion/pipeline.py` | PROTOTYPE | High-level convenience over MaskedMAF. |

**Decision point:** the `harmonize.py` `COMMON_SCHEMA` has US-specific variable names. Either move to microplex-us or make the mappings country-configurable.

### Calibration (three modules, overlapping)

| Primitive | File | Status | Notes |
|---|---|---|---|
| `Calibrator` (IPF, chi-square, entropy) | `calibration.py` (2011 lines) | WIRED | Core calibration class. Classical survey calibration. |
| `LinearConstraint` | `calibration.py` | WIRED | Explicit linear constraint rows. |
| `Reweighter` (L0/L1/L2 sparse) | `reweighting.py` (506 lines) | PROTOTYPE | Sparse L0/L1/L2 with scipy and cvxpy backends. Geographic hierarchy support. |
| `microcalibrate` (external) | PolicyEngine package | WIRED (via microplex-us callers externally) | PolicyEngine's gradient-descent chi-squared library. |

**Decision point (load-bearing):** three calibrators partly cover the same problem.

- **Recommendation:** `Calibrator` (classical, identity-preserving) is the mainline for the cross-section pipeline, because it preserves all entity IDs by construction. `Reweighter` is the **optional sparse deployment selector** applied *after* Calibrator to produce a web-app-sized subsample. `microcalibrate` stays as an external dependency only if it offers something `Calibrator` does not (gradient-descent scalability beyond ~1M rows?) — otherwise retire it.
- **Must settle before any wiring commit lands** because migration step 2 depends on choosing the mainline.

### Hierarchical synthesis (`microplex.hierarchical`)

| Primitive | File | Status | Notes |
|---|---|---|---|
| `HouseholdSchema`, hierarchical household→person two-pass | `hierarchical.py` (1155 lines) | PROTOTYPE | Different meaning than "hierarchical calibration." This is two-pass synthesis: household skeleton first, then person attributes conditioned on household context. |
| `TaxUnitOptimizer` | `hierarchical.py` | WIRED | Already used by microplex-us. |

### Geography (`microplex.geography`)

| Primitive | File | Status | Notes |
|---|---|---|---|
| `AtomicGeographyCrosswalk` | `geography.py` | WIRED | — |
| `GeographyProvider`, `StaticGeographyProvider` | `geography.py` | WIRED | — |
| `ProbabilisticAtomicGeographyAssigner` | `geography.py` | WIRED | — |
| `GeographyAssignmentPlan` | `geography.py` | WIRED | — |

**Note:** US-specific GEOID constants (`STATE_LEN`, `COUNTY_LEN`, `TRACT_LEN`, `BLOCK_LEN`) are in core. Comment says "kept as compatibility constants" — probably deletable after UK port proves the abstraction is truly country-agnostic.

### Generative building blocks

| Primitive | File | Status | Notes |
|---|---|---|---|
| `Synthesizer` | `synthesizer.py` (728 lines) | WIRED | Main conditional synthesis class. Uses normalizing flows. |
| `ConditionalMAF` | `flows.py` (526 lines) | PROTOTYPE | Conditional MAF normalizing flow primitive. |
| `DGP` learning | `dgp.py`, `dgp_methods.py` | UNKNOWN | Population data-generating-process learning from multiple partial surveys. Distinct from fusion; claims to be "not statistical matching" and "not imputation" but learn true joint. |
| `StatMatchSynthesizer` | `statmatch_backend.py` | PROTOTYPE | Wraps py-statmatch NND hot-deck. Useful for PUMS ↔ CPS graft. |
| `MultiVariableTransformer` | `transforms.py` | WIRED | — |
| `BinaryModel`, `DiscreteModelCollection` | `discrete.py` | WIRED | — |

### Data sources (`microplex.data_sources`)

| Source | Location | Country-appropriate? | Notes |
|---|---|---|---|
| `cps`, `cps_mappings`, `cps_transform` | `data_sources/cps.py` et al | **No** (US-specific) | Should move to microplex-us. |
| `puf` | `data_sources/puf.py` | **No** (US-specific) | Should move to microplex-us. |
| `psid` | `data_sources/psid.py` | **No** (US-specific) | Should move to microplex-us. |

**Cleanup:** these three belong in `microplex-us/src/microplex_us/data_sources/` (where microplex-us already has its own `cps.py`, `puf.py`, etc.). Core has US-specific data loaders sitting in what should be a country-agnostic package.

### Validation (`microplex.validation`)

| Primitive | File | Country-appropriate? | Notes |
|---|---|---|---|
| `baseline` | `validation/baseline.py` | Likely generic | Needs review. |
| `soi` | `validation/soi.py` | **No** (US-specific) | Should move to microplex-us. |

### Targets (`microplex.targets`)

| Primitive | File | Status | Notes |
|---|---|---|---|
| `TargetSpec`, `TargetSet` | `targets/spec.py` | WIRED | — |
| `TargetProvider` protocol | `targets/provider.py` | WIRED | — |
| `TargetQuery` | `targets/provider.py` | WIRED | — |
| `assert_valid_benchmark_artifact_manifest` | `targets/artifacts.py` | WIRED | — |
| `rac_mapping`, `database`, `bundles`, `benchmarking` | `targets/*` | UNKNOWN | Need review. |

## What microplex-us currently imports from core

Used (from grep of imports):

```
microplex.calibration           (Calibrator, LinearConstraint)
microplex.core                  (EntityType, ObservationFrame, SourceProvider, SourceQuery,
                                 SourceManifest, SourceArchetype, SourceVariableCapability)
microplex.core.semantics        (subset of exports)
microplex.fusion                (FusionPlan only — not the actual fusion synthesizers)
microplex.geography             (subset)
microplex.hierarchical          (TaxUnitOptimizer)
microplex.synthesizer           (Synthesizer base)
microplex.targets               (TargetQuery, TargetSpec, TargetSet,
                                 assert_valid_benchmark_artifact_manifest)
```

Unused but implemented in core:

```
microplex.transitions           (all of it — Mortality, Marriage, Divorce, Disability)
microplex.models                (all trajectory / panel evolution models)
microplex.fusion.MaskedMAF      (neural fusion synthesizer)
microplex.fusion.MultiSourceFusion
microplex.fusion.harmonize      (stack_surveys, harmonize_surveys)
microplex.reweighting           (Reweighter — sparse L0)
microplex.statmatch_backend     (StatMatchSynthesizer — for PUMS graft)
microplex.hierarchical          (HouseholdSchema, hierarchical synthesis pipeline)
microplex.core.periods.Period   (period axis)
microplex.data_sources.psid
microplex.dgp                   (DGP learning)
```

## Gaps — what genuinely needs to be built

Against the H+ proposal, what is NOT already in core (in any form):

1. **Identity-preserving calibrator protocol.** Concept only exists as a note; `Calibrator` and `Reweighter` are concrete classes with different contracts. A shared protocol that declares "output retains all input entity IDs" is missing.
2. **Spatial smoothness regularization** for local-area calibration. Neither `Calibrator` nor `Reweighter` currently penalizes weight differences across adjacent geographies.
3. **Fay-Herriot / composite estimator** for small-area estimation. Not present.
4. **Held-out target evaluation harness.** Calibrate-on vs validate-on split is not a first-class concept in the existing harness.
5. **Forbes backbone integration** for top-income records. PE is adding this upstream; microplex has no equivalent.
6. **`TemporalDonorSpec` unification.** `transitions/*` classes and `PanelEvolutionModel` are two overlapping takes; a reconciled canonical abstraction does not exist.

Everything else in the H+ proposal is at minimum a PROTOTYPE in core.

## Three-way calibration overlap — decision required

```
microplex.calibration.Calibrator    classical: IPF / chi-square / entropy     WIRED
microplex.reweighting.Reweighter    sparse:    L0 / L1 / L2                   UNUSED
microcalibrate (external)           gradient-descent chi-squared              UNUSED
```

Recommended resolution:

- **Mainline:** `Calibrator` (identity-preserving, classical). Used for every production calibration.
- **Optional sparse post-step:** `Reweighter` (L0). Applied after `Calibrator` when a deployment subsample is needed (e.g., 50k-record web app artifact).
- **Retire:** `microcalibrate` external dependency, unless benchmarking shows it does something `Calibrator` does not (e.g., gradient-descent scalability past ~1M rows on realistic constraint matrices).

This choice is load-bearing for migration step 2. It needs a yes/no before any wiring commits land.

## Migration order

| # | Swap | Gate | Blocked by |
|---|---|---|---|
| 0 | Resolve `codex/core-semantic-guards` branch state in microplex | microplex core tree clean on main | — |
| 1 | Adopt `microplex.core.periods.Period` in microplex-us | microplex-us compiles with single period type | 0 |
| 2 | Adopt `Calibrator` end-to-end, retire staged solve_now/solve_later | Cross-section beats current checkpoint on PE-native loss | 0, calibrator decision |
| 3 | Adopt `MultiSourceFusion` + `MaskedMAF`; retire donor-block system | Neural fusion parity-evaluated vs block donors | 2 |
| 4 | Adopt `statmatch_backend` for ACS PUMS ↔ CPS graft | PUMA-level local scaffold exists | 3 |
| 5 | Adopt `Reweighter` as optional sparse deployment selector | 50k-record web-app artifact | 4 |
| 6 | Adopt `transitions/*` for Phase 2 trivial forward projection | 1-year forward projection runs | 5 |
| 7 | Consolidate `transitions/*` and `PanelEvolutionModel` into one canonical form | Unified AR model beats separate hazards on PSID validation | 6 |
| 8 | Adopt `TrajectoryTransformer` / `TrajectoryVAE` | Neural trajectory beats interval-specific QRF on age-earnings | 7 |

Steps 1–2 alone could clear G1 (national cross-section beats ECPS).

## Prerequisite cleanup (microplex core)

Before any wiring commits land in core:

1. **Review `codex/core-semantic-guards` branch** (last commit 2026-04-02). It has useful-looking work (semantic transforms, sparse calibration frontier analysis, referee feedback) but ~200 uncommitted/deleted files. Either:
   - Land the useful pieces, or
   - Hard-reset to clean origin/main and cherry-pick, or
   - Abandon the branch and start fresh.
2. **Relocate US-specific code out of core:**
   - `microplex/data_sources/cps*`, `puf.py`, `psid.py` → microplex-us
   - `microplex/validation/soi.py` → microplex-us
   - SSA hardcoded tables in `transitions/*` → microplex-us (or make pluggable)
   - GEOID length constants in `geography.py` → microplex-us or private helper
3. **Delete the compatibility shims** in core root (`unified_calibration.py`, `target_registry.py`, `pe_targets.py`, `data.py`, `cps_synthetic.py`, `calibration_harness.py`) once all callers have migrated to microplex-us imports. Right now they stay as shims.

## Risks

1. **"Unused" ≠ "ready."** Every PROTOTYPE entry above likely has at least one production-blocking gap. Expect 20–40% of wiring effort to be "finish the core primitive" rather than "integrate."
2. **US-specific rates baked into "generic" core.** `transitions/*` has SSA life tables and CPS rates hardcoded at core level. Wiring microplex-us to those is easy; porting microplex-uk to them is impossible without first decoupling.
3. **Three-way calibrator overlap may hide performance differences.** Before choosing `Calibrator` as mainline, run one apples-to-apples benchmark against `Reweighter` and `microcalibrate` on a representative constraint matrix.
4. **`codex/core-semantic-guards` abandonment.** The stale branch may contain work that materially improves these primitives. Losing it to a hard-reset could waste thought. Reviewing before discarding is cheap insurance.

## Concrete next actions

1. Decide the codex branch's fate (land / rebase / abandon).
2. Settle the three-way calibrator question (benchmark or decision document).
3. Write PSID → ObservationFrame adapter in microplex-us data_sources (if not already done — needs check).
4. Prototype migration step 2 on a small slice: CPS + QRF via `MultiSourceFusion` + `Calibrator` → compare to current microplex-us pipeline at 2000-record smoke scale.
5. Once smoke passes, land step 1 (Period adoption) as the first wiring commit.

## Provenance

This audit reads core as of commit `71f270e` on branch `codex/core-semantic-guards` (microplex core). It does not execute any of the primitives, so READY / PARTIAL / PROTOTYPE classifications are based on interface inspection and file-size heuristics. Each classification needs empirical confirmation before commitment.
