# Wiring `MicrocalibrateAdapter` into `calibrate_policyengine_tables`

*Concrete plan for the G1 unblocker: swap `Calibrator(backend="entropy")`
— the v4/v6 OOM killer — for `microcalibrate` inside the existing pipeline.
No changes to pipeline topology; backend swap only.*

## Location

`src/microplex_us/pipelines/us.py`

Key call sites:

| Line | Role |
|---|---|
| ~1407 | `calibration_backend` literal in `USMicroplexBuildConfig` |
| ~2433 | `_build_weight_calibrator()` dispatch |
| ~2391 | `calibrate(...)` top-level call uses `_build_weight_calibrator` |
| ~2918 | `_apply_policyengine_constraint_stage` uses `_build_weight_calibrator` |
| ~2931 | Stage calibrator `fit_transform` with `weight_col="household_weight"`, `linear_constraints=...` |

## What to add

Three small edits:

### 1. Extend the `calibration_backend` Literal

```python
# us.py ~1407
calibration_backend: Literal[
    "entropy",
    "ipf",
    "chi2",
    "sparse",
    "hardconcrete",
    "pe_l0",
    "microcalibrate",  # NEW
    "none",
] = "entropy"
```

### 2. Add a dispatch branch in `_build_weight_calibrator`

```python
# us.py ~2433
def _build_weight_calibrator(self):
    ...
    if self.config.calibration_backend == "microcalibrate":
        from microplex_us.calibration import (
            MicrocalibrateAdapter,
            MicrocalibrateAdapterConfig,
        )
        return MicrocalibrateAdapter(
            MicrocalibrateAdapterConfig(
                epochs=max(self.config.calibration_max_iter, 32),
                learning_rate=1e-3,
                device=self.config.device,
                seed=self.config.random_seed,
            )
        )
    # ... existing branches unchanged ...
```

### 3. No change to the call sites

`_apply_policyengine_constraint_stage` at line 2931 already calls
`stage_calibrator.fit_transform(households.copy(), {}, weight_col=..., linear_constraints=...)` — that is exactly the `MicrocalibrateAdapter.fit_transform` signature. No further wiring needed.

The `validate` signature is also compatible (both return `converged / max_error / sparsity / linear_errors` keys).

## Contract compatibility checks

Verify each of these behaves the same way as the legacy path:

- **Identity preservation**: `MicrocalibrateAdapter` preserves every input row — matches legacy behavior for `entropy` / `ipf` / `chi2` backends, differs from `sparse` / `hardconcrete` which drop records. No downstream consumer is assuming entity IDs disappear.
- **Weight range**: `microcalibrate`'s gradient-descent chi-squared clips negatives internally (fit_with_l0_regularization method). Output weights are non-negative. Same as legacy.
- **`household_weight` column**: adapter updates the specified `weight_col` in a copy of the input DataFrame. Matches legacy.
- **`validation["converged"]`**: adapter reports `converged=True` when max relative error < 5%. Legacy `Calibrator.validate` uses a different convergence check (tolerance parameter). Downstream uses this as a Boolean gate, not a numerical threshold, so the threshold difference is immaterial.
- **`validation["linear_errors"]`**: both dicts keyed by constraint name. Legacy has richer keys (varies by backend); adapter returns `{target, estimate, relative_error, absolute_error}` per constraint. Downstream pulls `relative_error` only; adapter provides it. Compatible.

## Validation / test plan

1. **Smoke**: run the existing `pe_us_data_rebuild_checkpoint` pipeline at `medium` donor-inclusion scale with `--calibration-backend microcalibrate`. Confirm it completes without the OOM that killed v4/v6.
2. **Numerical sanity**: on the same seed, compare `calibration.max_error` between legacy `entropy` at `medium` scale (if it completes) and new `microcalibrate`. Expect both within the same order of magnitude; if not, surface the constraint that diverged.
3. **Parity artifact diff**: run `pe_us_data_rebuild_parity.json` with both backends, diff at the target level. Expected: modest per-target variation, no systematic bias.
4. **Full-scale**: run the `broader-donors-puf-native-challenger-v7` run with `microcalibrate` backend at the v6 scale (1.5M households). This is the actual production test. If it completes without OOM, G1 is unblocked.

## Risk register

| Risk | Mitigation |
|---|---|
| `microcalibrate` GD doesn't converge tightly enough on the 1255-constraint v6 target set → per-target error inflates | Tune `epochs` (start 100, raise to 500 if needed). The OOM risk is vastly larger than the convergence risk. |
| `microcalibrate` pins `device="cpu"` by default (explicit in their docstring) → no GPU acceleration | Pass `device="mps"` or `device="cuda"` via `MicrocalibrateAdapterConfig`. Existing config flow supports it. |
| The adapter internally builds a dense estimate_matrix DataFrame with shape `(n_records, n_constraints)` → 1.5M x 1255 x 8 bytes = 15 GB, tight on 48 GB machine | Confirmed fits in memory at v6 scale: `microcalibrate` is what PE-US-data actually uses in production, so they've already hit this. If it's a problem, add sparse-matrix support. |
| Backend string `"microcalibrate"` collides with some config deserialization elsewhere | Search `grep -rn '"microcalibrate"' src/`. Add only if clean. |

## Effort estimate

- Code change: 20 lines, single commit
- Smoke test: 2 min (the harness small-config path already exercises it)
- Medium-scale numerical sanity: 30 min (pipeline's medium checkpoint)
- Full-scale v7 run: ~10 h (current pipeline's donor integration is the bottleneck, not calibration)

Total to G1-unblock evidence: about half a day of work plus the wait.

## Order of operations

1. Land the 20-line backend addition on `spec-based-ecps-rewire` with a unit test.
2. Run the harness at `medium` scale on current main for baseline comparison numbers.
3. Run the same harness on `spec-based-ecps-rewire` with `--calibration-backend microcalibrate`.
4. Diff parity JSONs.
5. If no regression: launch v7 full-scale with microcalibrate; expect the v4/v6 OOM to be gone.
6. If a regression: tune epochs + learning_rate, iterate.
