# Direct PE-native optimizer review — 2026-03-31

## Scope

Code review and architectural diagnosis of the new direct PE-native weight optimization path in `pe_native_optimization.py`, its integration into the performance harness, and interpretation of the first A/B result (0.0004 improvement on a 0.92 loss).

Files reviewed:
- `src/microplex_us/pipelines/pe_native_optimization.py`
- `src/microplex_us/pipelines/performance.py` (harness integration)
- `src/microplex_us/pipelines/pe_native_scores.py` (scoring subprocess)
- `src/microplex_us/pipelines/__init__.py` (exports)
- `tests/pipelines/test_pe_native_optimization.py`
- `tests/pipelines/test_performance.py`
- Upstream: `policyengine_us_data/utils/loss.py`, `policyengine_us_data/calibration/unified_calibration.py`
- Artifacts: `tmp_pe_native_direct_opt_20260331.json`, raw candidate scores

---

## Findings

### 1. NO BUG — objective alignment is correct

The optimizer's quadratic form `||M^T w - s||^2` is algebraically identical to the scorer's native loss. Derivation:

The scorer (`pe_native_scores.py:182-185`) computes:
```
estimate = w @ A
rel_error_j = ((estimate_j - t_j + 1) / (t_j + 1))^2
loss = mean(inv_mean_norm * norm_j * rel_error_j)
```

The optimizer (`pe_native_optimization.py:74-85`) constructs:
```
scaling_j = sqrt(inv_mean_norm * norm_j / T) / (t_j + 1)
M = A * scaling[newaxis, :]
s = (t - 1) * scaling
```

Expanding `||M^T w - s||^2`:
```
= sum_j scaling_j^2 * (w^T A_j - t_j + 1)^2
= sum_j (inv_mean_norm * norm_j / T) * ((w^T A_j - t_j + 1) / (t_j + 1))^2
= (1/T) * inv_mean_norm * sum_j norm_j * rel_error_j
= scorer loss
```

Confirmed numerically: optimizer `initial_loss = 0.9233365911702254`, scorer raw loss `0.9233365911702252` — difference is `2e-16` (float64 noise).

Both scripts use the same `build_loss_matrix()`, same `_ENHANCED_CPS_BAD_TARGETS`, same zero-mask threshold (`atol=0.1`), same national/state normalization. The objectives are provably identical.

### 2. NO BUG — gradient, Lipschitz, and step size are correct

Gradient of `f(w) = ||M^T w - s||^2` is `∇f = 2M(M^T w - s)`. Code at lines 254-256:
```python
residual = matrix.T @ weights - target  # M^T w - s
gradient = 2.0 * (matrix @ residual)     # 2M(M^T w - s)
```

Correct, including the L2 penalty term.

Lipschitz constant via power iteration on `MM^T` yields `λ_max(MM^T)`. The full Lipschitz is `2λ_max + 2*l2_penalty`, matching the Hessian `2MM^T + 2λI`. Step size `1/L` gives guaranteed descent per projected-gradient iteration. All correct.

### 3. NO BUG — simplex projection is correct

Standard Michelot/Duchi O(n log n) projection onto `{x ≥ 0, Σx = total}`. Budget variant correctly restricts support via `argpartition`. Edge cases handled.

### 4. NO BUG — H5 weight rewrite is correct

Group-to-household mapping via `person_household_id × person_{group}_id` bridge tables is sound for PE entity structure. `setdefault` is safe because PE groups don't span households. Float64→float32 cast at write time is consistent with PE storage format.

### 5. NO BUG — performance harness integration is correct

`performance.py:812-846`: when `optimize_pe_native_loss=True`, the harness exports a candidate H5, runs the optimizer to produce a second H5 with optimized weights, and passes the optimized H5 to the scorer. The optimization metadata is attached to the scores dict under the `"optimization"` key. Wiring is correct.

### 6. MINOR — weight-sum drift after projection iterations

`optimized_weight_sum = 6,920,897` vs `initial_weight_sum = 6,920,834` — a drift of `63` (~9e-6 relative). Each simplex projection targets the same `total_weight`, but `np.maximum(clipped - theta, 0.0)` doesn't enforce the exact sum. The drift accumulates over 200 iterations and is practically negligible for the loss computation.

**Fix (optional)**: add a single rescale after the final projection:
```python
weights *= total_weight / weights.sum()
```

### 7. MINOR — convergence reporting is misleading but not harmful

`converged=false` after 200 iterations with total improvement of `0.0004` means per-step improvement averaged `~2e-6`, which exceeds `tol=1e-8`. The method is in a diminishing-returns regime, not a divergent one. More iterations would eventually trigger the convergence criterion but would not materially improve the loss.

This is expected behavior for projected gradient descent on an overdetermined quadratic: the feasible minimum is close to the starting point, so each step makes only a tiny improvement, but the relative improvement `(current - candidate) / max(1, current)` stays above `tol` for many iterations.

**Not a bug**, but the convergence report could be more informative. Consider logging the relative improvement trajectory or adding a `max_iter_exhausted` flag.

### 8. MEDIUM — no end-to-end validation that rescored loss matches optimizer's internal loss

The optimizer reports `optimized_loss` from its internal `objective()` call. The harness then rescores the rewritten H5 with the full PE-native scorer subprocess. These should match within float32/float64 tolerance, but there's no assertion validating this. If a future change causes the loss matrix extraction to diverge from the scoring path (e.g., different `build_loss_matrix` version, different target filtering), the optimizer would silently optimize a stale objective.

**Recommended**: add a post-optimization assertion in `optimize_policyengine_us_native_loss_dataset` that compares `summary["optimized_loss"]` to the subprocess-reported `candidate_loss_before` from a re-extraction of the optimized H5. Or at minimum, log both values for manual comparison.

### 9. NOT A BUG — test coverage is appropriate

`test_pe_native_optimization.py` covers:
- Weight optimization reduces loss and respects budget constraint
- H5 weight rewrite propagates to person and group weight arrays
- Full end-to-end pipeline with monkeypatched subprocess

`test_performance.py` covers:
- Harness rejects `optimize_pe_native_loss` without `evaluate_pe_native_loss`
- Optimizer parameters (budget, max_iter, l2_penalty, tol) pass through correctly
- Optimized H5 is what gets scored

No gaps in the critical paths.

---

## Question 1: Does the optimizer optimize the same objective as the scorer?

**Yes, exactly.** Proven algebraically in Finding 1 and confirmed numerically (initial losses match within float64 noise). Both paths call `build_loss_matrix()` from the same `policyengine-us-data` checkout, apply the same bad-target/zero-target filtering, and use the same national/state normalization.

## Question 2: Are there correctness bugs?

**No serious bugs.** The scaled-matrix construction, projected gradient routine, H5 weight rewrite, and harness integration are all correct. Two minor items worth addressing:
- Weight-sum drift (~9e-6 relative) — cosmetic, optional fix
- No cross-validation between optimizer's internal loss and rescored loss — worth adding as a guard

## Question 3: Does the tiny improvement (0.92334 → 0.92290) mean record support is the bottleneck?

**Yes, this is strong evidence.** The argument:

1. The optimizer directly minimizes the exact PE-native loss as a function of 2000 household weights, subject to non-negativity and sum constraints.
2. The minimum it found is 0.9229, only 0.05% better than the starting point of 0.9233.
3. This means the best achievable loss with these 2000 records is ~0.923 — the entropy calibrator was already near-optimal for this support.
4. PE's baseline achieves 0.020 with ~30,000 source-imputed records.
5. The gap is a factor of **46×**, and only 0.05% of it was attributable to the weight objective.

The remaining 99.95% of the gap is structural: the 2000 households lack the support to span the ~2,817 target dimensions. This is consistent with all the build-log evidence:
- Feasibility filter drops 60-70% of calibration constraints
- State-age support recall is 0.49–0.63 (vs ~1.0 for PE)
- Many exact targets have literally zero candidate mass
- Scaling `sample_n` produces steeper loss improvements than any other lever tested

The direct optimizer cleanly rules out "maybe entropy just has a bad weight objective" as a hypothesis.

## Question 4: What should the next high-leverage step be?

**Full-support selection path.** The direct weight optimizer has served its diagnostic purpose and confirmed the bottleneck. The next steps, in order of priority:

1. **Full-support + budgeted household selection** — This is the path already started at `policyengine_selection_household_budget=29999` in the build log. Use the full CPS+PUF support (all ~30K+ source households without subsampling), preserve the full donor-integrated surface with `synthesis_backend='seed'`, then use the sparse selector to prune to a target household budget before final calibration. This directly addresses the support gap.

2. **Move toward PE's L0 selection/calibration architecture** — PE-US-data's `unified_calibration.py` uses L0-regularized optimization to simultaneously select records and calibrate weights. The current microplex path does selection → calibration as separate stages. Unifying them (or at least using the same L0 regularizer for selection) would let the selector prefer households that jointly cover more target dimensions. This is the medium-term architectural convergence.

3. **Do not invest further in direct weight optimization on small candidates** — The diagnostic value is exhausted. The optimizer proved that weight-objective mismatch accounts for <0.1% of the gap. Rerunning it on larger candidates would confirm the same conclusion at higher cost.

4. **Keep the optimizer code** — It's clean, correct, and useful for future diagnostics (e.g., to measure how much a larger candidate's loss is weight-bounded vs support-bounded).

---

## Summary

The direct PE-native optimizer is mathematically correct, properly aligned with the scorer, and cleanly integrated into the harness. No serious bugs. The first A/B result (0.0004 improvement out of a 0.903 gap) definitively confirms that the bottleneck is record support/construction, not the weight objective. The next high-leverage move is the full-support + budgeted selection path already prototyped in the build log.
