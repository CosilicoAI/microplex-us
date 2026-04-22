# B2 downstream validation (v11-per-stage-lambda)

Run date: 2026-04-22  
Artifact: `artifacts/live_pe_us_data_rebuild_checkpoint_20260421_v11_per_stage_lambda/v11-per-stage-lambda/policyengine_us.h5`  
Period: 2024  
Method: `scripts/run_b2_batched.py` with batch_size=50_000 for income_tax, 100_000 for aca_ptc, full-dataset for the rest.  
Comparison framework: `microplex_us.validation.downstream.DOWNSTREAM_BENCHMARKS_2024`.

## Results

| Variable | Computed | Benchmark | Rel error | Source |
|----------|---------:|----------:|---------:|--------|
| income_tax | $2,089.7B | $2,400.0B | −12.9% | IRS SOI 2022 ~$2.22T; CBO 2024 projection ~$2.4T |
| eitc | $64.2B | $64.0B | +0.3% | IRS SOI 2023 (Table 2.5) |
| snap | $101.8B | $100.0B | +1.8% | USDA FNS FY2024 |
| ctc | $151.9B | $115.0B | +32.1% | IRS SOI 2023 (pre-OBBBA $2,000/qc) |
| ssi | $108.2B | $66.0B | +64.0% | SSA SSI Annual Statistical Report 2024 |
| aca_ptc | $14.1B | $60.0B | −76.4% | CMS/IRS ACA PTC 2024 (IRA-enhanced) |

## Reading

- **Within ±15%** of benchmark: income_tax (−12.9%), eitc (+0.3%), snap (+1.8%). The tax-mechanics chain and the two largest means-tested programs reconcile to published totals once calibrated weights are applied.
- **Elevated +30% to +65%**: ctc and ssi. ctc = 32% above IRS SOI suggests either more qualifying children per household than IRS counts, or the synthesis pulled CTC-eligible families with higher frequency than the population-level CTC claim rate; ssi at +64% is the cleanest outlier and points to either over-representation of the aged / disabled low-income subpopulation or a missed means-test gate in the synthesis-then-materialize step.
- **Under at −76%**: aca_ptc. The `has_marketplace_health_coverage` flag is in the synthesis target set, but the reconciled PTC depends on a policy-output chain (MAGI, federal poverty line, premium contribution). Either marketplace enrollment is under-represented at the income bands where PTC is largest, or the IRA-enhanced subsidy schedule isn't firing as it does in production IRS data.

## Interpretation for the paper's B2 section

Three headline aggregates reconcile within single-digit or low-teens relative error. The three that don't (ctc, ssi, aca_ptc) are individually diagnosable — each points to a specific shortfall in the synthesis step rather than a structural problem in the calibration framework. A follow-up calibration pass can add direct targets on these aggregates (CTC disbursed, SSI disbursed, ACA PTC disbursed) to drive them in.

The income_tax reconciliation at −12.9% is the most important single number: it's the paper's headline claim that the calibrated synthesis produces a PolicyEngine-US-readable frame whose downstream tax-output reconciles to IRS administrative totals within a credible tolerance.

## Reproduction

```bash
# All variables except income_tax and aca_ptc fit in the full-dataset path:
for var in ssi snap eitc ctc; do
  .venv/bin/python -u scripts/run_b2_validation_single_var.py \
    --dataset <h5> --output <json_path> --variable "$var" --period 2024
done

# income_tax and aca_ptc need batching to avoid 30+ GB peak RSS:
.venv/bin/python -u scripts/run_b2_batched.py \
  --dataset <h5> --output <json_path> --variable income_tax \
  --period 2024 --batch-size 50000

.venv/bin/python -u scripts/run_b2_batched.py \
  --dataset <h5> --output <json_path> --variable aca_ptc \
  --period 2024 --batch-size 100000
```
