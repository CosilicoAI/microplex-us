"""Tests for the staged reduced benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from microplex.core import EntityType
from microplex.targets import FilterOperator

from microplex_us.pipelines.performance import (
    USMicroplexPerformanceHarnessConfig,
    USMicroplexPerformanceHarnessResult,
)
from microplex_us.pipelines.reduced_benchmark import (
    USMicroplexReducedBenchmarkHarnessConfig,
    USMicroplexReducedBenchmarkSpec,
    USMicroplexReducedCalibrationReport,
    USMicroplexReducedDimensionSpec,
    USMicroplexReducedMeasureSpec,
    calibrate_and_evaluate_us_reduced_benchmarks,
    default_us_atomic_rung1_benchmarks,
    default_us_atomic_rung2_calibration,
    evaluate_us_reduced_benchmark,
    reduced_benchmark_to_calibration_targets,
    run_us_microplex_reduced_benchmark_harness,
)
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexBuildResult,
    USMicroplexTargets,
)
from microplex_us.policyengine import (
    PolicyEngineUSEntityTableBundle,
    build_policyengine_us_time_period_arrays,
    write_policyengine_us_time_period_dataset,
)


def _sample_bundle(
    *,
    household_weights: tuple[float, ...],
    state_fips: tuple[int, ...],
    ages_by_household: tuple[tuple[float, ...], ...],
    female_by_household: tuple[tuple[bool, ...], ...] | None = None,
    employment_income_by_household: tuple[tuple[float, ...], ...] | None = None,
) -> PolicyEngineUSEntityTableBundle:
    household_ids = list(range(1, len(household_weights) + 1))
    households = pd.DataFrame(
        {
            "household_id": household_ids,
            "household_weight": list(household_weights),
            "state_fips": list(state_fips),
        }
    )
    person_rows: list[dict[str, int | float]] = []
    person_id = 10
    female_groups = female_by_household or tuple(
        tuple(False for _ in ages) for ages in ages_by_household
    )
    employment_groups = employment_income_by_household or tuple(
        tuple(0.0 for _ in ages) for ages in ages_by_household
    )
    for household_id, ages, female_flags, incomes in zip(
        household_ids,
        ages_by_household,
        female_groups,
        employment_groups,
        strict=True,
    ):
        for age, is_female, employment_income in zip(
            ages,
            female_flags,
            incomes,
            strict=True,
        ):
            person_rows.append(
                {
                    "person_id": person_id,
                    "household_id": household_id,
                    "tax_unit_id": household_id * 100,
                    "spm_unit_id": household_id * 1000,
                    "family_id": household_id * 5000,
                    "marital_unit_id": household_id * 7000,
                    "age": age,
                    "is_female": is_female,
                    "employment_income_before_lsr": employment_income,
                }
            )
            person_id += 1
    persons = pd.DataFrame(person_rows)
    return PolicyEngineUSEntityTableBundle(
        households=households,
        persons=persons,
        tax_units=None,
        spm_units=None,
        families=None,
        marital_units=None,
    )


def _write_dataset(bundle: PolicyEngineUSEntityTableBundle, path: Path) -> Path:
    person_variable_map = {"age": "age"}
    if "is_female" in bundle.persons.columns:
        person_variable_map["is_female"] = "is_female"
    if "employment_income_before_lsr" in bundle.persons.columns:
        person_variable_map["employment_income_before_lsr"] = (
            "employment_income_before_lsr"
        )
    arrays = build_policyengine_us_time_period_arrays(
        bundle,
        period=2024,
        household_variable_map={"state_fips": "state_fips"},
        person_variable_map=person_variable_map,
    )
    return write_policyengine_us_time_period_dataset(arrays, path)


def test_evaluate_us_reduced_benchmark_compares_weighted_household_counts(tmp_path):
    baseline_path = _write_dataset(
        _sample_bundle(
            household_weights=(2.0, 1.0),
            state_fips=(6, 36),
            ages_by_household=((10.0, 40.0), (70.0,)),
        ),
        tmp_path / "baseline.h5",
    )
    candidate_path = _write_dataset(
        _sample_bundle(
            household_weights=(1.0, 1.0),
            state_fips=(6, 36),
            ages_by_household=((40.0,), (70.0,)),
        ),
        tmp_path / "candidate.h5",
    )
    spec = USMicroplexReducedBenchmarkSpec(
        name="household_count_by_state",
        entity="household",
        dimensions=(
            USMicroplexReducedDimensionSpec(variable="state_fips", zero_pad=2),
        ),
        measures=(
            USMicroplexReducedMeasureSpec(name="weighted_household_count"),
        ),
    )

    report = evaluate_us_reduced_benchmark(
        candidate_path,
        baseline_path,
        spec,
        period=2024,
    )

    summary = report.measure_summaries["weighted_household_count"]
    assert summary["candidate_total"] == pytest.approx(2.0)
    assert summary["baseline_total"] == pytest.approx(3.0)
    assert summary["support_recall"] == pytest.approx(1.0)
    assert report.top_cell_gaps["weighted_household_count"][0]["state_fips"] == "06"
    assert report.top_cell_gaps["weighted_household_count"][0]["delta"] == pytest.approx(
        -1.0
    )


def test_evaluate_us_reduced_benchmark_supports_binned_person_counts(tmp_path):
    baseline_path = _write_dataset(
        _sample_bundle(
            household_weights=(2.0, 1.0),
            state_fips=(6, 36),
            ages_by_household=((10.0, 40.0), (70.0,)),
        ),
        tmp_path / "baseline.h5",
    )
    candidate_path = _write_dataset(
        _sample_bundle(
            household_weights=(1.0, 1.0),
            state_fips=(6, 36),
            ages_by_household=((40.0,), (70.0,)),
        ),
        tmp_path / "candidate.h5",
    )
    spec = USMicroplexReducedBenchmarkSpec(
        name="person_count_by_state_age",
        entity="person",
        dimensions=(
            USMicroplexReducedDimensionSpec(variable="state_fips", zero_pad=2),
            USMicroplexReducedDimensionSpec(
                variable="age",
                label="age_bucket",
                bins=(0.0, 18.0, 65.0, 200.0),
                bin_labels=("0_to_17", "18_to_64", "65_plus"),
            ),
        ),
        measures=(
            USMicroplexReducedMeasureSpec(name="weighted_person_count"),
        ),
    )

    report = evaluate_us_reduced_benchmark(
        candidate_path,
        baseline_path,
        spec,
        period=2024,
    )

    top_gap = report.top_cell_gaps["weighted_person_count"][0]
    assert top_gap["state_fips"] == "06"
    assert top_gap["age_bucket"] == "0_to_17"
    assert top_gap["delta"] == pytest.approx(-2.0)
    assert report.summary["n_dimensions"] == 2


def test_run_us_microplex_reduced_benchmark_harness_wraps_performance_harness(
    monkeypatch,
    tmp_path,
):
    baseline_path = _write_dataset(
        _sample_bundle(
            household_weights=(2.0, 1.0),
            state_fips=(6, 36),
            ages_by_household=((10.0, 40.0), (70.0,)),
        ),
        tmp_path / "baseline.h5",
    )
    candidate_path = _write_dataset(
        _sample_bundle(
            household_weights=(1.0, 1.0),
            state_fips=(6, 36),
            ages_by_household=((40.0,), (70.0,)),
        ),
        tmp_path / "candidate.h5",
    )
    captured: dict[str, USMicroplexPerformanceHarnessConfig] = {}

    def _fake_run_us_microplex_performance_harness(
        providers,
        *,
        config,
        queries=None,
        **kwargs,
    ):
        _ = providers
        _ = queries
        _ = kwargs
        captured["config"] = config
        build_config = USMicroplexBuildConfig()
        build_result = USMicroplexBuildResult(
            config=build_config,
            seed_data=pd.DataFrame(),
            synthetic_data=pd.DataFrame(),
            calibrated_data=pd.DataFrame(),
            targets=USMicroplexTargets(marginal={}, continuous={}),
            calibration_summary={"backend": "policyengine_db_none"},
            policyengine_tables=_sample_bundle(
                household_weights=(1.0, 1.0),
                state_fips=(6, 36),
                ages_by_household=((40.0,), (70.0,)),
            ),
            source_frame=None,
            source_frames=(),
            fusion_plan=None,
        )
        return USMicroplexPerformanceHarnessResult(
            config=config,
            build_config=build_config,
            build_result=build_result,
            source_names=("stub",),
            stage_timings={"write_policyengine_dataset": 0.1},
            total_seconds=0.2,
            policyengine_dataset_path=str(candidate_path),
        )

    monkeypatch.setattr(
        "microplex_us.pipelines.reduced_benchmark.run_us_microplex_performance_harness",
        _fake_run_us_microplex_performance_harness,
    )
    spec = USMicroplexReducedBenchmarkSpec(
        name="household_count_by_state",
        entity="household",
        dimensions=(
            USMicroplexReducedDimensionSpec(variable="state_fips", zero_pad=2),
        ),
        measures=(
            USMicroplexReducedMeasureSpec(name="weighted_household_count"),
        ),
    )
    output_json = tmp_path / "reduced_harness.json"
    result = run_us_microplex_reduced_benchmark_harness(
        [SimpleNamespace(descriptor=SimpleNamespace(name="stub"))],
        config=USMicroplexReducedBenchmarkHarnessConfig(
            performance_config=USMicroplexPerformanceHarnessConfig(
                baseline_dataset=baseline_path,
                evaluate_parity=True,
                evaluate_pe_native_loss=True,
            ),
            benchmark_specs=(spec,),
            output_json_path=output_json,
        ),
    )

    assert captured["config"].evaluate_parity is False
    assert captured["config"].evaluate_pe_native_loss is False
    assert captured["config"].output_policyengine_dataset_path is not None
    assert result.candidate_dataset_path == str(candidate_path)
    assert "household_count_by_state" in result.benchmark_reports
    payload = json.loads(output_json.read_text())
    assert "benchmark_reports" in payload
    assert payload["benchmark_reports"]["household_count_by_state"]["summary"]["n_cells"] == 2


def test_evaluate_us_reduced_benchmark_weighted_sum(tmp_path):
    """Weighted sum aggregation correctly sums age * weight per state."""
    baseline_path = _write_dataset(
        _sample_bundle(
            household_weights=(2.0, 1.0),
            state_fips=(6, 36),
            ages_by_household=((10.0, 40.0), (70.0,)),
        ),
        tmp_path / "baseline.h5",
    )
    candidate_path = _write_dataset(
        _sample_bundle(
            household_weights=(1.0, 1.0),
            state_fips=(6, 36),
            ages_by_household=((40.0,), (70.0,)),
        ),
        tmp_path / "candidate.h5",
    )
    spec = USMicroplexReducedBenchmarkSpec(
        name="person_age_sum_by_state",
        entity="person",
        dimensions=(
            USMicroplexReducedDimensionSpec(variable="state_fips", zero_pad=2),
        ),
        measures=(
            USMicroplexReducedMeasureSpec(
                name="weighted_age_sum",
                aggregation="weighted_sum",
                variable="age",
            ),
        ),
    )

    report = evaluate_us_reduced_benchmark(
        candidate_path,
        baseline_path,
        spec,
        period=2024,
    )

    summary = report.measure_summaries["weighted_age_sum"]
    # Baseline: state 06 has person age 10 (w=2) + age 40 (w=2) = 100,
    #           state 36 has person age 70 (w=1) = 70  → total 170
    # Candidate: state 06 has person age 40 (w=1) = 40,
    #            state 36 has person age 70 (w=1) = 70  → total 110
    assert summary["baseline_total"] == pytest.approx(170.0)
    assert summary["candidate_total"] == pytest.approx(110.0)
    assert summary["total_delta"] == pytest.approx(-60.0)


def test_validate_duplicate_dimension_output_names():
    """Duplicate dimension output names are rejected."""
    spec = USMicroplexReducedBenchmarkSpec(
        name="bad_spec",
        entity="person",
        dimensions=(
            USMicroplexReducedDimensionSpec(variable="state_fips", zero_pad=2),
            USMicroplexReducedDimensionSpec(
                variable="state_fips", label="state_fips", zero_pad=2
            ),
        ),
    )
    with pytest.raises(ValueError, match="duplicate dimension output name"):
        from microplex_us.pipelines.reduced_benchmark import (
            _validate_reduced_benchmark_spec,
        )

        _validate_reduced_benchmark_spec(spec)


def test_evaluate_us_reduced_benchmark_weighted_mean_by_state_sex(tmp_path):
    baseline_path = _write_dataset(
        _sample_bundle(
            household_weights=(2.0, 1.0),
            state_fips=(6, 36),
            ages_by_household=((10.0, 40.0), (70.0,)),
            female_by_household=((True, False), (True,)),
            employment_income_by_household=((100.0, 80.0), (40.0,)),
        ),
        tmp_path / "baseline.h5",
    )
    candidate_path = _write_dataset(
        _sample_bundle(
            household_weights=(1.0, 1.0),
            state_fips=(6, 36),
            ages_by_household=((10.0, 40.0), (70.0,)),
            female_by_household=((True, False), (True,)),
            employment_income_by_household=((120.0, 60.0), (20.0,)),
        ),
        tmp_path / "candidate.h5",
    )
    spec = USMicroplexReducedBenchmarkSpec(
        name="employment_income_mean_by_state_sex",
        entity="person",
        dimensions=(
            USMicroplexReducedDimensionSpec(variable="state_fips", zero_pad=2),
            USMicroplexReducedDimensionSpec(variable="is_female"),
        ),
        measures=(
            USMicroplexReducedMeasureSpec(
                name="weighted_employment_income_mean",
                aggregation="weighted_mean",
                variable="employment_income_before_lsr",
            ),
        ),
    )

    report = evaluate_us_reduced_benchmark(
        candidate_path,
        baseline_path,
        spec,
        period=2024,
    )

    summary = report.measure_summaries["weighted_employment_income_mean"]
    assert summary["baseline_nonzero_cell_count"] == 3
    assert summary["candidate_nonzero_cell_count"] == 3
    assert report.top_cell_gaps["weighted_employment_income_mean"][0]["state_fips"] == "36"
    assert report.top_cell_gaps["weighted_employment_income_mean"][0]["is_female"] is True
    assert report.top_cell_gaps["weighted_employment_income_mean"][0]["delta"] == pytest.approx(
        -20.0
    )


def test_evaluate_us_reduced_benchmark_weighted_mean_asymmetric_cells(tmp_path):
    """Weighted mean with asymmetric cell coverage produces valid MARE, not NaN."""
    baseline_path = _write_dataset(
        _sample_bundle(
            household_weights=(2.0, 1.0),
            state_fips=(6, 36),
            ages_by_household=((10.0, 40.0), (70.0,)),
            female_by_household=((True, False), (True,)),
            employment_income_by_household=((100.0, 80.0), (40.0,)),
        ),
        tmp_path / "baseline.h5",
    )
    # Candidate has only state 06 — state 36 cell will be missing.
    candidate_path = _write_dataset(
        _sample_bundle(
            household_weights=(1.0,),
            state_fips=(6,),
            ages_by_household=((40.0, 20.0),),
            female_by_household=((True, False),),
            employment_income_by_household=((120.0, 60.0),),
        ),
        tmp_path / "candidate.h5",
    )
    spec = USMicroplexReducedBenchmarkSpec(
        name="income_mean_by_state_sex",
        entity="person",
        dimensions=(
            USMicroplexReducedDimensionSpec(variable="state_fips", zero_pad=2),
            USMicroplexReducedDimensionSpec(variable="is_female"),
        ),
        measures=(
            USMicroplexReducedMeasureSpec(
                name="weighted_employment_income_mean",
                aggregation="weighted_mean",
                variable="employment_income_before_lsr",
            ),
        ),
    )

    report = evaluate_us_reduced_benchmark(
        candidate_path,
        baseline_path,
        spec,
        period=2024,
    )

    mare = report.measure_summaries["weighted_employment_income_mean"]
    assert not np.isnan(mare["mean_abs_relative_error"])
    assert not np.isnan(mare["max_abs_relative_error"])
    assert not np.isnan(report.summary["mean_measure_mare"])
    # The missing (36, True) cell should surface in top gaps with NaN candidate.
    assert mare["n_cells"] == 3
    assert mare["shared_nonzero_cell_count"] == 2


def test_default_us_atomic_rung1_benchmarks_returns_expected_specs():
    specs = default_us_atomic_rung1_benchmarks()
    assert [spec.name for spec in specs] == [
        "person_count_by_state_sex",
        "employment_income_sum_by_state",
        "employment_income_mean_by_state_sex",
    ]


def test_reduced_benchmark_to_calibration_targets_emits_state_count_targets(tmp_path):
    baseline_path = _write_dataset(
        _sample_bundle(
            household_weights=(2.0, 1.0),
            state_fips=(6, 36),
            ages_by_household=((10.0,), (70.0,)),
        ),
        tmp_path / "baseline.h5",
    )
    spec = USMicroplexReducedBenchmarkSpec(
        name="household_count_by_state",
        entity="household",
        dimensions=(
            USMicroplexReducedDimensionSpec(variable="state_fips", zero_pad=2),
        ),
        measures=(USMicroplexReducedMeasureSpec(name="weighted_household_count"),),
    )

    targets = reduced_benchmark_to_calibration_targets(spec, baseline_path, period=2024)

    assert len(targets) == 2
    assert all(target.entity is EntityType.HOUSEHOLD for target in targets)
    assert all(target.aggregation.value == "count" for target in targets)
    assert {target.value for target in targets} == {1.0, 2.0}
    state_filters = {
        target.filters[0].value: target.filters[0].operator for target in targets
    }
    assert state_filters == {6: FilterOperator.EQ, 36: FilterOperator.EQ}


def test_calibrate_and_evaluate_us_reduced_benchmarks_improves_state_count_surface(
    tmp_path,
):
    baseline_path = _write_dataset(
        _sample_bundle(
            household_weights=(2.0, 1.0),
            state_fips=(6, 36),
            ages_by_household=((10.0, 40.0), (70.0,)),
            female_by_household=((True, False), (True,)),
            employment_income_by_household=((100.0, 80.0), (40.0,)),
        ),
        tmp_path / "baseline.h5",
    )
    candidate_path = _write_dataset(
        _sample_bundle(
            household_weights=(1.0, 1.0),
            state_fips=(6, 36),
            ages_by_household=((10.0, 40.0), (70.0,)),
            female_by_household=((True, False), (True,)),
            employment_income_by_household=((100.0, 80.0), (40.0,)),
        ),
        tmp_path / "candidate.h5",
    )
    calibration_spec, evaluation_specs = default_us_atomic_rung2_calibration()
    output_path = tmp_path / "reweighted.h5"

    report = calibrate_and_evaluate_us_reduced_benchmarks(
        candidate_path,
        baseline_path,
        calibration_spec,
        evaluation_specs=(evaluation_specs[0], evaluation_specs[1]),
        period=2024,
        output_reweighted_dataset_path=output_path,
    )

    assert isinstance(report, USMicroplexReducedCalibrationReport)
    assert report.reweighting_summary["constraint_count"] == 2
    assert report.reweighted_dataset_path == str(output_path.resolve())
    assert output_path.exists()
    state_spec_name = evaluation_specs[0].name
    age_spec_name = evaluation_specs[1].name
    assert (
        report.benchmark_deltas[state_spec_name]["post_mean_measure_mare"]
        < report.benchmark_deltas[state_spec_name]["pre_mean_measure_mare"]
    )
    assert (
        report.benchmark_deltas[age_spec_name]["post_mean_measure_mare"]
        < report.benchmark_deltas[age_spec_name]["pre_mean_measure_mare"]
    )
