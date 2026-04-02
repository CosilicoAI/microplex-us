"""Tests for the staged reduced benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from microplex_us.pipelines.performance import (
    USMicroplexPerformanceHarnessConfig,
    USMicroplexPerformanceHarnessResult,
)
from microplex_us.pipelines.reduced_benchmark import (
    USMicroplexReducedBenchmarkHarnessConfig,
    USMicroplexReducedBenchmarkSpec,
    USMicroplexReducedDimensionSpec,
    USMicroplexReducedMeasureSpec,
    evaluate_us_reduced_benchmark,
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
    for household_id, ages in zip(household_ids, ages_by_household, strict=True):
        for age in ages:
            person_rows.append(
                {
                    "person_id": person_id,
                    "household_id": household_id,
                    "tax_unit_id": household_id * 100,
                    "spm_unit_id": household_id * 1000,
                    "family_id": household_id * 5000,
                    "marital_unit_id": household_id * 7000,
                    "age": age,
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
    arrays = build_policyengine_us_time_period_arrays(
        bundle,
        period=2024,
        household_variable_map={"state_fips": "state_fips"},
        person_variable_map={"age": "age"},
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
