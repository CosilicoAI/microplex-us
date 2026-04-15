"""Tests for the persistent PE-US comparison harness."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from microplex.core import EntityType
from microplex.targets import (
    FilterOperator,
    StaticTargetProvider,
    TargetFilter,
    TargetQuery,
    TargetSet,
    TargetSpec,
)

import microplex_us.policyengine.comparison as comparison_module
import microplex_us.policyengine.harness as harness_module
from microplex_us.policyengine import (
    PolicyEngineUSEntityTableBundle,
    PolicyEngineUSHarnessRun,
    PolicyEngineUSHarnessSlice,
    PolicyEngineUSHarnessSliceResult,
    PolicyEngineUSMaterializationError,
    build_policyengine_us_time_period_arrays,
    default_policyengine_us_db_all_target_slices,
    default_policyengine_us_db_harness_slices,
    default_policyengine_us_db_parity_slices,
    evaluate_policyengine_us_harness,
    filter_nonempty_policyengine_us_harness_slices,
    write_policyengine_us_time_period_dataset,
)
from microplex_us.policyengine.comparison import (
    PolicyEngineUSTargetComparisonReport,
    PolicyEngineUSTargetEvaluation,
    PolicyEngineUSTargetEvaluationReport,
)


def _candidate_tables() -> PolicyEngineUSEntityTableBundle:
    return PolicyEngineUSEntityTableBundle(
        households=pd.DataFrame(
            {
                "household_id": [1, 2],
                "household_weight": [2.0, 1.0],
                "state_fips": [6, 36],
                "snap": [100.0, 50.0],
            }
        ),
        persons=pd.DataFrame(
            {
                "person_id": [10, 11, 20],
                "household_id": [1, 1, 2],
                "tax_unit_id": [100, 100, 200],
                "spm_unit_id": [1000, 1000, 2000],
                "family_id": [5000, 5000, 6000],
                "marital_unit_id": [7000, 7000, 8000],
                "age": [40.0, 10.0, 30.0],
                "employment_income": [30_000.0, 0.0, 20_000.0],
            }
        ),
        tax_units=pd.DataFrame(
            {
                "tax_unit_id": [100, 200],
                "household_id": [1, 2],
                "filing_status": ["JOINT", "SINGLE"],
            }
        ),
        spm_units=pd.DataFrame(
            {
                "spm_unit_id": [1000, 2000],
                "household_id": [1, 2],
            }
        ),
        families=pd.DataFrame(
            {
                "family_id": [5000, 6000],
                "household_id": [1, 2],
            }
        ),
        marital_units=pd.DataFrame(
            {
                "marital_unit_id": [7000, 8000],
                "household_id": [1, 2],
            }
        ),
    )


def _baseline_dataset(tmp_path: Path) -> Path:
    tables = PolicyEngineUSEntityTableBundle(
        households=pd.DataFrame(
            {
                "household_id": [1, 2],
                "household_weight": [1.0, 1.0],
                "state_fips": [6, 36],
                "snap": [75.0, 50.0],
            }
        ),
        persons=pd.DataFrame(
            {
                "person_id": [10, 11, 20],
                "household_id": [1, 1, 2],
                "tax_unit_id": [100, 100, 200],
                "spm_unit_id": [1000, 1000, 2000],
                "family_id": [5000, 5000, 6000],
                "marital_unit_id": [7000, 7000, 8000],
                "age": [40.0, 10.0, 30.0],
                "employment_income": [30_000.0, 0.0, 20_000.0],
            }
        ),
        tax_units=pd.DataFrame(
            {
                "tax_unit_id": [100, 200],
                "household_id": [1, 2],
                "filing_status": ["JOINT", "SINGLE"],
            }
        ),
        spm_units=pd.DataFrame(
            {
                "spm_unit_id": [1000, 2000],
                "household_id": [1, 2],
            }
        ),
        families=pd.DataFrame(
            {
                "family_id": [5000, 6000],
                "household_id": [1, 2],
            }
        ),
        marital_units=pd.DataFrame(
            {
                "marital_unit_id": [7000, 8000],
                "household_id": [1, 2],
            }
        ),
    )
    arrays = build_policyengine_us_time_period_arrays(
        tables,
        period=2024,
        household_variable_map={"state_fips": "state_fips", "snap": "snap"},
        person_variable_map={"age": "age", "employment_income": "employment_income"},
        tax_unit_variable_map={"filing_status": "filing_status"},
    )
    dataset_path = tmp_path / "baseline.h5"
    write_policyengine_us_time_period_dataset(arrays, dataset_path)
    return dataset_path


def test_evaluate_policyengine_us_harness_scores_candidate_against_baseline(tmp_path):
    provider = StaticTargetProvider(
        TargetSet(
            [
                TargetSpec(
                    name="ca_households",
                    entity=EntityType.HOUSEHOLD,
                    value=2.0,
                    period=2024,
                    aggregation="count",
                    filters=(TargetFilter("state_fips", FilterOperator.EQ, 6),),
                ),
                TargetSpec(
                    name="snap_total",
                    entity=EntityType.HOUSEHOLD,
                    value=250.0,
                    period=2024,
                    measure="snap",
                    aggregation="sum",
                ),
            ]
        )
    )
    slices = [
        PolicyEngineUSHarnessSlice(
            name="counts",
            tags=("national", "counts"),
            query=TargetQuery(period=2024, names=("ca_households",)),
        ),
        PolicyEngineUSHarnessSlice(
            name="snap",
            tags=("national", "programs"),
            query=TargetQuery(period=2024, names=("snap_total",)),
        ),
    ]

    run = evaluate_policyengine_us_harness(
        _candidate_tables(),
        provider,
        slices,
        baseline_dataset=_baseline_dataset(tmp_path),
        dataset_year=2024,
        candidate_label="candidate",
        baseline_label="baseline",
        metadata={"git_commit": "abc123"},
    )

    assert run.candidate_label == "candidate"
    assert run.baseline_label == "baseline"
    assert len(run.slice_results) == 2
    assert run.mean_abs_relative_error_delta is not None
    assert run.mean_abs_relative_error_delta < 0.0
    assert run.benchmark_suite.mean_abs_relative_error_delta is not None
    assert run.benchmark_suite.mean_abs_relative_error_delta < 0.0
    assert run.candidate_composite_parity_loss is not None
    assert run.baseline_composite_parity_loss is not None
    assert run.composite_parity_loss_delta is not None
    assert run.composite_parity_loss_delta < 0.0
    assert run.slice_win_rate == 1.0
    assert run.benchmark_suite.slice_win_rate == 1.0
    assert run.target_win_rate == 1.0
    assert run.supported_target_rate == 1.0
    assert run.tag_summaries["national"]["supported_target_rate"] == 1.0
    assert run.tag_summaries["national"]["candidate_composite_parity_loss"] is not None
    assert run.parity_scorecard["overall"]["candidate_beats_baseline"] is True
    assert run.parity_scorecard["national"]["candidate_beats_baseline"] is True
    assert run.attribute_cell_summaries
    assert run.metadata["git_commit"] == "abc123"


def test_policyengine_us_harness_run_round_trips_json(tmp_path):
    provider = StaticTargetProvider(
        TargetSet(
            [
                TargetSpec(
                    name="ca_households",
                    entity=EntityType.HOUSEHOLD,
                    value=2.0,
                    period=2024,
                    aggregation="count",
                    filters=(TargetFilter("state_fips", FilterOperator.EQ, 6),),
                ),
                TargetSpec(
                    name="snap_total",
                    entity=EntityType.HOUSEHOLD,
                    value=250.0,
                    period=2024,
                    measure="snap",
                    aggregation="sum",
                ),
            ]
        )
    )
    run = evaluate_policyengine_us_harness(
        _candidate_tables(),
        provider,
        [
            PolicyEngineUSHarnessSlice(
                name="core",
                query=TargetQuery(period=2024, names=("ca_households", "snap_total")),
                description="Core parity slice",
                tags=("national", "programs"),
            )
        ],
        baseline_dataset=_baseline_dataset(tmp_path),
        dataset_year=2024,
    )

    output_path = run.save(tmp_path / "reports" / "harness.json")
    loaded = PolicyEngineUSHarnessRun.load(output_path)
    payload = json.loads(output_path.read_text())

    assert loaded.candidate_label == run.candidate_label
    assert loaded.baseline_label == run.baseline_label
    assert loaded.period == run.period
    assert loaded.slice_results[0].slice.name == "core"
    assert loaded.slice_results[0].slice.description == "Core parity slice"
    assert loaded.slice_results[0].slice.tags == ("national", "programs")
    assert loaded.slice_results[0].comparison.candidate.evaluations[0].target.filters
    assert (
        loaded.slice_results[0]
        .comparison.candidate.evaluations[0]
        .target.filters[0]
        .feature
        == "state_fips"
    )
    assert loaded.slice_win_rate == run.slice_win_rate
    assert loaded.candidate_composite_parity_loss == run.candidate_composite_parity_loss
    assert payload["slices"][0]["summary"]["candidate_supported_target_count"] == 2
    assert payload["slices"][0]["summary"]["baseline_supported_target_count"] == 2


def test_policyengine_us_harness_preserves_duplicate_target_names_across_slices():
    target = TargetSpec(
        name="population",
        entity=EntityType.HOUSEHOLD,
        value=100.0,
        period=2024,
        aggregation="count",
        metadata={
            "geo_level": "state",
            "domain_variable": "age",
        },
    )
    run = PolicyEngineUSHarnessRun(
        candidate_label="candidate",
        baseline_label="baseline",
        period=2024,
        slice_results=[
            PolicyEngineUSHarnessSliceResult(
                slice=PolicyEngineUSHarnessSlice(
                    name="slice_a",
                    query=TargetQuery(period=2024, names=("population",)),
                ),
                comparison=PolicyEngineUSTargetComparisonReport(
                    candidate=PolicyEngineUSTargetEvaluationReport(
                        label="candidate",
                        period=2024,
                        evaluations=[
                            PolicyEngineUSTargetEvaluation(
                                target=target,
                                actual_value=100.0,
                            )
                        ],
                    ),
                    baseline=PolicyEngineUSTargetEvaluationReport(
                        label="baseline",
                        period=2024,
                        evaluations=[
                            PolicyEngineUSTargetEvaluation(
                                target=target,
                                actual_value=80.0,
                            )
                        ],
                    ),
                ),
            ),
            PolicyEngineUSHarnessSliceResult(
                slice=PolicyEngineUSHarnessSlice(
                    name="slice_b",
                    query=TargetQuery(period=2024, names=("population",)),
                ),
                comparison=PolicyEngineUSTargetComparisonReport(
                    candidate=PolicyEngineUSTargetEvaluationReport(
                        label="candidate",
                        period=2024,
                        evaluations=[
                            PolicyEngineUSTargetEvaluation(
                                target=target,
                                actual_value=50.0,
                            )
                        ],
                    ),
                    baseline=PolicyEngineUSTargetEvaluationReport(
                        label="baseline",
                        period=2024,
                        evaluations=[
                            PolicyEngineUSTargetEvaluation(
                                target=target,
                                actual_value=60.0,
                            )
                        ],
                    ),
                ),
            ),
        ],
    )

    cell_key = "geo=state|entity=household|aggregation=count|feature=household_count|domain=age"

    assert run.candidate_micro_mean_abs_relative_error == pytest.approx(0.25)
    assert run.baseline_micro_mean_abs_relative_error == pytest.approx(0.30)
    assert run.attribute_cell_summaries[cell_key]["candidate_target_count"] == 2
    assert run.attribute_cell_summaries[cell_key]["baseline_target_count"] == 2


def test_evaluate_policyengine_us_harness_raises_on_strict_materialization_failure(
    tmp_path,
):
    provider = StaticTargetProvider(
        TargetSet(
            [
                TargetSpec(
                    name="snap_total",
                    entity=EntityType.HOUSEHOLD,
                    value=250.0,
                    period=2024,
                    measure="snap",
                    aggregation="sum",
                ),
            ]
        )
    )
    baseline_tables = PolicyEngineUSEntityTableBundle(
        households=_candidate_tables().households.drop(columns=["snap"]),
        persons=_candidate_tables().persons,
        tax_units=_candidate_tables().tax_units,
        spm_units=_candidate_tables().spm_units,
        families=_candidate_tables().families,
        marital_units=_candidate_tables().marital_units,
    )
    arrays = build_policyengine_us_time_period_arrays(
        baseline_tables,
        period=2024,
        household_variable_map={"state_fips": "state_fips"},
        person_variable_map={"age": "age", "employment_income": "employment_income"},
        tax_unit_variable_map={"filing_status": "filing_status"},
    )
    baseline_path = tmp_path / "baseline_missing_snap.h5"
    write_policyengine_us_time_period_dataset(arrays, baseline_path)

    class FakeEntity:
        def __init__(self, key: str):
            self.key = key

    class FakeVariable:
        def __init__(self, entity: FakeEntity, formulas: dict[str, object] | None = None):
            self.entity = entity
            self.formulas = formulas or {}

        def is_input_variable(self) -> bool:
            return not self.formulas

    class FakeTaxBenefitSystem:
        variables = {
            "state_fips": FakeVariable(FakeEntity("household")),
            "snap": FakeVariable(FakeEntity("household"), formulas={"2024": object()}),
        }

    class FailingSimulation:
        tax_benefit_system = FakeTaxBenefitSystem()

        def __init__(self, dataset, dataset_year=None, **kwargs):
            assert Path(dataset).exists()
            assert dataset_year == 2024
            _ = kwargs

        def calculate(self, variable, period=None, map_to=None):
            assert variable == "snap"
            assert period == 2024
            assert map_to is None
            raise RuntimeError("snap materialization unavailable")

    with pytest.raises(PolicyEngineUSMaterializationError, match="baseline"):
        evaluate_policyengine_us_harness(
            _candidate_tables(),
            provider,
            [
                PolicyEngineUSHarnessSlice(
                    name="snap",
                    query=TargetQuery(period=2024, names=("snap_total",)),
                )
            ],
            baseline_dataset=baseline_path,
            dataset_year=2024,
            simulation_cls=FailingSimulation,
            candidate_label="candidate",
            baseline_label="baseline",
            strict_materialization=True,
        )


def test_filter_nonempty_policyengine_us_harness_slices_drops_empty_queries():
    provider = StaticTargetProvider(
        TargetSet(
            [
                TargetSpec(
                    name="snap_total",
                    entity=EntityType.HOUSEHOLD,
                    value=250.0,
                    period=2024,
                    measure="snap",
                    aggregation="sum",
                ),
            ]
        )
    )

    filtered = filter_nonempty_policyengine_us_harness_slices(
        provider,
        (
            PolicyEngineUSHarnessSlice(
                name="counts",
                query=TargetQuery(period=2024, names=("household_count",)),
            ),
            PolicyEngineUSHarnessSlice(
                name="snap",
                query=TargetQuery(period=2024, names=("snap_total",)),
            ),
        ),
    )

    assert [slice_spec.name for slice_spec in filtered] == ["snap"]


def test_evaluate_policyengine_us_harness_reuses_union_evaluation(tmp_path, monkeypatch):
    provider = StaticTargetProvider(
        TargetSet(
            [
                TargetSpec(
                    name="ca_households",
                    entity=EntityType.HOUSEHOLD,
                    value=2.0,
                    period=2024,
                    aggregation="count",
                    filters=(TargetFilter("state_fips", FilterOperator.EQ, 6),),
                ),
                TargetSpec(
                    name="snap_total",
                    entity=EntityType.HOUSEHOLD,
                    value=250.0,
                    period=2024,
                    measure="snap",
                    aggregation="sum",
                ),
            ]
        )
    )
    evaluate_calls: list[tuple[str, ...]] = []
    real_evaluate = comparison_module.evaluate_policyengine_us_target_set

    def record_evaluate(*args, **kwargs):
        targets = args[1]
        if isinstance(targets, TargetSet):
            target_names = tuple(target.name for target in targets.targets)
        else:
            target_names = tuple(target.name for target in targets)
        evaluate_calls.append(target_names)
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(comparison_module, "evaluate_policyengine_us_target_set", record_evaluate)
    monkeypatch.setattr(harness_module, "evaluate_policyengine_us_target_set", record_evaluate)

    run = evaluate_policyengine_us_harness(
        _candidate_tables(),
        provider,
        [
            PolicyEngineUSHarnessSlice(
                name="counts",
                query=TargetQuery(period=2024, names=("ca_households",)),
            ),
            PolicyEngineUSHarnessSlice(
                name="snap",
                query=TargetQuery(period=2024, names=("snap_total",)),
            ),
        ],
        baseline_dataset=_baseline_dataset(tmp_path),
        dataset_year=2024,
    )

    assert run.slice_win_rate == 1.0
    assert evaluate_calls == [
        ("ca_households", "snap_total"),
        ("ca_households", "snap_total"),
    ]


def test_evaluate_policyengine_us_harness_passes_candidate_direct_override_variables(
    tmp_path,
    monkeypatch,
):
    provider = StaticTargetProvider(
        TargetSet(
            [
                TargetSpec(
                    name="snap_total",
                    entity=EntityType.HOUSEHOLD,
                    value=250.0,
                    period=2024,
                    measure="snap",
                    aggregation="sum",
                ),
            ]
        )
    )
    captured: list[tuple[str, ...]] = []
    real_evaluate = comparison_module.evaluate_policyengine_us_target_sets

    def record_evaluate(*args, **kwargs):
        captured.append(tuple(kwargs.get("direct_override_variables", ())))
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(
        comparison_module,
        "evaluate_policyengine_us_target_sets",
        record_evaluate,
    )
    monkeypatch.setattr(
        harness_module,
        "evaluate_policyengine_us_target_sets",
        record_evaluate,
    )

    evaluate_policyengine_us_harness(
        _candidate_tables(),
        provider,
        [
            PolicyEngineUSHarnessSlice(
                name="snap",
                query=TargetQuery(period=2024, names=("snap_total",)),
            ),
        ],
        baseline_dataset=_baseline_dataset(tmp_path),
        dataset_year=2024,
        candidate_direct_override_variables=("snap", "ssi"),
    )

    assert captured == [("snap", "ssi")]


def test_evaluate_policyengine_us_harness_excludes_zero_common_slices_from_suite(
    tmp_path,
):
    provider = StaticTargetProvider(
        TargetSet(
            [
                TargetSpec(
                    name="ca_households",
                    entity=EntityType.HOUSEHOLD,
                    value=2.0,
                    period=2024,
                    aggregation="count",
                    filters=(TargetFilter("state_fips", FilterOperator.EQ, 6),),
                ),
                TargetSpec(
                    name="district_households",
                    entity=EntityType.HOUSEHOLD,
                    value=2.0,
                    period=2024,
                    aggregation="count",
                    filters=(
                        TargetFilter(
                            "congressional_district_geoid",
                            FilterOperator.EQ,
                            601,
                        ),
                    ),
                ),
            ]
        )
    )

    run = evaluate_policyengine_us_harness(
        _candidate_tables(),
        provider,
        [
            PolicyEngineUSHarnessSlice(
                name="state",
                query=TargetQuery(period=2024, names=("ca_households",)),
                tags=("local", "state"),
            ),
            PolicyEngineUSHarnessSlice(
                name="district",
                query=TargetQuery(period=2024, names=("district_households",)),
                tags=("local", "district"),
            ),
        ],
        baseline_dataset=_baseline_dataset(tmp_path),
        dataset_year=2024,
        strict_materialization=False,
    )

    assert len(run.slice_results) == 2
    assert run.slice_results[1].comparison.benchmark_comparison is None
    assert [result.slice.name for result in run.benchmark_suite.slice_results] == ["state"]
    assert run.metadata["excluded_slice_names"] == ["district"]


def test_default_policyengine_us_db_harness_slices_tracks_provider_filters():
    slices = default_policyengine_us_db_harness_slices(
        period=2024,
        variables=("snap", "household_count"),
        domain_variables=("snap",),
        geo_levels=("state",),
        reform_id=3,
    )

    assert [slice_spec.name for slice_spec in slices] == [
        "all_targets",
        "snap",
        "household_count",
    ]
    assert slices[0].query.provider_filters == {
        "reform_id": 3,
        "variables": ["snap", "household_count"],
        "domain_variables": ["snap"],
        "geo_levels": ["state"],
    }
    assert slices[1].query.provider_filters["variables"] == ["snap"]
    assert slices[2].query.provider_filters["variables"] == ["household_count"]


def test_default_policyengine_us_db_all_target_slices_span_all_active_targets():
    slices = default_policyengine_us_db_all_target_slices(period=2024, reform_id=3)

    assert [slice_spec.name for slice_spec in slices] == ["all_targets"]
    assert slices[0].tags == ("benchmark", "all_targets")
    assert slices[0].query.provider_filters == {"reform_id": 3}


def test_default_policyengine_us_db_parity_slices_track_tags_and_filters():
    slices = default_policyengine_us_db_parity_slices(
        period=2024,
        variables=("snap", "household_count"),
        domain_variables=("snap",),
        geo_levels=("state", "district"),
        reform_id=3,
    )

    assert [slice_spec.name for slice_spec in slices] == [
        "state_programs_core",
        "district_snap_households",
    ]
    assert slices[0].tags == ("parity", "local", "state", "programs")
    assert slices[0].query.provider_filters == {
        "reform_id": 3,
        "variables": ["household_count"],
        "domain_variable_values": ["snap"],
        "geo_levels": ["state"],
    }
    assert slices[-1].tags == ("parity", "local", "district", "programs", "snap")
    assert slices[-1].query.provider_filters["geo_levels"] == ["district"]
    assert slices[-1].query.provider_filters["domain_variable_values"] == ["snap"]
