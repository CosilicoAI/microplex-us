"""Tests for the US microplex performance harness."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd
from microplex.core import EntityType
from microplex.targets import TargetQuery, TargetSet, TargetSpec

from microplex_us.pipelines.performance import (
    USMicroplexPerformanceHarnessConfig,
    USMicroplexPerformanceSession,
    default_fast_calibration_target_variables,
    run_us_microplex_performance_harness,
    warm_us_microplex_parity_cache,
)
from microplex_us.pipelines.us import USMicroplexBuildConfig
from microplex_us.policyengine import (
    PolicyEngineUSComparisonCache,
    PolicyEngineUSEntityTableBundle,
)


class _DummyProvider:
    def __init__(self, name: str):
        self.descriptor = SimpleNamespace(name=name)

    def load_frame(self, query=None):
        _ = query
        return SimpleNamespace(source=SimpleNamespace(name=self.descriptor.name))


@dataclass
class _FakeHarnessRun:
    candidate_composite_parity_loss: float = 0.4
    baseline_composite_parity_loss: float = 0.5
    candidate_mean_abs_relative_error: float = 0.2
    baseline_mean_abs_relative_error: float = 0.25
    target_win_rate: float = 0.75
    slice_win_rate: float = 1.0


class _FakePipeline:
    def __init__(
        self,
        config=None,
        stage_log: list[str] | None = None,
        stage_log_style: str = "full",
    ):
        self.config = config
        self.stage_log = stage_log
        self.stage_log_style = stage_log_style

    def _log(self, message: str) -> None:
        if self.stage_log is not None:
            self.stage_log.append(message)

    def prepare_source_input(self, frame):
        if self.stage_log_style == "short":
            self._log(f"prepare_source_input:{frame.source.name}")
        else:
            self._log("prepare_source_input")
        return SimpleNamespace(frame=frame)

    def _select_scaffold_source(self, source_inputs):
        self._log("select_scaffold")
        return source_inputs[0]

    def prepare_seed_data_from_source(self, source_input):
        _ = source_input
        self._log("prepare_seed" if self.stage_log_style == "short" else "prepare_seed_data")
        return pd.DataFrame({"household_id": [1], "income": [1.0], "hh_weight": [1.0]})

    def _integrate_donor_sources(self, seed_data, *, scaffold_input, donor_inputs):
        _ = scaffold_input
        _ = donor_inputs
        self._log(
            "integrate_donors"
            if self.stage_log_style == "short"
            else "integrate_donor_sources"
        )
        if self.stage_log is not None:
            return {
                "seed_data": seed_data.assign(dividend_income=[2.0]),
                "integrated_variables": ["dividend_income"],
            }
        return {"seed_data": seed_data, "integrated_variables": []}

    def build_targets(self, seed_data):
        _ = seed_data
        self._log("build_targets")
        return SimpleNamespace(marginal={}, continuous={})

    def _resolve_synthesis_variables(
        self,
        source_input=None,
        *,
        fusion_plan=None,
        include_all_observed_targets=False,
        available_columns=None,
    ):
        _ = source_input
        _ = fusion_plan
        _ = include_all_observed_targets
        _ = available_columns
        self._log("resolve_synthesis_variables")
        return SimpleNamespace(condition_vars=("age",), target_vars=("income",))

    def synthesize(self, seed_data, synthesis_variables=None):
        _ = synthesis_variables
        self._log("synthesize")
        return seed_data.assign(weight=[1.0]), None, {"backend": "bootstrap"}

    def ensure_target_support(self, synthetic_data, seed_data, targets):
        _ = seed_data
        _ = targets
        self._log("ensure_target_support")
        return synthetic_data

    def build_policyengine_entity_tables(self, population):
        _ = population
        self._log(
            "build_policyengine_entity_tables"
            if self.stage_log_style == "short"
            else "build_policyengine_tables"
        )
        if self.stage_log is not None:
            return PolicyEngineUSEntityTableBundle(
                households=pd.DataFrame({"household_id": [1], "household_weight": [1.0]}),
                persons=None,
                tax_units=None,
                spm_units=None,
                families=None,
                marital_units=None,
            )
        return SimpleNamespace(households=pd.DataFrame({"household_id": [1]}))

    def calibrate_policyengine_tables(self, tables):
        _ = tables
        self._log("calibrate_policyengine_tables")
        if self.stage_log is not None:
            return (
                PolicyEngineUSEntityTableBundle(
                    households=pd.DataFrame({"household_id": [1], "household_weight": [1.0]}),
                    persons=None,
                    tax_units=None,
                    spm_units=None,
                    families=None,
                    marital_units=None,
                ),
                pd.DataFrame({"weight": [1.0]}),
                {"backend": "policyengine_db_entropy"},
            )
        return (
            SimpleNamespace(households=pd.DataFrame({"household_id": [1]})),
            pd.DataFrame({"weight": [1.0]}),
            {"backend": "policyengine_db_entropy"},
        )

    def export_policyengine_dataset(
        self,
        result,
        path,
        *,
        period=None,
        direct_override_variables=None,
    ):
        _ = result
        _ = period
        self._log(f"export_policyengine_dataset:{tuple(direct_override_variables or ())}")
        path.write_text("stub")
        return path


def _patch_fake_harness(
    monkeypatch,
    *,
    stage_log: list[str] | None = None,
    stage_log_style: str = "full",
) -> None:
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.USMicroplexPipeline",
        lambda config=None: _FakePipeline(
            config=config,
            stage_log=stage_log,
            stage_log_style=stage_log_style,
        ),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.FusionPlan.from_sources",
        lambda sources: SimpleNamespace(source_names=tuple(source.name for source in sources)),
    )


def test_default_fast_calibration_target_variables_prefers_income_tax_over_agi():
    assert default_fast_calibration_target_variables(
        ("adjusted_gross_income", "income_tax", "dividend_income")
    ) == ("income_tax", "dividend_income")
    assert default_fast_calibration_target_variables(
        ("adjusted_gross_income", "dividend_income")
    ) == ("adjusted_gross_income", "dividend_income")


def test_run_us_microplex_performance_harness_returns_stage_timings(monkeypatch):
    stage_log: list[str] = []
    cache_refs: list[object] = []
    _patch_fake_harness(
        monkeypatch,
        stage_log=stage_log,
        stage_log_style="short",
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.PolicyEngineUSDBTargetProvider",
        lambda path: SimpleNamespace(path=path),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.default_policyengine_us_db_harness_slices",
        lambda **kwargs: (SimpleNamespace(name="all_targets", query=SimpleNamespace(period=kwargs["period"])),),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.filter_nonempty_policyengine_us_harness_slices",
        lambda provider, slices, cache=None: cache_refs.append(cache) or tuple(slices),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.evaluate_policyengine_us_harness",
        lambda *args, **kwargs: cache_refs.append(kwargs.get("cache")) or _FakeHarnessRun(),
    )
    comparison_cache = PolicyEngineUSComparisonCache()

    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps"), _DummyProvider("puf")],
        config=USMicroplexPerformanceHarnessConfig(
            targets_db="/tmp/policy_data.db",
            baseline_dataset="/tmp/enhanced_cps.h5",
        ),
        comparison_cache=comparison_cache,
    )

    assert result.source_names == ("cps", "puf")
    assert result.candidate_composite_parity_loss == 0.4
    assert result.baseline_composite_parity_loss == 0.5
    assert result.target_win_rate == 0.75
    assert result.slice_win_rate == 1.0
    assert result.total_seconds >= 0.0
    assert cache_refs == [comparison_cache, comparison_cache]
    assert set(result.stage_timings) >= {
        "load_frames",
        "prepare_source_inputs",
        "prepare_seed_data",
        "integrate_donor_sources",
        "build_targets",
        "resolve_synthesis_variables",
        "synthesize",
        "ensure_target_support",
        "build_policyengine_tables",
        "calibrate_policyengine_tables",
        "evaluate_parity_harness",
    }
    assert stage_log == [
        "prepare_source_input:cps",
        "prepare_source_input:puf",
        "select_scaffold",
        "prepare_seed",
        "integrate_donors",
        "build_targets",
        "resolve_synthesis_variables",
        "synthesize",
        "ensure_target_support",
        "build_policyengine_entity_tables",
        "calibrate_policyengine_tables",
    ]


def test_run_us_microplex_performance_harness_requires_targets_db_and_baseline_for_parity():
    try:
        run_us_microplex_performance_harness(
            providers=[_DummyProvider("cps")],
            config=USMicroplexPerformanceHarnessConfig(),
        )
    except ValueError as exc:
        assert "requires both targets_db and baseline_dataset" in str(exc)
    else:
        raise AssertionError("Expected ValueError when parity inputs are missing")


def test_run_us_microplex_performance_harness_can_skip_parity(monkeypatch):
    _patch_fake_harness(monkeypatch)

    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(evaluate_parity=False),
    )

    assert result.parity_run is None
    assert "evaluate_parity_harness" not in result.stage_timings


def test_run_us_microplex_performance_harness_can_enable_fast_calibration_targets(
    monkeypatch,
):
    _patch_fake_harness(monkeypatch)

    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            fast_inner_loop_calibration=True,
            target_variables=("adjusted_gross_income", "income_tax", "dividend_income"),
        ),
    )

    assert result.build_config.policyengine_target_variables == (
        "adjusted_gross_income",
        "income_tax",
        "dividend_income",
    )
    assert result.build_config.policyengine_calibration_target_variables == (
        "income_tax",
        "dividend_income",
    )


def test_run_us_microplex_performance_harness_can_keep_exact_calibration_targets(
    monkeypatch,
):
    _patch_fake_harness(monkeypatch)

    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            fast_inner_loop_calibration=False,
            target_variables=("adjusted_gross_income", "income_tax", "dividend_income"),
        ),
    )

    assert result.build_config.policyengine_calibration_target_variables == (
        "adjusted_gross_income",
        "income_tax",
        "dividend_income",
    )


def test_run_us_microplex_performance_harness_preserves_target_profiles(monkeypatch):
    _patch_fake_harness(monkeypatch)

    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            target_profile="pe_native_broad",
            calibration_target_profile="pe_native_broad",
        ),
    )

    assert result.build_config.policyengine_target_profile == "pe_native_broad"
    assert result.build_config.policyengine_calibration_target_profile == "pe_native_broad"
    assert result.build_config.policyengine_target_variables == ()
    assert result.build_config.policyengine_target_geo_levels == ()
    assert result.build_config.policyengine_calibration_target_variables == ()
    assert result.build_config.policyengine_calibration_target_geo_levels == ()


def test_run_us_microplex_performance_harness_can_evaluate_native_loss(monkeypatch):
    _patch_fake_harness(monkeypatch)
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.compute_us_pe_native_scores",
        lambda **kwargs: {
            "summary": {
                "candidate_enhanced_cps_native_loss": 0.2,
                "baseline_enhanced_cps_native_loss": 0.3,
                "enhanced_cps_native_loss_delta": -0.1,
            },
            "kwargs": kwargs,
        },
    )

    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            evaluate_pe_native_loss=True,
            baseline_dataset="/tmp/enhanced_cps.h5",
            policyengine_us_data_repo="/tmp/policyengine-us-data",
        ),
    )

    assert result.candidate_enhanced_cps_native_loss == 0.2
    assert result.baseline_enhanced_cps_native_loss == 0.3
    assert result.enhanced_cps_native_loss_delta == -0.1
    assert "evaluate_pe_native_loss" in result.stage_timings


def test_run_us_microplex_performance_harness_passes_export_direct_overrides(monkeypatch):
    stage_log: list[str] = []
    _patch_fake_harness(monkeypatch, stage_log=stage_log)
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.compute_us_pe_native_scores",
        lambda **kwargs: {"summary": {}, "kwargs": kwargs},
    )

    run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            evaluate_pe_native_loss=True,
            baseline_dataset="/tmp/enhanced_cps.h5",
            policyengine_us_data_repo="/tmp/policyengine-us-data",
            build_config=USMicroplexBuildConfig(
                policyengine_direct_override_variables=("filing_status", "snap")
            ),
        ),
    )

    assert "export_policyengine_dataset:('filing_status', 'snap')" in stage_log


def test_warm_us_microplex_parity_cache_preloads_baseline(monkeypatch):
    cache = PolicyEngineUSComparisonCache()
    load_target_set_calls: list[tuple[int, tuple[str, ...] | None]] = []
    baseline_calls: list[dict[str, object]] = []

    class FakeProvider:
        def load_target_set(self, query=None):
            period = query.period if query is not None else 2024
            names = tuple(query.names) if query is not None else ()
            load_target_set_calls.append((period, names or None))
            return TargetSet(
                [
                    TargetSpec(
                        name="policyengine_us_target_1",
                        entity=EntityType.HOUSEHOLD,
                        value=1.0,
                        period=period,
                        aggregation="count",
                    )
                ]
            )

    monkeypatch.setattr(
        "microplex_us.pipelines.performance.PolicyEngineUSDBTargetProvider",
        lambda path: FakeProvider(),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.default_policyengine_us_db_harness_slices",
        lambda **kwargs: (
            SimpleNamespace(
                name="all_targets",
                query=TargetQuery(period=kwargs["period"]),
            ),
        ),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.PolicyEngineUSComparisonCache.load_baseline_report",
        lambda self, **kwargs: baseline_calls.append(kwargs) or SimpleNamespace(),
    )

    warmed_cache = warm_us_microplex_parity_cache(
        config=USMicroplexPerformanceHarnessConfig(
            targets_db="/tmp/policy_data.db",
            baseline_dataset="/tmp/enhanced_cps.h5",
        ),
        comparison_cache=cache,
    )

    assert warmed_cache is cache
    assert load_target_set_calls == [(2024, None)]
    assert baseline_calls
    assert baseline_calls[0]["baseline_dataset"] == "/tmp/enhanced_cps.h5"


def test_warm_us_microplex_parity_cache_uses_resolved_scope_for_named_profile(monkeypatch):
    cache = PolicyEngineUSComparisonCache()
    slice_kwargs: dict[str, object] = {}
    baseline_calls: list[dict[str, object]] = []

    class FakeProvider:
        def load_target_set(self, query=None):
            period = query.period if query is not None else 2024
            return TargetSet(
                [
                    TargetSpec(
                        name="policyengine_us_target_1",
                        entity=EntityType.HOUSEHOLD,
                        value=1.0,
                        period=period,
                        aggregation="count",
                    )
                ]
            )

    monkeypatch.setattr(
        "microplex_us.pipelines.performance.PolicyEngineUSDBTargetProvider",
        lambda path: FakeProvider(),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.default_policyengine_us_db_harness_slices",
        lambda **kwargs: slice_kwargs.update(kwargs)
        or (
            SimpleNamespace(
                name="all_targets",
                query=TargetQuery(period=kwargs["period"]),
            ),
        ),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.PolicyEngineUSComparisonCache.load_baseline_report",
        lambda self, **kwargs: baseline_calls.append(kwargs) or SimpleNamespace(),
    )

    warm_us_microplex_parity_cache(
        config=USMicroplexPerformanceHarnessConfig(
            targets_db="/tmp/policy_data.db",
            baseline_dataset="/tmp/enhanced_cps.h5",
            target_profile="pe_native_broad",
            calibration_target_profile="pe_native_broad",
            build_config=USMicroplexBuildConfig(
                policyengine_target_profile="pe_native_broad",
                policyengine_calibration_target_profile="pe_native_broad",
            ),
        ),
        comparison_cache=cache,
    )

    assert slice_kwargs["variables"] == ()
    assert slice_kwargs["domain_variables"] == ()
    assert slice_kwargs["geo_levels"] == ()
    assert baseline_calls


def test_us_microplex_performance_session_reuses_comparison_cache(monkeypatch):
    session = USMicroplexPerformanceSession()
    run_calls: list[tuple[PolicyEngineUSComparisonCache, object, object, object]] = []

    def fake_run(
        providers,
        *,
        config,
        queries=None,
        comparison_cache=None,
        frame_cache=None,
        precalibration_cache=None,
        calibration_cache=None,
    ):
        _ = providers
        _ = config
        _ = queries
        run_calls.append(
            (
                comparison_cache,
                frame_cache,
                precalibration_cache,
                calibration_cache,
            )
        )
        return "ok"

    monkeypatch.setattr(
        "microplex_us.pipelines.performance.run_us_microplex_performance_harness",
        fake_run,
    )

    result = session.run(
        [_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(evaluate_parity=False),
    )

    assert result == "ok"
    assert run_calls == [
        (
            session.comparison_cache,
            session.frame_cache,
            session.precalibration_cache,
            session.calibration_cache,
        )
    ]


def test_us_microplex_performance_session_reuses_loaded_frames(monkeypatch):
    session = USMicroplexPerformanceSession()
    load_calls: list[str] = []

    class CountingProvider(_DummyProvider):
        def load_frame(self, query=None):
            _ = query
            load_calls.append(self.descriptor.name)
            return SimpleNamespace(source=SimpleNamespace(name=self.descriptor.name))
    _patch_fake_harness(monkeypatch)

    provider = CountingProvider("cps")
    config = USMicroplexPerformanceHarnessConfig(evaluate_parity=False)

    first = session.run([provider], config=config)
    second = session.run([provider], config=config)

    assert first.source_names == ("cps",)
    assert second.source_names == ("cps",)
    assert load_calls == ["cps"]


def test_us_microplex_performance_session_reuses_precalibration_state(monkeypatch):
    stage_calls: list[str] = []

    class CountingProvider(_DummyProvider):
        def load_frame(self, query=None):
            _ = query
            return SimpleNamespace(source=SimpleNamespace(name=self.descriptor.name))
    _patch_fake_harness(monkeypatch, stage_log=stage_calls)

    provider = CountingProvider("cps")
    config = USMicroplexPerformanceHarnessConfig(evaluate_parity=False)

    frame_cache = {}
    precalibration_cache = {}

    first = run_us_microplex_performance_harness(
        [provider],
        config=config,
        frame_cache=frame_cache,
        precalibration_cache=precalibration_cache,
        calibration_cache=None,
    )
    second = run_us_microplex_performance_harness(
        [provider],
        config=config,
        frame_cache=frame_cache,
        precalibration_cache=precalibration_cache,
        calibration_cache=None,
    )

    assert first.source_names == ("cps",)
    assert second.source_names == ("cps",)
    assert stage_calls.count("prepare_source_input") == 1
    assert stage_calls.count("prepare_seed_data") == 1
    assert stage_calls.count("integrate_donor_sources") == 1
    assert stage_calls.count("build_targets") == 1
    assert stage_calls.count("resolve_synthesis_variables") == 1
    assert stage_calls.count("synthesize") == 1
    assert stage_calls.count("ensure_target_support") == 1
    assert stage_calls.count("build_policyengine_tables") == 1
    assert stage_calls.count("calibrate_policyengine_tables") == 2


def test_us_microplex_performance_session_reuses_calibration_state(monkeypatch):
    session = USMicroplexPerformanceSession()
    stage_calls: list[str] = []

    class CountingProvider(_DummyProvider):
        def load_frame(self, query=None):
            _ = query
            return SimpleNamespace(source=SimpleNamespace(name=self.descriptor.name))
    _patch_fake_harness(monkeypatch, stage_log=stage_calls)

    provider = CountingProvider("cps")
    config = USMicroplexPerformanceHarnessConfig(evaluate_parity=False)

    first = session.run([provider], config=config)
    second = session.run([provider], config=config)

    assert first.source_names == ("cps",)
    assert second.source_names == ("cps",)
    assert stage_calls.count("prepare_source_input") == 1
    assert stage_calls.count("calibrate_policyengine_tables") == 1
    assert len(session.calibration_cache) == 1
    assert first.stage_timings["calibrate_policyengine_tables"] >= 0.0
    assert second.stage_timings["calibrate_policyengine_tables"] == 0.0
