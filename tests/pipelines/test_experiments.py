"""Tests for source-mix PE-US experiment runners."""

from __future__ import annotations

from pathlib import Path

from microplex_us.pipelines.artifacts import USMicroplexArtifactPaths
from microplex_us.pipelines.experiments import (
    USMicroplexExperimentReport,
    USMicroplexSourceExperimentSpec,
    default_us_source_mix_experiments,
    run_us_microplex_source_experiments,
)
from microplex_us.pipelines.performance import USMicroplexPerformanceHarnessConfig
from microplex_us.pipelines.registry import USMicroplexRunRegistryEntry


class _DummyProvider:
    def __init__(self, name: str):
        self.descriptor = type("Descriptor", (), {"name": name})()


def _artifact_paths(root: Path, name: str) -> USMicroplexArtifactPaths:
    output_dir = root / name
    return USMicroplexArtifactPaths(
        output_dir=output_dir,
        version_id=name,
        seed_data=output_dir / "seed.parquet",
        synthetic_data=output_dir / "synthetic.parquet",
        calibrated_data=output_dir / "calibrated.parquet",
        targets=output_dir / "targets.json",
        manifest=output_dir / "manifest.json",
        synthesizer=None,
        policyengine_dataset=output_dir / "policyengine.h5",
        policyengine_harness=output_dir / "policyengine_harness.json",
        run_registry=root / "run_registry.jsonl",
    )


def _entry(
    artifact_id: str,
    *,
    composite_loss: float,
    source_names: tuple[str, ...],
) -> USMicroplexRunRegistryEntry:
    return USMicroplexRunRegistryEntry(
        created_at="2026-03-25T12:00:00+00:00",
        artifact_id=artifact_id,
        artifact_dir=f"/tmp/{artifact_id}",
        manifest_path=f"/tmp/{artifact_id}/manifest.json",
        candidate_mean_abs_relative_error=composite_loss + 0.1,
        baseline_mean_abs_relative_error=0.5,
        mean_abs_relative_error_delta=composite_loss - 0.25,
        candidate_composite_parity_loss=composite_loss,
        baseline_composite_parity_loss=0.5,
        composite_parity_loss_delta=composite_loss - 0.5,
        source_names=source_names,
    )


def test_run_us_microplex_source_experiments_saves_report_and_sorts(monkeypatch, tmp_path):
    call_log: list[dict[str, object]] = []

    def fake_build_and_save(
        providers,
        output_root,
        *,
        config=None,
        queries=None,
        frontier_metric="candidate_composite_parity_loss",
        policyengine_comparison_cache=None,
        policyengine_target_provider=None,
        policyengine_baseline_dataset=None,
        policyengine_harness_slices=None,
        policyengine_harness_metadata=None,
        run_registry_path=None,
        run_registry_metadata=None,
    ):
        experiment_name = run_registry_metadata["experiment_name"]
        call_log.append(
            {
                "name": experiment_name,
                "frontier_metric": frontier_metric,
                "policyengine_comparison_cache": policyengine_comparison_cache,
                "run_registry_metadata": dict(run_registry_metadata),
                "policyengine_harness_metadata": dict(policyengine_harness_metadata),
            }
        )
        composite_loss = 0.35 if experiment_name == "cps+puf" else 0.45
        current_entry = _entry(
            experiment_name,
            composite_loss=composite_loss,
            source_names=tuple(provider.descriptor.name for provider in providers),
        )
        return type(
            "FakeArtifacts",
            (),
            {
                "artifact_paths": _artifact_paths(Path(output_root), experiment_name),
                "current_entry": current_entry,
                "frontier_entry": current_entry,
                "frontier_delta": 0.0,
            },
        )()

    monkeypatch.setattr(
        "microplex_us.pipelines.experiments.build_and_save_versioned_us_microplex_from_source_providers",
        fake_build_and_save,
    )

    report = run_us_microplex_source_experiments(
        [
            USMicroplexSourceExperimentSpec(
                name="cps-only",
                providers=(_DummyProvider("cps"),),
                metadata={"family": "baseline"},
            ),
            USMicroplexSourceExperimentSpec(
                name="cps+puf",
                providers=(_DummyProvider("cps"), _DummyProvider("puf")),
                metadata={"family": "tax"},
            ),
        ],
        tmp_path / "experiments",
        metadata={"suite": "parity-search"},
    )

    assert report.best_result is not None
    assert report.best_result.name == "cps+puf"
    assert [result.name for result in report.leaderboard] == ["cps+puf", "cps-only"]
    assert report.metadata["suite"] == "parity-search"
    assert len(call_log) == 2
    assert call_log[0]["frontier_metric"] == "candidate_composite_parity_loss"
    assert call_log[0]["policyengine_comparison_cache"] is call_log[1]["policyengine_comparison_cache"]
    assert call_log[0]["run_registry_metadata"]["experiment_name"] == "cps-only"
    assert call_log[1]["policyengine_harness_metadata"]["experiment_name"] == "cps+puf"

    report_path = tmp_path / "experiments" / "experiment_report.json"
    assert report_path.exists()
    loaded = USMicroplexExperimentReport.load(report_path)
    assert loaded.best_result is not None
    assert loaded.best_result.name == "cps+puf"
    assert loaded.leaderboard[0].current_entry is not None
    assert loaded.leaderboard[0].current_entry.candidate_composite_parity_loss == 0.35


def test_run_us_microplex_source_experiments_requires_at_least_one_experiment(tmp_path):
    try:
        run_us_microplex_source_experiments([], tmp_path / "experiments")
    except ValueError as exc:
        assert "at least one experiment" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty experiment batch")


def test_default_us_source_mix_experiments_builds_standard_ladder():
    cps_provider = _DummyProvider("cps")
    puf_provider = _DummyProvider("puf")
    psid_provider = _DummyProvider("psid")

    experiments = default_us_source_mix_experiments(
        cps_provider=cps_provider,
        puf_provider=puf_provider,
        psid_provider=psid_provider,
    )

    assert [experiment.name for experiment in experiments] == [
        "cps-only",
        "cps+puf",
        "cps+psid",
        "cps+puf+psid",
    ]
    assert experiments[0].metadata["sources"] == ["cps"]
    assert experiments[-1].metadata["sources"] == ["cps", "puf", "psid"]


def test_run_us_microplex_source_experiments_can_use_performance_session(
    monkeypatch,
    tmp_path,
):
    calls: dict[str, list[object]] = {
        "warm": [],
        "run": [],
        "save": [],
    }

    class FakeSession:
        def __init__(self):
            self.comparison_cache = object()

        def warm_parity_cache(self, *, config):
            calls["warm"].append(config)

        def run(self, providers, *, config, queries=None):
            calls["run"].append(
                {
                    "providers": tuple(provider.descriptor.name for provider in providers),
                    "config": config,
                    "queries": queries,
                }
            )
            return type(
                "FakePerformanceResult",
                (),
                {"build_result": f"build:{'+'.join(provider.descriptor.name for provider in providers)}"},
            )()

    def fake_save_build_result(
        build_result,
        output_root,
        *,
        frontier_metric="candidate_composite_parity_loss",
        policyengine_comparison_cache=None,
        policyengine_target_provider=None,
        policyengine_baseline_dataset=None,
        policyengine_harness_slices=None,
        policyengine_harness_metadata=None,
        run_registry_path=None,
        run_registry_metadata=None,
        version_id=None,
    ):
        experiment_name = run_registry_metadata["experiment_name"]
        calls["save"].append(
            {
                "build_result": build_result,
                "output_root": Path(output_root),
                "frontier_metric": frontier_metric,
                "policyengine_comparison_cache": policyengine_comparison_cache,
                "policyengine_harness_metadata": dict(policyengine_harness_metadata),
                "run_registry_metadata": dict(run_registry_metadata),
                "version_id": version_id,
            }
        )
        current_entry = _entry(
            experiment_name,
            composite_loss=0.3 if experiment_name == "cps+puf" else 0.4,
            source_names=tuple(run_registry_metadata["sources"])
            if "sources" in run_registry_metadata
            else (experiment_name,),
        )
        return type(
            "FakeArtifacts",
            (),
            {
                "artifact_paths": _artifact_paths(Path(output_root), experiment_name),
                "current_entry": current_entry,
                "frontier_entry": current_entry,
                "frontier_delta": 0.0,
            },
        )()

    monkeypatch.setattr(
        "microplex_us.pipelines.experiments.save_versioned_us_microplex_build_result",
        fake_save_build_result,
    )

    session = FakeSession()
    performance_config = USMicroplexPerformanceHarnessConfig(
        sample_n=25,
        n_synthetic=25,
        targets_db="/tmp/policy_data.db",
        baseline_dataset="/tmp/baseline.h5",
        evaluate_parity=True,
    )
    report = run_us_microplex_source_experiments(
        [
            USMicroplexSourceExperimentSpec(
                name="cps-only",
                providers=(_DummyProvider("cps"),),
                metadata={"sources": ["cps"]},
            ),
            USMicroplexSourceExperimentSpec(
                name="cps+puf",
                providers=(_DummyProvider("cps"), _DummyProvider("puf")),
                metadata={"sources": ["cps", "puf"]},
            ),
        ],
        tmp_path / "experiments",
        performance_harness_config=performance_config,
        performance_session=session,
    )

    assert report.best_result is not None
    assert report.best_result.name == "cps+puf"
    assert len(calls["warm"]) == 1
    assert len(calls["run"]) == 2
    assert len(calls["save"]) == 2
    assert calls["run"][0]["config"].evaluate_parity is False
    assert calls["save"][0]["policyengine_comparison_cache"] is session.comparison_cache
    assert calls["save"][1]["run_registry_metadata"]["experiment_name"] == "cps+puf"
    assert calls["save"][1]["policyengine_harness_metadata"]["experiment_name"] == "cps+puf"


def test_run_us_microplex_source_experiments_requires_performance_config_for_session(
    tmp_path,
):
    class FakeSession:
        comparison_cache = object()

    try:
        run_us_microplex_source_experiments(
            [
                USMicroplexSourceExperimentSpec(
                    name="cps-only",
                    providers=(_DummyProvider("cps"),),
                )
            ],
            tmp_path / "experiments",
            performance_session=FakeSession(),
        )
    except ValueError as exc:
        assert "performance_harness_config is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError when session is provided without config")
