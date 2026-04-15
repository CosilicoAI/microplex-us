"""Tests for the canonical US version-bump benchmark CLI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from microplex_us.pipelines.version_benchmark import (
    _resolve_site_snapshot_path,
    main,
)


def test_resolve_site_snapshot_path_defaults_to_artifacts_root(tmp_path) -> None:
    output_root = tmp_path / "artifacts" / "live_run"
    resolved = _resolve_site_snapshot_path(
        output_root=output_root,
        site_snapshot_path=None,
    )
    assert resolved == (tmp_path / "artifacts" / "site_snapshot_us.json").resolve()


def test_main_writes_default_site_snapshot(monkeypatch, tmp_path) -> None:
    recorded_snapshot_paths: list[Path] = []
    recorded_build_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        "microplex_us.pipelines.version_benchmark.CPSASECParquetSourceProvider",
        lambda data_dir: ("cps", data_dir),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.version_benchmark.build_and_save_versioned_us_microplex_from_source_providers",
        lambda **_kwargs: (
            recorded_build_kwargs.update(_kwargs)
            or SimpleNamespace(
                artifact_paths=SimpleNamespace(
                    output_dir=tmp_path / "artifacts" / "live_run" / "run-1",
                    run_registry=tmp_path / "artifacts" / "live_run" / "run_registry.jsonl",
                ),
                current_entry=SimpleNamespace(
                    candidate_enhanced_cps_native_loss=0.2,
                    baseline_enhanced_cps_native_loss=0.3,
                    enhanced_cps_native_loss_delta=-0.1,
                ),
            )
        ),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.version_benchmark.write_us_microplex_site_snapshot",
        lambda artifact_dir, output_path: recorded_snapshot_paths.append(Path(output_path)),
    )

    main(
        [
            "--output-root",
            str(tmp_path / "artifacts" / "live_run"),
            "--cps-parquet-dir",
            str(tmp_path / "cps"),
            "--baseline-dataset",
            str(tmp_path / "baseline.h5"),
            "--targets-db",
            str(tmp_path / "targets.duckdb"),
        ]
    )

    assert recorded_snapshot_paths == [
        (tmp_path / "artifacts" / "site_snapshot_us.json").resolve()
    ]
    assert recorded_build_kwargs["frontier_metric"] == "enhanced_cps_native_loss_delta"
    assert recorded_build_kwargs["require_policyengine_native_score"] is True
    config = recorded_build_kwargs["config"]
    assert config.policyengine_target_profile == "pe_native_broad"
    assert config.policyengine_calibration_target_profile == "pe_native_broad"


def test_main_can_require_beating_pe_native_loss(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "microplex_us.pipelines.version_benchmark.CPSASECParquetSourceProvider",
        lambda data_dir: ("cps", data_dir),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.version_benchmark.build_and_save_versioned_us_microplex_from_source_providers",
        lambda **_kwargs: SimpleNamespace(
            artifact_paths=SimpleNamespace(
                output_dir=tmp_path / "artifacts" / "live_run" / "run-1",
                run_registry=tmp_path / "artifacts" / "live_run" / "run_registry.jsonl",
            ),
            current_entry=SimpleNamespace(
                candidate_enhanced_cps_native_loss=0.2,
                baseline_enhanced_cps_native_loss=0.1,
                enhanced_cps_native_loss_delta=0.1,
            ),
        ),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.version_benchmark.write_us_microplex_site_snapshot",
        lambda *_args, **_kwargs: None,
    )

    try:
        main(
            [
                "--output-root",
                str(tmp_path / "artifacts" / "live_run"),
                "--cps-parquet-dir",
                str(tmp_path / "cps"),
                "--baseline-dataset",
                str(tmp_path / "baseline.h5"),
                "--targets-db",
                str(tmp_path / "targets.duckdb"),
                "--require-beat-pe-native-loss",
            ]
        )
    except SystemExit as exc:
        assert str(exc) == (
            "US version-bump benchmark did not beat PE on PE-native enhanced-CPS "
            "loss: candidate=0.200000, baseline=0.100000, delta=0.100000"
        )
    else:
        raise AssertionError("expected SystemExit when native loss does not beat PE")
