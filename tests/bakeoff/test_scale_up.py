"""Smoke tests for the synthesizer scale-up harness.

These tests exercise the harness on a deliberately tiny slice of real
enhanced_cps_2024. They do NOT constitute the scale-up benchmark itself;
that lives behind the CLI and takes significantly longer.

The goal here is: does the harness load data, fit a synthesizer, compute
metrics, and return a populated ScaleUpResult without crashing?
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microplex_us.bakeoff import (
    DEFAULT_CONDITION_COLS,
    DEFAULT_TARGET_COLS,
    ScaleUpRunner,
    ScaleUpStageConfig,
    stage1_config,
)

_ENHANCED_CPS_PATH = (
    Path.home()
    / "PolicyEngine/policyengine-us-data/policyengine_us_data/storage/enhanced_cps_2024.h5"
)

pytestmark = [
    pytest.mark.skipif(
        not _ENHANCED_CPS_PATH.exists(),
        reason="enhanced_cps_2024.h5 not available locally",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("prdc") is None,
        reason="prdc package not installed (uv pip install prdc)",
    ),
]


@pytest.fixture(scope="module")
def small_config() -> ScaleUpStageConfig:
    """Tiny config — a handful of columns, ~500 rows, one fast method."""
    base = stage1_config()
    return ScaleUpStageConfig(
        stage="smoke",
        n_rows=500,
        methods=("ZI-QRF",),
        condition_cols=("age", "is_female"),
        target_cols=(
            "employment_income_last_year",
            "self_employment_income_last_year",
            "snap_reported",
        ),
        holdout_frac=0.2,
        seed=0,
        k=5,
        n_generate=400,
        data_path=base.data_path,
        year=base.year,
        rare_cell_checks=(),  # skip rare-cell checks in smoke
    )


def test_load_frame_returns_expected_shape(small_config: ScaleUpStageConfig) -> None:
    runner = ScaleUpRunner(small_config)
    df = runner.load_frame()
    # n_rows is the upper bound after subsampling; if fewer in source, we get fewer.
    assert len(df) <= small_config.n_rows + 1
    assert len(df) > 100  # still a real sample
    expected_cols = set(small_config.condition_cols) | set(small_config.target_cols)
    assert expected_cols <= set(df.columns)


def test_split_train_holdout_shapes(small_config: ScaleUpStageConfig) -> None:
    runner = ScaleUpRunner(small_config)
    df = runner.load_frame()
    train, holdout = runner.split(df)
    assert len(train) + len(holdout) == len(df)
    # 20 % holdout within ±1
    expected_holdout = int(len(df) * 0.2)
    assert abs(len(holdout) - expected_holdout) <= 1


def test_fit_and_generate_returns_dataframe(
    small_config: ScaleUpStageConfig,
) -> None:
    runner = ScaleUpRunner(small_config)
    df = runner.load_frame()
    train, _ = runner.split(df)
    synthetic, timing = runner.fit_and_generate("ZI-QRF", train, n_generate=200)

    assert isinstance(synthetic, pd.DataFrame)
    assert len(synthetic) == 200
    assert timing["fit_wall_seconds"] >= 0
    assert timing["generate_wall_seconds"] >= 0
    assert timing["peak_rss_gb_during_fit"] > 0


def test_run_returns_populated_result(small_config: ScaleUpStageConfig) -> None:
    runner = ScaleUpRunner(small_config)
    results = runner.run()
    assert len(results) == 1
    r = results[0]
    assert r.method == "ZI-QRF"
    assert r.stage == "smoke"
    # PRDC values in [0, 1].
    for val in (r.precision, r.density, r.coverage):
        assert 0.0 <= val <= 1.0 + 1e-9
    # Zero-rate MAE in [0, 1].
    assert 0.0 <= r.zero_rate_mae <= 1.0
    assert r.n_train_rows > 0
    assert r.n_holdout_rows > 0
    assert r.n_cols == 5  # 2 cond + 3 target


def test_missing_column_raises_cleanly() -> None:
    cfg = ScaleUpStageConfig(
        stage="smoke",
        n_rows=100,
        methods=("ZI-QRF",),
        condition_cols=("age", "definitely_not_a_real_column"),
        target_cols=("employment_income_last_year",),
        data_path=_ENHANCED_CPS_PATH,
        rare_cell_checks=(),
    )
    runner = ScaleUpRunner(cfg)
    with pytest.raises(KeyError, match="definitely_not_a_real_column"):
        runner.load_frame()


def test_default_column_sets_are_sensible() -> None:
    """Sanity check on the curated default column list."""
    total = set(DEFAULT_CONDITION_COLS) | set(DEFAULT_TARGET_COLS)
    assert len(total) == len(DEFAULT_CONDITION_COLS) + len(DEFAULT_TARGET_COLS), (
        "Default conditioning and target columns overlap"
    )
    assert len(DEFAULT_CONDITION_COLS) >= 5
    assert len(DEFAULT_TARGET_COLS) >= 20
    assert len(total) <= 60, "Stage-1 default exceeds ~50-column budget"


def test_incremental_jsonl_persists_each_method(
    small_config: ScaleUpStageConfig, tmp_path: Path
) -> None:
    """Each completed method gets written as JSONL before the next starts."""
    import json as _json

    runner = ScaleUpRunner(small_config)
    incremental = tmp_path / "stage_incremental.jsonl"
    results = runner.run(incremental_path=incremental)

    assert incremental.exists()
    lines = [ln for ln in incremental.read_text().splitlines() if ln.strip()]
    assert len(lines) == len(results)
    # Round-trip: each line decodes to a ScaleUpResult-shaped dict.
    for line in lines:
        d = _json.loads(line)
        assert {"method", "stage", "coverage", "fit_wall_seconds"} <= set(d)
