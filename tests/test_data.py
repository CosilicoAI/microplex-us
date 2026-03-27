"""Tests for the US-specific CPS data helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from microplex_us.data import (
    create_sample_data,
    get_data_info,
    load_cps_asec,
    load_cps_for_synthesis,
)


def test_create_sample_data_returns_households_and_persons() -> None:
    households, persons = create_sample_data(n_households=64, seed=123)

    assert len(households) == 64
    assert households["household_id"].is_unique
    assert persons["person_id"].is_unique
    assert set(["household_id", "n_persons", "hh_weight"]).issubset(households.columns)
    assert set(["person_id", "household_id", "age", "income"]).issubset(persons.columns)


def test_load_cps_asec_reads_preprocessed_parquet(tmp_path: Path) -> None:
    households, persons = create_sample_data(n_households=32, seed=7)
    households.to_parquet(tmp_path / "cps_asec_households.parquet")
    persons.to_parquet(tmp_path / "cps_asec_persons.parquet")

    loaded_households, loaded_persons = load_cps_asec(data_dir=tmp_path)

    pd.testing.assert_frame_equal(loaded_households, households)
    pd.testing.assert_frame_equal(loaded_persons, persons)


def test_load_cps_for_synthesis_samples_households_consistently(tmp_path: Path) -> None:
    households, persons = create_sample_data(n_households=200, seed=11)
    households.to_parquet(tmp_path / "cps_asec_households.parquet")
    persons.to_parquet(tmp_path / "cps_asec_persons.parquet")

    sampled_households, sampled_persons = load_cps_for_synthesis(
        data_dir=tmp_path,
        sample_fraction=0.25,
        random_state=99,
    )

    assert 0 < len(sampled_households) < len(households)
    assert set(sampled_persons["household_id"]) <= set(sampled_households["household_id"])


def test_get_data_info_reports_missing_files(tmp_path: Path) -> None:
    info = get_data_info(data_dir=tmp_path)

    assert info["households"] == {"exists": False}
    assert info["persons"] == {"exists": False}


def test_load_cps_asec_raises_helpful_error_for_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="CPS ASEC data files not found"):
        load_cps_asec(data_dir=tmp_path)
