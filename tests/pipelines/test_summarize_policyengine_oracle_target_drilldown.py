"""Tests for one-artifact calibration-oracle target drilldowns."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
from microplex.targets import TargetQuery

from microplex_us.pipelines.summarize_policyengine_oracle_target_drilldown import (
    summarize_us_policyengine_oracle_target_drilldown,
)
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    _policyengine_target_ledger_entry,
)
from microplex_us.policyengine import (
    PolicyEngineUSDBTargetProvider,
    PolicyEngineUSEntityTableBundle,
    build_policyengine_us_time_period_arrays,
    compute_policyengine_us_definition_hash,
    write_policyengine_us_time_period_dataset,
)


def test_summarize_us_policyengine_oracle_target_drilldown_filters_saved_artifact(
    tmp_path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    db_path = tmp_path / "policy_data.db"
    dataset_path = bundle_dir / "policyengine_us.h5"

    _create_policyengine_targets_db(db_path)
    _write_policyengine_dataset(dataset_path)

    provider = PolicyEngineUSDBTargetProvider(db_path)
    target = provider.load_target_set(
        TargetQuery(period=2024, provider_filters={"variables": ["household_count"]})
    ).targets[0]
    target_ledger = [
        _policyengine_target_ledger_entry(
            target=target,
            stage="solve_now",
            reason="selected_stage_1",
            household_count=2,
        )
    ]

    config = USMicroplexBuildConfig(
        policyengine_targets_db=str(db_path),
        policyengine_target_period=2024,
        policyengine_target_variables=("household_count",),
        policyengine_calibration_target_variables=("household_count",),
        calibration_backend="entropy",
        policyengine_dataset_year=2024,
    )
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "artifacts": {"policyengine_dataset": dataset_path.name},
                "calibration": {
                    "oracle_relative_error_cap": 10.0,
                    "materialized_variables": [],
                    "target_ledger": target_ledger,
                },
            }
        )
    )

    summary = summarize_us_policyengine_oracle_target_drilldown(bundle_dir, top_k=5)

    assert summary["summary"]["targetCount"] == 1
    assert summary["summary"]["supportedTargetCount"] == 1
    assert summary["summary"]["unsupportedTargetCount"] == 0
    assert summary["summary"]["stageCounts"] == {"solve_now": 1}
    assert summary["summary"]["largestFamiliesByCappedError"] == [
        {
            "group": "household_count|domain=household_count",
            "cappedErrorMass": 0.6,
            "count": 1,
            "meanCappedError": 0.6,
        }
    ]
    assert summary["summary"]["largestGeographiesByCappedError"] == [
        {
            "group": "state:CA",
            "cappedErrorMass": 0.6,
            "count": 1,
            "meanCappedError": 0.6,
        }
    ]
    assert summary["topRows"][0]["stage"] == "solve_now"
    assert summary["topRows"][0]["loss_family"] == "household_count|domain=household_count"
    assert summary["topRows"][0]["loss_geography"] == "state:CA"
    assert summary["topRows"][0]["actual_value"] == 2.0
    assert summary["topRows"][0]["target_value"] == 5.0
    assert summary["topRows"][0]["driver_variable"] == "household_count"
    assert summary["topRows"][0]["provenance_class"] == "stored_input"

    family_summary = summarize_us_policyengine_oracle_target_drilldown(
        bundle_dir,
        family="household_count|domain=household_count",
        geography="state:CA",
        stage="solve_now",
        top_k=5,
    )
    assert family_summary["summary"]["targetCount"] == 1
    assert family_summary["topRows"][0]["target_name"] == summary["topRows"][0]["target_name"]


def test_summarize_us_policyengine_oracle_target_drilldown_marks_rematerialized_formula(
    tmp_path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    db_path = tmp_path / "policy_data.db"
    dataset_path = bundle_dir / "policyengine_us.h5"

    _create_policyengine_targets_db(
        db_path,
        variable="snap",
        value=250.0,
        domain_variable="snap",
    )
    _write_policyengine_dataset(dataset_path, include_raw_snap=True)

    provider = PolicyEngineUSDBTargetProvider(db_path)
    target = provider.load_target_set(
        TargetQuery(period=2024, provider_filters={"variables": ["snap"]})
    ).targets[0]
    target_ledger = [
        _policyengine_target_ledger_entry(
            target=target,
            stage="solve_now",
            reason="selected_stage_1",
            household_count=2,
        )
    ]

    config = USMicroplexBuildConfig(
        policyengine_targets_db=str(db_path),
        policyengine_target_period=2024,
        policyengine_target_variables=("snap",),
        policyengine_calibration_target_variables=("snap",),
        calibration_backend="entropy",
        policyengine_dataset_year=2024,
    )
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "artifacts": {"policyengine_dataset": dataset_path.name},
                "calibration": {
                    "oracle_relative_error_cap": 10.0,
                    "materialized_variables": [],
                    "target_ledger": target_ledger,
                },
            }
        )
    )

    summary = summarize_us_policyengine_oracle_target_drilldown(bundle_dir, top_k=5)

    assert summary["topRows"][0]["driver_variable"] == "snap"
    assert summary["topRows"][0]["driver_is_materialized"] is True
    assert summary["topRows"][0]["provenance_class"] == "policyengine_materialized"


def _write_policyengine_dataset(path: Path, *, include_raw_snap: bool = False) -> None:
    household_data = {
        "household_id": [1, 2],
        "household_weight": [1.0, 1.0],
        "state_fips": [6, 6],
    }
    household_variable_map = {"state_fips": "state_fips"}
    if include_raw_snap:
        household_data["snap"] = [100.0, 0.0]
        household_variable_map["snap"] = "snap"
    tables = PolicyEngineUSEntityTableBundle(
        households=pd.DataFrame(household_data),
        persons=pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [35, 40],
            }
        ),
    )
    arrays = build_policyengine_us_time_period_arrays(
        tables,
        period=2024,
        household_variable_map=household_variable_map,
        person_variable_map={"age": "age"},
    )
    write_policyengine_us_time_period_dataset(arrays, path)


def _create_policyengine_targets_db(
    path: Path,
    *,
    variable: str = "household_count",
    value: float = 5.0,
    domain_variable: str = "household_count",
) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        f"""
        CREATE TABLE strata (
            stratum_id INTEGER PRIMARY KEY,
            definition_hash TEXT,
            parent_stratum_id INTEGER
        );

        CREATE TABLE stratum_constraints (
            stratum_id INTEGER NOT NULL,
            constraint_variable TEXT NOT NULL,
            operation TEXT NOT NULL,
            value TEXT NOT NULL
        );

        CREATE TABLE targets (
            target_id INTEGER PRIMARY KEY,
            variable TEXT NOT NULL,
            period INTEGER NOT NULL,
            stratum_id INTEGER NOT NULL,
            reform_id INTEGER NOT NULL DEFAULT 0,
            value REAL,
            active BOOLEAN NOT NULL DEFAULT 1,
            tolerance REAL,
            source TEXT,
            notes TEXT
        );

        CREATE VIEW target_overview AS
        SELECT
            t.target_id,
            t.stratum_id,
            t.variable,
            t.value,
            t.period,
            t.active,
            'state' AS geo_level,
            '06' AS geographic_id,
            '{domain_variable}' AS domain_variable
        FROM targets AS t;
        """
    )
    conn.execute(
        """
        INSERT INTO strata (stratum_id, definition_hash, parent_stratum_id)
        VALUES (?, ?, NULL)
        """,
        (1, compute_policyengine_us_definition_hash(())),
    )
    conn.execute(
        """
        INSERT INTO targets (
            target_id,
            variable,
            period,
            stratum_id,
            reform_id,
            value,
            active,
            tolerance,
            source,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (1, variable, 2024, 1, 0, value, 1, None, "test", variable),
    )
    conn.commit()
    conn.close()
