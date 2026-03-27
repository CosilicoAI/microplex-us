"""Tests for pipeline artifact persistence."""

import json
import sqlite3
from pathlib import Path

import h5py
import pandas as pd
from microplex.core import EntityType
from microplex.targets import StaticTargetProvider, TargetQuery, TargetSet, TargetSpec

from microplex_us.pipelines.artifacts import save_us_microplex_artifacts
from microplex_us.pipelines.registry import load_us_microplex_run_registry
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexBuildResult,
    USMicroplexTargets,
)
from microplex_us.policyengine import (
    PolicyEngineUSEntityTableBundle,
    PolicyEngineUSHarnessSlice,
    build_policyengine_us_time_period_arrays,
    compute_policyengine_us_definition_hash,
    write_policyengine_us_time_period_dataset,
)


def _write_baseline_dataset(
    path: Path,
    tables: PolicyEngineUSEntityTableBundle,
) -> Path:
    arrays = build_policyengine_us_time_period_arrays(
        tables,
        period=2024,
        household_variable_map={"state_fips": "state_fips", "snap": "snap"},
        person_variable_map={"age": "age", "income": "employment_income"},
        tax_unit_variable_map={"filing_status": "filing_status"},
    )
    write_policyengine_us_time_period_dataset(arrays, path)
    return path


def _create_policyengine_targets_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
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
            CASE
                WHEN t.variable = 'snap' THEN 'state'
                ELSE 'district'
            END AS geo_level,
            CASE
                WHEN t.variable = 'snap' THEN '06'
                ELSE '0601'
            END AS geographic_id,
            CASE
                WHEN t.variable = 'snap' THEN 'snap'
                WHEN t.variable = 'household_count' THEN 'snap'
                ELSE NULL
            END AS domain_variable
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
    conn.executemany(
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
        [
            (1, "household_count", 2024, 1, 0, 3.0, 1, None, "test", "count"),
            (2, "snap", 2024, 1, 0, 250.0, 1, None, "test", "snap"),
        ],
    )
    conn.commit()
    conn.close()


class TestSaveUSMicroplexArtifacts:
    """Test saving pipeline artifacts."""

    def test_writes_expected_files(self, tmp_path):
        result = USMicroplexBuildResult(
            config=USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            ),
            seed_data=pd.DataFrame({"income": [10.0], "hh_weight": [1.0]}),
            synthetic_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [1.0, 1.0]}),
            calibrated_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [0.5, 1.5]}),
            targets=USMicroplexTargets(
                marginal={"state": {"CA": 2.0}},
                continuous={"income": 30.0},
            ),
            calibration_summary={"max_error": 0.01, "mean_error": 0.005},
            synthesis_metadata={"backend": "bootstrap"},
            synthesizer=None,
            policyengine_tables=PolicyEngineUSEntityTableBundle(
                households=pd.DataFrame(
                    {"household_id": [1, 2], "household_weight": [0.5, 1.5]}
                ),
                persons=pd.DataFrame(
                    {
                        "person_id": [10, 11],
                        "household_id": [1, 2],
                        "tax_unit_id": [101, 102],
                        "spm_unit_id": [201, 202],
                        "family_id": [301, 302],
                        "marital_unit_id": [401, 402],
                        "age": [35, 62],
                        "taxable_interest_income": [125.0, 250.0],
                    }
                ),
                tax_units=pd.DataFrame(
                    {
                        "tax_unit_id": [101, 102],
                        "household_id": [1, 2],
                        "filing_status": ["SINGLE", "JOINT"],
                    }
                ),
                spm_units=pd.DataFrame(
                    {"spm_unit_id": [201, 202], "household_id": [1, 2]}
                ),
                families=pd.DataFrame(
                    {"family_id": [301, 302], "household_id": [1, 2]}
                ),
                marital_units=pd.DataFrame(
                    {"marital_unit_id": [401, 402], "household_id": [1, 2]}
                ),
            ),
        )

        paths = save_us_microplex_artifacts(result, tmp_path)

        assert paths.output_dir == tmp_path
        assert paths.seed_data.exists()
        assert paths.synthetic_data.exists()
        assert paths.calibrated_data.exists()
        assert paths.targets.exists()
        assert paths.manifest.exists()
        assert paths.synthesizer is None
        assert paths.policyengine_dataset is not None
        assert paths.policyengine_dataset.exists()

        manifest = json.loads(paths.manifest.read_text())
        assert manifest["rows"]["synthetic"] == 2
        assert manifest["weights"]["nonzero"] == 2
        assert manifest["config"]["synthesis_backend"] == "bootstrap"
        assert manifest["artifacts"]["policyengine_dataset"] == "policyengine_us.h5"

        with h5py.File(paths.policyengine_dataset, "r") as handle:
            assert "household_id" in handle
            assert "person_household_id" in handle
            assert "tax_unit_id" in handle
            assert "taxable_interest_income" in handle
            assert "filing_status" in handle

    def test_writes_model_when_present(self, tmp_path):
        class FakeSynthesizer:
            def __init__(self):
                self.saved_path = None

            def save(self, path):
                self.saved_path = path
                path.write_text("model")

        fake = FakeSynthesizer()
        result = USMicroplexBuildResult(
            config=USMicroplexBuildConfig(
                n_synthetic=1,
                synthesis_backend="synthesizer",
                calibration_backend="entropy",
            ),
            seed_data=pd.DataFrame({"income": [10.0], "hh_weight": [1.0]}),
            synthetic_data=pd.DataFrame({"income": [10.0], "weight": [1.0]}),
            calibrated_data=pd.DataFrame({"income": [10.0], "weight": [1.0]}),
            targets=USMicroplexTargets(marginal={}, continuous={"income": 10.0}),
            calibration_summary={"max_error": 0.0, "mean_error": 0.0},
            synthesis_metadata={"backend": "synthesizer"},
            synthesizer=fake,
            policyengine_tables=None,
        )

        paths = save_us_microplex_artifacts(result, tmp_path)

        assert paths.synthesizer is not None
        assert paths.synthesizer.exists()
        assert fake.saved_path == paths.synthesizer

    def test_writes_policyengine_harness_when_baseline_and_targets_are_provided(
        self, tmp_path
    ):
        result = USMicroplexBuildResult(
            config=USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                policyengine_dataset_year=2024,
            ),
            seed_data=pd.DataFrame({"income": [10.0], "hh_weight": [1.0]}),
            synthetic_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [1.0, 1.0]}),
            calibrated_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [0.5, 1.5]}),
            targets=USMicroplexTargets(
                marginal={"state": {"CA": 2.0}},
                continuous={"income": 30.0},
            ),
            calibration_summary={"max_error": 0.01, "mean_error": 0.005},
            synthesis_metadata={"backend": "bootstrap"},
            synthesizer=None,
            policyengine_tables=PolicyEngineUSEntityTableBundle(
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
                        "tax_unit_id": [101, 101, 102],
                        "spm_unit_id": [201, 201, 202],
                        "family_id": [301, 301, 302],
                        "marital_unit_id": [401, 401, 402],
                        "age": [40.0, 10.0, 30.0],
                        "income": [30_000.0, 0.0, 20_000.0],
                    }
                ),
                tax_units=pd.DataFrame(
                    {
                        "tax_unit_id": [101, 102],
                        "household_id": [1, 2],
                        "filing_status": ["JOINT", "SINGLE"],
                    }
                ),
                spm_units=pd.DataFrame(
                    {"spm_unit_id": [201, 202], "household_id": [1, 2]}
                ),
                families=pd.DataFrame(
                    {"family_id": [301, 302], "household_id": [1, 2]}
                ),
                marital_units=pd.DataFrame(
                    {"marital_unit_id": [401, 402], "household_id": [1, 2]}
                ),
            ),
        )
        baseline_dataset = _write_baseline_dataset(
            tmp_path / "baseline.h5",
            PolicyEngineUSEntityTableBundle(
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
                        "tax_unit_id": [101, 101, 102],
                        "spm_unit_id": [201, 201, 202],
                        "family_id": [301, 301, 302],
                        "marital_unit_id": [401, 401, 402],
                        "age": [40.0, 10.0, 30.0],
                        "income": [30_000.0, 0.0, 20_000.0],
                    }
                ),
                tax_units=pd.DataFrame(
                    {
                        "tax_unit_id": [101, 102],
                        "household_id": [1, 2],
                        "filing_status": ["JOINT", "SINGLE"],
                    }
                ),
                spm_units=pd.DataFrame(
                    {"spm_unit_id": [201, 202], "household_id": [1, 2]}
                ),
                families=pd.DataFrame(
                    {"family_id": [301, 302], "household_id": [1, 2]}
                ),
                marital_units=pd.DataFrame(
                    {"marital_unit_id": [401, 402], "household_id": [1, 2]}
                ),
            ),
        )
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

        paths = save_us_microplex_artifacts(
            result,
            tmp_path / "bundle",
            policyengine_target_provider=provider,
            policyengine_baseline_dataset=baseline_dataset,
            policyengine_harness_slices=(
                PolicyEngineUSHarnessSlice(
                    name="snap",
                    description="SNAP parity",
                    query=TargetQuery(period=2024, names=("snap_total",)),
                ),
            ),
            policyengine_harness_metadata={"baseline_dataset": baseline_dataset.name},
        )

        assert paths.policyengine_harness is not None
        assert paths.policyengine_harness.exists()

        manifest = json.loads(paths.manifest.read_text())
        assert manifest["artifacts"]["policyengine_harness"] == "policyengine_harness.json"
        assert manifest["policyengine_harness"]["slice_win_rate"] == 1.0
        assert manifest["policyengine_harness"]["target_win_rate"] == 1.0
        assert manifest["policyengine_harness"]["candidate_composite_parity_loss"] is not None

        harness_payload = json.loads(paths.policyengine_harness.read_text())
        assert harness_payload["metadata"]["baseline_dataset"] == "baseline.h5"
        assert harness_payload["metadata"]["policyengine_us_runtime_version"] is not None
        assert harness_payload["summary"]["slice_win_rate"] == 1.0
        assert harness_payload["summary"]["candidate_composite_parity_loss"] is not None

    def test_writes_policyengine_harness_from_build_config_defaults(self, tmp_path):
        targets_db = tmp_path / "policyengine_targets.db"
        _create_policyengine_targets_db(targets_db)
        baseline_dataset = _write_baseline_dataset(
            tmp_path / "baseline.h5",
            PolicyEngineUSEntityTableBundle(
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
                        "tax_unit_id": [101, 101, 102],
                        "spm_unit_id": [201, 201, 202],
                        "family_id": [301, 301, 302],
                        "marital_unit_id": [401, 401, 402],
                        "age": [40.0, 10.0, 30.0],
                        "income": [30_000.0, 0.0, 20_000.0],
                    }
                ),
                tax_units=pd.DataFrame(
                    {
                        "tax_unit_id": [101, 102],
                        "household_id": [1, 2],
                        "filing_status": ["JOINT", "SINGLE"],
                    }
                ),
                spm_units=pd.DataFrame(
                    {"spm_unit_id": [201, 202], "household_id": [1, 2]}
                ),
                families=pd.DataFrame(
                    {"family_id": [301, 302], "household_id": [1, 2]}
                ),
                marital_units=pd.DataFrame(
                    {"marital_unit_id": [401, 402], "household_id": [1, 2]}
                ),
            ),
        )
        result = USMicroplexBuildResult(
            config=USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                policyengine_dataset_year=2024,
                policyengine_targets_db=str(targets_db),
                policyengine_baseline_dataset=str(baseline_dataset),
                policyengine_target_variables=("snap", "household_count"),
            ),
            seed_data=pd.DataFrame({"income": [10.0], "hh_weight": [1.0]}),
            synthetic_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [1.0, 1.0]}),
            calibrated_data=pd.DataFrame({"income": [10.0, 20.0], "weight": [0.5, 1.5]}),
            targets=USMicroplexTargets(
                marginal={"state": {"CA": 2.0}},
                continuous={"income": 30.0},
            ),
            calibration_summary={"max_error": 0.01, "mean_error": 0.005},
            synthesis_metadata={"backend": "bootstrap"},
            synthesizer=None,
            policyengine_tables=PolicyEngineUSEntityTableBundle(
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
                        "tax_unit_id": [101, 101, 102],
                        "spm_unit_id": [201, 201, 202],
                        "family_id": [301, 301, 302],
                        "marital_unit_id": [401, 401, 402],
                        "age": [40.0, 10.0, 30.0],
                        "income": [30_000.0, 0.0, 20_000.0],
                    }
                ),
                tax_units=pd.DataFrame(
                    {
                        "tax_unit_id": [101, 102],
                        "household_id": [1, 2],
                        "filing_status": ["JOINT", "SINGLE"],
                    }
                ),
                spm_units=pd.DataFrame(
                    {"spm_unit_id": [201, 202], "household_id": [1, 2]}
                ),
                families=pd.DataFrame(
                    {"family_id": [301, 302], "household_id": [1, 2]}
                ),
                marital_units=pd.DataFrame(
                    {"marital_unit_id": [401, 402], "household_id": [1, 2]}
                ),
            ),
        )

        paths = save_us_microplex_artifacts(result, tmp_path / "bundle")

        assert paths.policyengine_harness is not None
        assert paths.policyengine_harness.exists()
        assert paths.run_registry is not None
        assert paths.run_registry.exists()

        manifest = json.loads(paths.manifest.read_text())
        assert manifest["policyengine_harness"]["slice_win_rate"] == 1.0
        assert manifest["policyengine_harness"]["target_win_rate"] == 1.0
        assert manifest["policyengine_harness"]["candidate_composite_parity_loss"] is not None
        assert manifest["policyengine_harness"]["parity_scorecard"]["overall"][
            "candidate_beats_baseline"
        ] is True
        assert manifest["run_registry"]["artifact_id"] == "bundle"
        assert manifest["run_registry"]["improved_candidate_frontier"] is True
        assert manifest["run_registry"]["improved_composite_frontier"] is True
        assert (
            manifest["run_registry"]["default_frontier_metric"]
            == "candidate_composite_parity_loss"
        )

        harness_payload = json.loads(paths.policyengine_harness.read_text())
        assert harness_payload["metadata"]["baseline_dataset"] == "baseline.h5"
        assert harness_payload["metadata"]["targets_db"] == "policyengine_targets.db"
        assert harness_payload["metadata"]["harness_suite"] == "policyengine_us_all_targets"
        assert harness_payload["metadata"]["harness_slice_names"] == ["all_targets"]
        assert harness_payload["metadata"]["target_variables"] == [
            "snap",
            "household_count",
        ]
        assert harness_payload["metadata"]["policyengine_us_runtime_version"] is not None
        assert [slice_payload["name"] for slice_payload in harness_payload["slices"]] == [
            "all_targets",
        ]
        registry_entries = load_us_microplex_run_registry(paths.run_registry)
        assert len(registry_entries) == 1
        assert registry_entries[0].artifact_id == "bundle"
        assert registry_entries[0].policyengine_us_runtime_version is not None
        assert registry_entries[0].supported_target_rate == 1.0
        assert registry_entries[0].candidate_composite_parity_loss is not None
        assert registry_entries[0].tag_summaries["all_targets"]["target_win_rate"] == 1.0
