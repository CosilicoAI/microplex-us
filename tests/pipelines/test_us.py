"""Tests for the US microplex pipeline library."""


import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from microplex.core import (
    EntityObservation,
    EntityRelationship,
    EntityType,
    ObservationFrame,
    RelationshipCardinality,
    Shareability,
    SourceDescriptor,
    SourceQuery,
    SourceVariableCapability,
    StaticSourceProvider,
    TimeStructure,
)
from microplex.targets import TargetAggregation, TargetQuery, TargetSpec

from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexBuildResult,
    USMicroplexPipeline,
    USMicroplexTargets,
    _select_feasible_policyengine_calibration_constraints,
    _summarize_weight_diagnostics,
    build_us_microplex,
)
from microplex_us.policyengine.us import (
    PolicyEngineUSConstraint,
    build_policyengine_us_export_variable_maps,
    compute_policyengine_us_definition_hash,
)


def _create_policyengine_calibration_db(path) -> None:
    national_constraints: tuple[PolicyEngineUSConstraint, ...] = ()
    california_constraints = (
        PolicyEngineUSConstraint("state_fips", "==", "6"),
    )
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
        """
    )
    conn.executemany(
        """
        INSERT INTO strata (stratum_id, definition_hash, parent_stratum_id)
        VALUES (?, ?, ?)
        """,
        [
            (
                1,
                compute_policyengine_us_definition_hash(national_constraints),
                None,
            ),
            (
                2,
                compute_policyengine_us_definition_hash(
                    california_constraints,
                    parent_stratum_id=1,
                ),
                1,
            ),
        ],
    )
    conn.executemany(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            constraint_variable,
            operation,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        [(2, "state_fips", "==", "6")],
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
            (1, "household_count", 2024, 1, 0, 450.0, 1, None, "test", "national"),
            (2, "household_count", 2024, 2, 0, 225.0, 1, None, "test", "ca"),
        ],
    )
    conn.commit()
    conn.close()


def _create_policyengine_calibration_db_with_unsupported_target(path) -> None:
    unsupported_constraints = (
        PolicyEngineUSConstraint("age", ">=", "18"),
    )
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
        """
    )
    conn.executemany(
        """
        INSERT INTO strata (stratum_id, definition_hash, parent_stratum_id)
        VALUES (?, ?, ?)
        """,
        [
            (1, compute_policyengine_us_definition_hash(()), None),
            (2, compute_policyengine_us_definition_hash(unsupported_constraints), None),
        ],
    )
    conn.execute(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            constraint_variable,
            operation,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        (2, "age", ">=", "18"),
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
            (10, "household_count", 2024, 1, 0, 450.0, 1, 0.0, "test", "All households"),
            (
                11,
                "tax_unit_count",
                2024,
                2,
                0,
                2.0,
                1,
                0.0,
                "test",
                "Adult tax units",
            ),
        ],
    )
    conn.commit()
    conn.close()


class TestUSMicroplexBuildConfig:
    """Test pipeline configuration."""

    def test_defaults(self):
        config = USMicroplexBuildConfig()

        assert config.synthesis_backend == "synthesizer"
        assert config.calibration_backend == "entropy"
        assert config.n_synthetic == 100_000
        assert config.random_seed == 42

    def test_custom_values(self):
        config = USMicroplexBuildConfig(
            n_synthetic=250,
            synthesis_backend="seed",
            calibration_backend="ipf",
            synthesizer_epochs=12,
            policyengine_selection_household_budget=500,
        )

        assert config.n_synthetic == 250
        assert config.synthesis_backend == "seed"
        assert config.calibration_backend == "ipf"
        assert config.synthesizer_epochs == 12
        assert config.policyengine_selection_household_budget == 500


class TestUSMicroplexPipeline:
    """Test orchestration for US microplex builds."""

    @pytest.fixture
    def households(self):
        return pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "state_fips": [6, 36, 48],
                "county_fips": [6037, 36061, 48201],
                "hh_weight": [100.0, 150.0, 200.0],
                "tenure": [1, 2, 1],
            }
        )

    @pytest.fixture
    def persons(self):
        return pd.DataFrame(
            {
                "person_id": [10, 11, 12, 13, 14, 15],
                "household_id": [1, 1, 2, 2, 3, 3],
                "age": [34, 12, 47, 43, 68, 30],
                "sex": [1, 2, 2, 1, 1, 2],
                "education": [3, 1, 4, 4, 2, 4],
                "employment_status": [1, 0, 1, 1, 2, 1],
                "income": [55_000.0, 0.0, 72_000.0, 40_000.0, 18_000.0, 65_000.0],
            }
        )

    def test_prepare_seed_data(self, persons, households):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())

        seed = pipeline.prepare_seed_data(persons, households)

        assert len(seed) == len(persons)
        assert "state" in seed.columns
        assert "county_fips" in seed.columns
        assert "age_group" in seed.columns
        assert "income_bracket" in seed.columns
        assert set(seed["state"]) == {"CA", "NY", "TX"}
        assert set(seed["age_group"].astype(str)) == {"0-17", "18-34", "35-54", "65+"}

    def test_build_targets(self, persons, households):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        seed = pipeline.prepare_seed_data(persons, households)

        targets = pipeline.build_targets(seed)

        assert isinstance(targets, USMicroplexTargets)
        assert set(targets.marginal.keys()) == {"state", "age_group", "income_bracket"}
        assert targets.marginal["state"]["CA"] == 200.0
        assert targets.marginal["state"]["NY"] == 300.0
        assert targets.marginal["state"]["TX"] == 400.0
        expected_income = float((seed["hh_weight"] * seed["income"]).sum())
        assert targets.continuous["income"] == expected_income

    def test_build_with_bootstrap_backend(self, persons, households):
        config = USMicroplexBuildConfig(
            n_synthetic=12,
            synthesis_backend="bootstrap",
            calibration_backend="entropy",
            random_seed=7,
        )
        result = build_us_microplex(persons, households, config)

        assert len(result.seed_data) == len(persons)
        assert result.synthetic_data["household_id"].nunique() == 12
        assert result.calibrated_data["household_id"].nunique() == 12
        assert len(result.synthetic_data) > 12
        assert len(result.calibrated_data) > 12
        assert "weight" in result.calibrated_data.columns
        assert result.calibration_summary["max_error"] < 0.05
        assert result.synthesizer is None
        assert result.policyengine_tables is not None
        assert result.source_frame is not None
        assert result.fusion_plan is not None
        assert len(result.policyengine_tables.households) == 12
        assert len(result.policyengine_tables.persons) == len(result.calibrated_data)
        assert len(result.policyengine_tables.tax_units) > 0
        assert len(result.policyengine_tables.spm_units) > 0
        assert len(result.policyengine_tables.families) > 0
        assert len(result.policyengine_tables.marital_units) > 0
        assert result.synthesis_metadata["bootstrap_strata_columns"] == []

    def test_bootstrap_infers_state_strata_from_target_scope(self, persons, households):
        config = USMicroplexBuildConfig(
            synthesis_backend="bootstrap",
            policyengine_calibration_target_geo_levels=("state",),
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households)

        assert pipeline._resolve_bootstrap_strata_columns(seed) == ("state_fips",)

    def test_bootstrap_preserves_state_support_when_state_targets_are_requested(self):
        seed = pd.DataFrame(
            {
                "household_id": [0, 1],
                "person_id": [0, 1],
                "hh_weight": [1000.0, 1.0],
                "state_fips": [6, 36],
                "age": [40, 41],
                "sex": [1, 2],
                "education": [3, 3],
                "employment_status": [1, 1],
                "tenure": [1, 1],
                "income": [50_000.0, 60_000.0],
            }
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="bootstrap",
                random_seed=1,
                policyengine_calibration_target_geo_levels=("state",),
            )
        )

        synthetic = pipeline._synthesize_bootstrap(
            seed,
            initial_weight=1.0,
            strata_columns=pipeline._resolve_bootstrap_strata_columns(seed),
        )

        assert synthetic["state_fips"].nunique() == 2

    def test_bootstrap_explicit_missing_strata_column_raises(self, persons, households):
        config = USMicroplexBuildConfig(
            synthesis_backend="bootstrap",
            bootstrap_strata_columns=("county_fips",),
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households)

        with pytest.raises(ValueError, match="bootstrap_strata_columns"):
            pipeline._resolve_bootstrap_strata_columns(seed)

    def test_build_with_synthesizer_backend(self, persons, households):
        config = USMicroplexBuildConfig(
            n_synthetic=10,
            synthesis_backend="synthesizer",
            calibration_backend="entropy",
            synthesizer_epochs=5,
            synthesizer_n_layers=2,
            synthesizer_hidden_dim=16,
            random_seed=11,
        )
        result = build_us_microplex(persons, households, config)

        assert len(result.synthetic_data) == 10
        assert result.synthesizer is not None
        assert result.synthesis_metadata["backend"] == "synthesizer"
        assert result.synthesis_metadata["condition_vars"] == [
            "age",
            "sex",
            "education",
            "employment_status",
            "state_fips",
            "tenure",
        ]
        assert result.synthesis_metadata["target_vars"] == ["income"]
        assert result.fusion_plan is not None
        assert set(result.fusion_plan.output_entities) == {
            EntityType.HOUSEHOLD,
            EntityType.PERSON,
        }
        assert (result.synthetic_data["income"] >= 0).all()
        assert result.policyengine_tables is not None

    def test_build_policyengine_entity_tables(self, persons, households):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        seed = pipeline.prepare_seed_data(persons, households)
        synthetic = pipeline._finalize_synthetic_population(seed, initial_weight=1.0)

        tables = pipeline.build_policyengine_entity_tables(synthetic)

        assert set(tables.households.columns) >= {"household_id", "household_weight"}
        assert set(tables.persons.columns) >= {
            "person_id",
            "household_id",
            "tax_unit_id",
            "spm_unit_id",
            "family_id",
            "marital_unit_id",
        }
        assert set(tables.tax_units.columns) >= {"tax_unit_id", "household_id"}
        assert set(tables.spm_units.columns) >= {"spm_unit_id", "household_id"}
        assert set(tables.families.columns) >= {"family_id", "household_id"}
        assert set(tables.marital_units.columns) >= {"marital_unit_id", "household_id"}
        assert tables.persons["tax_unit_id"].notna().all()
        assert tables.persons["spm_unit_id"].notna().all()
        assert tables.persons["family_id"].notna().all()
        assert tables.persons["marital_unit_id"].notna().all()
        assert set(tables.tax_units["filing_status"]).issubset(
            {"SINGLE", "JOINT", "SEPARATE", "HEAD_OF_HOUSEHOLD", "SURVIVING_SPOUSE"}
        )

    def test_build_policyengine_entity_tables_derives_tax_input_columns(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2],
                "household_id": [10, 10],
                "weight": [1.0, 1.0],
                "age": [45, 43],
                "sex": [1, 2],
                "race": [4, 2],
                "hispanic": [2, 1],
                "income": [60_000.0, 15_000.0],
                "wage_income": [50_000.0, 10_000.0],
                "self_employment_income": [5_000.0, 0.0],
                "taxable_interest_income": [100.0, 20.0],
                "ordinary_dividend_income": [80.0, 30.0],
                "qualified_dividend_income": [30.0, 5.0],
                "short_term_capital_gains": [10.0, 0.0],
                "long_term_capital_gains": [40.0, 5.0],
                "rental_income": [200.0, 0.0],
                "gross_social_security": [0.0, 800.0],
                "ssi": [0.0, 600.0],
                "taxable_pension_income": [0.0, 300.0],
                "unemployment_compensation": [0.0, 150.0],
                "medicaid": [0.0, 1_250.0],
                "medicaid_enrolled": [False, True],
                "state_income_tax_paid": [400.0, 50.0],
                "filing_status": ["JOINT", "JOINT"],
                "relationship_to_head": [0, 1],
                "state_fips": [6, 6],
                "tenure": [1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)

        assert person_rows["employment_income_before_lsr"].tolist() == [50_000.0, 10_000.0]
        assert person_rows["self_employment_income_before_lsr"].tolist() == [5_000.0, 0.0]
        assert person_rows["taxable_interest_income"].tolist() == [100.0, 20.0]
        assert person_rows["dividend_income"].tolist() == [80.0, 30.0]
        assert person_rows["qualified_dividend_income"].tolist() == [30.0, 5.0]
        assert person_rows["non_qualified_dividend_income"].tolist() == [50.0, 25.0]
        assert person_rows["short_term_capital_gains"].tolist() == [10.0, 0.0]
        assert person_rows["long_term_capital_gains_before_response"].tolist() == [40.0, 5.0]
        assert person_rows["social_security_retirement"].tolist() == [0.0, 800.0]
        assert person_rows["ssi"].tolist() == [0.0, 600.0]
        assert person_rows["taxable_private_pension_income"].tolist() == [0.0, 300.0]
        assert person_rows["unemployment_compensation"].tolist() == [0.0, 150.0]
        assert person_rows["is_female"].tolist() == [False, True]
        assert person_rows["cps_race"].tolist() == [4, 2]
        assert person_rows["is_hispanic"].tolist() == [False, True]
        assert person_rows["medicaid"].tolist() == [0.0, 1_250.0]
        assert person_rows["medicaid_enrolled"].tolist() == [False, True]
        assert person_rows["state_income_tax_reported"].tolist() == [400.0, 50.0]

    def test_build_policyengine_entity_tables_fallback_employment_excludes_transfer_income(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1],
                "household_id": [10],
                "weight": [1.0],
                "age": [62],
                "sex": [2],
                "income": [18_000.0],
                "ssi": [9_000.0],
                "public_assistance": [3_000.0],
                "gross_social_security": [2_000.0],
                "filing_status": ["SINGLE"],
                "relationship_to_head": [0],
                "state_fips": [6],
                "tenure": [1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_row = tables.persons.iloc[0]

        assert person_row["employment_income_before_lsr"] == 4_000.0
        assert person_row["ssi"] == 9_000.0
        assert person_row["social_security_retirement"] == 2_000.0

    def test_build_policyengine_entity_tables_derives_dividend_totals_from_atomic_components(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1],
                "household_id": [10],
                "weight": [1.0],
                "age": [45],
                "income": [60_000.0],
                "wage_income": [50_000.0],
                "ordinary_dividend_income": [50.0],
                "dividend_income": [0.0],
                "qualified_dividend_income": [30.0],
                "non_qualified_dividend_income": [12.0],
                "filing_status": ["SINGLE"],
                "relationship_to_head": [0],
                "state_fips": [6],
                "tenure": [1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_row = tables.persons.iloc[0]

        assert person_row["qualified_dividend_income"] == 30.0
        assert person_row["non_qualified_dividend_income"] == 12.0
        assert person_row["ordinary_dividend_income"] == 42.0
        assert person_row["dividend_income"] == 42.0

    def test_build_policyengine_entity_tables_derives_relationships_from_family_relationship(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 10],
                "weight": [1.0, 1.0, 1.0],
                "age": [45, 43, 12],
                "income": [60_000.0, 15_000.0, 0.0],
                "family_relationship": [0, 1, 2],
                "marital_status": [1, 1, 7],
                "state_fips": [6, 6, 6],
                "tenure": [1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert person_rows["relationship_to_head"].tolist() == [0, 1, 2]
        assert len(tax_units) == 1
        assert tax_units.iloc[0]["filing_status"] == "JOINT"
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_derives_relationships_from_one_based_family_relationship(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 10],
                "weight": [1.0, 1.0, 1.0],
                "age": [45, 43, 12],
                "income": [60_000.0, 15_000.0, 0.0],
                "family_relationship": [1, 2, 3],
                "marital_status": [1, 1, 7],
                "state_fips": [6, 6, 6],
                "tenure": [1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert person_rows["relationship_to_head"].tolist() == [0, 1, 2]
        assert len(tax_units) == 1
        assert tax_units.iloc[0]["filing_status"] == "JOINT"
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_uses_spouse_and_dependent_flags_when_relationship_missing(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 10],
                "weight": [1.0, 1.0, 1.0],
                "age": [45, 43, 12],
                "income": [60_000.0, 15_000.0, 0.0],
                "is_spouse": [0, 1, 0],
                "is_dependent": [0, 0, 1],
                "state_fips": [6, 6, 6],
                "tenure": [1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert person_rows["relationship_to_head"].tolist() == [0, 1, 2]
        assert len(tax_units) == 1
        assert tax_units.iloc[0]["filing_status"] == "JOINT"
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_prefers_richer_family_relationship_over_collapsed_relationship_to_head(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 10],
                "weight": [1.0, 1.0, 1.0],
                "age": [45, 43, 12],
                "income": [60_000.0, 15_000.0, 0.0],
                "family_relationship": [0, 1, 2],
                "relationship_to_head": [0, 3, 3],
                "marital_status": [1, 1, 7],
                "state_fips": [6, 6, 6],
                "tenure": [1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert person_rows["relationship_to_head"].tolist() == [0, 1, 2]
        assert len(tax_units) == 1
        assert tax_units.iloc[0]["filing_status"] == "JOINT"
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_policyengine_entity_tables_repairs_households_without_a_head(self):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())
        population = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "household_id": [10, 10, 10],
                "weight": [1.0, 1.0, 1.0],
                "age": [45, 43, 12],
                "income": [60_000.0, 15_000.0, 0.0],
                "relationship_to_head": [1, 1, 2],
                "marital_status": [1, 1, 7],
                "state_fips": [6, 6, 6],
                "tenure": [1, 1, 1],
            }
        )

        tables = pipeline.build_policyengine_entity_tables(population)
        person_rows = tables.persons.sort_values("person_id").reset_index(drop=True)
        tax_units = tables.tax_units.sort_values("tax_unit_id").reset_index(drop=True)

        assert person_rows["relationship_to_head"].tolist() == [0, 1, 2]
        assert len(tax_units) == 1
        assert tax_units.iloc[0]["filing_status"] == "JOINT"
        assert tax_units.iloc[0]["n_dependents"] == 1

    def test_build_from_source_providers_accepts_year_specific_query_keys(self):
        households = pd.DataFrame(
            {
                "household_id": ["1"],
                "state_fips": [6],
                "household_weight": [1.0],
                "year": [2024],
            }
        )
        persons = pd.DataFrame(
            {
                "person_id": ["1:1"],
                "household_id": ["1"],
                "age": [40],
                "sex": [1],
                "education": [3],
                "employment_status": [1],
                "income": [50_000.0],
                "weight": [1.0],
                "year": [2024],
            }
        )

        descriptor = SourceDescriptor(
            name="toy_source",
            shareability=Shareability.PUBLIC,
            time_structure=TimeStructure.REPEATED_CROSS_SECTION,
            observations=(
                EntityObservation(
                    entity=EntityType.HOUSEHOLD,
                    key_column="household_id",
                    variable_names=("state_fips",),
                    weight_column="household_weight",
                    period_column="year",
                ),
                EntityObservation(
                    entity=EntityType.PERSON,
                    key_column="person_id",
                    variable_names=("age", "sex", "education", "employment_status", "income"),
                    weight_column="weight",
                    period_column="year",
                ),
            ),
        )
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="toy_source_2024",
                shareability=descriptor.shareability,
                time_structure=descriptor.time_structure,
                observations=descriptor.observations,
            ),
            tables={
                EntityType.HOUSEHOLD: households,
                EntityType.PERSON: persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )

        class YearNamedProvider:
            year = 2024
            _descriptor_cache = None

            @property
            def descriptor(self):
                return self._descriptor_cache or descriptor

            def load_frame(self, query=None):
                self.last_query = query
                self._descriptor_cache = frame.source
                return frame

        provider = YearNamedProvider()
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=1,
                synthesis_backend="bootstrap",
            )
        )

        result = pipeline.build_from_source_providers(
            [provider],
            queries={
                "toy_source_2024": SourceQuery(
                    provider_filters={"sample_n": 1, "random_seed": 7}
                )
            },
        )

        assert provider.last_query is not None
        assert provider.last_query.provider_filters["sample_n"] == 1
        assert result.source_frame is not None
        assert result.source_frame.source.name == "toy_source_2024"

    def test_integrate_donor_sources_models_dividends_compositionally(
        self,
        monkeypatch,
    ):
        captured: dict[str, object] = {}

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 19],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [60_000.0, 12_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [80.0, 90.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002],
                "household_id": [101, 102],
                "age": [44, 21],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [58_000.0, 13_000.0],
                "qualified_dividend_income": [20.0, 7.0],
                "non_qualified_dividend_income": [8.0, 3.0],
                "ordinary_dividend_income": [28.0, 10.0],
                "dividend_income": [500.0, 200.0],
            }
        )

        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "qualified_dividend_income",
                            "non_qualified_dividend_income",
                            "ordinary_dividend_income",
                            "dividend_income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )

        class FakeSynthesizer:
            def __init__(self, *args, **kwargs):
                _ = args
                captured["init_kwargs"] = dict(kwargs)
                self.target_vars = kwargs.get("target_vars", [])

            def fit(self, *args, **kwargs):
                _ = args
                captured["fit_kwargs"] = dict(kwargs)

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                if "dividend_income" in self.target_vars:
                    result["dividend_income"] = [28.0, 10.0]
                if "qualified_dividend_share" in self.target_vars:
                    result["qualified_dividend_share"] = [20.0 / 28.0, 0.7]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                donor_imputer_epochs=7,
                donor_imputer_batch_size=33,
                donor_imputer_learning_rate=5e-4,
                donor_imputer_n_layers=3,
                donor_imputer_hidden_dim=48,
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert integration["integrated_variables"] == [
            "non_qualified_dividend_income",
            "qualified_dividend_income",
        ]
        assert integration["seed_data"]["qualified_dividend_income"].round(6).tolist() == [
            20.0,
            7.0,
        ]
        assert integration["seed_data"]["non_qualified_dividend_income"].round(6).tolist() == [
            8.0,
            3.0,
        ]
        assert integration["seed_data"]["ordinary_dividend_income"].round(6).tolist() == [
            28.0,
            10.0,
        ]
        assert integration["seed_data"]["dividend_income"].round(6).tolist() == [
            28.0,
            10.0,
        ]
        assert "qualified_dividend_share" not in integration["seed_data"].columns
        assert captured["init_kwargs"]["n_layers"] == 3
        assert captured["init_kwargs"]["hidden_dim"] == 48
        assert captured["fit_kwargs"]["epochs"] == 7
        assert captured["fit_kwargs"]["batch_size"] == 33
        assert captured["fit_kwargs"]["learning_rate"] == 5e-4

    def test_integrate_donor_sources_models_unrelated_tax_variables_in_separate_blocks(
        self,
        monkeypatch,
    ):
        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 19],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [60_000.0, 12_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [80.0, 90.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002],
                "household_id": [101, 102],
                "age": [44, 21],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [58_000.0, 13_000.0],
                "qualified_dividend_income": [20.0, 7.0],
                "non_qualified_dividend_income": [8.0, 3.0],
                "partnership_s_corp_income": [1_000.0, 200.0],
            }
        )

        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "qualified_dividend_income",
                            "non_qualified_dividend_income",
                            "partnership_s_corp_income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )

        target_var_calls: list[tuple[str, ...]] = []

        class FakeSynthesizer:
            def __init__(self, *args, **kwargs):
                _ = args
                self.target_vars = tuple(kwargs.get("target_vars", []))
                target_var_calls.append(self.target_vars)

            def fit(self, *args, **kwargs):
                _ = args
                _ = kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                if self.target_vars == ("dividend_income", "qualified_dividend_share"):
                    result["dividend_income"] = [28.0, 10.0]
                    result["qualified_dividend_share"] = [20.0 / 28.0, 0.7]
                if self.target_vars == ("partnership_s_corp_income",):
                    result["partnership_s_corp_income"] = [1_000.0, 200.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert target_var_calls == [
            ("dividend_income", "qualified_dividend_share"),
            ("partnership_s_corp_income",),
        ]
        assert integration["seed_data"]["qualified_dividend_income"].round(6).tolist() == [
            20.0,
            7.0,
        ]
        assert integration["seed_data"]["non_qualified_dividend_income"].round(6).tolist() == [
            8.0,
            3.0,
        ]
        assert integration["seed_data"]["partnership_s_corp_income"].round(6).tolist() == [
            1_000.0,
            200.0,
        ]

    def test_integrate_donor_sources_can_use_zi_qrf_backend(self, monkeypatch):
        captured: dict[str, object] = {}

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 19],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [60_000.0, 12_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [80.0, 90.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002],
                "household_id": [101, 102],
                "age": [44, 21],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [58_000.0, 13_000.0],
                "public_assistance": [200.0, 0.0],
            }
        )

        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="benefit_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "public_assistance",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )

        class FakeQRFImputer:
            def __init__(self, **kwargs):
                captured["init_kwargs"] = kwargs

            def fit(self, frame, **kwargs):
                captured["fit_columns"] = list(frame.columns)
                captured["fit_kwargs"] = kwargs
                return self

            def generate(self, frame, seed=None):
                _ = seed
                return frame.assign(public_assistance=[190.0, 10.0])

        monkeypatch.setattr(
            "microplex_us.pipelines.us.ColumnwiseQRFDonorImputer",
            FakeQRFImputer,
        )

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                donor_imputer_backend="zi_qrf",
                donor_imputer_qrf_n_estimators=77,
                donor_imputer_qrf_zero_threshold=0.1,
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert integration["integrated_variables"] == ["public_assistance"]
        assert captured["init_kwargs"]["n_estimators"] == 77
        assert captured["init_kwargs"]["zero_threshold"] == 0.1
        assert captured["init_kwargs"]["zero_inflated_vars"] == {"public_assistance"}
        assert captured["init_kwargs"]["nonnegative_vars"] == {"public_assistance"}
        assert "weight" in captured["fit_columns"]
        assert captured["fit_kwargs"]["weight_col"] == "weight"
        assert set(integration["seed_data"]["public_assistance"].tolist()) <= {0.0, 200.0}

    def test_integrate_donor_sources_preserves_informative_scaffold_values(self, monkeypatch):
        cps_households = pd.DataFrame(
            {
                "household_id": [1],
                "hh_weight": [100.0],
                "state_fips": [6],
                "tenure": [1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10],
                "household_id": [1],
                "age": [45],
                "sex": [1],
                "education": [3],
                "employment_status": [1],
                "income": [60_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101],
                "hh_weight": [80.0],
                "state_fips": [6],
                "tenure": [1],
                "household_weight": [999.0],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001],
                "household_id": [101],
                "age": [44],
                "sex": [1],
                "education": [3],
                "employment_status": [0],
                "income": [5.0],
                "qualified_dividend_income": [20.0],
                "non_qualified_dividend_income": [8.0],
                "tax_unit_id": [12345],
            }
        )

        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure", "household_weight"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "qualified_dividend_income",
                            "non_qualified_dividend_income",
                            "tax_unit_id",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )

        class FakeSynthesizer:
            def __init__(self, *args, **kwargs):
                _ = args, kwargs
                self.target_vars = tuple(kwargs.get("target_vars", []))

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                if self.target_vars == ("dividend_income", "qualified_dividend_share"):
                    result["dividend_income"] = [28.0]
                    result["qualified_dividend_share"] = [20.0 / 28.0]
                if self.target_vars == ("income",):
                    result["income"] = [5.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(n_synthetic=1, synthesis_backend="bootstrap")
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)
        seed_data["income"] = [60_000.0]

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert "household_weight" not in integration["integrated_variables"]
        assert "tax_unit_id" not in integration["integrated_variables"]
        assert "income" not in integration["integrated_variables"]
        assert integration["seed_data"]["income"].tolist() == [60_000.0]

    def test_export_policyengine_dataset(self, persons, households, tmp_path):
        config = USMicroplexBuildConfig(
            n_synthetic=8,
            synthesis_backend="bootstrap",
            calibration_backend="entropy",
            policyengine_dataset_year=2024,
        )
        result = build_us_microplex(persons, households, config)
        pipeline = USMicroplexPipeline(config)

        output_path = pipeline.export_policyengine_dataset(result, tmp_path / "us_microplex.h5")

        assert output_path.exists()

    def test_export_policyengine_dataset_passes_direct_overrides(
        self,
        persons,
        households,
        tmp_path,
        monkeypatch,
    ):
        captured: list[tuple[str, ...]] = []

        original_build_maps = build_policyengine_us_export_variable_maps

        def _capture_build_maps(*args, **kwargs):
            captured.append(tuple(kwargs.get("direct_override_variables", ())))
            return original_build_maps(*args, **kwargs)

        monkeypatch.setattr(
            "microplex_us.pipelines.us.build_policyengine_us_export_variable_maps",
            _capture_build_maps,
        )

        config = USMicroplexBuildConfig(
            n_synthetic=8,
            synthesis_backend="bootstrap",
            calibration_backend="entropy",
            policyengine_dataset_year=2024,
            policyengine_direct_override_variables=("filing_status",),
        )
        result = build_us_microplex(persons, households, config)
        pipeline = USMicroplexPipeline(config)

        output_path = pipeline.export_policyengine_dataset(result, tmp_path / "us_microplex.h5")

        assert output_path.exists()
        assert captured == [("filing_status",)]

    def test_calibrate_policyengine_tables_from_db(self, persons, households, tmp_path):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, calibrated_persons, summary = (
            pipeline.calibrate_policyengine_tables(tables)
        )

        household_weights = calibrated_tables.households.set_index("household_id")[
            "household_weight"
        ]
        california_weight = calibrated_tables.households.loc[
            calibrated_tables.households["state_fips"] == 6,
            "household_weight",
        ].sum()

        assert summary["backend"] == "policyengine_db_entropy"
        assert summary["n_constraints"] == 2
        assert summary["max_error"] < 1e-6
        assert summary["weight_collapse_suspected"] is False
        assert summary["household_weight_diagnostics"]["total_weight"] == pytest.approx(
            450.0,
            rel=1e-6,
        )
        assert summary["household_weight_diagnostics"]["positive_count"] == summary[
            "household_weight_diagnostics"
        ]["row_count"]
        assert household_weights.sum() == pytest.approx(450.0, rel=1e-6)
        assert california_weight == pytest.approx(225.0, rel=1e-6)
        assert calibrated_persons.loc[
            calibrated_persons["state_fips"] == 6, "weight"
        ].iloc[0] == pytest.approx(225.0, rel=1e-6)

    def test_calibrate_policyengine_tables_from_db_with_sparse_backend(
        self,
        persons,
        households,
        tmp_path,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        config = USMicroplexBuildConfig(
            calibration_backend="sparse",
            target_sparsity=0.0,
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert summary["backend"] == "policyengine_db_sparse"
        assert summary["n_constraints"] == 2
        assert summary["max_error"] < 1e-5
        assert summary["converged"] is True
        assert summary["sparsity"] == pytest.approx(0.0, abs=1e-9)
        assert calibrated_tables.households["household_weight"].sum() == pytest.approx(
            450.0,
            rel=1e-5,
        )

    def test_synthesize_seed_backend_preserves_seed_support(self, persons, households):
        config = USMicroplexBuildConfig(
            synthesis_backend="seed",
            n_synthetic=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households)

        synthetic, synthesizer, metadata = pipeline.synthesize(seed)

        assert synthesizer is None
        assert metadata["backend"] == "seed"
        assert metadata["n_seed_records"] == len(seed)
        assert len(synthetic) == len(seed)
        assert synthetic["household_id"].nunique() == seed["household_id"].nunique()
        assert synthetic["weight"].tolist() == pytest.approx(seed["hh_weight"].tolist())

    def test_calibrate_policyengine_tables_can_prune_to_household_budget(
        self,
        persons,
        households,
        tmp_path,
        monkeypatch,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)

        class StubSparseSelector:
            def __init__(self, **_kwargs):
                pass

            def fit_transform(
                self,
                frame,
                *_args,
                weight_col,
                linear_constraints=None,
                **_kwargs,
            ):
                result = frame.copy()
                result[weight_col] = np.array([10.0, 8.0, 0.0])
                self._constraints = tuple(linear_constraints or ())
                return result

            def validate(self, _frame):
                return {
                    "max_error": 0.1,
                    "mean_error": 0.05,
                    "converged": True,
                    "sparsity": 1 / 3,
                    "linear_errors": {
                        constraint.name: {
                            "actual": float(constraint.target),
                            "target": float(constraint.target),
                            "relative_error": 0.0,
                        }
                        for constraint in getattr(self, "_constraints", ())
                    },
                }

        monkeypatch.setattr(
            "microplex_us.pipelines.us.SparseCalibrator",
            StubSparseSelector,
        )
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
            policyengine_selection_household_budget=2,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, calibrated_persons, summary = (
            pipeline.calibrate_policyengine_tables(tables)
        )

        assert len(calibrated_tables.households) == 2
        assert set(calibrated_tables.households["household_id"]) == {1, 2}
        assert set(calibrated_persons["household_id"]) == {1, 2}
        assert summary["selection"]["applied"] is True
        assert summary["selection"]["selected_household_count"] == 2
        assert summary["selection"]["selector_positive_selected_count"] == 2
        assert "pre_selection" in summary["feasibility_filter"]

    def test_calibrate_policyengine_tables_from_db_with_hardconcrete_backend(
        self,
        persons,
        households,
        tmp_path,
        monkeypatch,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        seen_constraints = {}

        class StubHardConcreteCalibrator:
            def __init__(self, **_kwargs):
                self._constraints = ()

            def fit_transform(
                self,
                frame,
                *_args,
                weight_col,
                linear_constraints=None,
                **_kwargs,
            ):
                self._constraints = tuple(linear_constraints or ())
                seen_constraints["count"] = len(self._constraints)
                return frame.copy()

            def validate(self, _frame):
                return {
                    "max_error": 0.0,
                    "mean_error": 0.0,
                    "converged": True,
                    "sparsity": 0.25,
                    "linear_errors": {
                        constraint.name: {
                            "actual": float(constraint.target),
                            "target": float(constraint.target),
                            "relative_error": 0.0,
                        }
                        for constraint in self._constraints
                    },
                }

        monkeypatch.setattr(
            "microplex_us.pipelines.us.HardConcreteCalibrator",
            StubHardConcreteCalibrator,
        )
        config = USMicroplexBuildConfig(
            calibration_backend="hardconcrete",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert seen_constraints["count"] == 2
        assert summary["backend"] == "policyengine_db_hardconcrete"
        assert summary["n_constraints"] == 2
        assert summary["converged"] is True
        assert summary["sparsity"] == pytest.approx(0.25)
        assert calibrated_tables.households["household_weight"].sum() == pytest.approx(
            450.0,
            rel=1e-6,
        )

    def test_calibrate_policyengine_tables_flags_weight_collapse(
        self,
        persons,
        households,
        tmp_path,
        monkeypatch,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)

        class CollapsingCalibrator:
            def __init__(self, method):
                self.method = method

            def fit_transform(
                self,
                frame,
                *_args,
                weight_col,
                **_kwargs,
            ):
                collapsed = frame.copy()
                collapsed[weight_col] = 1e-10
                return collapsed

            def validate(self, _frame):
                return {
                    "max_error": 1.0,
                    "mean_error": 1.0,
                    "converged": False,
                    "linear_errors": {},
                }

        monkeypatch.setattr(
            "microplex_us.pipelines.us.Calibrator",
            CollapsingCalibrator,
        )
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        _, calibrated_persons, summary = pipeline.calibrate_policyengine_tables(tables)

        assert summary["weight_collapse_suspected"] is True
        assert summary["household_weight_diagnostics"]["tiny_count"] == summary[
            "household_weight_diagnostics"
        ]["row_count"]
        assert summary["household_weight_diagnostics"]["total_weight"] == pytest.approx(
            summary["household_weight_diagnostics"]["row_count"] * 1e-10
        )
        assert summary["person_weight_diagnostics"]["tiny_count"] == len(calibrated_persons)

    def test_summarize_weight_diagnostics_flags_low_effective_sample_ratio(self):
        summary = _summarize_weight_diagnostics([100.0, 100.0] + [1e-10] * 10)

        assert summary["tiny_share"] < 0.95
        assert summary["effective_sample_ratio"] < 0.25
        assert summary["collapse_suspected"] is True

    def test_select_feasible_policyengine_calibration_constraints_caps_budget(self):
        targets = [
            TargetSpec(
                name="national_count",
                entity=EntityType.HOUSEHOLD,
                value=100.0,
                period=2024,
                aggregation=TargetAggregation.COUNT,
                metadata={"geo_level": "national"},
            ),
            TargetSpec(
                name="state_count",
                entity=EntityType.HOUSEHOLD,
                value=50.0,
                period=2024,
                aggregation=TargetAggregation.COUNT,
                metadata={"geo_level": "state"},
            ),
            TargetSpec(
                name="state_sum",
                entity=EntityType.HOUSEHOLD,
                value=25.0,
                period=2024,
                measure="snap",
                aggregation=TargetAggregation.SUM,
                metadata={"geo_level": "state"},
            ),
        ]
        constraints = (
            SimpleNamespace(coefficients=np.array([1.0, 1.0])),
            SimpleNamespace(coefficients=np.array([1.0, 0.0])),
            SimpleNamespace(coefficients=np.array([1.0, 1.0])),
        )

        selected_targets, selected_constraints, summary = (
            _select_feasible_policyengine_calibration_constraints(
                targets,
                constraints,
                household_count=2,
                max_constraints=None,
                max_constraints_per_household=1.0,
                min_active_households=1,
            )
        )

        assert [target.name for target in selected_targets] == [
            "national_count",
            "state_count",
        ]
        assert len(selected_constraints) == 2
        assert summary["feasibility_filter_applied"] is True
        assert summary["requested_max_constraints"] == 2
        assert summary["n_constraints_before_feasibility_filter"] == 3
        assert summary["n_constraints_after_feasibility_filter"] == 2
        assert summary["n_constraints_dropped_over_capacity"] == 1
        assert summary["constraint_drop_share"] == pytest.approx(1 / 3)
        assert summary["warning_messages"]

    def test_select_feasible_policyengine_calibration_constraints_drops_low_support_rows(self):
        targets = [
            TargetSpec(
                name="dense_state_count",
                entity=EntityType.HOUSEHOLD,
                value=50.0,
                period=2024,
                aggregation=TargetAggregation.COUNT,
                metadata={"geo_level": "state"},
            ),
            TargetSpec(
                name="thin_state_count",
                entity=EntityType.HOUSEHOLD,
                value=25.0,
                period=2024,
                aggregation=TargetAggregation.COUNT,
                metadata={"geo_level": "state"},
            ),
        ]
        constraints = (
            SimpleNamespace(coefficients=np.array([1.0, 1.0, 1.0, 1.0, 1.0])),
            SimpleNamespace(coefficients=np.array([0.0, 0.0, 0.0, 0.0, 1.0])),
        )

        selected_targets, _, summary = _select_feasible_policyengine_calibration_constraints(
            targets,
            constraints,
            household_count=5,
            max_constraints=None,
            max_constraints_per_household=None,
            min_active_households=5,
        )

        assert [target.name for target in selected_targets] == ["dense_state_count"]
        assert summary["n_constraints_dropped_low_support"] == 1
        assert summary["n_constraints_after_feasibility_filter"] == 1

    def test_calibrate_policyengine_tables_applies_feasibility_constraint_budget(
        self,
        persons,
        households,
        tmp_path,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_max_constraints_per_household=0.5,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert summary["n_constraints"] == 1
        assert summary["feasibility_filter"]["feasibility_filter_applied"] is True
        assert summary["feasibility_filter"]["requested_max_constraints"] == 1
        assert summary["feasibility_filter"]["n_constraints_before_feasibility_filter"] == 2
        assert summary["feasibility_filter"]["n_constraints_after_feasibility_filter"] == 1
        assert calibrated_tables.households["household_weight"].sum() == pytest.approx(
            450.0,
            rel=1e-6,
        )

    def test_calibrate_policyengine_tables_warns_when_many_constraints_are_dropped(
        self,
        persons,
        households,
        tmp_path,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_max_constraints_per_household=0.5,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        with pytest.warns(
            UserWarning,
            match="Calibration feasibility filter dropped",
        ):
            _, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert summary["warnings"]

    def test_calibrate_policyengine_tables_skips_structurally_unsupported_targets(
        self,
        persons,
        households,
        tmp_path,
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db_with_unsupported_target(db_path)
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert summary["n_loaded_targets"] == 2
        assert summary["n_supported_targets"] == 1
        assert summary["n_unsupported_targets"] == 1
        assert summary["n_constraints"] == 1
        assert calibrated_tables.households["household_weight"].sum() == pytest.approx(
            450.0,
            rel=1e-6,
        )

    def test_policyengine_target_provider_returns_canonical_specs(
        self, persons, households, tmp_path
    ):
        db_path = tmp_path / "policyengine_targets.db"
        _create_policyengine_calibration_db(db_path)
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("household_count",),
            policyengine_target_period=2024,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)
        provider = pipeline.config.policyengine_targets_db
        assert provider is not None
        bindings = pipeline._infer_policyengine_variable_bindings(tables)

        from microplex_us.policyengine.us import PolicyEngineUSDBTargetProvider

        targets = PolicyEngineUSDBTargetProvider(provider).load_target_set(
            TargetQuery(
                period=2024,
                provider_filters={
                    "variables": ["household_count"],
                    "reform_id": 0,
                    "entity_overrides": {
                        variable: binding.entity for variable, binding in bindings.items()
                    },
                },
            )
        )

        assert targets.targets
        assert all(isinstance(target, TargetSpec) for target in targets.targets)

    def test_calibrate_policyengine_tables_from_db_with_simulated_variable(
        self, persons, households, tmp_path
    ):
        db_path = tmp_path / "policyengine_targets.db"
        conn = sqlite3.connect(db_path)
        national_constraints: tuple[PolicyEngineUSConstraint, ...] = ()
        snap_positive_constraints = (
            PolicyEngineUSConstraint("snap", ">", "0"),
        )
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
            """
        )
        conn.executemany(
            """
            INSERT INTO strata (stratum_id, definition_hash, parent_stratum_id)
            VALUES (?, ?, ?)
            """,
            [
                (
                    1,
                    compute_policyengine_us_definition_hash(national_constraints),
                    None,
                ),
                (
                    2,
                    compute_policyengine_us_definition_hash(
                        snap_positive_constraints,
                        parent_stratum_id=1,
                    ),
                    1,
                ),
            ],
        )
        conn.execute(
            """
            INSERT INTO stratum_constraints (
                stratum_id,
                constraint_variable,
                operation,
                value
            ) VALUES (?, ?, ?, ?)
            """,
            (2, "snap", ">", "0"),
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
                (1, "snap", 2024, 1, 0, 200.0, 1, None, "test", "national"),
                (2, "household_count", 2024, 2, 0, 2.0, 1, None, "test", "positive"),
            ],
        )
        conn.commit()
        conn.close()

        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity, formulas=None):
                self.entity = entity
                self.formulas = formulas or {}

            def is_input_variable(self):
                return not self.formulas

        class FakeSystem:
            variables = {
                "employment_income": FakeVariable(FakeEntity("person")),
                "state_fips": FakeVariable(FakeEntity("household")),
                "snap": FakeVariable(
                    FakeEntity("household"),
                    formulas={"2024": object()},
                ),
            }

        class FakeSimulation:
            tax_benefit_system = FakeSystem()

            def __init__(self, dataset, dataset_year=None, **kwargs):
                self.dataset = dataset
                self.dataset_year = dataset_year
                _ = kwargs

            def calculate(self, variable, period=None, map_to=None):
                assert period == 2024
                assert self.dataset_year == 2024
                assert map_to is None
                if variable == "snap":
                    return [100.0, 0.0, 0.0]
                raise KeyError(variable)

        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("snap", "household_count"),
            policyengine_target_period=2024,
            policyengine_dataset_year=2024,
            policyengine_simulation_cls=FakeSimulation,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight", "income": "employment_income"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, calibrated_persons, summary = (
            pipeline.calibrate_policyengine_tables(tables)
        )

        assert summary["backend"] == "policyengine_db_entropy"
        assert summary["n_constraints"] == 2
        assert summary["materialized_variables"] == ["snap"]
        assert summary["max_error"] < 1e-6
        positive_weight = calibrated_tables.households.loc[
            calibrated_tables.households["snap"] > 0,
            "household_weight",
        ].sum()
        assert (
            calibrated_tables.households["snap"]
            * calibrated_tables.households["household_weight"]
        ).sum() == pytest.approx(
            200.0,
            rel=1e-6,
        )
        assert positive_weight == pytest.approx(2.0, rel=1e-6)
        positive_household_id = int(
            calibrated_tables.households.loc[
                calibrated_tables.households["snap"] > 0,
                "household_id",
            ].iloc[0]
        )
        assert calibrated_persons.loc[
            calibrated_persons["household_id"] == positive_household_id, "weight"
        ].iloc[0] == pytest.approx(2.0, rel=1e-6)

    def test_calibrate_policyengine_tables_skips_failed_materialized_variables(
        self, persons, households, tmp_path
    ):
        db_path = tmp_path / "policyengine_targets.db"
        conn = sqlite3.connect(db_path)
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
            """
        )
        conn.execute(
            """
            INSERT INTO strata (stratum_id, definition_hash, parent_stratum_id)
            VALUES (?, ?, ?)
            """,
            (1, compute_policyengine_us_definition_hash(()), None),
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
                (1, "snap", 2024, 1, 0, 200.0, 1, None, "test", "national"),
                (
                    2,
                    "adjusted_gross_income",
                    2024,
                    1,
                    0,
                    1_000.0,
                    1,
                    None,
                    "test",
                    "agi",
                ),
            ],
        )
        conn.commit()
        conn.close()

        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity, formulas=None):
                self.entity = entity
                self.formulas = formulas or {}

            def is_input_variable(self):
                return not self.formulas

        class FakeSystem:
            variables = {
                "employment_income": FakeVariable(FakeEntity("person")),
                "state_fips": FakeVariable(FakeEntity("household")),
                "snap": FakeVariable(
                    FakeEntity("household"),
                    formulas={"2024": object()},
                ),
                "adjusted_gross_income": FakeVariable(
                    FakeEntity("tax_unit"),
                    formulas={"2024": object()},
                ),
            }

        class FakeSimulation:
            tax_benefit_system = FakeSystem()

            def __init__(self, dataset, dataset_year=None, **kwargs):
                self.dataset = dataset
                self.dataset_year = dataset_year
                _ = kwargs

            def calculate(self, variable, period=None, map_to=None):
                assert period == 2024
                assert self.dataset_year == 2024
                assert map_to is None
                if variable == "snap":
                    return [100.0, 0.0, 0.0]
                if variable == "adjusted_gross_income":
                    raise RuntimeError("invalid state metadata")
                raise KeyError(variable)

        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("snap", "adjusted_gross_income"),
            policyengine_target_period=2024,
            policyengine_dataset_year=2024,
            policyengine_simulation_cls=FakeSimulation,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight", "income": "employment_income"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert summary["n_loaded_targets"] == 2
        assert summary["n_supported_targets"] == 1
        assert summary["n_constraints"] == 1
        assert summary["materialized_variables"] == ["snap"]
        assert summary["materialization_failures"] == {
            "adjusted_gross_income": "RuntimeError: invalid state metadata"
        }

    def test_calibrate_policyengine_tables_uses_calibration_target_filters(
        self, persons, households, tmp_path
    ):
        db_path = tmp_path / "policyengine_targets.db"
        conn = sqlite3.connect(db_path)
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
            """
        )
        conn.execute(
            """
            INSERT INTO strata (stratum_id, definition_hash, parent_stratum_id)
            VALUES (?, ?, ?)
            """,
            (1, compute_policyengine_us_definition_hash(()), None),
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
                (1, "snap", 2024, 1, 0, 200.0, 1, None, "test", "national"),
                (
                    2,
                    "adjusted_gross_income",
                    2024,
                    1,
                    0,
                    1_000.0,
                    1,
                    None,
                    "test",
                    "agi",
                ),
            ],
        )
        conn.commit()
        conn.close()

        class FakeEntity:
            def __init__(self, key):
                self.key = key

        class FakeVariable:
            def __init__(self, entity, formulas=None):
                self.entity = entity
                self.formulas = formulas or {}

            def is_input_variable(self):
                return not self.formulas

        class FakeSystem:
            variables = {
                "employment_income": FakeVariable(FakeEntity("person")),
                "state_fips": FakeVariable(FakeEntity("household")),
                "snap": FakeVariable(
                    FakeEntity("household"),
                    formulas={"2024": object()},
                ),
                "adjusted_gross_income": FakeVariable(
                    FakeEntity("tax_unit"),
                    formulas={"2024": object()},
                ),
            }

        class FakeSimulation:
            tax_benefit_system = FakeSystem()

            def __init__(self, dataset, dataset_year=None, **kwargs):
                self.dataset = dataset
                self.dataset_year = dataset_year
                _ = kwargs

            def calculate(self, variable, period=None, map_to=None):
                assert period == 2024
                assert self.dataset_year == 2024
                assert map_to is None
                if variable == "snap":
                    return [100.0, 0.0, 0.0]
                if variable == "adjusted_gross_income":
                    raise RuntimeError("invalid state metadata")
                raise KeyError(variable)

        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            policyengine_targets_db=str(db_path),
            policyengine_target_variables=("snap", "adjusted_gross_income"),
            policyengine_calibration_target_variables=("snap",),
            policyengine_target_period=2024,
            policyengine_dataset_year=2024,
            policyengine_simulation_cls=FakeSimulation,
            policyengine_calibration_min_active_households=1,
        )
        pipeline = USMicroplexPipeline(config)
        seed = pipeline.prepare_seed_data(persons, households).rename(
            columns={"hh_weight": "weight", "income": "employment_income"}
        )
        tables = pipeline.build_policyengine_entity_tables(seed)

        calibrated_tables, _, summary = pipeline.calibrate_policyengine_tables(tables)

        assert summary["n_loaded_targets"] == 1
        assert summary["n_supported_targets"] == 1
        assert summary["n_constraints"] == 1
        assert summary["target_variables"] == ["snap"]
        assert summary["materialized_variables"] == ["snap"]
        assert summary["materialization_failures"] == {}
        assert (
            calibrated_tables.households["snap"]
            * calibrated_tables.households["household_weight"]
        ).sum() == pytest.approx(200.0, rel=1e-6)
        assert (
            calibrated_tables.households["snap"]
            * calibrated_tables.households["household_weight"]
        ).sum() == pytest.approx(200.0, rel=1e-6)

    def test_build_policyengine_target_query_includes_named_target_profile(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                policyengine_target_profile="pe_native_broad",
            )
        )

        query = pipeline._build_policyengine_target_query({}, period=2024)

        assert query.provider_filters["target_profile"] == "pe_native_broad"
        assert query.provider_filters["target_cells"]
        assert {
            cell["geo_level"] for cell in query.provider_filters["target_cells"]
        } <= {"national", "state"}

    def test_build_policyengine_target_query_prefers_calibration_profile_override(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                policyengine_target_profile="pe_native_broad",
                policyengine_calibration_target_profile="pe_native_broad",
                policyengine_calibration_target_variables=("snap",),
            )
        )

        query = pipeline._build_policyengine_target_query(
            {},
            period=2024,
            for_calibration=True,
        )

        assert query.provider_filters["target_profile"] == "pe_native_broad"
        assert query.provider_filters["variables"] == ["snap"]
        assert query.provider_filters["target_cells"]

    def test_load_inputs_from_directory(self, persons, households, tmp_path):
        households.rename(columns={"hh_weight": "household_weight"}).to_parquet(
            tmp_path / "cps_asec_households.parquet",
            index=False,
        )
        persons.to_parquet(tmp_path / "cps_asec_persons.parquet", index=False)

        config = USMicroplexBuildConfig(
            n_synthetic=8,
            synthesis_backend="bootstrap",
            calibration_backend="entropy",
        )
        pipeline = USMicroplexPipeline(config)
        result = pipeline.build_from_data_dir(tmp_path)

        assert result.synthetic_data["household_id"].nunique() == 8
        assert len(result.synthetic_data) > 8
        assert result.seed_data["hh_weight"].sum() == pytest.approx(900.0)

    def test_build_weight_calibrator_respects_iteration_and_tolerance_config(self):
        config = USMicroplexBuildConfig(
            calibration_backend="entropy",
            calibration_tol=1e-4,
            calibration_max_iter=777,
        )
        pipeline = USMicroplexPipeline(config)

        calibrator = pipeline._build_weight_calibrator()

        assert calibrator.tol == pytest.approx(1e-4)
        assert calibrator.max_iter == 777

    def test_build_from_data_dir_can_prefer_cached_cps_asec_source(
        self,
        persons,
        households,
        tmp_path,
        monkeypatch,
    ):
        households.rename(columns={"hh_weight": "household_weight"}).to_parquet(
            tmp_path / "cps_asec_households.parquet",
            index=False,
        )
        persons.to_parquet(tmp_path / "cps_asec_persons.parquet", index=False)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "cps_asec_2023_processed.parquet").write_text("stub")

        class FakeCachedProvider:
            def __init__(self, *, year, cache_dir, download):
                self.year = year
                self.cache_dir = cache_dir
                self.download = download
                self.descriptor = SourceDescriptor(
                    name="cps_asec",
                    shareability=Shareability.PUBLIC,
                    time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                    observations=(
                        EntityObservation(
                            entity=EntityType.HOUSEHOLD,
                            key_column="household_id",
                            variable_names=("state_fips",),
                        ),
                    ),
                )

        class FakeParquetProvider:
            def __init__(self, *, data_dir):
                self.data_dir = data_dir
                self.descriptor = SourceDescriptor(
                    name="cps_asec_parquet",
                    shareability=Shareability.PUBLIC,
                    time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                    observations=(
                        EntityObservation(
                            entity=EntityType.HOUSEHOLD,
                            key_column="household_id",
                            variable_names=("state_fips",),
                        ),
                    ),
                )

        monkeypatch.setattr(
            "microplex_us.data_sources.cps.CPSASECSourceProvider",
            FakeCachedProvider,
        )
        monkeypatch.setattr(
            "microplex_us.data_sources.cps.CPSASECParquetSourceProvider",
            FakeParquetProvider,
        )

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                prefer_cached_cps_asec_source=True,
                cps_asec_cache_dir=str(cache_dir),
                cps_asec_source_year=2023,
            )
        )
        chosen: dict[str, object] = {}

        def fake_build_from_source_provider(provider):
            chosen["provider"] = provider
            return "cached"

        monkeypatch.setattr(pipeline, "build_from_source_provider", fake_build_from_source_provider)

        result = pipeline.build_from_data_dir(tmp_path)

        assert result == "cached"
        assert chosen["provider"].descriptor.name == "cps_asec"

    def test_build_from_source_provider(self, persons, households):
        provider_households = households.rename(
            columns={
                "household_id": "hh_id",
                "hh_weight": "household_weight",
            }
        )
        provider_persons = persons.rename(
            columns={
                "person_id": "person_key",
                "household_id": "hh_id",
            }
        )
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="test_cps",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="hh_id",
                        weight_column="household_weight",
                        variable_names=tuple(
                            column
                            for column in provider_households.columns
                            if column != "hh_id"
                        ),
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_key",
                        variable_names=tuple(
                            column
                            for column in provider_persons.columns
                            if column not in {"person_key", "hh_id"}
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: provider_households,
                EntityType.PERSON: provider_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="hh_id",
                    child_key="hh_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        provider = StaticSourceProvider(frame)
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=8,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )

        result = pipeline.build_from_source_provider(provider)

        assert result.synthetic_data["household_id"].nunique() == 8
        assert len(result.synthetic_data) > 8
        assert result.source_frame is not None
        assert result.source_frame.source.name == "test_cps"
        assert result.fusion_plan is not None
        assert result.fusion_plan.source_names == ("test_cps",)
        assert result.seed_data["hh_weight"].sum() == pytest.approx(900.0)
        assert {"person_id", "household_id"}.issubset(result.seed_data.columns)

    def test_build_from_source_provider_requires_household_person_relationship(
        self, persons, households
    ):
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="test_cps",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "hh_weight", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: households,
                EntityType.PERSON: persons,
            },
            relationships=(),
        )
        provider = StaticSourceProvider(frame)
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())

        with pytest.raises(
            ValueError,
            match="one-to-many household-to-person relationship",
        ):
            pipeline.build_from_source_provider(provider)

    def test_build_from_frames_prefers_scaffold_with_valid_geography(self):
        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 19],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [60_000.0, 12_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [50.0, 75.0, 80.0],
                "state_fips": [0, 0, 0],
                "tenure": [1, 2, 1],
                "extra_household_var": [1.0, 2.0, 3.0],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [51, 34, 28],
                "sex": [1, 2, 1],
                "education": [4, 3, 2],
                "employment_status": [1, 1, 0],
                "income": [80_000.0, 40_000.0, 20_000.0],
                "extra_person_var": [9.0, 8.0, 7.0],
            }
        )

        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure", "extra_household_var"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "extra_person_var",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )

        result = pipeline.build_from_frames([cps_frame, donor_frame])

        assert result.source_frame is not None
        assert result.source_frame.source.name == "cps_like"
        assert result.seed_data["state_fips"].tolist() == [6, 36]

    def test_build_from_frames_prefers_scaffold_with_state_program_proxies(self):
        proxy_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        proxy_persons = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [1, 2],
                "age": [45, 19],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [60_000.0, 12_000.0],
                "has_medicaid": [1, 0],
                "public_assistance": [0.0, 250.0],
                "ssi": [0.0, 0.0],
                "social_security": [0.0, 0.0],
            }
        )
        wider_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [90.0, 110.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
                "extra_household_var": [1.0, 2.0],
            }
        )
        wider_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002],
                "household_id": [101, 102],
                "age": [44, 21],
                "sex": [1, 2],
                "education": [3, 2],
                "employment_status": [1, 0],
                "income": [58_000.0, 13_000.0],
                "extra_person_var": [9.0, 8.0],
                "another_extra_var": [5.0, 6.0],
            }
        )

        proxy_frame = ObservationFrame(
            source=SourceDescriptor(
                name="proxy_rich_cps",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "has_medicaid",
                            "public_assistance",
                            "ssi",
                            "social_security",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: proxy_households,
                EntityType.PERSON: proxy_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        wider_frame = ObservationFrame(
            source=SourceDescriptor(
                name="wider_but_proxy_poor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure", "extra_household_var"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "extra_person_var",
                            "another_extra_var",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: wider_households,
                EntityType.PERSON: wider_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )

        result = pipeline.build_from_frames([proxy_frame, wider_frame])

        assert result.source_frame is not None
        assert result.source_frame.source.name == "proxy_rich_cps"
        assert result.synthesis_metadata["state_program_support_proxies"]["available"] == [
            "has_medicaid",
            "public_assistance",
            "social_security",
            "ssi",
        ]
        assert result.synthesis_metadata["condition_vars"] == [
            "age",
            "sex",
            "education",
            "employment_status",
            "state_fips",
            "tenure",
            "has_medicaid",
        ]
        assert "has_medicaid" not in result.synthesis_metadata["target_vars"]
        assert "public_assistance" in result.synthesis_metadata["target_vars"]
        assert "ssi" in result.synthesis_metadata["target_vars"]
        assert "social_security" in result.synthesis_metadata["target_vars"]

    def test_build_from_source_provider_promotes_state_program_proxies_to_conditions(self):
        households = pd.DataFrame(
            {
                "household_key": [1, 2, 3],
                "household_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        persons = pd.DataFrame(
            {
                "person_key": [10, 11, 12],
                "household_key": [1, 2, 3],
                "age": [45, 19, 62],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [60_000.0, 12_000.0, 40_000.0],
                "has_medicaid": [1, 0, 1],
                "public_assistance": [0.0, 250.0, 0.0],
                "ssi": [0.0, 0.0, 900.0],
                "social_security": [0.0, 0.0, 1200.0],
            }
        )
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="proxy_rich_single_source",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_key",
                        variable_names=("state_fips", "tenure"),
                        weight_column="household_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_key",
                        variable_names=(
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "has_medicaid",
                            "public_assistance",
                            "ssi",
                            "social_security",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: households,
                EntityType.PERSON: persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_key",
                    child_key="household_key",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )

        result = pipeline.build_from_source_provider(StaticSourceProvider(frame))

        assert result.synthesis_metadata["condition_vars"] == [
            "age",
            "sex",
            "education",
            "employment_status",
            "state_fips",
            "tenure",
            "has_medicaid",
        ]
        assert result.synthesis_metadata["target_vars"] == [
            "income",
            "public_assistance",
            "ssi",
            "social_security",
        ]

    def test_build_from_frames_skips_non_numeric_donor_imputation_targets(self):
        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20, 30],
                "household_id": [1, 2, 3],
                "age": [45, 19, 62],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [60_000.0, 12_000.0, 40_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 90.0, 110.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [44, 21, 61],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [58_000.0, 13_000.0, 41_000.0],
                "taxable_interest_income": [100.0, 50.0, 25.0],
                "all_zero_income": [0.0, 0.0, 0.0],
                "filing_status": ["SINGLE", "JOINT", "SINGLE"],
            }
        )

        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "taxable_interest_income",
                            "all_zero_income",
                            "filing_status",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert "taxable_interest_income" in integration["seed_data"].columns
        assert "all_zero_income" not in integration["seed_data"].columns
        assert "filing_status" not in integration["seed_data"].columns
        assert integration["integrated_variables"] == ["taxable_interest_income"]

    def test_integrate_donor_sources_restricts_puf_to_authoritative_variables(self):
        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20, 30],
                "household_id": [1, 2, 3],
                "age": [45, 19, 62],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [60_000.0, 12_000.0, 40_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 90.0, 110.0],
                "state_fips": [0, 0, 0],
                "tenure": [0, 0, 0],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [44, 21, 61],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [58_000.0, 13_000.0, 41_000.0],
                "employment_income": [55_000.0, 12_500.0, 39_000.0],
                "taxable_interest_income": [0.0, 25.0, 100.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="irs_soi_puf_2024",
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "employment_income",
                            "taxable_interest_income",
                        ),
                    ),
                ),
                variable_capabilities={
                    "state_fips": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "tenure": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "income": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "employment_status": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "employment_income": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "taxable_interest_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=True,
                    ),
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert "taxable_interest_income" in integration["integrated_variables"]
        assert "employment_income" not in integration["integrated_variables"]
        assert "taxable_interest_income" in integration["seed_data"].columns
        assert "employment_income" not in integration["seed_data"].columns

    def test_integrate_donor_sources_respects_excluded_variables(self, monkeypatch):
        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = condition_vars, kwargs
                self.target_vars = tuple(target_vars)

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["taxable_interest_income"] = [10.0] * len(result)
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20, 30],
                "household_id": [1, 2, 3],
                "age": [45, 19, 62],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [60_000.0, 12_000.0, 40_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 90.0, 110.0],
                "state_fips": [0, 0, 0],
                "tenure": [1, 2, 1],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [44, 21, 61],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [58_000.0, 13_000.0, 41_000.0],
                "taxable_interest_income": [0.0, 25.0, 100.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="irs_soi_puf_2024",
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "taxable_interest_income",
                        ),
                    ),
                ),
                variable_capabilities={
                    "state_fips": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "tenure": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "income": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "employment_status": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "taxable_interest_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=True,
                    ),
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                donor_imputer_excluded_variables=("taxable_interest_income",),
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert integration["integrated_variables"] == []
        assert "taxable_interest_income" not in integration["seed_data"].columns

    def test_default_build_config_excludes_filing_status_code_from_donor_imputation(self):
        config = USMicroplexBuildConfig()

        assert "filing_status_code" in config.donor_imputer_excluded_variables

    def test_build_config_can_opt_back_into_filing_status_code_donor_imputation(self):
        config = USMicroplexBuildConfig(donor_imputer_excluded_variables=())

        assert "filing_status_code" not in config.donor_imputer_excluded_variables

    def test_integrate_donor_sources_drops_constant_donor_conditions(self, monkeypatch):
        captured: list[tuple[str, ...]] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = kwargs
                self.target_vars = tuple(target_vars)
                self.condition_vars = tuple(condition_vars)
                captured.append(self.condition_vars)

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["taxable_interest_income"] = [10.0] * len(result)
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20, 30],
                "household_id": [1, 2, 3],
                "age": [45, 19, 62],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [60_000.0, 12_000.0, 40_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 90.0, 110.0],
                "state_fips": [0, 0, 0],
                "tenure": [1, 2, 1],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [44, 21, 61],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [58_000.0, 13_000.0, 41_000.0],
                "taxable_interest_income": [0.0, 25.0, 100.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="irs_soi_puf_2024",
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "taxable_interest_income",
                        ),
                    ),
                ),
                variable_capabilities={
                    "state_fips": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "tenure": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=True,
                    ),
                    "income": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "employment_status": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "taxable_interest_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=True,
                    ),
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert captured
        assert "state_fips" not in captured[0]
        assert "tenure" in captured[0]

    def test_integrate_donor_sources_selects_top_correlated_condition_vars(
        self,
        monkeypatch,
    ):
        captured: list[tuple[str, ...]] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = target_vars, kwargs
                captured.append(tuple(condition_vars))

            def fit(self, *args, **kwargs):
                _ = args, kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["taxable_interest_income"] = [10.0, 20.0, 30.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20, 30],
                "household_id": [1, 2, 3],
                "age": [25, 45, 65],
                "sex": [1, 2, 1],
                "education": [2, 2, 2],
                "employment_status": [1, 1, 1],
                "income": [30_000.0, 40_000.0, 50_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 90.0, 110.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [24, 44, 64],
                "sex": [1, 1, 1],
                "education": [2, 2, 2],
                "employment_status": [1, 1, 1],
                "income": [10_000.0, 80_000.0, 20_000.0],
                "taxable_interest_income": [5.0, 15.0, 25.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="irs_soi_puf_2024",
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "taxable_interest_income",
                        ),
                    ),
                ),
                variable_capabilities={
                    "state_fips": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "income": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=True,
                    ),
                    "taxable_interest_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=True,
                    ),
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                donor_imputer_max_condition_vars=1,
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert captured == [("age",)]

    def test_integrate_donor_sources_projects_tax_unit_native_blocks_when_ids_present(
        self,
        monkeypatch,
    ):
        captured_conditions: list[tuple[str, ...]] = []
        captured_fit_rows: list[int] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = target_vars, kwargs
                captured_conditions.append(tuple(condition_vars))

            def fit(self, frame, *args, **kwargs):
                _ = args, kwargs
                captured_fit_rows.append(len(frame))

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["taxable_interest_income"] = [25.0, 75.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": ["1:1", "1:2", "2:1"],
                "household_id": [1, 1, 2],
                "tax_unit_id": [100, 100, 200],
                "age": [45, 43, 19],
                "sex": [1, 2, 1],
                "education": [3, 3, 2],
                "employment_status": [1, 1, 1],
                "income": [60_000.0, 15_000.0, 12_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [80.0, 90.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": ["101:1", "101:2", "102:1"],
                "household_id": [101, 101, 102],
                "tax_unit_id": [900, 900, 901],
                "age": [44, 42, 21],
                "sex": [1, 2, 1],
                "education": [3, 3, 2],
                "employment_status": [1, 1, 1],
                "income": [58_000.0, 14_000.0, 13_000.0],
                "taxable_interest_income": [0.0, 0.0, 100.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "tax_unit_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "tax_unit_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "taxable_interest_income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert len(captured_conditions) == 1
        assert {"age", "income", "state_fips", "tenure"}.issubset(
            set(captured_conditions[0])
        )
        assert captured_fit_rows == [2]
        assert integration["seed_data"]["taxable_interest_income"].tolist() == [0.0, 0.0, 100.0]

    def test_integrate_donor_sources_allows_person_conditions_for_labor_tax_unit_blocks(
        self,
        monkeypatch,
    ):
        captured_conditions: list[tuple[str, ...]] = []
        captured_fit_rows: list[int] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = target_vars, kwargs
                captured_conditions.append(tuple(condition_vars))

            def fit(self, frame, *args, **kwargs):
                _ = args, kwargs
                captured_fit_rows.append(len(frame))

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["self_employment_income"] = [0.0, 15.0, 90.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": ["1:1", "1:2", "2:1", "3:1"],
                "household_id": [1, 1, 2, 3],
                "tax_unit_id": [100, 100, 200, 300],
                "age": [25, 23, 45, 65],
                "sex": [1, 1, 1, 1],
                "education": [2, 2, 2, 2],
                "employment_status": [1, 1, 1, 1],
                "income": [20_000.0, 5_000.0, 50_000.0, 90_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 90.0, 110.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": ["101:1", "101:2", "102:1", "103:1"],
                "household_id": [101, 101, 102, 103],
                "tax_unit_id": [900, 900, 901, 902],
                "age": [24, 22, 44, 64],
                "sex": [1, 1, 1, 1],
                "education": [2, 2, 2, 2],
                "employment_status": [1, 1, 1, 1],
                "income": [18_000.0, 4_000.0, 52_000.0, 92_000.0],
                "self_employment_income": [0.0, 0.0, 20.0, 100.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "tax_unit_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "tax_unit_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "self_employment_income",
                        ),
                    ),
                ),
                variable_capabilities={
                    "state_fips": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "income": SourceVariableCapability(
                        authoritative=False,
                        usable_as_condition=False,
                    ),
                    "self_employment_income": SourceVariableCapability(
                        authoritative=True,
                        usable_as_condition=True,
                    ),
                },
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
                donor_imputer_max_condition_vars=1,
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert captured_conditions == [("age",)]
        assert captured_fit_rows == [3]

    def test_project_frame_to_entity_uses_variable_projection_aggregation(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        frame = pd.DataFrame(
            {
                "tax_unit_id": [100, 100, 200],
                "age": [25, 45, 65],
                "income": [20_000.0, 5_000.0, 90_000.0],
                "tenure": [1, 1, 2],
            }
        )

        projected = pipeline._project_frame_to_entity(
            frame,
            entity=EntityType.TAX_UNIT,
            variables={"age", "income", "tenure"},
        )

        assert projected["tax_unit_id"].tolist() == [100, 200]
        assert projected["age"].tolist() == [45, 65]
        assert projected["income"].tolist() == [25_000.0, 90_000.0]
        assert projected["tenure"].tolist() == [1, 2]

    def test_integrate_donor_sources_projects_spm_unit_native_blocks_when_ids_missing(
        self,
        monkeypatch,
    ):
        captured_conditions: list[tuple[str, ...]] = []
        captured_fit_rows: list[int] = []

        class FakeSynthesizer:
            def __init__(self, *, target_vars, condition_vars, **kwargs):
                _ = target_vars, kwargs
                captured_conditions.append(tuple(condition_vars))

            def fit(self, frame, *args, **kwargs):
                _ = args, kwargs
                captured_fit_rows.append(len(frame))

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["snap"] = [120.0, 0.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2],
                "hh_weight": [100.0, 120.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": ["1:1", "1:2", "2:1"],
                "household_id": [1, 1, 2],
                "relationship_to_head": [0, 2, 0],
                "age": [40, 10, 55],
                "sex": [1, 2, 1],
                "education": [3, 1, 4],
                "employment_status": [1, 0, 1],
                "income": [40_000.0, 0.0, 35_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102],
                "hh_weight": [80.0, 90.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": ["101:1", "101:2", "102:1"],
                "household_id": [101, 101, 102],
                "relationship_to_head": [0, 2, 0],
                "age": [42, 11, 57],
                "sex": [1, 2, 1],
                "education": [3, 1, 4],
                "employment_status": [1, 0, 1],
                "income": [38_000.0, 0.0, 34_000.0],
                "snap": [120.0, 120.0, 0.0],
            }
        )
        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "relationship_to_head",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="spm_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "relationship_to_head",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "snap",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert len(captured_conditions) == 1
        assert {"age", "income", "state_fips", "tenure"}.issubset(
            set(captured_conditions[0])
        )
        assert captured_fit_rows == [2]
        assert "spm_unit_id" in integration["seed_data"].columns
        assert integration["seed_data"]["snap"].tolist() == [120.0, 120.0, 0.0]

    def test_build_from_frames_rank_matches_generated_donor_values(
        self,
        monkeypatch,
    ):
        cps_households = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "hh_weight": [100.0, 120.0, 140.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        cps_persons = pd.DataFrame(
            {
                "person_id": [10, 20, 30],
                "household_id": [1, 2, 3],
                "age": [45, 19, 62],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [60_000.0, 12_000.0, 40_000.0],
            }
        )
        donor_households = pd.DataFrame(
            {
                "household_id": [101, 102, 103],
                "hh_weight": [80.0, 90.0, 110.0],
                "state_fips": [6, 36, 12],
                "tenure": [1, 2, 1],
            }
        )
        donor_persons = pd.DataFrame(
            {
                "person_id": [1001, 1002, 1003],
                "household_id": [101, 102, 103],
                "age": [44, 21, 61],
                "sex": [1, 2, 1],
                "education": [3, 2, 4],
                "employment_status": [1, 0, 1],
                "income": [58_000.0, 13_000.0, 41_000.0],
                "taxable_interest_income": [0.0, 0.0, 100.0],
            }
        )

        cps_frame = ObservationFrame(
            source=SourceDescriptor(
                name="cps_like",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: cps_households,
                EntityType.PERSON: cps_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        donor_frame = ObservationFrame(
            source=SourceDescriptor(
                name="tax_donor",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=("state_fips", "tenure"),
                        weight_column="hh_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=(
                            "household_id",
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "taxable_interest_income",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: donor_households,
                EntityType.PERSON: donor_persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )

        class FakeSynthesizer:
            def __init__(self, *args, **kwargs):
                _ = args
                _ = kwargs

            def fit(self, *args, **kwargs):
                _ = args
                _ = kwargs

            def generate(self, frame, seed=None):
                _ = seed
                result = frame.copy()
                result["taxable_interest_income"] = [1e12, -1e12, 500.0]
                return result

        monkeypatch.setattr("microplex_us.pipelines.us.Synthesizer", FakeSynthesizer)

        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=6,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        cps_input = pipeline.prepare_source_input(cps_frame)
        donor_input = pipeline.prepare_source_input(donor_frame)
        seed_data = pipeline.prepare_seed_data_from_source(cps_input)

        integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=cps_input,
            donor_inputs=[donor_input],
        )

        assert integration["seed_data"]["taxable_interest_income"].tolist() == [100.0, 0.0, 0.0]

    def test_rank_match_donor_values_preserves_zero_inflated_positive_support(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        scores = pd.Series([0.1, 0.2, 0.9, 1.0], dtype=float)
        donor_values = pd.Series([0.0, 0.0, 10.0, 20.0], dtype=float)
        donor_weights = pd.Series([1.0, 1.0, 1.0, 1.0], dtype=float)

        matched = pipeline._rank_match_donor_values(
            scores,
            donor_values=donor_values,
            donor_weights=donor_weights,
            rng=np.random.default_rng(42),
        )

        assert matched.tolist() == [0.0, 0.0, 10.0, 20.0]

    def test_rank_match_donor_values_respects_weighted_positive_rate(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=5,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        scores = pd.Series([0.1, 0.2, 0.3, 0.9, 1.0], dtype=float)
        donor_values = pd.Series([0.0, 0.0, 10.0], dtype=float)
        donor_weights = pd.Series([4.0, 4.0, 2.0], dtype=float)

        matched = pipeline._rank_match_donor_values(
            scores,
            donor_values=donor_values,
            donor_weights=donor_weights,
            rng=np.random.default_rng(42),
        )

        assert (matched > 0).sum() == 1
        assert matched.iloc[-1] > 0.0
        assert matched.iloc[:-1].eq(0.0).all()

    def test_build_from_source_provider_defaults_missing_optional_variables(self):
        households = pd.DataFrame(
            {
                "household_key": [1, 2],
                "household_weight": [125.0, 175.0],
                "region_code": [1, 2],
            }
        )
        persons = pd.DataFrame(
            {
                "person_key": [10, 11],
                "household_key": [1, 2],
                "age": [45, 19],
                "income": [60_000.0, 12_000.0],
            }
        )
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="sparse_provider",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_key",
                        variable_names=("region_code",),
                        weight_column="household_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_key",
                        variable_names=("age", "income"),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: households,
                EntityType.PERSON: persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_key",
                    child_key="household_key",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        provider = StaticSourceProvider(frame)
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )

        result = pipeline.build_from_source_provider(provider)

        assert result.seed_data["tenure"].eq(0).all()
        assert result.seed_data["employment_status"].eq(0).all()
        assert set(result.seed_data["state"]) == {"UNK"}
        assert result.seed_data["hh_weight"].sum() == pytest.approx(300.0)

    def test_build_from_source_provider_prefers_household_scoped_merge_columns(self):
        households = pd.DataFrame(
            {
                "household_key": [1, 2],
                "household_weight": [125.0, 175.0],
                "state_fips": [6, 36],
                "tenure": [1, 2],
            }
        )
        persons = pd.DataFrame(
            {
                "person_key": [10, 11],
                "household_key": [1, 2],
                "age": [45, 19],
                "income": [60_000.0, 12_000.0],
                "state_fips": [99, 99],
                "tenure": [9, 9],
            }
        )
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="overlapping_columns",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_key",
                        variable_names=("state_fips", "tenure"),
                        weight_column="household_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_key",
                        variable_names=("age", "income", "state_fips", "tenure"),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: households,
                EntityType.PERSON: persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_key",
                    child_key="household_key",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        provider = StaticSourceProvider(frame)
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )

        result = pipeline.build_from_source_provider(provider)

        assert result.seed_data["state_fips"].tolist() == [6, 36]
        assert result.seed_data["tenure"].tolist() == [1, 2]

    def test_synthesizer_uses_observed_source_coverage(self):
        households = pd.DataFrame(
            {
                "household_key": [1, 2, 3],
                "household_weight": [100.0, 120.0, 140.0],
                "region_code": [1, 2, 3],
            }
        )
        persons = pd.DataFrame(
            {
                "person_key": [10, 11, 12],
                "household_key": [1, 2, 3],
                "age": [45, 19, 62],
                "income": [60_000.0, 12_000.0, 40_000.0],
            }
        )
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="sparse_provider",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_key",
                        variable_names=("region_code",),
                        weight_column="household_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_key",
                        variable_names=("age", "income"),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: households,
                EntityType.PERSON: persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_key",
                    child_key="household_key",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        provider = StaticSourceProvider(frame)
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=3,
                synthesis_backend="synthesizer",
                calibration_backend="entropy",
                synthesizer_epochs=2,
                synthesizer_n_layers=2,
                synthesizer_hidden_dim=8,
                random_seed=5,
            )
        )

        result = pipeline.build_from_source_provider(provider)

        assert result.synthesis_metadata["condition_vars"] == ["age"]
        assert result.synthesis_metadata["target_vars"] == ["income"]
        assert result.synthesizer is not None

    def test_synthesizer_handles_state_program_proxy_condition_vars(self):
        households = pd.DataFrame(
            {
                "household_key": [1, 2, 3, 4],
                "household_weight": [100.0, 120.0, 140.0, 160.0],
                "state_fips": [6, 6, 36, 36],
                "tenure": [1, 2, 1, 2],
            }
        )
        persons = pd.DataFrame(
            {
                "person_key": [10, 11, 12, 13],
                "household_key": [1, 2, 3, 4],
                "age": [45, 19, 62, 35],
                "sex": [1, 2, 1, 2],
                "education": [4, 2, 3, 1],
                "employment_status": [1, 0, 1, 1],
                "income": [60_000.0, 12_000.0, 40_000.0, 22_000.0],
                "has_medicaid": [1.0, 0.0, 0.0, 1.0],
                "public_assistance": [0.0, 150.0, 0.0, 0.0],
                "ssi": [0.0, 0.0, 0.0, 0.0],
                "social_security": [0.0, 0.0, 900.0, 0.0],
            }
        )
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="state_program_proxy_provider",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_key",
                        variable_names=("state_fips", "tenure"),
                        weight_column="household_weight",
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_key",
                        variable_names=(
                            "age",
                            "sex",
                            "education",
                            "employment_status",
                            "income",
                            "has_medicaid",
                            "public_assistance",
                            "ssi",
                            "social_security",
                        ),
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: households,
                EntityType.PERSON: persons,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_key",
                    child_key="household_key",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
            ),
        )
        provider = StaticSourceProvider(frame)
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="synthesizer",
                calibration_backend="entropy",
                synthesizer_epochs=2,
                synthesizer_n_layers=2,
                synthesizer_hidden_dim=8,
                random_seed=7,
            )
        )

        result = pipeline.build_from_source_provider(provider)

        assert result.synthesizer is not None
        assert result.synthesis_metadata["condition_vars"] == [
            "age",
            "sex",
            "education",
            "employment_status",
            "state_fips",
            "tenure",
            "has_medicaid",
        ]
        assert len(result.synthetic_data) == 4

    def test_constant_has_medicaid_is_not_auto_promoted_to_condition_var(self):
        frame = pd.DataFrame(
            {
                "age": [25, 40, 55, 32],
                "sex": [1, 2, 1, 2],
                "education": [2, 3, 4, 1],
                "employment_status": [1, 1, 0, 1],
                "state_fips": [6, 6, 36, 36],
                "tenure": [1, 2, 1, 2],
                "income": [50_000.0, 30_000.0, 20_000.0, 80_000.0],
                "has_medicaid": [0.0, 0.0, 0.0, 0.0],
                "weight": [1.0, 1.0, 1.0, 1.0],
            }
        )
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=4,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )

        condition_vars = pipeline._resolve_synthesis_condition_vars(
            frame.columns,
            observed_frame=frame,
        )

        assert "has_medicaid" not in condition_vars

    def test_ensure_target_support_handles_bool_destination_columns(self):
        pipeline = USMicroplexPipeline(
            USMicroplexBuildConfig(
                n_synthetic=2,
                synthesis_backend="bootstrap",
                calibration_backend="entropy",
            )
        )
        synthetic_data = pd.DataFrame(
            {
                "person_id": [0, 1],
                "household_id": [0, 1],
                "state_fips": [6, 36],
                "tenure": [1, 2],
                "age": [40, 50],
                "sex": [1, 2],
                "education": [3, 4],
                "employment_status": [1, 1],
                "income": [40_000.0, 60_000.0],
                "has_medicaid": pd.Series([False, False], dtype=bool),
                "weight": [1.0, 1.0],
            }
        )
        seed_data = pd.DataFrame(
            {
                "person_id": [10, 20],
                "household_id": [10, 20],
                "state_fips": [6, 36],
                "tenure": [1, 2],
                "age": [41, 51],
                "sex": [1, 2],
                "education": [3, 4],
                "employment_status": [1, 1],
                "income": [42_000.0, 61_000.0],
                "has_medicaid": [1.0, 0.0],
                "weight": [1.0, 1.0],
            }
        )
        targets = USMicroplexTargets(
            marginal={"has_medicaid": ["1.0"]},
            continuous={},
        )

        result = pipeline.ensure_target_support(synthetic_data, seed_data, targets)

        assert pd.to_numeric(result["has_medicaid"], errors="coerce").max() == 1.0

    def test_build_from_missing_directory_raises(self, tmp_path):
        pipeline = USMicroplexPipeline(USMicroplexBuildConfig())

        with pytest.raises(FileNotFoundError, match="CPS ASEC data files not found"):
            pipeline.build_from_data_dir(tmp_path)


class TestUSMicroplexBuildResult:
    """Test build result helpers."""

    @pytest.fixture
    def result(self):
        config = USMicroplexBuildConfig(
            n_synthetic=3,
            synthesis_backend="bootstrap",
            calibration_backend="entropy",
        )
        seed = pd.DataFrame({"income": [1.0], "hh_weight": [1.0]})
        synthetic = pd.DataFrame({"income": [1.0, 2.0, 3.0], "weight": [1.0, 1.0, 1.0]})
        calibrated = synthetic.copy()
        calibrated["weight"] = [0.0, 2.0, 3.0]

        return USMicroplexBuildResult(
            config=config,
            seed_data=seed,
            synthetic_data=synthetic,
            calibrated_data=calibrated,
            targets=USMicroplexTargets(marginal={}, continuous={"income": 6.0}),
            calibration_summary={"max_error": 0.0, "mean_error": 0.0},
            synthesis_metadata={"backend": "bootstrap"},
            synthesizer=None,
            policyengine_tables=None,
        )

    def test_nonzero_weight_count(self, result):
        assert result.n_nonzero_weights == 2

    def test_total_weighted_population(self, result):
        assert result.total_weighted_population == 5.0
