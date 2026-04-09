"""Tests for raw source-stage parity auditing."""

from __future__ import annotations

import h5py
import pandas as pd
import pytest
from microplex.core import (
    EntityObservation,
    EntityRelationship,
    EntityType,
    ObservationFrame,
    RelationshipCardinality,
    Shareability,
    SourceArchetype,
    SourceDescriptor,
    TimeStructure,
)

from microplex_us.pipelines.source_stage_parity import (
    SourceStageParityVariableSpec,
    build_us_cps_source_stage_parity_audit,
    build_us_puf_source_stage_parity_audit,
    build_us_source_stage_parity_audit,
    observation_frame_to_policyengine_entity_bundle,
)


def _write_period_dataset(path, data: dict[str, list | tuple], *, period: int = 2023) -> None:
    with h5py.File(path, "w") as handle:
        for variable, values in data.items():
            group = handle.create_group(variable)
            group.create_dataset(str(period), data=values)


def _write_flat_dataset(path, data: dict[str, list | tuple]) -> None:
    with h5py.File(path, "w") as handle:
        for variable, values in data.items():
            handle.create_dataset(variable, data=values)


def _build_test_frame(*, source_name: str = "test_source") -> ObservationFrame:
    households = pd.DataFrame(
        {
            "household_id": ["1", "2"],
            "household_weight": [10.0, 20.0],
            "state_fips": [1, 2],
            "county_fips": [11, 22],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": ["101", "102", "103"],
            "household_id": ["1", "1", "2"],
            "tax_unit_id": ["11", "11", "12"],
            "spm_unit_id": ["21", "21", "22"],
            "family_id": ["31", "31", "32"],
            "weight": [10.0, 10.0, 20.0],
            "age": [30, 10, 50],
            "wage_income": [100.0, 0.0, 200.0],
            "employment_income": [100.0, 0.0, 200.0],
            "is_hispanic": [1, 0, 0],
            "filing_status": ["SINGLE", "SINGLE", "JOINT"],
        }
    )
    descriptor = SourceDescriptor(
        name=source_name,
        shareability=Shareability.PUBLIC,
        time_structure=TimeStructure.REPEATED_CROSS_SECTION,
        archetype=SourceArchetype.HOUSEHOLD_INCOME,
        observations=(
            EntityObservation(
                entity=EntityType.HOUSEHOLD,
                key_column="household_id",
                variable_names=("state_fips",),
                weight_column="household_weight",
            ),
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=("age",),
                weight_column="weight",
            ),
        ),
    )
    frame = ObservationFrame(
        source=descriptor,
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
    frame.validate()
    return frame


def test_observation_frame_to_policyengine_entity_bundle_derives_group_tables() -> None:
    frame = _build_test_frame()
    bundle = observation_frame_to_policyengine_entity_bundle(frame)

    assert bundle.tax_units is not None
    assert bundle.spm_units is not None
    assert bundle.families is not None
    assert bundle.tax_units["tax_unit_id"].tolist() == ["11", "12"]
    assert bundle.tax_units["household_id"].tolist() == ["1", "2"]
    assert bundle.spm_units["spm_unit_id"].tolist() == ["21", "22"]
    assert bundle.families["family_id"].tolist() == ["31", "32"]


def test_build_us_source_stage_parity_audit_reports_weighted_alias_and_structure(
    tmp_path,
) -> None:
    reference_path = tmp_path / "reference.h5"
    _write_period_dataset(
        reference_path,
        {
            "household_id": [1, 2],
            "household_weight": [10.0, 20.0],
            "person_id": [101, 102, 103],
            "person_household_id": [1, 1, 2],
            "tax_unit_id": [11, 12],
            "person_tax_unit_id": [11, 11, 12],
            "age": [30, 10, 50],
            "state_fips": [1, 2],
            "county_fips": [11, 22],
            "employment_income": [100.0, 0.0, 200.0],
            "is_hispanic": [1, 0, 0],
        },
    )

    bundle = observation_frame_to_policyengine_entity_bundle(_build_test_frame())
    audit = build_us_source_stage_parity_audit(
        bundle,
        reference_path,
        source_id="cps_asec",
        period=2023,
        focus_variables=(
            SourceStageParityVariableSpec(
                "employment_income",
                "wage_income",
                "employment_income",
            ),
            SourceStageParityVariableSpec("state_fips", "state_fips"),
            SourceStageParityVariableSpec("age", "age", value_kind="numeric"),
        ),
    )

    assert audit["entityStructure"]["candidate"]["weighted_mean_household_size"] == pytest.approx(
        4.0 / 3.0
    )
    assert audit["householdSizeDistribution"]["candidate"]["shares"]["1"] == pytest.approx(
        2.0 / 3.0
    )
    assert audit["householdSizeDistribution"]["candidate"]["shares"]["2"] == pytest.approx(
        1.0 / 3.0
    )
    employment = audit["focusVariables"]["employment_income"]
    assert employment["candidate_variable"] == "wage_income"
    assert employment["reference_variable"] == "employment_income"
    assert employment["comparison"]["weighted_sum_ratio"] == pytest.approx(1.0)
    age = audit["focusVariables"]["age"]
    assert age["candidate"]["kind"] == "numeric"
    assert age["reference"]["kind"] == "numeric"
    assert age["comparison"]["type"] == "numeric"
    assert age["comparison"]["weighted_mean_ratio"] == pytest.approx(1.0)
    assert audit["focusVariables"]["state_fips"]["candidate_entity"] == "household"
    assert (
        audit["schema"]["entities"]["person"]["extra_in_candidate_count"] >= 1
    )


def test_build_us_source_stage_parity_audit_reads_flat_reference_h5(tmp_path) -> None:
    reference_path = tmp_path / "reference_flat.h5"
    _write_flat_dataset(
        reference_path,
        {
            "household_id": [1, 2],
            "household_weight": [10.0, 20.0],
            "person_id": [101, 102, 103],
            "person_household_id": [1, 1, 2],
            "tax_unit_id": [11, 12],
            "person_tax_unit_id": [11, 11, 12],
            "age": [30, 10, 50],
            "employment_income": [100.0, 0.0, 200.0],
            "weird_metric": [1.0, 2.0, 3.0, 4.0],
        },
    )

    bundle = observation_frame_to_policyengine_entity_bundle(_build_test_frame())
    audit = build_us_source_stage_parity_audit(
        bundle,
        reference_path,
        source_id="cps_asec",
        period=2023,
        focus_variables=(
            SourceStageParityVariableSpec(
                "employment_income",
                "wage_income",
                "employment_income",
            ),
        ),
    )

    assert audit["entityStructure"]["reference"]["weighted_mean_household_size"] == pytest.approx(
        4.0 / 3.0
    )
    assert (
        audit["focusVariables"]["employment_income"]["comparison"]["weighted_sum_ratio"]
        == pytest.approx(1.0)
    )


def test_build_us_cps_source_stage_parity_audit_uses_provider_frame(
    monkeypatch,
    tmp_path,
) -> None:
    reference_path = tmp_path / "reference.h5"
    _write_period_dataset(
        reference_path,
        {
            "household_id": [1, 2],
            "household_weight": [10.0, 20.0],
            "person_id": [101, 102, 103],
            "person_household_id": [1, 1, 2],
            "tax_unit_id": [11, 12],
            "person_tax_unit_id": [11, 11, 12],
            "age": [30, 10, 50],
        },
    )

    monkeypatch.setattr(
        "microplex_us.pipelines.source_stage_parity.CPSASECSourceProvider.load_frame",
        lambda self, query=None: _build_test_frame(source_name="mock_cps"),
    )
    audit = build_us_cps_source_stage_parity_audit(
        reference_path,
        year=2023,
        download=False,
        sample_n=5,
        random_seed=7,
        focus_variables=(SourceStageParityVariableSpec("age", "age"),),
    )

    assert audit["sourceId"] == "cps_asec"
    assert audit["candidate"]["metadata"]["candidateSourceName"] == "mock_cps"
    assert audit["candidate"]["metadata"]["providerFilters"]["sample_n"] == 5


def test_build_us_puf_source_stage_parity_audit_uses_provider_frame(
    monkeypatch,
    tmp_path,
) -> None:
    reference_path = tmp_path / "reference.h5"
    _write_period_dataset(
        reference_path,
        {
            "household_id": [1, 2],
            "household_weight": [10.0, 20.0],
            "person_id": [101, 102, 103],
            "person_household_id": [1, 1, 2],
            "tax_unit_id": [11, 12],
            "person_tax_unit_id": [11, 11, 12],
            "employment_income": [100.0, 0.0, 200.0],
        },
        period=2024,
    )

    monkeypatch.setattr(
        "microplex_us.pipelines.source_stage_parity.PUFSourceProvider.load_frame",
        lambda self, query=None: _build_test_frame(source_name="mock_puf"),
    )
    audit = build_us_puf_source_stage_parity_audit(
        reference_path,
        target_year=2024,
        sample_n=8,
        random_seed=3,
        focus_variables=(
            SourceStageParityVariableSpec(
                "employment_income",
                "employment_income",
            ),
        ),
    )

    assert audit["sourceId"] == "irs_soi_puf"
    assert audit["candidate"]["metadata"]["candidateSourceName"] == "mock_puf"
    assert audit["candidate"]["metadata"]["providerFilters"]["sample_n"] == 8
