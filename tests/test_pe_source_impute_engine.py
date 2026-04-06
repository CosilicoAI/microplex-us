"""Tests for the PE source-impute block engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from microplex.core import EntityType

from microplex_us.pe_source_impute_engine import (
    PE_SOURCE_IMPUTE_BLOCK_ENGINE,
    PESourceImputeBlockRunRequest,
    PESourceImputeConditionedBlockRunRequest,
    PESourceImputePreparedBlockInputs,
)
from microplex_us.variables import DonorImputationBlockSpec


def test_prepare_condition_surface_and_predictors_for_acs_block() -> None:
    donor_frame = pd.DataFrame(
        {
            "household_id": [1, 1, 2],
            "age": [45, 12, 70],
            "sex": [1, 2, 2],
            "is_head": [1.0, 0.0, 1.0],
            "tenure": [1, 1, 2],
            "employment_income": [50_000.0, 0.0, 12_000.0],
            "self_employment_income": [5_000.0, 0.0, 0.0],
            "gross_social_security": [0.0, 0.0, 20_000.0],
            "taxable_pension_income": [0.0, 0.0, 15_000.0],
            "state_fips": [6, 6, 36],
            "rent": [1_200.0, 0.0, 950.0],
        }
    )
    current_frame = donor_frame.copy()

    surface = PE_SOURCE_IMPUTE_BLOCK_ENGINE.prepare_condition_surface(
        donor_frame=donor_frame,
        current_frame=current_frame,
        donor_source_name="acs_2022",
        donor_block=("rent",),
    )

    assert surface is not None
    assert surface.spec.key == "acs"
    assert surface.donor_frame["is_household_head"].tolist() == [1.0, 0.0, 1.0]
    assert surface.donor_frame["pension_income"].tolist() == [0.0, 0.0, 15_000.0]
    assert surface.compatible_predictors(
        compatibility_fn=lambda donor, current: donor.notna().all() and current.notna().all(),
    ) == list(surface.spec.predictors)


def test_prepare_condition_surface_returns_none_for_unmapped_block() -> None:
    frame = pd.DataFrame({"tip_income": [1.0], "employment_income": [10.0]})

    surface = PE_SOURCE_IMPUTE_BLOCK_ENGINE.prepare_condition_surface(
        donor_frame=frame,
        current_frame=frame,
        donor_source_name="unknown_source",
        donor_block=("tip_income",),
    )

    assert surface is None


def test_run_prepared_block_executes_fit_generate_and_assignment() -> None:
    donor_frame = pd.DataFrame(
        {
            "household_id": [1, 1, 2],
            "age": [45, 12, 70],
            "sex": [1, 2, 2],
            "is_head": [1.0, 0.0, 1.0],
            "tenure": [1, 1, 2],
            "employment_income": [50_000.0, 0.0, 12_000.0],
            "self_employment_income": [5_000.0, 0.0, 0.0],
            "gross_social_security": [0.0, 0.0, 20_000.0],
            "taxable_pension_income": [0.0, 0.0, 15_000.0],
            "state_fips": [6, 6, 36],
            "rent": [1_200.0, 0.0, 950.0],
            "hh_weight": [100.0, 100.0, 120.0],
        }
    )
    current_frame = donor_frame.drop(columns=["rent"]).copy()
    surface = PE_SOURCE_IMPUTE_BLOCK_ENGINE.prepare_condition_surface(
        donor_frame=donor_frame,
        current_frame=current_frame,
        donor_source_name="acs_2022",
        donor_block=("rent",),
    )

    class _FakeImputer:
        def fit(self, frame, *, weight_col, **kwargs):
            self.fit_frame = frame
            self.weight_col = weight_col
            self.fit_kwargs = kwargs

        def generate(self, frame, *, seed):
            self.generate_frame = frame
            self.seed = seed
            return pd.DataFrame({"rent": [400.0, 100.0, 300.0]}, index=frame.index)

    fake_imputer = _FakeImputer()
    built: dict[str, object] = {}

    def _build_imputer(condition_vars, target_vars):
        built["condition_vars"] = tuple(condition_vars)
        built["target_vars"] = tuple(target_vars)
        return fake_imputer

    def _rank_match(scores, *, donor_values, donor_weights, rng, strategy):
        built["rank_scores"] = scores.tolist()
        built["rank_donor_values"] = donor_values.tolist()
        built["rank_strategy"] = strategy
        return scores.astype(float)

    assert surface is not None
    result = PE_SOURCE_IMPUTE_BLOCK_ENGINE.run_prepared_block(
        surface=surface,
        request=PESourceImputeBlockRunRequest(
            donor_block_spec=DonorImputationBlockSpec(
                model_variables=("rent",),
                restored_variables=("rent",),
            ),
            donor_fit_source=donor_frame,
            current_generation_source=current_frame,
            current_frame=current_frame,
            entity_key=None,
        ),
        build_imputer=_build_imputer,
        rank_match=_rank_match,
        compatibility_fn=lambda donor, current: donor.notna().all() and current.notna().all(),
        fit_kwargs={"epochs": 5, "batch_size": 32, "learning_rate": 0.01, "verbose": False},
        seed=17,
        rng=np.random.default_rng(0),
    )

    assert result is not None
    assert built["target_vars"] == ("rent",)
    assert built["condition_vars"] == tuple(surface.spec.predictors)
    assert fake_imputer.weight_col == "weight"
    assert fake_imputer.seed == 17
    assert result.updated_frame["rent"].tolist() == [400.0, 100.0, 300.0]
    assert result.integrated_variables == ("rent",)


def test_run_conditioned_block_executes_generic_donor_path() -> None:
    donor_frame = pd.DataFrame(
        {
            "age": [45, 12, 70],
            "state_fips": [6, 6, 36],
            "rent": [1_200.0, 0.0, 950.0],
            "hh_weight": [100.0, 100.0, 120.0],
        }
    )
    current_frame = donor_frame.drop(columns=["rent"]).copy()

    class _FakeImputer:
        def fit(self, frame, *, weight_col, **kwargs):
            self.fit_frame = frame
            self.weight_col = weight_col
            self.fit_kwargs = kwargs

        def generate(self, frame, *, seed):
            self.generate_frame = frame
            self.seed = seed
            return pd.DataFrame({"rent": [500.0, 200.0, 350.0]}, index=frame.index)

    fake_imputer = _FakeImputer()
    built: dict[str, object] = {}

    def _build_imputer(condition_vars, target_vars):
        built["condition_vars"] = tuple(condition_vars)
        built["target_vars"] = tuple(target_vars)
        return fake_imputer

    def _rank_match(scores, *, donor_values, donor_weights, rng, strategy):
        built["rank_scores"] = scores.tolist()
        built["rank_donor_values"] = donor_values.tolist()
        built["rank_strategy"] = strategy
        return scores.astype(float)

    result = PE_SOURCE_IMPUTE_BLOCK_ENGINE.run_conditioned_block(
        request=PESourceImputeConditionedBlockRunRequest(
            block_request=PESourceImputeBlockRunRequest(
                donor_block_spec=DonorImputationBlockSpec(
                    model_variables=("rent",),
                    restored_variables=("rent",),
                ),
                donor_fit_source=donor_frame,
                current_generation_source=current_frame,
                current_frame=current_frame,
                entity_key=None,
            ),
            donor_condition_source=donor_frame,
            current_condition_source=current_frame,
            condition_vars=("age", "state_fips"),
        ),
        build_imputer=_build_imputer,
        rank_match=_rank_match,
        fit_kwargs={"epochs": 5, "batch_size": 32, "learning_rate": 0.01, "verbose": False},
        seed=23,
        rng=np.random.default_rng(0),
    )

    assert result is not None
    assert built["target_vars"] == ("rent",)
    assert built["condition_vars"] == ("age", "state_fips")
    assert fake_imputer.weight_col == "weight"
    assert fake_imputer.seed == 23
    assert result.updated_frame["rent"].tolist() == [500.0, 200.0, 350.0]
    assert result.integrated_variables == ("rent",)


def test_prepare_block_inputs_projects_entity_and_preserves_compatible_shared_vars() -> None:
    donor_seed = pd.DataFrame(
        {
            "person_id": [10, 11, 20],
            "household_id": [1, 1, 2],
            "age": [45, 12, 70],
            "state_fips": [6, 6, 36],
            "tenure": [1, 1, 2],
            "rent": [1200.0, 0.0, 950.0],
            "hh_weight": [100.0, 100.0, 120.0],
        }
    )
    current_frame = donor_seed.drop(columns=["rent"]).copy()

    def _can_project(current, donor, entity):
        return entity is EntityType.HOUSEHOLD

    def _project(frame, *, entity, variables):
        assert entity is EntityType.HOUSEHOLD
        columns = [
            "household_id",
            *sorted(variable for variable in variables if variable != "household_id"),
        ]
        keep_columns = list(columns)
        if "person_id" in frame.columns and "person_id" not in columns:
            columns.append("person_id")
        projected = frame[columns].copy().sort_values(
            ["household_id", "person_id"],
            kind="mergesort",
        )
        aggregations = {
            column: ("first" if column in {"state_fips", "tenure", "hh_weight", "person_id"} else "max")
            for column in projected.columns
            if column != "household_id"
        }
        projected = projected.groupby("household_id", as_index=False).agg(aggregations)
        return projected[keep_columns]

    prepared = PE_SOURCE_IMPUTE_BLOCK_ENGINE.prepare_block_inputs(
        donor_seed=donor_seed,
        current_frame=current_frame,
        shared_vars=["age", "state_fips", "tenure"],
        donor_block_spec=DonorImputationBlockSpec(
            model_variables=("rent",),
            restored_variables=("rent",),
            native_entity=EntityType.HOUSEHOLD,
            condition_entities=(EntityType.HOUSEHOLD,),
        ),
        donor_source_name=None,
        prepare_pe_surface=False,
        can_project_to_entity=_can_project,
        project_frame_to_entity=_project,
        entity_key_fn=lambda entity: "household_id" if entity is EntityType.HOUSEHOLD else None,
    )

    assert isinstance(prepared, PESourceImputePreparedBlockInputs)
    assert prepared.entity_key == "household_id"
    assert prepared.shared_vars_for_block == ("age", "state_fips", "tenure")
    assert prepared.condition_surface is None
    assert prepared.donor_fit_source["household_id"].tolist() == [1, 2]
    assert prepared.current_generation_source.columns.tolist() == [
        "household_id",
        "age",
        "state_fips",
        "tenure",
    ]


def test_prepare_block_inputs_builds_condition_surface_when_requested() -> None:
    donor_seed = pd.DataFrame(
        {
            "household_id": [1, 1, 2],
            "age": [45, 12, 70],
            "sex": [1, 2, 2],
            "is_head": [1.0, 0.0, 1.0],
            "tenure": [1, 1, 2],
            "employment_income": [50_000.0, 0.0, 12_000.0],
            "self_employment_income": [5_000.0, 0.0, 0.0],
            "gross_social_security": [0.0, 0.0, 20_000.0],
            "taxable_pension_income": [0.0, 0.0, 15_000.0],
            "state_fips": [6, 6, 36],
            "rent": [1_200.0, 0.0, 950.0],
        }
    )
    current_frame = donor_seed.copy()

    prepared = PE_SOURCE_IMPUTE_BLOCK_ENGINE.prepare_block_inputs(
        donor_seed=donor_seed,
        current_frame=current_frame,
        shared_vars=["age", "sex", "state_fips"],
        donor_block_spec=DonorImputationBlockSpec(
            model_variables=("rent",),
            restored_variables=("rent",),
        ),
        donor_source_name="acs_2022",
        prepare_pe_surface=True,
        can_project_to_entity=lambda current, donor, entity: False,
        project_frame_to_entity=lambda frame, *, entity, variables: frame,
        entity_key_fn=lambda entity: None,
    )

    assert prepared.condition_surface is not None
    assert prepared.condition_surface.spec.key == "acs"
    assert prepared.shared_vars_for_block == ("age", "sex", "state_fips")
