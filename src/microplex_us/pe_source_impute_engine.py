"""Execution helpers for PE source-impute donor blocks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microplex_us.pe_source_impute_specs import (
    PESourceImputeBlockSpec,
    load_pe_source_impute_block_specs,
    prepare_pe_source_impute_condition_frame,
    resolve_pe_source_impute_block_key,
)
from microplex_us.variables import (
    DonorImputationBlockSpec,
    apply_donor_variable_semantics,
)

DonorConditionCompatibilityFn = Callable[[pd.Series, pd.Series], bool]
DonorImputerBuilderFn = Callable[[list[str], tuple[str, ...]], object]
DonorRankMatcherFn = Callable[..., pd.Series]


@dataclass(frozen=True)
class PESourceImputeConditionSurface:
    """Prepared donor/current condition frames for one PE donor block."""

    spec: PESourceImputeBlockSpec
    donor_frame: pd.DataFrame
    current_frame: pd.DataFrame

    def compatible_predictors(
        self,
        *,
        compatibility_fn: DonorConditionCompatibilityFn,
    ) -> list[str]:
        """Return the manifest predictor surface filtered to compatible columns."""
        return [
            variable
            for variable in self.spec.predictors
            if variable in self.donor_frame.columns
            and variable in self.current_frame.columns
            and compatibility_fn(self.donor_frame[variable], self.current_frame[variable])
        ]


@dataclass(frozen=True)
class PESourceImputeBlockRunRequest:
    """Inputs needed to execute one PE donor block once its surface is resolved."""

    donor_block_spec: DonorImputationBlockSpec
    donor_fit_source: pd.DataFrame
    current_generation_source: pd.DataFrame
    current_frame: pd.DataFrame
    entity_key: str | None


@dataclass(frozen=True)
class PESourceImputeBlockRunResult:
    """Updated seed frame after executing one PE donor block."""

    updated_frame: pd.DataFrame
    integrated_variables: tuple[str, ...]
    condition_vars: tuple[str, ...]


@dataclass(frozen=True)
class PESourceImputeBlockEngine:
    """Centralized resolver for PE donor-block specs and condition surfaces."""

    specs: dict[str, PESourceImputeBlockSpec]

    @classmethod
    def default(cls) -> PESourceImputeBlockEngine:
        return cls(specs=load_pe_source_impute_block_specs())

    def resolve_spec(
        self,
        *,
        donor_source_name: str | None,
        donor_block: tuple[str, ...],
    ) -> PESourceImputeBlockSpec | None:
        """Resolve one donor source/block pair to a PE source-impute spec."""
        key = resolve_pe_source_impute_block_key(
            donor_source_name=donor_source_name,
            donor_block=donor_block,
        )
        if key is None:
            return None
        return self.specs[key]

    def prepare_condition_surface(
        self,
        *,
        donor_frame: pd.DataFrame,
        current_frame: pd.DataFrame,
        donor_source_name: str | None,
        donor_block: tuple[str, ...],
    ) -> PESourceImputeConditionSurface | None:
        """Prepare the PE prespecified donor/current condition frames for one block."""
        spec = self.resolve_spec(
            donor_source_name=donor_source_name,
            donor_block=donor_block,
        )
        if spec is None:
            return None
        return PESourceImputeConditionSurface(
            spec=spec,
            donor_frame=prepare_pe_source_impute_condition_frame(donor_frame, spec),
            current_frame=prepare_pe_source_impute_condition_frame(current_frame, spec),
        )

    def run_prepared_block(
        self,
        *,
        surface: PESourceImputeConditionSurface,
        request: PESourceImputeBlockRunRequest,
        build_imputer: DonorImputerBuilderFn,
        rank_match: DonorRankMatcherFn,
        compatibility_fn: DonorConditionCompatibilityFn,
        fit_kwargs: dict[str, int | float | bool],
        seed: int,
        rng: np.random.Generator,
    ) -> PESourceImputeBlockRunResult | None:
        """Run one PE prespecified donor block from fit through matched assignment."""
        condition_vars = surface.compatible_predictors(
            compatibility_fn=compatibility_fn,
        )
        if not condition_vars:
            return None

        fit_frame = surface.donor_frame[
            condition_vars + list(request.donor_block_spec.model_variables) + ["hh_weight"]
        ].copy()
        fit_frame = fit_frame.rename(columns={"hh_weight": "weight"})
        imputer = build_imputer(
            condition_vars=condition_vars,
            target_vars=request.donor_block_spec.model_variables,
        )
        imputer.fit(
            fit_frame,
            weight_col="weight",
            **fit_kwargs,
        )
        generated = imputer.generate(
            surface.current_frame[condition_vars].copy(),
            seed=seed,
        )
        updated = request.current_frame.copy()
        for variable in request.donor_block_spec.model_variables:
            donor_support = (
                pd.to_numeric(request.donor_fit_source[variable], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            generated_scores = pd.to_numeric(
                generated[variable],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)
            if donor_support.empty:
                updated[variable] = generated_scores.fillna(0.0).astype(float)
                continue
            donor_weights = pd.to_numeric(
                request.donor_fit_source.loc[donor_support.index, "hh_weight"],
                errors="coerce",
            ).fillna(0.0)
            matched_values = rank_match(
                generated_scores.fillna(float(donor_support.median())).astype(float),
                donor_values=donor_support.astype(float),
                donor_weights=donor_weights.astype(float),
                rng=rng,
                strategy=request.donor_block_spec.strategy_for(variable),
            )
            if (
                request.entity_key is not None
                and request.entity_key in request.current_generation_source.columns
            ):
                entity_values = pd.Series(
                    matched_values.to_numpy(dtype=float),
                    index=request.current_generation_source[request.entity_key].to_numpy(),
                    dtype=float,
                )
                updated[variable] = pd.to_numeric(
                    updated[request.entity_key].map(entity_values),
                    errors="coerce",
                ).fillna(0.0)
            else:
                updated[variable] = matched_values
        if request.donor_block_spec.restore_frame is not None:
            updated = request.donor_block_spec.restore_frame(updated)
        updated = apply_donor_variable_semantics(
            updated,
            request.donor_block_spec.restored_variables,
        )
        return PESourceImputeBlockRunResult(
            updated_frame=updated,
            integrated_variables=request.donor_block_spec.restored_variables,
            condition_vars=tuple(condition_vars),
        )


PE_SOURCE_IMPUTE_BLOCK_ENGINE = PESourceImputeBlockEngine.default()
