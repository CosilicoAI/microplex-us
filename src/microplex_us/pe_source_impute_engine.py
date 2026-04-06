"""Execution helpers for PE source-impute donor blocks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from microplex_us.pe_source_impute_specs import (
    PESourceImputeBlockSpec,
    load_pe_source_impute_block_specs,
    prepare_pe_source_impute_condition_frame,
    resolve_pe_source_impute_block_key,
)

DonorConditionCompatibilityFn = Callable[[pd.Series, pd.Series], bool]


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


PE_SOURCE_IMPUTE_BLOCK_ENGINE = PESourceImputeBlockEngine.default()
