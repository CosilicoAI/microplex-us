"""Shared PE source-impute donor block specs loaded from manifest data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from microplex.core import SourceArchetype


@dataclass(frozen=True)
class PESourceImputeBlockSpec:
    """Declarative contract for one PE donor-survey block."""

    key: str
    survey_name: str
    block_name: str | None
    default_year: int
    archetype: SourceArchetype | None
    household_variables: tuple[str, ...]
    person_variables: tuple[str, ...]
    target_variables: tuple[str, ...]
    predictors: tuple[str, ...]

    @property
    def descriptor_name(self) -> str:
        if self.block_name is None:
            return self.survey_name
        return f"{self.survey_name}_{self.block_name}"

    def source_name(self, year: int) -> str:
        return f"{self.descriptor_name}_{year}"


def _manifest_path() -> Path:
    return Path(__file__).resolve().parent / "manifests" / "pe_source_impute_blocks.json"


def _archetype_from_name(value: str | None) -> SourceArchetype | None:
    if value is None:
        return None
    return SourceArchetype(value)


def _spec_from_payload(key: str, payload: dict[str, Any]) -> PESourceImputeBlockSpec:
    return PESourceImputeBlockSpec(
        key=key,
        survey_name=str(payload["survey_name"]),
        block_name=payload.get("block_name"),
        default_year=int(payload["default_year"]),
        archetype=_archetype_from_name(payload.get("archetype")),
        household_variables=tuple(payload["household_variables"]),
        person_variables=tuple(payload["person_variables"]),
        target_variables=tuple(payload["target_variables"]),
        predictors=tuple(payload["predictors"]),
    )


@cache
def load_pe_source_impute_block_specs() -> dict[str, PESourceImputeBlockSpec]:
    """Load the PE donor-block spec manifest."""
    with _manifest_path().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    blocks = payload.get("blocks", {})
    return {
        key: _spec_from_payload(key, value)
        for key, value in blocks.items()
    }


def get_pe_source_impute_block_spec(key: str) -> PESourceImputeBlockSpec:
    """Return one named PE donor-block spec."""
    specs = load_pe_source_impute_block_specs()
    try:
        return specs[key]
    except KeyError as error:
        available = ", ".join(sorted(specs))
        raise KeyError(f"Unknown PE source-impute block '{key}'. Expected one of: {available}") from error


def resolve_sipp_source_impute_block_spec(block: str) -> PESourceImputeBlockSpec:
    """Resolve one SIPP donor block by short block name."""
    return get_pe_source_impute_block_spec(f"sipp_{block}")


def resolve_pe_source_impute_block_key(
    *,
    donor_source_name: str | None,
    donor_block: tuple[str, ...],
) -> str | None:
    """Map a donor source name and target block to one manifest block key."""
    normalized_name = (donor_source_name or "").strip().lower()
    block_set = set(donor_block)
    for key, spec in load_pe_source_impute_block_specs().items():
        if spec.survey_name not in normalized_name:
            continue
        if block_set <= set(spec.target_variables):
            return key
    return None
