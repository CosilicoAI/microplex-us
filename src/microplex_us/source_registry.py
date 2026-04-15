"""Declarative source-variable capability registry for US source providers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from microplex.core import SourceVariableCapability

from microplex_us.variables import resolve_variable_semantic_capabilities


@dataclass(frozen=True)
class SourceVariablePolicy:
    """Declarative overrides for how one source variable should be used."""

    authoritative: bool | None = None
    usable_as_condition: bool | None = None
    notes: str | None = None

    def apply(self, base: SourceVariableCapability | None = None) -> SourceVariableCapability:
        """Resolve this policy against an optional base capability."""
        base = base or SourceVariableCapability()
        return SourceVariableCapability(
            authoritative=(
                base.authoritative
                if self.authoritative is None
                else self.authoritative
            ),
            usable_as_condition=(
                base.usable_as_condition
                if self.usable_as_condition is None
                else self.usable_as_condition
            ),
            notes=self.notes if self.notes is not None else base.notes,
        )


@dataclass(frozen=True)
class SourceVariablePolicySpec:
    """Variable-usage policy for one source family."""

    source_prefixes: tuple[str, ...]
    variable_policies: Mapping[str, SourceVariablePolicy]

    def matches(self, source_name: str) -> bool:
        return any(
            source_name == prefix or source_name.startswith(f"{prefix}_")
            for prefix in self.source_prefixes
        )


PUF_SOURCE_VARIABLE_POLICY = SourceVariablePolicySpec(
    source_prefixes=("irs_soi_puf",),
    variable_policies={
        "state_fips": SourceVariablePolicy(
            authoritative=False,
            usable_as_condition=False,
            notes="PUF does not carry usable state geography in the current microdata build.",
        ),
        "tenure": SourceVariablePolicy(
            authoritative=False,
            usable_as_condition=False,
            notes="PUF tenure is scaffold filler rather than a native source attribute.",
        ),
        "income": SourceVariablePolicy(
            authoritative=False,
            usable_as_condition=False,
            notes="PUF income is a derived convenience column, not an atomic donor target.",
        ),
        "employment_status": SourceVariablePolicy(
            authoritative=False,
            usable_as_condition=False,
            notes="PUF employment status is derived from tax-line amounts, not observed directly.",
        ),
        "employment_income": SourceVariablePolicy(
            authoritative=True,
            usable_as_condition=False,
            notes="PUF wage income is source-native but should not be used as a shared donor condition.",
        ),
        "filing_status_code": SourceVariablePolicy(
            authoritative=True,
            usable_as_condition=False,
            notes="PUF filing status is source-native tax-unit structure and should survive rebuild donor integration.",
        ),
    },
)

SURVEY_DONOR_FILLER_POLICY = SourceVariablePolicySpec(
    source_prefixes=("sipp", "scf"),
    variable_policies={
        "state_fips": SourceVariablePolicy(
            authoritative=False,
            usable_as_condition=False,
            notes="SIPP/SCF donor survey adapters do not carry real state geography in the rebuild path.",
        ),
        "tenure": SourceVariablePolicy(
            authoritative=False,
            usable_as_condition=False,
            notes="SIPP/SCF donor survey adapters use filler tenure only to satisfy the household schema.",
        ),
    },
)

DEFAULT_SOURCE_VARIABLE_POLICIES: tuple[SourceVariablePolicySpec, ...] = (
    PUF_SOURCE_VARIABLE_POLICY,
    SURVEY_DONOR_FILLER_POLICY,
)


def resolve_source_variable_capabilities(
    source_name: str,
    variable_names: Iterable[str],
    *,
    policy_specs: Sequence[SourceVariablePolicySpec] = DEFAULT_SOURCE_VARIABLE_POLICIES,
) -> dict[str, SourceVariableCapability]:
    """Build per-variable capabilities for a source from declarative policy specs."""
    variables = tuple(dict.fromkeys(variable_names))
    resolved = resolve_variable_semantic_capabilities(variables)
    matching_specs = [spec for spec in policy_specs if spec.matches(source_name)]

    for variable in variables:
        capability = resolved.get(variable, SourceVariableCapability())
        for spec in matching_specs:
            policy = spec.variable_policies.get(variable)
            if policy is None:
                continue
            capability = policy.apply(capability)
        if capability != SourceVariableCapability():
            resolved[variable] = capability
    return resolved
