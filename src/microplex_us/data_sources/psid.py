"""PSID (Panel Study of Income Dynamics) data source for microplex.

Provides loading, processing, and transition rate extraction from PSID panel data.
PSID is the longest-running longitudinal household survey in the world (1968-present),
making it ideal for calibrating demographic transition models.

Key features:
- Load PSID panel data via the `psid` Python package
- Extract empirical transition rates (marriage, divorce, etc.)
- Calibrate microplex transition models with PSID data
- Use as a source in MultiSourceFusion for coverage evaluation

Example:
    >>> from microplex.data_sources.psid import load_psid_panel, extract_transition_rates
    >>>
    >>> # Load PSID data
    >>> dataset = load_psid_panel(data_dir="./psid_data")
    >>>
    >>> # Extract transition rates for model calibration
    >>> transitions = psid.get_household_transitions(dataset.panel)
    >>> rates = extract_transition_rates(transitions)
    >>>
    >>> # Calibrate marriage model
    >>> from microplex.transitions import MarriageTransition
>>> marriage_rates = calibrate_marriage_rates(rates["marriage_by_age"])
>>> model = MarriageTransition(base_rates={"male": 0.05, "female": 0.06})
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from microplex.core import (
    EntityObservation,
    EntityRelationship,
    EntityType,
    ObservationFrame,
    RelationshipCardinality,
    Shareability,
    SourceArchetype,
    SourceDescriptor,
    SourceQuery,
    TimeStructure,
    apply_source_query,
)

# Variable mapping from PSID names to microplex conventions
PSID_TO_MICROPLEX_VARS = {
    # Demographics
    "age": "age",
    "sex": "is_male",  # Will need transformation (1=male in PSID)
    "marital_status": "marital_status",
    "education": "education",
    "race": "race",

    # Income
    "total_family_income": "total_income",
    "head_labor_income": "head_labor_income",
    "wife_labor_income": "spouse_labor_income",

    # Wealth
    "total_wealth": "total_wealth",

    # Identifiers
    "person_id": "person_id",
    "interview_number": "household_id",
    "year": "year",
    "relationship": "relationship",
}


@dataclass
class PSIDDataset:
    """Container for PSID panel data.

    Attributes:
        persons: DataFrame with person-year observations
        source: Data source identifier (path or "mock")
        panel: Optional Panel object from psid package
    """

    persons: pd.DataFrame
    source: str
    panel: Any = None

    @property
    def n_persons(self) -> int:
        """Number of unique persons in dataset."""
        if "person_id" in self.persons.columns:
            return self.persons["person_id"].nunique()
        return 0

    @property
    def n_observations(self) -> int:
        """Total number of person-year observations."""
        return len(self.persons)

    @property
    def years(self) -> list[int]:
        """List of years in the dataset."""
        if "year" in self.persons.columns:
            return sorted(self.persons["year"].unique().tolist())
        return []

    def summary(self) -> dict:
        """Return summary statistics."""
        return {
            "n_persons": self.n_persons,
            "n_observations": self.n_observations,
            "years": self.years,
            "source": self.source,
        }


def load_psid_panel(
    data_dir: str | Path,
    years: list[int] | None = None,
    family_vars: dict[str, str] | None = None,
    individual_vars: dict[str, str] | None = None,
    sample: str | None = None,
) -> PSIDDataset:
    """Load PSID panel data using the psid package.

    Args:
        data_dir: Directory containing PSID data files
        years: List of survey years to include (None = all available)
        family_vars: Dict mapping variable names to crosswalk lookups
        individual_vars: Dict mapping individual-level variables
        sample: Sample type filter ("SRC", "SEO", "IMMIGRANT", or None for all)

    Returns:
        PSIDDataset with loaded panel data

    Raises:
        FileNotFoundError: If data_dir doesn't exist
        ValueError: If psid package is not installed
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"PSID data directory not found: {data_dir}")

    try:
        import psid
    except ImportError:
        raise ValueError(
            "psid package not installed. Install with: pip install psid"
        )

    # Build panel using psid package
    panel = psid.build_panel(
        data_dir=str(data_dir),
        years=years,
        family_vars=family_vars,
        individual_vars=individual_vars,
        sample=sample,
    )

    # Convert to DataFrame with microplex variable names
    df = panel.data.copy()

    # Rename columns to microplex conventions
    rename_map = {}
    for psid_name, microplex_name in PSID_TO_MICROPLEX_VARS.items():
        if psid_name in df.columns:
            rename_map[psid_name] = microplex_name

    df = df.rename(columns=rename_map)

    # Transform sex to is_male boolean
    if "is_male" in df.columns:
        df["is_male"] = df["is_male"] == 1

    return PSIDDataset(
        persons=df,
        source=str(data_dir),
        panel=panel,
    )


def extract_transition_rates(
    transitions_df: pd.DataFrame,
    transition_types: list[str] | None = None,
) -> dict[str, float]:
    """Extract overall transition rates from PSID transition data.

    Args:
        transitions_df: DataFrame from psid.get_household_transitions()
        transition_types: Types to extract (None = all available)

    Returns:
        Dict mapping transition type to annual probability
    """
    if "type" not in transitions_df.columns:
        raise ValueError("transitions_df must have 'type' column")

    total = len(transitions_df)
    if total == 0:
        return {}

    counts = transitions_df["type"].value_counts()

    if transition_types is None:
        transition_types = counts.index.tolist()

    rates = {}
    for t_type in transition_types:
        if t_type in counts.index:
            rates[t_type] = counts[t_type] / total
        else:
            rates[t_type] = 0.0

    return rates


def get_age_specific_rates(
    transitions_df: pd.DataFrame,
    transition_type: str,
    age_bins: list[tuple[int, int]],
    age_col: str = "age_from",
) -> dict[tuple[int, int], float]:
    """Extract age-specific transition rates.

    Args:
        transitions_df: DataFrame from psid.get_household_transitions()
        transition_type: Type of transition (e.g., "marriage", "divorce")
        age_bins: List of (age_min, age_max) tuples
        age_col: Column name for age

    Returns:
        Dict mapping age range to rate
    """
    if age_col not in transitions_df.columns:
        return {}

    rates = {}

    for age_min, age_max in age_bins:
        # Filter to age bin
        mask = (transitions_df[age_col] >= age_min) & (transitions_df[age_col] <= age_max)
        bin_data = transitions_df[mask]

        if len(bin_data) == 0:
            rates[(age_min, age_max)] = 0.0
            continue

        # Count transitions of specified type
        type_count = (bin_data["type"] == transition_type).sum()
        rates[(age_min, age_max)] = type_count / len(bin_data)

    return rates


def calibrate_marriage_rates(
    psid_rates: dict[tuple[int, int], float],
    gender_adjustment: dict[str, float] | None = None,
) -> dict[tuple[int, int], float]:
    """Convert PSID-derived rates to MarriageTransition format.

    Args:
        psid_rates: Dict from get_age_specific_rates() for marriage
        gender_adjustment: Optional {"male": factor, "female": factor}

    Returns:
        Dict compatible with MarriageTransition base_rates
    """
    # PSID rates are already in the right format: (age_min, age_max) -> rate
    calibrated = {}

    for age_range, rate in psid_rates.items():
        # Ensure rate is a valid probability
        calibrated[age_range] = float(np.clip(rate, 0.0, 1.0))

    return calibrated


def calibrate_divorce_rates(
    psid_rates: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    """Convert PSID-derived rates to DivorceTransition format.

    Args:
        psid_rates: Dict from get_age_specific_rates() for divorce

    Returns:
        Dict compatible with DivorceTransition age_effects
    """
    # Same format as marriage rates
    calibrated = {}

    for age_range, rate in psid_rates.items():
        calibrated[age_range] = float(np.clip(rate, 0.0, 1.0))

    return calibrated


def create_psid_fusion_source(
    dataset: PSIDDataset,
    source_vars: list[str] | None = None,
) -> dict:
    """Create configuration for adding PSID to MultiSourceFusion.

    Args:
        dataset: PSIDDataset from load_psid_panel()
        source_vars: Variables to include (None = common set)

    Returns:
        Dict with parameters for fusion.add_source()
    """
    if source_vars is None:
        # Default to variables commonly available in PSID
        source_vars = ["age", "total_income"]

        # Add others if present
        optional = ["is_male", "marital_status", "education", "total_wealth"]
        for var in optional:
            if var in dataset.persons.columns:
                source_vars.append(var)

    # Determine number of periods per person
    if "year" in dataset.persons.columns and "person_id" in dataset.persons.columns:
        periods_per_person = dataset.persons.groupby("person_id")["year"].nunique()
        n_periods = int(periods_per_person.median())
    else:
        n_periods = 1

    return {
        "name": "psid",
        "data": dataset.persons,
        "source_vars": source_vars,
        "n_periods": n_periods,
        "person_id_col": "person_id",
        "period_col": "year",
    }


def _sample_psid_households(
    persons: pd.DataFrame,
    *,
    sample_n: int | None,
    random_seed: int,
) -> pd.DataFrame:
    """Sample linked households after selecting one survey year."""
    if sample_n is None:
        return persons.reset_index(drop=True)
    household_ids = persons["household_id"].drop_duplicates()
    if sample_n >= len(household_ids):
        return persons.reset_index(drop=True)
    sampled_households = household_ids.sample(
        n=sample_n,
        random_state=random_seed,
        replace=False,
    )
    return persons[persons["household_id"].isin(set(sampled_households))].reset_index(
        drop=True
    )


def _normalize_psid_cross_section(
    persons: pd.DataFrame,
    *,
    survey_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project PSID panel data to one cross-sectional frame."""
    cross_section = persons.copy()
    if "year" in cross_section.columns:
        cross_section = cross_section[cross_section["year"] == survey_year].copy()
    if cross_section.empty:
        raise ValueError(f"PSID data has no observations for survey year {survey_year}")

    raw_household = cross_section["household_id"].astype(str)
    raw_person = cross_section["person_id"].astype(str)
    cross_section["household_id"] = f"{survey_year}:" + raw_household
    cross_section["person_id"] = f"{survey_year}:" + raw_person
    cross_section["year"] = survey_year

    def numeric_series(column: str, default: float = 0.0) -> pd.Series:
        if column in cross_section.columns:
            return pd.to_numeric(cross_section[column], errors="coerce")
        return pd.Series(default, index=cross_section.index, dtype=float)

    income = numeric_series("total_income", default=np.nan)
    if income.isna().all():
        income = numeric_series("head_labor_income").fillna(0) + numeric_series(
            "spouse_labor_income"
        ).fillna(0)
    cross_section["income"] = income.fillna(0.0).astype(float)
    if "is_male" in cross_section.columns:
        cross_section["sex"] = np.where(cross_section["is_male"], 1, 2)
    else:
        cross_section["sex"] = 0
    cross_section["education"] = numeric_series("education").fillna(0).astype(int)
    cross_section["employment_status"] = (
        cross_section["income"].astype(float) > 0
    ).astype(int)
    cross_section["age"] = numeric_series("age").fillna(0).astype(int)
    cross_section["weight"] = numeric_series("weight", default=1.0).fillna(1.0)

    households = (
        cross_section.groupby("household_id", as_index=False)
        .agg({"year": "first", "weight": "sum"})
        .rename(columns={"weight": "household_weight"})
    )
    households["state_fips"] = 0
    households["tenure"] = 0

    return households, cross_section


def _build_psid_observation_frame(
    *,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    source_name: str,
    shareability: Shareability,
) -> ObservationFrame:
    """Build an observation frame from a PSID cross section."""
    descriptor = SourceDescriptor(
        name=source_name,
        shareability=shareability,
        time_structure=TimeStructure.PANEL,
        archetype=SourceArchetype.LONGITUDINAL_SOCIOECONOMIC,
        observations=(
            EntityObservation(
                entity=EntityType.HOUSEHOLD,
                key_column="household_id",
                variable_names=("state_fips", "tenure"),
                weight_column="household_weight",
                period_column="year",
            ),
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=tuple(
                    column
                    for column in persons.columns
                    if column not in {"person_id", "household_id", "weight", "year"}
                ),
                weight_column="weight",
                period_column="year",
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


@dataclass
class PSIDSourceProvider:
    """Source-provider wrapper around PSID panel extracts."""

    data_dir: str | Path
    survey_year: int | None = None
    years: list[int] | None = None
    sample: str | None = None
    shareability: Shareability = Shareability.RESTRICTED
    loader: Callable[..., PSIDDataset] | None = None
    _descriptor_cache: SourceDescriptor | None = None

    @property
    def descriptor(self) -> SourceDescriptor:
        if self._descriptor_cache is not None:
            return self._descriptor_cache
        return SourceDescriptor(
            name="psid",
            shareability=self.shareability,
            time_structure=TimeStructure.PANEL,
            archetype=SourceArchetype.LONGITUDINAL_SOCIOECONOMIC,
            observations=(
                EntityObservation(
                    entity=EntityType.HOUSEHOLD,
                    key_column="household_id",
                    variable_names=("state_fips", "tenure"),
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

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        query = query or SourceQuery()
        provider_filters = query.provider_filters
        loader = self.loader or load_psid_panel
        dataset = loader(
            data_dir=provider_filters.get("data_dir", self.data_dir),
            years=provider_filters.get("years", self.years),
            sample=provider_filters.get("sample", self.sample),
        )
        persons = dataset.persons.copy()
        available_years = sorted(persons["year"].dropna().astype(int).unique().tolist())
        survey_year = int(
            provider_filters.get(
                "survey_year",
                query.period
                if query.period is not None
                else (
                    self.survey_year if self.survey_year is not None else available_years[-1]
                ),
            )
        )
        households, persons = _normalize_psid_cross_section(
            persons,
            survey_year=survey_year,
        )
        persons = _sample_psid_households(
            persons,
            sample_n=provider_filters.get("sample_n"),
            random_seed=int(provider_filters.get("random_seed", 0)),
        )
        households = households[
            households["household_id"].isin(set(persons["household_id"]))
        ].reset_index(drop=True)
        frame = _build_psid_observation_frame(
            households=households,
            persons=persons,
            source_name=f"psid_{survey_year}",
            shareability=self.shareability,
        )
        self._descriptor_cache = frame.source
        return apply_source_query(frame, query)
