"""IRS Public Use File (PUF) loader, processing, and source-provider wrapper.

Downloads PUF from HuggingFace, uprates 2015 → target year,
and maps to common variable schema for multi-survey fusion.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from microplex.core import (
    EntityObservation,
    EntityRelationship,
    EntityType,
    ObservationFrame,
    RelationshipCardinality,
    Shareability,
    SourceDescriptor,
    SourceQuery,
    TimeStructure,
    apply_source_query,
)

from microplex_us.source_manifests import load_us_source_manifest
from microplex_us.source_registry import resolve_source_variable_capabilities
from microplex_us.variables import normalize_dividend_columns

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

PUF_VARIABLE_MAP = {
    column_spec.raw_column: column_spec.canonical_name
    for column_spec in load_us_source_manifest("puf")
    .observation_for(EntityType.TAX_UNIT)
    .columns
}

# SOI growth factors for uprating 2015 → 2024
# Based on IRS SOI aggregate growth rates
# These should be updated with actual SOI data
UPRATING_FACTORS = {
    "employment_income": 1.45,  # ~4.5% annual wage growth
    "self_employment_income": 1.35,
    "farm_income": 1.20,
    "taxable_interest_income": 2.50,  # Interest rates rose significantly
    "tax_exempt_interest_income": 1.80,
    "ordinary_dividend_income": 1.60,
    "qualified_dividend_income": 1.60,
    "short_term_capital_gains": 1.80,
    "long_term_capital_gains": 2.20,  # Stock market growth
    "capital_gains_distributions": 1.80,
    "partnership_s_corp_income": 1.50,
    "rental_income_positive": 1.40,
    "rental_income_negative": 1.40,
    "ira_distributions": 1.60,
    "total_pension_income": 1.40,
    "taxable_pension_income": 1.40,
    "gross_social_security": 1.45,
    "taxable_social_security": 1.45,
    "unemployment_compensation": 0.30,  # Down from COVID peak
    "alimony_income": 0.50,  # Declining due to tax law change
    "medical_expense_agi_floor": 1.50,
    "state_income_tax_paid": 1.40,
    "real_estate_tax_paid": 1.35,
    "mortgage_interest_paid": 1.30,
    "charitable_cash": 1.40,
    "charitable_noncash": 1.40,
    "student_loan_interest": 1.20,
}

JOINT_HEAD_SHARE_ALLOCATION = {
    "employment_income": 0.6,
    "self_employment_income": 0.6,
}

JOINT_EQUAL_SHARE_ALLOCATION = (
    "farm_income",
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "ordinary_dividend_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "short_term_capital_gains",
    "long_term_capital_gains",
    "capital_gains_distributions",
    "partnership_s_corp_income",
    "rental_income",
    "ira_distributions",
    "total_pension_income",
    "taxable_pension_income",
    "gross_social_security",
    "taxable_social_security",
    "unemployment_compensation",
    "alimony_income",
    "medical_expense_agi_floor",
    "state_income_tax_paid",
    "real_estate_tax_paid",
    "mortgage_interest_paid",
    "charitable_cash",
    "charitable_noncash",
    "ira_deduction",
    "student_loan_interest",
)

MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS = {
    "health_insurance_premiums_without_medicare_part_b": 0.453,
    "other_medical_expenses": 0.325,
    "medicare_part_b_premiums": 0.137,
    "over_the_counter_health_expenses": 0.085,
}

def download_puf(cache_dir: Path | None = None) -> Path:
    """Download PUF from HuggingFace.

    Returns path to downloaded CSV file.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "microplex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    puf_path = cache_dir / "puf_2015.csv"
    demo_path = cache_dir / "demographics_2015.csv"

    # Prefer an already-present local copy over any remote resolution.
    if puf_path.exists():
        return puf_path, demo_path

    if not HF_AVAILABLE:
        raise ImportError("huggingface_hub required: pip install huggingface_hub")

    # Download PUF 2015
    puf_path = hf_hub_download(
        repo_id="policyengine/irs-soi-puf",
        filename="puf_2015.csv",
        repo_type="model",
        local_dir=cache_dir,
    )

    # Download demographics file
    demo_path = hf_hub_download(
        repo_id="policyengine/irs-soi-puf",
        filename="demographics_2015.csv",
        repo_type="model",
        local_dir=cache_dir,
    )

    return Path(puf_path), Path(demo_path)


def load_puf_raw(puf_path: Path, demographics_path: Path | None = None) -> pd.DataFrame:
    """Load raw PUF data from CSV."""
    print(f"Loading PUF from {puf_path}...")
    puf = pd.read_csv(puf_path)

    # Filter out aggregate records (MARS=0)
    puf = puf[puf["MARS"] != 0].copy()

    print(f"  Raw records: {len(puf):,}")

    # Load and merge demographics if available
    if demographics_path and demographics_path.exists():
        print(f"Loading demographics from {demographics_path}...")
        demo = pd.read_csv(demographics_path)

        # Demographics file has RECID to match
        if "RECID" in puf.columns and "RECID" in demo.columns:
            puf = puf.merge(demo, on="RECID", how="left", suffixes=("", "_demo"))
            print(f"  After demographics merge: {len(puf):,}")

    return puf


def map_puf_variables(
    puf: pd.DataFrame,
    *,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Map PUF variable codes to common names."""
    result = pd.DataFrame(index=puf.index)
    manifest = load_us_source_manifest("puf")
    observation = manifest.observation_for(EntityType.TAX_UNIT)

    for column_spec in observation.columns:
        if column_spec.raw_column in puf.columns:
            result[column_spec.canonical_name] = puf[column_spec.raw_column].fillna(0)
        else:
            result[column_spec.canonical_name] = 0

    # Fix weight (PUF stores in hundredths)
    if "weight" in result.columns:
        result["weight"] = result["weight"] / 100

    # Preserve rental losses as negative values so downstream PE targets can
    # recover rent-and-royalty loss cells.
    result["rental_income"] = (
        result.get("rental_income_positive", 0).fillna(0) +
        -result.get("rental_income_negative", 0).fillna(0)
    )
    if {"E26390", "E26400"}.issubset(set(puf.columns)):
        result["estate_income"] = puf["E26390"].fillna(0) - puf["E26400"].fillna(0)
    medical_expense_floor = result.get("medical_expense_agi_floor")
    if medical_expense_floor is not None:
        for variable, fraction in MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS.items():
            result[variable] = medical_expense_floor.fillna(0) * fraction

    # Map filing status code to string
    filing_status_map = {
        1: "SINGLE",
        2: "JOINT",
        3: "SEPARATE",
        4: "HEAD_OF_HOUSEHOLD",
        5: "WIDOW",
    }
    result["filing_status"] = result["filing_status_code"].map(filing_status_map).fillna("UNKNOWN")

    # Add age from demographics if available
    if "age" in puf.columns:
        result["age"] = puf["age"]
    elif "AGE_HEAD" in puf.columns:
        result["age"] = puf["AGE_HEAD"]
    else:
        # Impute age based on income patterns
        result["age"] = _impute_age(result, random_seed=random_seed)

    # Add sex from demographics if available
    if "is_male" in puf.columns:
        result["is_male"] = puf["is_male"]
    elif "GENDER" in puf.columns:
        result["is_male"] = (puf["GENDER"] == 1).astype(float)
    else:
        # Unknown - will be learned from CPS
        result["is_male"] = np.nan

    # Mark survey source
    result["_survey"] = "puf"

    return result


def _impute_age(
    df: pd.DataFrame,
    *,
    random_seed: int = 42,
) -> pd.Series:
    """Simple age imputation based on income patterns.

    This is a rough heuristic. The masked MAF will learn
    better age distributions from CPS.
    """
    # Base age on Social Security receipt and pension income
    age = pd.Series(40, index=df.index)  # Default

    # Social Security recipients tend to be older
    has_ss = df.get("gross_social_security", 0) > 0
    age = age.where(~has_ss, 68)

    # Pension recipients also older
    has_pension = df.get("taxable_pension_income", 0) > 0
    age = age.where(~has_pension | has_ss, 62)

    # IRA distributions suggest retirement age
    has_ira = df.get("ira_distributions", 0) > 0
    age = age.where(~has_ira | has_ss | has_pension, 60)

    # High earners tend to be prime working age
    high_wage = df.get("employment_income", 0) > 200_000
    age = age.where(~high_wage, 45)

    # Add some noise
    rng = np.random.default_rng(random_seed)
    noise = rng.normal(0, 5, len(age))
    age = (age + noise).clip(18, 95).astype(int)

    return age


def uprate_puf(df: pd.DataFrame, from_year: int = 2015, to_year: int = 2024) -> pd.DataFrame:
    """Uprate PUF income variables from one year to another.

    Uses SOI-based growth factors.
    """
    if from_year == to_year:
        return df

    # Simple scaling - in production, use year-specific factors
    year_factor = (to_year - from_year) / (2024 - 2015)

    result = df.copy()

    for var, factor in UPRATING_FACTORS.items():
        if var in result.columns:
            # Interpolate factor based on years
            scaled_factor = 1 + (factor - 1) * year_factor
            result[var] = result[var] * scaled_factor

    print(f"Uprated PUF from {from_year} to {to_year}")

    return result


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return df[column].fillna(0).astype(float)


def _add_derived_income_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result = normalize_dividend_columns(result)
    employment_income = _numeric_series(result, "employment_income")
    self_employment_income = _numeric_series(result, "self_employment_income")
    taxable_interest_income = _numeric_series(result, "taxable_interest_income")
    ordinary_dividend_income = _numeric_series(result, "ordinary_dividend_income")
    short_term_capital_gains = _numeric_series(result, "short_term_capital_gains")
    long_term_capital_gains = _numeric_series(result, "long_term_capital_gains")
    capital_gains_distributions = _numeric_series(
        result,
        "capital_gains_distributions",
    )
    taxable_pension_income = _numeric_series(result, "taxable_pension_income")
    gross_social_security = _numeric_series(result, "gross_social_security")
    rental_income = _numeric_series(result, "rental_income")
    unemployment_compensation = _numeric_series(
        result,
        "unemployment_compensation",
    )
    alimony_income = _numeric_series(result, "alimony_income")

    result["interest_income"] = taxable_interest_income
    result["dividend_income"] = ordinary_dividend_income
    result["capital_gains"] = (
        short_term_capital_gains
        + long_term_capital_gains
        + capital_gains_distributions
    )
    result["pension_income"] = taxable_pension_income
    result["social_security"] = gross_social_security
    result["income"] = (
        employment_income
        + self_employment_income
        + result["interest_income"]
        + result["dividend_income"]
        + rental_income
        + result["social_security"]
        + result["pension_income"]
        + unemployment_compensation
        + alimony_income
    )
    result["employment_status"] = (
        (employment_income + self_employment_income) > 0
    ).astype(int)
    return result


def _allocate_joint_tax_unit_amounts(
    row: pd.Series,
    head: pd.Series,
    spouse: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    for variable, head_share in JOINT_HEAD_SHARE_ALLOCATION.items():
        if variable not in row.index:
            continue
        amount = float(row[variable])
        head[variable] = amount * head_share
        spouse[variable] = amount * (1.0 - head_share)

    for variable in JOINT_EQUAL_SHARE_ALLOCATION:
        if variable not in row.index:
            continue
        amount = float(row[variable])
        head[variable] = amount * 0.5
        spouse[variable] = amount * 0.5

    return head, spouse


def expand_to_persons(df: pd.DataFrame) -> pd.DataFrame:
    """Expand tax unit records to person-level records.

    Each tax unit becomes 1-2 persons (filer + spouse if joint).
    This enables stacking with CPS person-level data.
    """
    records = []

    for idx, row in df.iterrows():
        filing_status = row.get("filing_status", "SINGLE")
        row.get("exemptions_count", 1)

        # Create head record
        head = row.copy()
        head["is_head"] = 1
        head["is_spouse"] = 0
        head["is_dependent"] = 0
        head["person_id"] = f"{idx}_head"
        head["tax_unit_id"] = idx
        records.append(head)

        # Create spouse record if joint filing
        if filing_status == "JOINT":
            spouse = row.copy()
            spouse["is_head"] = 0
            spouse["is_spouse"] = 1
            spouse["is_dependent"] = 0
            spouse["person_id"] = f"{idx}_spouse"
            spouse["tax_unit_id"] = idx

            head, spouse = _allocate_joint_tax_unit_amounts(row, head, spouse)
            # Spouse weight is same as head (we'll deduplicate in calibration)
            records.append(spouse)

    result = pd.DataFrame(records).reset_index(drop=True)
    result = _add_derived_income_columns(result)
    print(f"Expanded {len(df):,} tax units to {len(result):,} persons")

    return result


def load_puf(
    target_year: int = 2024,
    expand_persons: bool = True,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Load and process PUF for multi-survey fusion.

    Args:
        target_year: Year to uprate to
        expand_persons: If True, expand tax units to person records
        cache_dir: Directory to cache downloaded files

    Returns:
        DataFrame with common variable names, ready for stacking with CPS
    """
    # Download if needed
    puf_path, demo_path = download_puf(cache_dir)

    # Load raw data
    raw = load_puf_raw(puf_path, demo_path)

    # Map to common variables
    df = map_puf_variables(raw)

    # Uprate to target year
    df = uprate_puf(df, from_year=2015, to_year=target_year)

    # Expand to persons if requested
    if expand_persons:
        df = expand_to_persons(df)

    print(f"\nPUF loaded: {len(df):,} records")
    print(f"  Weight sum: {df['weight'].sum():,.0f}")

    return df


# Variables that PUF has but CPS doesn't (will be NaN in CPS)
PUF_EXCLUSIVE_VARS = [
    "short_term_capital_gains",
    "long_term_capital_gains",
    "capital_gains_distributions",
    "partnership_s_corp_income",
    "qualified_dividend_income",
    "tax_exempt_interest_income",
    "charitable_cash",
    "charitable_noncash",
    "mortgage_interest_paid",
    "state_income_tax_paid",
    "real_estate_tax_paid",
    "student_loan_interest",
    "ira_deduction",
]

# Variables that both surveys have (may differ in quality)
SHARED_VARS = [
    "employment_income",
    "self_employment_income",
    "taxable_interest_income",
    "ordinary_dividend_income",
    "rental_income",
    "gross_social_security",
    "taxable_pension_income",
    "unemployment_compensation",
    "age",
    "filing_status",
]


def _sample_tax_units(
    tax_units: pd.DataFrame,
    *,
    sample_n: int | None,
    random_seed: int,
) -> pd.DataFrame:
    """Sample tax units before expanding them to persons."""
    if sample_n is None or sample_n >= len(tax_units):
        return tax_units.reset_index(drop=True)
    sample_weights: pd.Series | None = None
    if "weight" in tax_units.columns:
        candidate_weights = (
            pd.to_numeric(tax_units["weight"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
        if candidate_weights.sum() > 0.0 and int((candidate_weights > 0.0).sum()) >= sample_n:
            sample_weights = candidate_weights
    try:
        return tax_units.sample(
            n=sample_n,
            random_state=random_seed,
            replace=False,
            weights=sample_weights,
        ).reset_index(drop=True)
    except ValueError:
        # Match CPS behavior: if weighted sampling without replacement is
        # infeasible at high sample sizes, fall back to deterministic uniform
        # sampling instead of failing the run.
        return tax_units.sample(
            n=sample_n,
            random_state=random_seed,
            replace=False,
            weights=None,
        ).reset_index(drop=True)


def _build_puf_tax_units(
    *,
    raw: pd.DataFrame,
    target_year: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Map raw PUF records into a normalized tax-unit table."""
    tax_units = map_puf_variables(raw, random_seed=random_seed)
    tax_units = uprate_puf(tax_units, from_year=2015, to_year=target_year)
    identifier = (
        raw["RECID"].astype(str).reset_index(drop=True)
        if "RECID" in raw.columns
        else pd.Series(np.arange(len(raw)).astype(str))
    )
    tax_units = tax_units.reset_index(drop=True)
    tax_units["household_id"] = identifier
    tax_units["year"] = target_year
    tax_units["state_fips"] = 0
    tax_units["tenure"] = 0
    tax_units["household_weight"] = tax_units["weight"].astype(float)
    tax_units = _add_derived_income_columns(tax_units)
    is_male = tax_units.get("is_male", pd.Series(np.nan, index=tax_units.index)).fillna(0)
    tax_units["sex"] = np.where(is_male > 0, 1, np.where(is_male == 0, 2, 0))
    tax_units["education"] = 0
    return tax_units


def _tax_units_to_persons(
    tax_units: pd.DataFrame,
    *,
    expand_persons_flag: bool,
) -> pd.DataFrame:
    """Expand tax units into a person table."""
    if expand_persons_flag:
        persons = expand_to_persons(tax_units)
    else:
        persons = tax_units.copy()
        persons["is_head"] = 1
        persons["is_spouse"] = 0
        persons["is_dependent"] = 0
        persons["person_id"] = persons["household_id"].astype(str) + ":head"
        persons["tax_unit_id"] = persons["household_id"].astype(str)
    persons = persons.reset_index(drop=True)
    persons["person_id"] = persons["person_id"].astype(str)
    persons["household_id"] = persons["household_id"].astype(str)
    persons["year"] = tax_units["year"].iloc[0] if not tax_units.empty else 2024
    if "income" not in persons.columns:
        persons["income"] = tax_units["income"]
    if "employment_status" not in persons.columns:
        persons["employment_status"] = tax_units["employment_status"]
    if "education" not in persons.columns:
        persons["education"] = 0
    if "age" not in persons.columns:
        persons["age"] = 0
    if "sex" not in persons.columns:
        persons["sex"] = 0
    return persons


def _build_puf_observation_frame(
    *,
    tax_units: pd.DataFrame,
    persons: pd.DataFrame,
    source_name: str,
    shareability: Shareability,
) -> ObservationFrame:
    """Build an observation frame from normalized PUF tax units."""
    manifest = load_us_source_manifest("puf")
    households = tax_units[
        ["household_id", "year", "state_fips", "tenure", "household_weight"]
    ].copy()
    person_variable_names = tuple(
        column
        for column in persons.columns
        if column not in {"person_id", "household_id", "weight", "year"}
    )
    descriptor = SourceDescriptor(
        name=source_name,
        shareability=shareability,
        time_structure=TimeStructure.REPEATED_CROSS_SECTION,
        archetype=manifest.archetype,
        population=manifest.population,
        description=manifest.description,
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
                variable_names=person_variable_names,
                weight_column="weight" if "weight" in persons.columns else None,
                period_column="year",
            ),
        ),
        variable_capabilities=resolve_source_variable_capabilities(
            source_name,
            ("state_fips", "tenure", *person_variable_names),
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
class PUFSourceProvider:
    """Source-provider wrapper around the IRS SOI PUF."""

    target_year: int = 2024
    cache_dir: Path | None = None
    puf_path: str | Path | None = None
    demographics_path: str | Path | None = None
    expand_persons: bool = True
    shareability: Shareability = Shareability.PUBLIC
    loader: Callable[[Path | None], tuple[Path, Path | None]] | None = None
    _descriptor_cache: SourceDescriptor | None = None

    @property
    def descriptor(self) -> SourceDescriptor:
        if self._descriptor_cache is not None:
            return self._descriptor_cache
        manifest = load_us_source_manifest("puf")
        person_variables = ("age", "sex", "income")
        return SourceDescriptor(
            name="irs_soi_puf",
            shareability=self.shareability,
            time_structure=TimeStructure.REPEATED_CROSS_SECTION,
            archetype=manifest.archetype,
            population=manifest.population,
            description=manifest.description,
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
                    variable_names=person_variables,
                    weight_column="weight",
                    period_column="year",
                ),
            ),
            variable_capabilities=resolve_source_variable_capabilities(
                "irs_soi_puf",
                ("state_fips", "tenure", *person_variables),
            ),
        )

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        query = query or SourceQuery()
        provider_filters = query.provider_filters
        target_year = int(provider_filters.get("target_year", self.target_year))
        expand_persons_flag = bool(
            provider_filters.get("expand_persons", self.expand_persons)
        )
        puf_path = provider_filters.get("puf_path", self.puf_path)
        demographics_path = provider_filters.get(
            "demographics_path",
            self.demographics_path,
        )
        if puf_path is None:
            loader = self.loader or download_puf
            loaded_puf_path, loaded_demographics_path = loader(self.cache_dir)
            puf_path = loaded_puf_path
            if demographics_path is None:
                demographics_path = loaded_demographics_path

        raw = load_puf_raw(
            Path(puf_path),
            Path(demographics_path) if demographics_path is not None else None,
        )
        raw = _sample_tax_units(
            raw,
            sample_n=provider_filters.get("sample_n"),
            random_seed=int(provider_filters.get("random_seed", 0)),
        )
        tax_units = _build_puf_tax_units(
            raw=raw,
            target_year=target_year,
            random_seed=int(provider_filters.get("random_seed", 0)),
        )
        persons = _tax_units_to_persons(
            tax_units,
            expand_persons_flag=expand_persons_flag,
        )
        frame = _build_puf_observation_frame(
            tax_units=tax_units,
            persons=persons,
            source_name=f"irs_soi_puf_{target_year}",
            shareability=self.shareability,
        )
        self._descriptor_cache = frame.source
        return apply_source_query(frame, query)


if __name__ == "__main__":
    # Test loading
    df = load_puf(target_year=2024)
    print("\nSample of loaded PUF:")
    print(df.head())

    print("\nIncome variable sums:")
    income_vars = [
        "employment_income", "self_employment_income",
        "long_term_capital_gains", "partnership_s_corp_income",
        "gross_social_security", "taxable_pension_income",
    ]
    for var in income_vars:
        if var in df.columns:
            total = (df[var] * df["weight"]).sum() / 1e9
            print(f"  {var}: ${total:.1f}B")
