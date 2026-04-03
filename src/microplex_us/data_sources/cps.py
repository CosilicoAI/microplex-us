"""
CPS ASEC (Annual Social and Economic Supplement) data loading.

The CPS ASEC is the primary source for income and poverty statistics in the US.
Released annually in March, it contains detailed income, employment, and
demographic information for ~100K households.

Data source: https://www.census.gov/data/datasets/time-series/demo/cps/cps-asec.html
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
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

from microplex_us.source_registry import resolve_source_variable_capabilities

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "microplex"
CPS_ASEC_PROCESSED_CACHE_VERSION = "20260403"

# CPS ASEC data URLs by year
CPS_URLS = {
    2023: "https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asecpub23csv.zip",
    2022: "https://www2.census.gov/programs-surveys/cps/datasets/2022/march/asecpub22csv.zip",
    2021: "https://www2.census.gov/programs-surveys/cps/datasets/2021/march/asecpub21csv.zip",
}

# Key variable mappings (Census variable name -> our name)
PERSON_VARIABLES = {
    # Demographics
    "A_AGE": "age",
    "A_SEX": "sex",
    "PRDTRACE": "race",
    "PEHSPNON": "hispanic",
    "PRDTHSP": "_cps_hispanic_code",
    "A_HGA": "education",
    "PEDISDRS": "_disability_dressing",
    "PEDISEAR": "_disability_hearing",
    "PEDISEYE": "_disability_vision",
    "PEDISOUT": "_disability_errands",
    "PEDISPHY": "_disability_physical",
    "PEDISREM": "_disability_cognitive",
    "DIS_VAL1": "_disability_income_1",
    "DIS_SC1": "_disability_income_code_1",
    "DIS_VAL2": "_disability_income_2",
    "DIS_SC2": "_disability_income_code_2",
    "RESNSS1": "_social_security_reason_1",
    "RESNSS2": "_social_security_reason_2",
    # Employment
    "A_CLSWKR": "class_of_worker",
    "A_WKSTAT": "work_status",
    "A_HRS1": "hours_worked",
    # Income (annual)
    "WSAL_VAL": "wage_income",
    "SEMP_VAL": "self_employment_income",
    "INT_VAL": "interest_income",
    "DIV_VAL": "dividend_income",
    "RNT_VAL": "rental_income",
    "SS_VAL": "social_security",
    "SSI_VAL": "ssi",
    "UC_VAL": "unemployment_compensation",
    "PTOTVAL": "total_person_income",
    "OI_OFF": "_other_income_code",
    "OI_VAL": "_other_income_value",
    # Benefits
    "PAW_VAL": "public_assistance",
    "CSP_VAL": "child_support_received",
    "CHSP_VAL": "child_support_expense",
    "MCARE": "has_medicare",
    "MCAID": "has_medicaid",
    "NOW_GRP": "has_esi",
    "NOW_MRK": "has_marketplace_health_coverage",
    "PHIP_VAL": "health_insurance_premiums_without_medicare_part_b",
    "POTC_VAL": "over_the_counter_health_expenses",
    "PMED_VAL": "other_medical_expenses",
    "PEMCPREM": "medicare_part_b_premiums",
    "WICYN": "_receives_wic",
    # Identifiers
    "PH_SEQ": "household_id",
    "GESTFIPS": "state_fips",
    "PF_SEQ": "family_id",
    "A_LINENO": "person_number",
    "A_FAMREL": "family_relationship",
    "A_MARITL": "marital_status",
    # Weights
    "A_FNLWGT": "weight",
    "MARSUPWT": "march_supplement_weight",
}

HOUSEHOLD_VARIABLES = {
    "H_SEQ": "household_id",
    "GESTFIPS": "state_fips",
    "GTCO": "county_fips",
    "GTCBSA": "cbsa",
    "HRHTYPE": "household_type",
    "H_NUMPER": "household_size",
    "HHINC": "household_income_bracket",
    "HTOTVAL": "household_total_income",
    "HSUP_WGT": "household_weight",
}

PERSON_OBSERVATION_EXCLUDED_COLUMNS = (
    "person_id",
    "household_id",
    "weight",
    "march_supplement_weight",
    "year",
)

HOUSEHOLD_OBSERVATION_EXCLUDED_COLUMNS = (
    "household_id",
    "household_weight",
    "year",
)

PERSON_NONNEGATIVE_VALUE_COLUMNS = (
    "wage_income",
    "self_employment_income",
    "interest_income",
    "dividend_income",
    "rental_income",
    "social_security",
    "ssi",
    "unemployment_compensation",
    "public_assistance",
    "total_person_income",
    "alimony_income",
    "child_support_received",
    "child_support_expense",
    "disability_benefits",
    "health_insurance_premiums_without_medicare_part_b",
    "over_the_counter_health_expenses",
    "other_medical_expenses",
    "medicare_part_b_premiums",
    "social_security_disability",
    "social_security_retirement",
    "social_security_survivors",
    "social_security_dependents",
)

PERSON_ZERO_DEFAULT_VALUE_COLUMNS = (
    "alimony_income",
    "child_support_received",
    "child_support_expense",
    "disability_benefits",
    "health_insurance_premiums_without_medicare_part_b",
    "over_the_counter_health_expenses",
    "other_medical_expenses",
    "medicare_part_b_premiums",
    "social_security_disability",
    "social_security_retirement",
    "social_security_survivors",
    "social_security_dependents",
)

PERSON_CACHE_REQUIRED_COLUMNS = (
    "state_fips",
    "county_fips",
    "cps_race",
    "is_hispanic",
    "is_disabled",
    "has_esi",
    "has_marketplace_health_coverage",
    "alimony_income",
    "child_support_received",
    "child_support_expense",
    "disability_benefits",
    "health_insurance_premiums_without_medicare_part_b",
    "other_medical_expenses",
    "over_the_counter_health_expenses",
    "medicare_part_b_premiums",
    "social_security_disability",
    "social_security_retirement",
    "social_security_survivors",
    "social_security_dependents",
    "receives_wic",
)

PERSON_CPS_DISABILITY_COLUMNS = (
    "_disability_dressing",
    "_disability_hearing",
    "_disability_vision",
    "_disability_errands",
    "_disability_physical",
    "_disability_cognitive",
)

WORKERS_COMP_DISABILITY_CODE = 1
ALIMONY_OTHER_INCOME_CODE = 20
SOCIAL_SECURITY_RETIREMENT_REASON_CODE = 1
SOCIAL_SECURITY_DISABILITY_REASON_CODE = 2
SOCIAL_SECURITY_SURVIVOR_REASON_CODES = (3, 5)
SOCIAL_SECURITY_DEPENDENT_REASON_CODES = (4, 6, 7)
MINIMUM_RETIREMENT_AGE = 62


def processed_cps_asec_cache_path(*, year: int, cache_dir: Path) -> Path:
    """Return the versioned processed-cache path for one CPS ASEC year."""
    return cache_dir / (
        f"cps_asec_{year}_processed_v{CPS_ASEC_PROCESSED_CACHE_VERSION}.parquet"
    )


def legacy_processed_cps_asec_cache_path(*, year: int, cache_dir: Path) -> Path:
    """Return the legacy unversioned processed-cache path for one CPS ASEC year."""
    return cache_dir / f"cps_asec_{year}_processed.parquet"


@dataclass
class CPSDataset:
    """Container for CPS ASEC data."""

    persons: pl.DataFrame
    households: pl.DataFrame
    year: int
    source: str

    @property
    def n_persons(self) -> int:
        return len(self.persons)

    @property
    def n_households(self) -> int:
        return len(self.households)

    def summary(self) -> dict:
        """Return summary statistics."""
        return {
            "year": self.year,
            "n_persons": self.n_persons,
            "n_households": self.n_households,
            "states": self.households["state_fips"].n_unique(),
            "total_weight": float(self.persons["weight"].sum()),
        }


def _descriptor_from_tables(
    *,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    name: str,
) -> SourceDescriptor:
    household_variables = tuple(
        column
        for column in households.columns
        if column not in HOUSEHOLD_OBSERVATION_EXCLUDED_COLUMNS
    )
    person_variables = tuple(
        column
        for column in persons.columns
        if column not in PERSON_OBSERVATION_EXCLUDED_COLUMNS
    )
    return SourceDescriptor(
        name=name,
        shareability=Shareability.PUBLIC,
        time_structure=TimeStructure.REPEATED_CROSS_SECTION,
        archetype=SourceArchetype.HOUSEHOLD_INCOME,
        observations=(
            EntityObservation(
                entity=EntityType.HOUSEHOLD,
                key_column="household_id",
                variable_names=household_variables,
                weight_column="household_weight" if "household_weight" in households.columns else None,
                period_column="year" if "year" in households.columns else None,
            ),
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=person_variables,
                weight_column="weight" if "weight" in persons.columns else None,
                period_column="year" if "year" in persons.columns else None,
            ),
        ),
        variable_capabilities=resolve_source_variable_capabilities(
            name,
            (*household_variables, *person_variables),
        ),
    )


def _ensure_person_ids(persons: pd.DataFrame) -> pd.DataFrame:
    result = persons.copy()
    if "person_id" in result.columns:
        return result
    if "person_number" in result.columns and "household_id" in result.columns:
        result["person_id"] = (
            result["household_id"].astype(str) + ":" + result["person_number"].astype(str)
        )
        return result
    if "household_id" in result.columns:
        result["person_id"] = (
            result["household_id"].astype(str)
            + ":"
            + result.groupby("household_id").cumcount().add(1).astype(str)
        )
        return result
    result["person_id"] = np.arange(len(result)).astype(str)
    return result


def _build_observation_frame(
    *,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    source_name: str,
) -> ObservationFrame:
    normalized_households = households.copy()
    normalized_persons = _ensure_person_ids(persons)
    descriptor = _descriptor_from_tables(
        households=normalized_households,
        persons=normalized_persons,
        name=source_name,
    )
    frame = ObservationFrame(
        source=descriptor,
        tables={
            EntityType.HOUSEHOLD: normalized_households,
            EntityType.PERSON: normalized_persons,
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


def _sample_households_and_persons(
    *,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    sample_n: int | None,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample households and keep all linked person records."""
    household_sort_columns = [
        column
        for column in ("household_id", "year")
        if column in households.columns
    ]
    person_sort_columns = [
        column
        for column in ("household_id", "person_id", "person_number", "year")
        if column in persons.columns
    ]
    if household_sort_columns:
        households = households.sort_values(
            household_sort_columns,
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        households = households.reset_index(drop=True)
    if person_sort_columns:
        persons = persons.sort_values(
            person_sort_columns,
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        persons = persons.reset_index(drop=True)
    if sample_n is None or sample_n >= len(households):
        return households, persons
    sample_weights: pd.Series | None = None
    if "household_weight" in households.columns:
        candidate_weights = (
            pd.to_numeric(households["household_weight"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
        if candidate_weights.sum() > 0.0 and int((candidate_weights > 0.0).sum()) >= sample_n:
            sample_weights = candidate_weights
    try:
        sampled_households = households.sample(
            n=sample_n,
            random_state=random_seed,
            replace=False,
            weights=sample_weights,
        ).copy()
    except ValueError:
        # Pandas can reject weighted sampling without replacement at large n
        # even when the positive-weight count looks sufficient. Fall back to
        # deterministic uniform sampling rather than failing the whole build.
        sampled_households = households.sample(
            n=sample_n,
            random_state=random_seed,
            replace=False,
            weights=None,
        ).copy()
    sampled_keys = set(sampled_households["household_id"])
    sampled_persons = persons[persons["household_id"].isin(sampled_keys)].copy()
    if household_sort_columns:
        sampled_households = sampled_households.sort_values(
            household_sort_columns,
            kind="mergesort",
        )
    if person_sort_columns:
        sampled_persons = sampled_persons.sort_values(
            person_sort_columns,
            kind="mergesort",
        )
    return sampled_households.reset_index(drop=True), sampled_persons.reset_index(drop=True)


@dataclass
class CPSASECSourceProvider:
    """Source-provider wrapper around the CPS ASEC Census loader."""

    year: int = 2023
    cache_dir: Path | None = None
    download: bool = True
    loader: Callable[..., CPSDataset] | None = None
    _descriptor_cache: SourceDescriptor | None = None

    @property
    def descriptor(self) -> SourceDescriptor:
        if self._descriptor_cache is not None:
            return self._descriptor_cache
        return SourceDescriptor(
            name="cps_asec",
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

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        query = query or SourceQuery()
        provider_filters = query.provider_filters
        loader = self.loader or load_cps_asec
        dataset = loader(
            year=int(provider_filters.get("year", self.year)),
            cache_dir=provider_filters.get("cache_dir", self.cache_dir),
            download=bool(provider_filters.get("download", self.download)),
        )
        households = dataset.households.to_pandas()
        persons = dataset.persons.to_pandas()
        households, persons = _sample_households_and_persons(
            households=households,
            persons=persons,
            sample_n=provider_filters.get("sample_n"),
            random_seed=int(provider_filters.get("random_seed", 0)),
        )
        frame = _build_observation_frame(
            households=households,
            persons=persons,
            source_name=f"cps_asec_{dataset.year}",
        )
        self._descriptor_cache = frame.source
        return apply_source_query(frame, query)


@dataclass
class CPSASECParquetSourceProvider:
    """Source-provider wrapper around split CPS household/person parquet files."""

    data_dir: str | Path
    year: int | None = None
    households_filename: str = "cps_asec_households.parquet"
    persons_filename: str = "cps_asec_persons.parquet"
    _descriptor_cache: SourceDescriptor | None = None

    @property
    def descriptor(self) -> SourceDescriptor:
        if self._descriptor_cache is not None:
            return self._descriptor_cache
        return SourceDescriptor(
            name="cps_asec_parquet",
            shareability=Shareability.PUBLIC,
            time_structure=TimeStructure.REPEATED_CROSS_SECTION,
            archetype=SourceArchetype.HOUSEHOLD_INCOME,
            observations=(
                EntityObservation(
                    entity=EntityType.HOUSEHOLD,
                    key_column="household_id",
                    variable_names=("state_fips",),
                ),
                EntityObservation(
                    entity=EntityType.PERSON,
                    key_column="person_id",
                    variable_names=("age",),
                ),
            ),
        )

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        data_dir = Path(self.data_dir)
        households_path = data_dir / self.households_filename
        persons_path = data_dir / self.persons_filename
        if not households_path.exists() or not persons_path.exists():
            raise FileNotFoundError(
                f"CPS ASEC data files not found in {data_dir}.\n"
                "Expected household/person parquet files in the source directory."
            )

        households = pd.read_parquet(households_path)
        persons = pd.read_parquet(persons_path)
        query = query or SourceQuery()
        provider_filters = query.provider_filters
        if self.year is not None:
            households = households.copy()
            persons = persons.copy()
            if "year" not in households.columns:
                households["year"] = self.year
            if "year" not in persons.columns:
                persons["year"] = self.year
        households, persons = _sample_households_and_persons(
            households=households,
            persons=persons,
            sample_n=provider_filters.get("sample_n"),
            random_seed=int(provider_filters.get("random_seed", 0)),
        )
        frame = _build_observation_frame(
            households=households,
            persons=persons,
            source_name="cps_asec_parquet",
        )
        self._descriptor_cache = frame.source
        return apply_source_query(frame, query)


def download_cps_asec(
    year: int,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download CPS ASEC data for a given year.

    Args:
        year: Year of CPS ASEC (e.g., 2023)
        cache_dir: Directory to cache downloads
        force: Re-download even if cached

    Returns:
        Path to downloaded/cached zip file
    """

    import httpx

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)

    if year not in CPS_URLS:
        available = ", ".join(str(y) for y in sorted(CPS_URLS.keys()))
        raise ValueError(f"CPS ASEC for {year} not available. Available: {available}")

    url = CPS_URLS[year]
    filename = f"cps_asec_{year}.zip"
    cache_path = cache_dir / filename

    if cache_path.exists() and not force:
        print(f"Using cached CPS ASEC {year} from {cache_path}")
        return cache_path

    print(f"Downloading CPS ASEC {year} from {url}...")

    with httpx.Client(follow_redirects=True, timeout=300) as client:
        response = client.get(url)
        response.raise_for_status()

        with open(cache_path, "wb") as f:
            f.write(response.content)

    print(f"Downloaded {len(response.content) / 1_000_000:.1f} MB to {cache_path}")
    return cache_path


def load_cps_asec(
    year: int = 2023,
    cache_dir: Path | None = None,
    download: bool = True,
) -> CPSDataset:
    """
    Load CPS ASEC data for a given year.

    Args:
        year: Year of CPS ASEC (e.g., 2023)
        cache_dir: Directory for cached data
        download: Whether to download if not cached

    Returns:
        CPSDataset with persons and households DataFrames
    """
    import zipfile

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    # Prefer a versioned processed cache so derivation-logic changes do not
    # silently reuse stale pre-sim columns.
    processed_path = processed_cps_asec_cache_path(year=year, cache_dir=cache_dir)
    legacy_processed_path = legacy_processed_cps_asec_cache_path(
        year=year,
        cache_dir=cache_dir,
    )
    if processed_path.exists():
        print(f"Loading processed CPS ASEC {year} from {processed_path}")
        persons = pl.read_parquet(processed_path)
        if _processed_persons_have_household_geography(persons):
            households = _derive_households(persons)
            return CPSDataset(
                persons=persons,
                households=households,
                year=year,
                source=str(processed_path),
            )
        print(
            f"Cached processed CPS ASEC {year} is missing state_fips; rebuilding from raw source"
        )
    elif legacy_processed_path.exists():
        print(
            "Ignoring legacy CPS ASEC processed cache "
            f"{legacy_processed_path} because cache version "
            f"{CPS_ASEC_PROCESSED_CACHE_VERSION} is required; rebuilding from raw source"
        )

    # Download if needed
    zip_path = cache_dir / f"cps_asec_{year}.zip"
    if not zip_path.exists():
        if not download:
            raise FileNotFoundError(
                f"CPS ASEC {year} not found at {zip_path}. "
                "Set download=True to fetch from Census."
            )
        zip_path = download_cps_asec(year, cache_dir)

    # Extract and parse
    print(f"Parsing CPS ASEC {year}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the person file (pppub*.csv)
        person_file = None
        household_file = None

        for name in zf.namelist():
            lower = name.lower()
            if "pppub" in lower and lower.endswith(".csv"):
                person_file = name
            elif "hhpub" in lower and lower.endswith(".csv"):
                household_file = name

        if person_file is None:
            raise ValueError(f"Could not find person file in {zip_path}")

        # Schema overrides for columns with large IDs that overflow int64
        schema_overrides = {
            "PERIDNUM": pl.Utf8,  # Person ID - too large for int64
            "H_IDNUM": pl.Utf8,  # Household ID - too large for int64
            "OCCURNUM": pl.Utf8,  # Occurrence number
            "QSTNUM": pl.Utf8,  # Questionnaire number
        }

        # Read person data
        with zf.open(person_file) as f:
            persons_raw = pl.read_csv(
                f,
                infer_schema_length=10000,
                schema_overrides=schema_overrides,
            )

        # Read household data if available
        if household_file:
            with zf.open(household_file) as f:
                households_raw = pl.read_csv(
                    f,
                    infer_schema_length=10000,
                    schema_overrides=schema_overrides,
                )
        else:
            households_raw = None

    # Process person data
    persons = _process_persons(persons_raw, year)

    # Process or derive household data
    if households_raw is not None:
        households = _process_households(households_raw, year)
    else:
        households = _derive_households(persons)

    persons = _attach_household_geography_to_persons(
        persons=persons,
        households=households,
    )

    # Cache processed data
    persons.write_parquet(processed_path)
    print(f"Cached processed data to {processed_path}")

    return CPSDataset(
        persons=persons,
        households=households,
        year=year,
        source=str(zip_path),
    )


def _process_persons(df: pl.DataFrame, year: int) -> pl.DataFrame:
    """Process raw person file into clean format."""
    selected = [
        pl.col(census_name).alias(our_name)
        for census_name, our_name in PERSON_VARIABLES.items()
        if census_name in df.columns
    ]
    if not selected:
        raise ValueError("No recognized variables found in person file")
    result = df.select(selected)

    # Scale weights: CPS ASEC weights have 2 implied decimal places
    # See CPS documentation: A_FNLWGT is expressed in units of 1/100
    # Divide by 100 to get actual population representation
    if "weight" in result.columns:
        result = result.with_columns(
            (pl.col("weight") / 100).alias("weight")
        )
    if "march_supplement_weight" in result.columns:
        result = result.with_columns(
            (pl.col("march_supplement_weight") / 100).alias("march_supplement_weight")
        )

    # Add derived columns
    if "age" in result.columns:
        result = result.with_columns([
            (pl.col("age") >= 18).alias("is_adult"),
            (pl.col("age") < 18).alias("is_child"),
            (pl.col("age") >= 65).alias("is_senior"),
        ])

    if "race" in result.columns and "cps_race" not in result.columns:
        result = result.with_columns(pl.col("race").alias("cps_race"))
    if "_cps_hispanic_code" in result.columns and "is_hispanic" not in result.columns:
        result = result.with_columns(
            (pl.col("_cps_hispanic_code") != 0).alias("is_hispanic")
        ).drop("_cps_hispanic_code")
    if {
        "_other_income_code",
        "_other_income_value",
    }.issubset(set(result.columns)) and "alimony_income" not in result.columns:
        result = result.with_columns(
            pl.when(pl.col("_other_income_code") == ALIMONY_OTHER_INCOME_CODE)
            .then(pl.col("_other_income_value"))
            .otherwise(0)
            .alias("alimony_income")
        ).drop(["_other_income_code", "_other_income_value"])
    else:
        drop_columns = [
            column
            for column in ("_other_income_code", "_other_income_value")
            if column in result.columns
        ]
        if drop_columns:
            result = result.drop(drop_columns)
    if {
        "_social_security_reason_1",
        "_social_security_reason_2",
        "social_security",
        "age",
    }.issubset(set(result.columns)) and (
        "social_security_disability" not in result.columns
        or "social_security_retirement" not in result.columns
        or "social_security_survivors" not in result.columns
        or "social_security_dependents" not in result.columns
    ):
        reason_1 = pl.col("_social_security_reason_1")
        reason_2 = pl.col("_social_security_reason_2")
        has_retirement_reason = (
            (reason_1 == SOCIAL_SECURITY_RETIREMENT_REASON_CODE)
            | (reason_2 == SOCIAL_SECURITY_RETIREMENT_REASON_CODE)
        )
        has_disability_reason = (
            (reason_1 == SOCIAL_SECURITY_DISABILITY_REASON_CODE)
            | (reason_2 == SOCIAL_SECURITY_DISABILITY_REASON_CODE)
        )
        has_survivor_reason = (
            reason_1.is_in(SOCIAL_SECURITY_SURVIVOR_REASON_CODES)
            | reason_2.is_in(SOCIAL_SECURITY_SURVIVOR_REASON_CODES)
        )
        has_dependent_reason = (
            reason_1.is_in(SOCIAL_SECURITY_DEPENDENT_REASON_CODES)
            | reason_2.is_in(SOCIAL_SECURITY_DEPENDENT_REASON_CODES)
        )
        unclassified_social_security = (
            (pl.col("social_security") > 0)
            & ~has_retirement_reason
            & ~has_disability_reason
            & ~has_survivor_reason
            & ~has_dependent_reason
        )
        derived_columns: list[pl.Expr] = []
        if "social_security_disability" not in result.columns:
            derived_columns.append(
                (
                    pl.when(has_disability_reason & ~has_retirement_reason)
                    .then(pl.col("social_security"))
                    .otherwise(0.0)
                    + pl.when(
                        unclassified_social_security
                        & (pl.col("age") < MINIMUM_RETIREMENT_AGE)
                    )
                    .then(pl.col("social_security"))
                    .otherwise(0.0)
                ).alias("social_security_disability")
            )
        if "social_security_retirement" not in result.columns:
            derived_columns.append(
                (
                    pl.when(has_retirement_reason & ~has_disability_reason)
                    .then(pl.col("social_security"))
                    .otherwise(0.0)
                    + pl.when(
                        unclassified_social_security
                        & (pl.col("age") >= MINIMUM_RETIREMENT_AGE)
                    )
                    .then(pl.col("social_security"))
                    .otherwise(0.0)
                ).alias("social_security_retirement")
            )
        if "social_security_survivors" not in result.columns:
            derived_columns.append(
                (
                    pl.when(
                        has_survivor_reason
                        & ~has_retirement_reason
                        & ~has_disability_reason
                    )
                    .then(pl.col("social_security"))
                    .otherwise(0.0)
                ).alias("social_security_survivors")
            )
        if "social_security_dependents" not in result.columns:
            derived_columns.append(
                (
                    pl.when(
                        has_dependent_reason
                        & ~has_retirement_reason
                        & ~has_disability_reason
                        & ~has_survivor_reason
                    )
                    .then(pl.col("social_security"))
                    .otherwise(0.0)
                ).alias("social_security_dependents")
            )
        result = result.with_columns(derived_columns).drop(
            ["_social_security_reason_1", "_social_security_reason_2"]
        )
    else:
        drop_columns = [
            column
            for column in (
                "_social_security_reason_1",
                "_social_security_reason_2",
            )
            if column in result.columns
        ]
        if drop_columns:
            result = result.drop(drop_columns)
    disability_columns = [
        column for column in PERSON_CPS_DISABILITY_COLUMNS if column in result.columns
    ]
    if disability_columns and "is_disabled" not in result.columns:
        result = result.with_columns(
            pl.any_horizontal(
                *[(pl.col(column) == 1) for column in disability_columns]
            ).alias("is_disabled")
        ).drop(disability_columns)
    elif disability_columns:
        result = result.drop(disability_columns)
    if {
        "_disability_income_1",
        "_disability_income_code_1",
        "_disability_income_2",
        "_disability_income_code_2",
    }.issubset(set(result.columns)) and "disability_benefits" not in result.columns:
        result = result.with_columns(
            (
                pl.when(pl.col("_disability_income_code_1") != WORKERS_COMP_DISABILITY_CODE)
                .then(pl.col("_disability_income_1"))
                .otherwise(0)
                +
                pl.when(pl.col("_disability_income_code_2") != WORKERS_COMP_DISABILITY_CODE)
                .then(pl.col("_disability_income_2"))
                .otherwise(0)
            ).alias("disability_benefits")
        ).drop(
            [
                "_disability_income_1",
                "_disability_income_code_1",
                "_disability_income_2",
                "_disability_income_code_2",
            ]
        )
    else:
        drop_columns = [
            column
            for column in (
                "_disability_income_1",
                "_disability_income_code_1",
                "_disability_income_2",
                "_disability_income_code_2",
            )
            if column in result.columns
        ]
        if drop_columns:
            result = result.drop(drop_columns)
    if "_receives_wic" in result.columns and "receives_wic" not in result.columns:
        result = result.with_columns(
            (pl.col("_receives_wic") == 1).alias("receives_wic")
        ).drop("_receives_wic")
    elif "_receives_wic" in result.columns:
        result = result.drop("_receives_wic")
    for value_column in PERSON_ZERO_DEFAULT_VALUE_COLUMNS:
        if value_column not in result.columns:
            result = result.with_columns(pl.lit(0.0).alias(value_column))
    for bool_column in (
        "has_esi",
        "has_marketplace_health_coverage",
        "receives_wic",
    ):
        if bool_column in result.columns:
            result = result.with_columns((pl.col(bool_column) == 1).alias(bool_column))
    for col in PERSON_NONNEGATIVE_VALUE_COLUMNS:
        if col in result.columns:
            result = result.with_columns(
                pl.when(pl.col(col) < 0)
                .then(0)
                .otherwise(pl.col(col))
                .alias(col)
            )

    # Add year
    result = result.with_columns(pl.lit(year).alias("year"))

    return result


def _processed_persons_have_household_geography(persons: pl.DataFrame) -> bool:
    """Whether cached processed person data can derive household geography."""
    required_columns = set(PERSON_CACHE_REQUIRED_COLUMNS)
    if not required_columns.issubset(set(persons.columns)):
        return False
    return len(persons["state_fips"].drop_nulls()) > 0


def _process_households(df: pl.DataFrame, year: int) -> pl.DataFrame:
    """Process raw household file into clean format."""
    selected = [
        pl.col(census_name).alias(our_name)
        for census_name, our_name in HOUSEHOLD_VARIABLES.items()
        if census_name in df.columns
    ]
    if not selected:
        raise ValueError("No recognized variables found in household file")
    result = df.select(selected)

    # Scale weights: CPS ASEC weights have 2 implied decimal places
    if "household_weight" in result.columns:
        result = result.with_columns(
            (pl.col("household_weight") / 100).alias("household_weight")
        )

    result = result.with_columns(pl.lit(year).alias("year"))

    return result


def _attach_household_geography_to_persons(
    *,
    persons: pl.DataFrame,
    households: pl.DataFrame,
) -> pl.DataFrame:
    """Propagate household geography onto cached person rows when needed."""
    if "household_id" not in households.columns:
        return persons
    geography_columns = [
        column
        for column in ("state_fips", "county_fips")
        if column in households.columns
    ]
    if not geography_columns:
        return persons
    joined = persons.join(
        households.select(["household_id", *geography_columns]).rename(
            {
                column: f"_household_{column}"
                for column in geography_columns
            }
        ),
        on="household_id",
        how="left",
    )
    for column in geography_columns:
        household_column = f"_household_{column}"
        if column in joined.columns:
            joined = joined.with_columns(
                pl.coalesce(column, household_column).alias(column)
            )
        else:
            joined = joined.with_columns(pl.col(household_column).alias(column))
        joined = joined.drop(household_column)
    return joined


def _derive_households(persons: pl.DataFrame) -> pl.DataFrame:
    """Derive household-level data from person records."""
    if "household_id" not in persons.columns:
        raise ValueError("Cannot derive households without household_id")

    aggregations = [
        pl.len().alias("household_size"),
        pl.col("weight").first().alias("household_weight"),
    ]
    if "state_fips" in persons.columns:
        aggregations.append(pl.col("state_fips").first().alias("state_fips"))
    else:
        aggregations.append(pl.lit(None).alias("state_fips"))
    if "county_fips" in persons.columns:
        aggregations.append(pl.col("county_fips").first().alias("county_fips"))
    else:
        aggregations.append(pl.lit(None).alias("county_fips"))
    if "total_person_income" in persons.columns:
        aggregations.append(
            pl.col("total_person_income").sum().alias("household_total_income")
        )
    else:
        aggregations.append(pl.lit(0).alias("household_total_income"))
    if "is_child" in persons.columns:
        aggregations.append(pl.col("is_child").sum().alias("num_children"))
    else:
        aggregations.append(pl.lit(0).alias("num_children"))
    if "is_adult" in persons.columns:
        aggregations.append(pl.col("is_adult").sum().alias("num_adults"))
    else:
        aggregations.append(pl.lit(0).alias("num_adults"))

    households = persons.group_by("household_id").agg(aggregations)

    if "year" in persons.columns:
        year_val = persons.select("year").unique().to_series()[0]
        households = households.with_columns(
            pl.lit(year_val).alias("year")
        )

    return households


def get_available_years() -> list[int]:
    """Return list of available CPS ASEC years."""
    return sorted(CPS_URLS.keys())
