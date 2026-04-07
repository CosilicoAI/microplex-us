"""IRS Public Use File (PUF) loader, processing, and source-provider wrapper.

Downloads PUF from HuggingFace, uprates 2015 → target year,
and maps to common variable schema for multi-survey fusion.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
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
    SourceDescriptor,
    SourceQuery,
    TimeStructure,
    apply_source_query,
)

from microplex_us.data_sources.cps import load_cps_asec
from microplex_us.data_sources.share_imputation import (
    GroupedShareModel,
    fit_grouped_share_model,
    predict_grouped_component_shares,
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
    "non_sch_d_capital_gains": 1.80,
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

MINIMUM_SOCIAL_SECURITY_RETIREMENT_AGE = 62
SOCIAL_SECURITY_SHARE_AGE_BINS = (-np.inf, 18.0, 30.0, 45.0, 62.0, 75.0, np.inf)
SOCIAL_SECURITY_SHARE_AGE_LABELS = (
    "under_18",
    "18_to_29",
    "30_to_44",
    "45_to_61",
    "62_to_74",
    "75_plus",
)
SOCIAL_SECURITY_SHARE_EXPLICIT_COMPONENTS = (
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
)
SOCIAL_SECURITY_SHARE_IMPLICIT_COMPONENT = "social_security_dependents"
SOCIAL_SECURITY_SHARE_COMPONENTS = (
    *SOCIAL_SECURITY_SHARE_EXPLICIT_COMPONENTS,
    SOCIAL_SECURITY_SHARE_IMPLICIT_COMPONENT,
)
SOCIAL_SECURITY_SPLIT_STRATEGY_GROUPED_SHARE = "grouped_share"
SOCIAL_SECURITY_SPLIT_STRATEGY_PE_QRF = "pe_qrf"
SOCIAL_SECURITY_SPLIT_STRATEGY_AGE_HEURISTIC = "age_heuristic"
PE_STYLE_SOCIAL_SECURITY_QRF_PREDICTORS = (
    "age",
    "is_male",
    "tax_unit_is_joint",
    "is_tax_unit_head",
    "is_tax_unit_dependent",
)
MIN_PE_STYLE_SOCIAL_SECURITY_QRF_TRAINING_RECORDS = 100

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
    "non_sch_d_capital_gains",
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

PUF_DEMOGRAPHIC_HELPER_COLUMNS = (
    "_puf_recid",
    "_puf_agerange",
    "_puf_earnsplit",
    "_puf_gender",
    "_puf_agedp1",
    "_puf_agedp2",
    "_puf_agedp3",
)

PUF_PERSON_EXPANSION_PRESERVE_COLUMNS = {
    "weight",
    "household_weight",
    "year",
    "household_id",
    "state_fips",
    "tenure",
    "filing_status",
    "filing_status_code",
    "exemptions_count",
    "eitc_children",
    "ctc_children",
    "age",
    "is_male",
    "employment_status",
    "income",
    "interest_income",
    "dividend_income",
    "capital_gains",
    "pension_income",
    "social_security",
    "social_security_retirement",
    *PUF_DEMOGRAPHIC_HELPER_COLUMNS,
}

MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS = {
    "health_insurance_premiums_without_medicare_part_b": 0.453,
    "other_medical_expenses": 0.325,
    "medicare_part_b_premiums": 0.137,
    "over_the_counter_health_expenses": 0.085,
}


@dataclass(frozen=True)
class PEStyleQRFShareModel:
    """PE-style QRF share model for PUF Social Security components."""

    predictors: tuple[str, ...]
    component_columns: tuple[str, ...]
    share_prediction_columns: tuple[str, ...]
    fitted_model: Any


SocialSecurityShareModel = GroupedShareModel | PEStyleQRFShareModel


@dataclass(frozen=True)
class PEStyleQRFImputationModel:
    """PE-style QRF imputation model for direct PUF variable imputation."""

    predictors: tuple[str, ...]
    imputed_variable: str
    fitted_model: Any

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
    impute_pre_tax_contributions: bool = False,
    pre_tax_contribution_model: PEStyleQRFImputationModel | None = None,
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
    if {"E00600", "E00650"}.issubset(set(puf.columns)):
        result["non_qualified_dividend_income"] = (
            puf["E00600"].fillna(0) - puf["E00650"].fillna(0)
        )
    if {"E26190", "E26180", "E25980", "E25960"}.issubset(set(puf.columns)):
        s_corp_income = puf["E26190"].fillna(0) - puf["E26180"].fillna(0)
        partnership_income = puf["E25980"].fillna(0) - puf["E25960"].fillna(0)
        result["partnership_s_corp_income"] = s_corp_income + partnership_income
    if {
        "E30400",
        "E30500",
        "E00900",
        "E02100",
        "E25940",
        "E25980",
        "E25920",
        "E25960",
    }.issubset(set(puf.columns)):
        se_deduction_factor = 0.9235
        taxable_se = puf["E30400"].fillna(0) + puf["E30500"].fillna(0)
        gross_se = taxable_se / se_deduction_factor
        schedule_c_f_income = puf["E00900"].fillna(0) + puf["E02100"].fillna(0)
        has_partnership = (
            puf["E25940"].fillna(0)
            + puf["E25980"].fillna(0)
            - puf["E25920"].fillna(0)
            - puf["E25960"].fillna(0)
        ) != 0
        result["partnership_se_income"] = np.where(
            has_partnership,
            gross_se - schedule_c_f_income,
            0.0,
        )
    if "T27800" in puf.columns:
        result["farm_income"] = puf["T27800"].fillna(0)
    if {"E26390", "E26400"}.issubset(set(puf.columns)):
        result["estate_income"] = puf["E26390"].fillna(0) - puf["E26400"].fillna(0)
    if {"E01500", "E01700"}.issubset(set(puf.columns)):
        result["tax_exempt_pension_income"] = (
            puf["E01500"].fillna(0) - puf["E01700"].fillna(0)
        )
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

    if "RECID" in puf.columns:
        result["_puf_recid"] = pd.to_numeric(puf["RECID"], errors="coerce")
    if "AGERANGE" in puf.columns:
        result["_puf_agerange"] = pd.to_numeric(puf["AGERANGE"], errors="coerce")
    if "EARNSPLIT" in puf.columns:
        result["_puf_earnsplit"] = pd.to_numeric(puf["EARNSPLIT"], errors="coerce")
    if "GENDER" in puf.columns:
        result["_puf_gender"] = pd.to_numeric(puf["GENDER"], errors="coerce")
    for dependent_idx in range(1, 4):
        raw_column = f"AGEDP{dependent_idx}"
        if raw_column in puf.columns:
            result[f"_puf_agedp{dependent_idx}"] = pd.to_numeric(
                puf[raw_column],
                errors="coerce",
            )

    if impute_pre_tax_contributions:
        model = pre_tax_contribution_model
        if model is None:
            try:
                model = _default_pe_style_puf_pre_tax_contribution_model()
            except (ImportError, ValueError):
                model = None
        if model is not None:
            predictor_frame = result.loc[:, model.predictors].copy()
            predictor_frame = predictor_frame.apply(
                lambda column: pd.to_numeric(column, errors="coerce").fillna(0.0)
            )
            predictions = model.fitted_model.predict(X_test=predictor_frame)
            result[model.imputed_variable] = pd.to_numeric(
                predictions[model.imputed_variable],
                errors="coerce",
            ).fillna(0.0)

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


def _decode_puf_filer_age(age_range: int | float | None, *, fallback: float = 40.0) -> int:
    if age_range is None or pd.isna(age_range):
        return int(fallback)
    age_code = int(age_range)
    if age_code == 0:
        return int(fallback)
    age_decode = {
        1: 18,
        2: 26,
        3: 35,
        4: 45,
        5: 55,
        6: 65,
        7: 80,
    }
    lower = age_decode.get(age_code)
    upper = age_decode.get(age_code + 1)
    if lower is None or upper is None:
        return int(fallback)
    return int(lower + (upper - lower) / 2)


def _decode_puf_dependent_age(age_range: int | float | None) -> int:
    if age_range is None or pd.isna(age_range):
        return 0
    age_code = int(age_range)
    if age_code == 0:
        return 0
    age_decode = {
        0: 0,
        1: 0,
        2: 5,
        3: 13,
        4: 17,
        5: 19,
        6: 25,
        7: 30,
    }
    lower = age_decode.get(age_code, 0)
    upper = age_decode.get(age_code + 1, lower)
    if upper <= lower:
        return int(lower)
    return int(lower + (upper - lower) / 2)


def _puf_joint_head_share(row: pd.Series, *, default: float = 0.6) -> float:
    earnsplit = row.get("_puf_earnsplit")
    if earnsplit is None or pd.isna(earnsplit):
        return default
    split_code = int(earnsplit)
    if split_code <= 0:
        return 1.0
    split_decodes = {
        1: 0.0,
        2: 0.25,
        3: 0.75,
        4: 1.0,
        5: 1.0,
    }
    lower = split_decodes.get(split_code)
    upper = split_decodes.get(split_code + 1)
    if lower is None or upper is None:
        return default
    return float(1.0 - ((lower + upper) / 2.0))


def _is_puf_numeric_split_column(df: pd.DataFrame, column: str) -> bool:
    if column in PUF_PERSON_EXPANSION_PRESERVE_COLUMNS:
        return False
    if column.startswith("_"):
        return False
    return pd.api.types.is_numeric_dtype(df[column])


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


def _default_cps_reference_year(target_year: int) -> int:
    return min(max(target_year - 1, 2021), 2023)


def _social_security_age_bucket(ages: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(ages, errors="coerce"),
        bins=SOCIAL_SECURITY_SHARE_AGE_BINS,
        labels=SOCIAL_SECURITY_SHARE_AGE_LABELS,
        right=False,
        include_lowest=True,
    )


def _normalize_social_security_split_strategy(strategy: str | None) -> str:
    resolved = (strategy or SOCIAL_SECURITY_SPLIT_STRATEGY_GROUPED_SHARE).strip().lower()
    allowed = {
        SOCIAL_SECURITY_SPLIT_STRATEGY_GROUPED_SHARE,
        SOCIAL_SECURITY_SPLIT_STRATEGY_PE_QRF,
        SOCIAL_SECURITY_SPLIT_STRATEGY_AGE_HEURISTIC,
    }
    if resolved not in allowed:
        raise ValueError(
            "social_security_split_strategy must be one of "
            f"{sorted(allowed)}; got {strategy!r}"
        )
    return resolved


def _build_pe_style_social_security_predictor_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    if "age" in frame.columns:
        result["age"] = pd.to_numeric(frame["age"], errors="coerce").astype(float)
    if "is_male" in frame.columns:
        result["is_male"] = pd.to_numeric(frame["is_male"], errors="coerce").astype(float)
    elif "sex" in frame.columns:
        sex = pd.to_numeric(frame["sex"], errors="coerce")
        result["is_male"] = pd.Series(
            np.where(sex == 1, 1.0, np.where(sex == 2, 0.0, np.nan)),
            index=frame.index,
            dtype=float,
        )
    if "tax_unit_is_joint" in frame.columns:
        result["tax_unit_is_joint"] = pd.to_numeric(
            frame["tax_unit_is_joint"], errors="coerce"
        ).astype(float)
    elif "filing_status" in frame.columns:
        filing_status = frame["filing_status"].astype(str)
        result["tax_unit_is_joint"] = (filing_status == "JOINT").astype(float)
    if "is_tax_unit_head" in frame.columns:
        result["is_tax_unit_head"] = pd.to_numeric(
            frame["is_tax_unit_head"], errors="coerce"
        ).astype(float)
    elif "is_head" in frame.columns:
        result["is_tax_unit_head"] = pd.to_numeric(
            frame["is_head"], errors="coerce"
        ).astype(float)
    if "is_tax_unit_dependent" in frame.columns:
        result["is_tax_unit_dependent"] = pd.to_numeric(
            frame["is_tax_unit_dependent"], errors="coerce"
        ).astype(float)
    elif "is_dependent" in frame.columns:
        result["is_tax_unit_dependent"] = pd.to_numeric(
            frame["is_dependent"], errors="coerce"
        ).astype(float)
    return result


def _fit_puf_social_security_share_model_from_reference(
    reference_persons: pd.DataFrame,
) -> GroupedShareModel:
    work = reference_persons.copy()
    if "weight" not in work.columns:
        work["weight"] = 1.0
    work["age_bucket"] = _social_security_age_bucket(
        work.get("age", pd.Series(np.nan, index=work.index))
    )
    return fit_grouped_share_model(
        work,
        explicit_component_columns=SOCIAL_SECURITY_SHARE_EXPLICIT_COMPONENTS,
        implicit_component_column=SOCIAL_SECURITY_SHARE_IMPLICIT_COMPONENT,
        feature_sets=(("age_bucket",),),
        weight_column="weight",
    )


def _default_puf_social_security_share_model(
    *,
    cps_reference_year: int,
    cache_dir: Path | None,
) -> GroupedShareModel:
    cps_dataset = load_cps_asec(
        year=cps_reference_year,
        cache_dir=cache_dir,
        download=True,
    )
    return _fit_puf_social_security_share_model_from_reference(
        cps_dataset.persons.to_pandas()
    )


def _fit_pe_style_puf_social_security_qrf_model_from_reference(
    reference_persons: pd.DataFrame,
    *,
    min_training_records: int = MIN_PE_STYLE_SOCIAL_SECURITY_QRF_TRAINING_RECORDS,
) -> PEStyleQRFShareModel:
    from microimpute.models.qrf import QRF

    total_social_security = _numeric_series(reference_persons, "social_security")
    has_social_security = total_social_security > 0.0
    if int(has_social_security.sum()) < min_training_records:
        raise ValueError(
            "PE-style QRF Social Security split requires at least "
            f"{min_training_records} positive training rows"
        )

    predictor_frame = _build_pe_style_social_security_predictor_frame(
        reference_persons.loc[has_social_security]
    )
    available_predictors = tuple(
        predictor
        for predictor in PE_STYLE_SOCIAL_SECURITY_QRF_PREDICTORS
        if predictor in predictor_frame.columns
        and predictor_frame[predictor].notna().any()
    )
    if not available_predictors:
        raise ValueError(
            "PE-style QRF Social Security split requires at least one predictor"
        )

    train = predictor_frame.loc[:, available_predictors].copy()
    total = total_social_security.loc[has_social_security].to_numpy(dtype=float)
    share_prediction_columns: list[str] = []
    for component in SOCIAL_SECURITY_SHARE_COMPONENTS:
        share_column = f"{component}_share"
        share_prediction_columns.append(share_column)
        component_values = _numeric_series(
            reference_persons.loc[has_social_security],
            component,
        ).to_numpy(dtype=float)
        train[share_column] = np.where(total > 0.0, component_values / total, 0.0)

    qrf = QRF(log_level="WARNING", memory_efficient=True)
    fitted_model = qrf.fit(
        X_train=train.loc[:, [*available_predictors, *share_prediction_columns]],
        predictors=list(available_predictors),
        imputed_variables=share_prediction_columns,
        n_jobs=1,
    )
    return PEStyleQRFShareModel(
        predictors=available_predictors,
        component_columns=SOCIAL_SECURITY_SHARE_COMPONENTS,
        share_prediction_columns=tuple(share_prediction_columns),
        fitted_model=fitted_model,
    )


def _default_pe_style_puf_social_security_share_model(
    *,
    cps_reference_year: int,
    cache_dir: Path | None,
) -> PEStyleQRFShareModel:
    cps_dataset = load_cps_asec(
        year=cps_reference_year,
        cache_dir=cache_dir,
        download=True,
    )
    return _fit_pe_style_puf_social_security_qrf_model_from_reference(
        cps_dataset.persons.to_pandas()
    )


@lru_cache(maxsize=1)
def _default_pe_style_puf_pre_tax_contribution_model() -> PEStyleQRFImputationModel:
    from microimpute.models.qrf import QRF
    from policyengine_us import Microsimulation
    from policyengine_us_data.datasets.cps import CPS_2021

    predictors = ("employment_income", "age", "is_male")
    cps = Microsimulation(dataset=CPS_2021)
    cps.subsample(10_000)
    cps_df = cps.calculate_dataframe(
        [*predictors, "household_weight", "pre_tax_contributions"]
    )
    train = cps_df.loc[:, [*predictors, "pre_tax_contributions"]].copy()
    train = train.apply(lambda column: pd.to_numeric(column, errors="coerce").fillna(0.0))

    qrf = QRF(log_level="WARNING", memory_efficient=True)
    fitted_model = qrf.fit(
        X_train=train,
        predictors=list(predictors),
        imputed_variables=["pre_tax_contributions"],
        n_jobs=1,
    )
    return PEStyleQRFImputationModel(
        predictors=predictors,
        imputed_variable="pre_tax_contributions",
        fitted_model=fitted_model,
    )


def _strategy_social_security_share_model_loader(
    strategy: str,
) -> Callable[[int, Path | None], SocialSecurityShareModel]:
    if strategy == SOCIAL_SECURITY_SPLIT_STRATEGY_PE_QRF:
        return lambda year, cache_dir: _default_pe_style_puf_social_security_share_model(
            cps_reference_year=year,
            cache_dir=cache_dir,
        )
    if strategy == SOCIAL_SECURITY_SPLIT_STRATEGY_AGE_HEURISTIC:
        return lambda year, cache_dir: _age_heuristic_puf_social_security_share_model()
    return lambda year, cache_dir: _default_puf_social_security_share_model(
        cps_reference_year=year,
        cache_dir=cache_dir,
    )


def _age_heuristic_puf_social_security_share_model() -> GroupedShareModel:
    reference = pd.DataFrame(
        {
            "age_bucket": list(SOCIAL_SECURITY_SHARE_AGE_LABELS),
            "weight": [1.0] * len(SOCIAL_SECURITY_SHARE_AGE_LABELS),
            "social_security_retirement": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            "social_security_disability": [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "social_security_survivors": [0.0] * len(SOCIAL_SECURITY_SHARE_AGE_LABELS),
            "social_security_dependents": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    return fit_grouped_share_model(
        reference,
        explicit_component_columns=SOCIAL_SECURITY_SHARE_EXPLICIT_COMPONENTS,
        implicit_component_column=SOCIAL_SECURITY_SHARE_IMPLICIT_COMPONENT,
        feature_sets=(("age_bucket",),),
        weight_column="weight",
    )


def _predict_puf_social_security_component_shares(
    persons: pd.DataFrame,
    *,
    share_model: SocialSecurityShareModel,
) -> pd.DataFrame:
    if isinstance(share_model, GroupedShareModel):
        features = persons.loc[:, []].copy()
        features["age_bucket"] = _social_security_age_bucket(
            persons.get("age", pd.Series(np.nan, index=persons.index))
        )
        return predict_grouped_component_shares(features, share_model)

    predictors = _build_pe_style_social_security_predictor_frame(persons)
    X_test = pd.DataFrame(index=persons.index)
    for predictor in share_model.predictors:
        if predictor in predictors.columns:
            X_test[predictor] = (
                pd.to_numeric(predictors[predictor], errors="coerce")
                .fillna(0.0)
                .astype(float)
            )
        else:
            X_test[predictor] = 0.0
    predictions = share_model.fitted_model.predict(X_test=X_test)

    shares = pd.DataFrame(index=persons.index)
    total = np.zeros(len(persons), dtype=float)
    for component, share_column in zip(
        share_model.component_columns,
        share_model.share_prediction_columns,
        strict=True,
    ):
        source_column = (
            share_column if share_column in predictions.columns else component
        )
        if source_column in predictions.columns:
            values = np.clip(
                pd.to_numeric(predictions[source_column], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float),
                0.0,
                1.0,
            )
        else:
            values = np.zeros(len(persons), dtype=float)
        shares[component] = values
        total += values

    positive_total = total > 0.0
    for component in share_model.component_columns:
        shares[component] = np.where(
            positive_total,
            shares[component].to_numpy(dtype=float) / total,
            0.0,
        )
    return shares


def _impute_puf_social_security_components(
    persons: pd.DataFrame,
    *,
    share_model: SocialSecurityShareModel,
) -> pd.DataFrame:
    result = persons.copy()
    total_social_security = _numeric_series(result, "social_security")
    if float(total_social_security.sum()) <= 0.0:
        for component in SOCIAL_SECURITY_SHARE_COMPONENTS:
            result[component] = 0.0
        return result

    shares = _predict_puf_social_security_component_shares(
        result,
        share_model=share_model,
    )
    for component in SOCIAL_SECURITY_SHARE_COMPONENTS:
        result[component] = total_social_security * shares[component]
    return result


def _add_derived_income_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result = normalize_dividend_columns(result)
    employment_income = _numeric_series(result, "employment_income")
    self_employment_income = _numeric_series(result, "self_employment_income")
    taxable_interest_income = _numeric_series(result, "taxable_interest_income")
    ordinary_dividend_income = _numeric_series(result, "ordinary_dividend_income")
    short_term_capital_gains = _numeric_series(result, "short_term_capital_gains")
    long_term_capital_gains = _numeric_series(result, "long_term_capital_gains")
    taxable_pension_income = _numeric_series(result, "taxable_pension_income")
    gross_social_security = _numeric_series(result, "gross_social_security")
    if "age" in result.columns:
        ages = pd.to_numeric(result["age"], errors="coerce").fillna(0.0).astype(float)
    else:
        ages = pd.Series(0.0, index=result.index, dtype=float)
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
    )
    result["pension_income"] = taxable_pension_income
    result["social_security"] = gross_social_security
    result["social_security_retirement"] = (
        gross_social_security.where(ages >= MINIMUM_SOCIAL_SECURITY_RETIREMENT_AGE, 0.0)
        .astype(float)
    )
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
    split_columns = [column for column in df.columns if _is_puf_numeric_split_column(df, column)]

    for idx, row in df.iterrows():
        filing_status = row.get("filing_status", "SINGLE")
        exemptions = int(pd.to_numeric(row.get("exemptions_count", 1), errors="coerce") or 1)
        has_pe_demographics = "_puf_agerange" in row.index and not pd.isna(row.get("_puf_agerange"))
        tax_unit_id = row.get("_puf_recid")
        if tax_unit_id is None or pd.isna(tax_unit_id):
            tax_unit_id = idx
        pe_tax_unit_id = str(int(tax_unit_id)) if pd.notna(tax_unit_id) else str(idx)

        # Create head record
        head = row.copy()
        head["is_head"] = 1
        head["is_spouse"] = 0
        head["is_dependent"] = 0
        head["person_id"] = f"{pe_tax_unit_id}:1" if has_pe_demographics else f"{idx}_head"
        head["tax_unit_id"] = pe_tax_unit_id if has_pe_demographics else str(idx)
        if has_pe_demographics:
            head["age"] = _decode_puf_filer_age(row.get("_puf_agerange"), fallback=row.get("age", 40.0))
            if pd.notna(row.get("_puf_gender")):
                head["is_male"] = float(int(row.get("_puf_gender")) == 1)
        records.append(head)

        # Create spouse record if joint filing
        if filing_status == "JOINT":
            spouse = row.copy()
            spouse["is_head"] = 0
            spouse["is_spouse"] = 1
            spouse["is_dependent"] = 0
            spouse["person_id"] = f"{pe_tax_unit_id}:2" if has_pe_demographics else f"{idx}_spouse"
            spouse["tax_unit_id"] = pe_tax_unit_id if has_pe_demographics else str(idx)

            if has_pe_demographics:
                spouse["age"] = _decode_puf_filer_age(row.get("_puf_agerange"), fallback=row.get("age", 40.0))
                if pd.notna(row.get("_puf_gender")):
                    spouse["is_male"] = float(int(row.get("_puf_gender")) != 1)
                head_share = _puf_joint_head_share(row)
                for column in split_columns:
                    amount = float(pd.to_numeric(row.get(column), errors="coerce") or 0.0)
                    head[column] = amount * head_share
                    spouse[column] = amount * (1.0 - head_share)
            else:
                head, spouse = _allocate_joint_tax_unit_amounts(row, head, spouse)
            # Spouse weight is same as head (we'll deduplicate in calibration)
            records.append(spouse)
            exemptions -= 1

        exemptions -= 1
        if has_pe_demographics:
            for dependent_idx in range(min(3, max(exemptions, 0))):
                dependent = row.copy()
                dependent["is_head"] = 0
                dependent["is_spouse"] = 0
                dependent["is_dependent"] = 1
                dependent["person_id"] = f"{pe_tax_unit_id}:{dependent_idx + 3}"
                dependent["tax_unit_id"] = pe_tax_unit_id
                dependent["age"] = _decode_puf_dependent_age(
                    row.get(f"_puf_agedp{dependent_idx + 1}")
                )
                dependent["is_male"] = 0.0
                for column in split_columns:
                    dependent[column] = 0.0
                records.append(dependent)

    result = pd.DataFrame(records).reset_index(drop=True)
    helper_columns = [column for column in result.columns if column in PUF_DEMOGRAPHIC_HELPER_COLUMNS]
    if helper_columns:
        result = result.drop(columns=helper_columns)
    result = _add_derived_income_columns(result)
    print(f"Expanded {len(df):,} tax units to {len(result):,} persons")

    return result


def load_puf(
    target_year: int = 2024,
    expand_persons: bool = True,
    cache_dir: Path | None = None,
    social_security_split_strategy: str = SOCIAL_SECURITY_SPLIT_STRATEGY_GROUPED_SHARE,
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
    df = map_puf_variables(raw, impute_pre_tax_contributions=True)

    # Uprate to target year
    df = uprate_puf(df, from_year=2015, to_year=target_year)

    # Expand to persons if requested
    if expand_persons:
        df = expand_to_persons(df)
        strategy = _normalize_social_security_split_strategy(
            social_security_split_strategy
        )
        try:
            if strategy == SOCIAL_SECURITY_SPLIT_STRATEGY_PE_QRF:
                share_model = _default_pe_style_puf_social_security_share_model(
                    cps_reference_year=_default_cps_reference_year(target_year),
                    cache_dir=cache_dir,
                )
            elif strategy == SOCIAL_SECURITY_SPLIT_STRATEGY_AGE_HEURISTIC:
                share_model = _age_heuristic_puf_social_security_share_model()
            else:
                share_model = _default_puf_social_security_share_model(
                    cps_reference_year=_default_cps_reference_year(target_year),
                    cache_dir=cache_dir,
                )
        except (FileNotFoundError, ImportError, ValueError):
            share_model = _age_heuristic_puf_social_security_share_model()
        df = _impute_puf_social_security_components(df, share_model=share_model)

    print(f"\nPUF loaded: {len(df):,} records")
    print(f"  Weight sum: {df['weight'].sum():,.0f}")

    return df


# Variables that PUF has but CPS doesn't (will be NaN in CPS)
PUF_EXCLUSIVE_VARS = [
    "pre_tax_contributions",
    "short_term_capital_gains",
    "long_term_capital_gains",
    "non_sch_d_capital_gains",
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
    tax_units = map_puf_variables(
        raw,
        random_seed=random_seed,
        impute_pre_tax_contributions=True,
    )
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
    cps_reference_year: int | None = None
    social_security_split_strategy: str = SOCIAL_SECURITY_SPLIT_STRATEGY_GROUPED_SHARE
    shareability: Shareability = Shareability.PUBLIC
    loader: Callable[[Path | None], tuple[Path, Path | None]] | None = None
    social_security_share_model_loader: (
        Callable[[int, Path | None], SocialSecurityShareModel] | None
    ) = None
    _descriptor_cache: SourceDescriptor | None = None
    _social_security_share_model_cache: dict[tuple[int, str], SocialSecurityShareModel] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

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
        cps_reference_year = int(
            provider_filters.get(
                "cps_reference_year",
                self.cps_reference_year or _default_cps_reference_year(target_year),
            )
        )
        social_security_split_strategy = _normalize_social_security_split_strategy(
            provider_filters.get(
                "social_security_split_strategy",
                self.social_security_split_strategy,
            )
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
        persons = _impute_puf_social_security_components(
            persons,
            share_model=self._load_social_security_share_model(
                cps_reference_year,
                social_security_split_strategy,
            ),
        )
        frame = _build_puf_observation_frame(
            tax_units=tax_units,
            persons=persons,
            source_name=f"irs_soi_puf_{target_year}",
            shareability=self.shareability,
        )
        self._descriptor_cache = frame.source
        return apply_source_query(frame, query)

    def _load_social_security_share_model(
        self,
        cps_reference_year: int,
        strategy: str,
    ) -> SocialSecurityShareModel:
        cache_key = (cps_reference_year, strategy)
        cached = self._social_security_share_model_cache.get(cache_key)
        if cached is not None:
            return cached
        loader = (
            self.social_security_share_model_loader
            or _strategy_social_security_share_model_loader(strategy)
        )
        try:
            model = loader(cps_reference_year, self.cache_dir)
        except (FileNotFoundError, ImportError, ValueError):
            model = _age_heuristic_puf_social_security_share_model()
        self._social_security_share_model_cache[cache_key] = model
        return model


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
