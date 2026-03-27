"""US-specific CPS ASEC data helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR_CANDIDATES = (
    PACKAGE_ROOT / "data",
    PACKAGE_ROOT.parent / "microplex" / "data",
)
DEFAULT_DATA_DIR = next(
    (candidate for candidate in DEFAULT_DATA_DIR_CANDIDATES if candidate.exists()),
    DEFAULT_DATA_DIR_CANDIDATES[0],
)


def load_cps_asec(
    data_dir: str | Path | None = None,
    households_only: bool = False,
    persons_only: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed CPS ASEC household and person parquet files."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    else:
        data_dir = Path(data_dir)

    household_path = data_dir / "cps_asec_households.parquet"
    person_path = data_dir / "cps_asec_persons.parquet"

    if not household_path.exists() or not person_path.exists():
        raise FileNotFoundError(
            f"CPS ASEC data files not found in {data_dir}.\n"
            "Run the downloader in `microplex-us` or provide preprocessed parquet files."
        )

    if households_only:
        return pd.read_parquet(household_path)
    if persons_only:
        return pd.read_parquet(person_path)

    households = pd.read_parquet(household_path)
    persons = pd.read_parquet(person_path)
    return households, persons


def load_cps_for_synthesis(
    data_dir: str | Path | None = None,
    sample_fraction: float | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CPS ASEC and normalize it for the hierarchical synthesizer."""
    households, persons = load_cps_asec(data_dir)
    households = _prepare_household_data(households)
    persons = _prepare_person_data(persons)

    if sample_fraction is not None and 0 < sample_fraction < 1:
        rng = np.random.default_rng(random_state)
        sampled_household_ids = rng.choice(
            households["household_id"].unique(),
            size=int(len(households) * sample_fraction),
            replace=False,
        )
        households = households[households["household_id"].isin(sampled_household_ids)]
        persons = persons[persons["household_id"].isin(sampled_household_ids)]

    return households, persons


def _prepare_household_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure CPS household data has the columns expected by synthesis code."""
    result = df.copy()
    required_cols = {
        "household_id": lambda: np.arange(len(result)),
        "n_persons": lambda: np.ones(len(result)),
        "n_adults": lambda: np.ones(len(result)),
        "n_children": lambda: np.zeros(len(result)),
        "state_fips": lambda: np.zeros(len(result)),
        "tenure": lambda: np.ones(len(result)),
        "hh_weight": lambda: np.ones(len(result)),
    }

    for column, default_factory in required_cols.items():
        if column not in result.columns:
            result[column] = default_factory()

    for column in ["n_persons", "n_adults", "n_children", "state_fips", "tenure"]:
        result[column] = (
            pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)
        )

    result["hh_weight"] = (
        pd.to_numeric(result["hh_weight"], errors="coerce").fillna(1).astype(float)
    )
    result["n_persons"] = result["n_persons"].clip(lower=1)
    result["n_adults"] = result["n_adults"].clip(lower=1)
    return result


def _prepare_person_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure CPS person data has the columns expected by synthesis code."""
    result = df.copy()
    required_cols = {
        "person_id": lambda: np.arange(len(result)),
        "household_id": lambda: np.zeros(len(result), dtype=int),
        "age": lambda: np.full(len(result), 30),
        "sex": lambda: np.ones(len(result)),
        "income": lambda: np.zeros(len(result)),
        "employment_status": lambda: np.zeros(len(result)),
        "education": lambda: np.ones(len(result)),
        "relationship_to_head": lambda: np.ones(len(result)),
    }

    for column, default_factory in required_cols.items():
        if column not in result.columns:
            result[column] = default_factory()

    for column in [
        "age",
        "sex",
        "employment_status",
        "education",
        "relationship_to_head",
    ]:
        result[column] = (
            pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)
        )
    result["income"] = (
        pd.to_numeric(result["income"], errors="coerce").fillna(0).astype(float)
    )
    result["age"] = result["age"].clip(0, 120)
    return result


def create_sample_data(
    n_households: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create CPS-shaped sample data for examples and tests."""
    rng = np.random.default_rng(seed)

    n_persons = rng.choice(
        [1, 2, 3, 4, 5, 6, 7],
        n_households,
        p=[0.28, 0.34, 0.16, 0.12, 0.06, 0.03, 0.01],
    )
    households = pd.DataFrame(
        {
            "household_id": np.arange(n_households),
            "n_persons": n_persons,
            "state_fips": rng.choice(
                [
                    6, 48, 12, 36, 42, 17, 39, 13, 37, 26, 4, 34, 51, 53, 25,
                    47, 29, 18, 55, 21, 24, 41, 8, 22, 5, 28, 20, 31, 35, 23,
                ],
                n_households,
            ),
            "tenure": rng.choice([1, 2, 3], n_households, p=[0.65, 0.34, 0.01]),
            "hh_weight": rng.lognormal(8, 0.5, n_households),
        }
    )
    households["n_children"] = np.minimum(
        rng.binomial(households["n_persons"], 0.25),
        households["n_persons"] - 1,
    )
    households["n_adults"] = households["n_persons"] - households["n_children"]

    people: list[dict[str, float | int]] = []
    person_id = 0
    for _, household in households.iterrows():
        household_id = household["household_id"]
        n_adults = int(household["n_adults"])
        n_children = int(household["n_children"])

        for adult_index in range(n_adults):
            age = int(rng.integers(18, 85))
            education = int(rng.choice([1, 2, 3, 4], p=[0.10, 0.28, 0.30, 0.32]))
            if rng.random() < 0.15:
                income = 0.0
            else:
                base_income = float(rng.lognormal(10.5, 1.0))
                age_factor = 1 + 0.02 * min(age - 18, 30) - 0.01 * max(age - 55, 0)
                education_factor = 1 + 0.3 * education
                income = max(0.0, base_income * age_factor * education_factor)
            people.append(
                {
                    "person_id": person_id,
                    "household_id": household_id,
                    "age": age,
                    "sex": int(rng.choice([1, 2])),
                    "income": income,
                    "employment_status": int(rng.choice([0, 1, 2], p=[0.35, 0.60, 0.05])),
                    "education": education,
                    "relationship_to_head": 1 if adult_index == 0 else (2 if adult_index == 1 else 3),
                }
            )
            person_id += 1

        for _child_index in range(n_children):
            people.append(
                {
                    "person_id": person_id,
                    "household_id": household_id,
                    "age": int(rng.integers(0, 18)),
                    "sex": int(rng.choice([1, 2])),
                    "income": 0.0,
                    "employment_status": 0,
                    "education": 1,
                    "relationship_to_head": 4,
                }
            )
            person_id += 1

    persons = pd.DataFrame(people)
    return households, persons


def get_data_info(data_dir: str | Path | None = None) -> dict:
    """Report availability and shape of local CPS ASEC parquet files."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    else:
        data_dir = Path(data_dir)

    info = {
        "data_dir": str(data_dir),
        "households": {"exists": False},
        "persons": {"exists": False},
    }

    household_path = data_dir / "cps_asec_households.parquet"
    person_path = data_dir / "cps_asec_persons.parquet"

    if household_path.exists():
        households = pd.read_parquet(household_path)
        info["households"] = {
            "exists": True,
            "path": str(household_path),
            "size_mb": household_path.stat().st_size / 1e6,
            "n_records": len(households),
            "columns": list(households.columns),
        }

    if person_path.exists():
        persons = pd.read_parquet(person_path)
        info["persons"] = {
            "exists": True,
            "path": str(person_path),
            "size_mb": person_path.stat().st_size / 1e6,
            "n_records": len(persons),
            "columns": list(persons.columns),
        }

    return info


__all__ = [
    "DEFAULT_DATA_DIR",
    "load_cps_asec",
    "load_cps_for_synthesis",
    "_prepare_household_data",
    "_prepare_person_data",
    "create_sample_data",
    "get_data_info",
]
