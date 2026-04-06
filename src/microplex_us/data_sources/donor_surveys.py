"""Spec-driven donor survey providers aligned with PE-US-data source-impute."""

from __future__ import annotations

import pickle
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

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

from microplex_us.pe_source_impute_specs import (
    PESourceImputeBlockSpec,
    get_pe_source_impute_block_spec,
    resolve_sipp_source_impute_block_spec,
)
from microplex_us.pipelines.pe_native_scores import (
    build_policyengine_us_data_subprocess_env,
    resolve_policyengine_us_data_python,
    resolve_policyengine_us_data_repo_root,
)
from microplex_us.source_registry import resolve_source_variable_capabilities

try:
    from huggingface_hub import hf_hub_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

PERSON_OBSERVATION_EXCLUDED_COLUMNS = (
    "person_id",
    "household_id",
    "weight",
    "year",
)
HOUSEHOLD_OBSERVATION_EXCLUDED_COLUMNS = (
    "household_id",
    "household_weight",
    "year",
)

PE_ACS_LOADER_SCRIPT = dedent(
    """
import pickle
import sys
import numpy as np
import pandas as pd

from policyengine_us_data.datasets.acs.acs import ACS_2022

out_path = sys.argv[1]
sample_n = None if sys.argv[2] == "None" else int(sys.argv[2])
random_seed = int(sys.argv[3])

data = ACS_2022().load_dataset()
household_index = pd.Index(data["household_id"])
person_households = pd.Index(data["person_household_id"])
household_to_row = pd.Series(
    np.arange(len(household_index), dtype=np.int64),
    index=household_index,
)
household_rows = household_to_row.loc[person_households].to_numpy()
persons = pd.DataFrame(
    {
        "person_id": data["person_id"],
        "household_id": person_households.to_numpy(),
        "age": data["age"],
        "sex": np.where(np.asarray(data["is_male"]).astype(bool), 1, 2),
        "is_male": np.asarray(data["is_male"]).astype(float),
        "is_household_head": np.asarray(data["is_household_head"]).astype(float),
        "employment_income": data["employment_income"],
        "self_employment_income": data["self_employment_income"],
        "social_security": data["social_security"],
        "taxable_pension_income": data["taxable_private_pension_income"],
        "rent": data["rent"],
        "real_estate_taxes": data["real_estate_taxes"],
        "state_fips": np.asarray(data["state_fips"])[household_rows],
        "weight": np.asarray(data["household_weight"])[household_rows],
        "year": np.full(len(data["person_id"]), 2022, dtype=np.int32),
    }
)
tenure_raw = pd.Series(np.asarray(data["tenure_type"])[household_rows]).map(
    lambda value: value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
)
persons["tenure_type"] = tenure_raw.map(
    {
        "OWNED_WITH_MORTGAGE": 1,
        "OWNED_OUTRIGHT": 1,
        "RENTED": 2,
        "NONE": 0,
    }
).fillna(0).astype(int)
persons["tenure"] = persons["tenure_type"]
persons["income"] = (
    pd.to_numeric(persons["employment_income"], errors="coerce").fillna(0.0)
    + pd.to_numeric(persons["self_employment_income"], errors="coerce").fillna(0.0)
    + pd.to_numeric(persons["social_security"], errors="coerce").fillna(0.0)
    + pd.to_numeric(persons["taxable_pension_income"], errors="coerce").fillna(0.0)
)

households = (
    persons[["household_id", "state_fips", "tenure", "weight", "year"]]
    .rename(columns={"weight": "household_weight"})
    .drop_duplicates(subset=["household_id"])
    .reset_index(drop=True)
)

if sample_n is not None and sample_n < len(households):
    sampled = households.sample(
        n=sample_n,
        random_state=random_seed,
        replace=False,
        weights=households["household_weight"],
    ).copy()
    keep = set(sampled["household_id"])
    households = sampled.sort_values(["household_id"]).reset_index(drop=True)
    persons = (
        persons[persons["household_id"].isin(keep)]
        .sort_values(["household_id", "person_id"])
        .reset_index(drop=True)
    )
else:
    households = households.sort_values(["household_id"]).reset_index(drop=True)
    persons = persons.sort_values(["household_id", "person_id"]).reset_index(drop=True)

with open(out_path, "wb") as handle:
    pickle.dump({"households": households, "persons": persons}, handle)
"""
)

PE_SCF_LOADER_SCRIPT = dedent(
    """
import pickle
import sys
import numpy as np
import pandas as pd

from policyengine_us_data.datasets.scf.scf import SCF_2022

out_path = sys.argv[1]
sample_n = None if sys.argv[2] == "None" else int(sys.argv[2])
random_seed = int(sys.argv[3])

data = SCF_2022().load_dataset()
persons = pd.DataFrame(
    {
        "household_id": np.arange(len(data["age"]), dtype=np.int64) + 1,
        "age": data["age"],
        "is_female": np.asarray(data["is_female"]).astype(float),
        "sex": np.where(np.asarray(data["is_female"]).astype(bool), 2, 1),
        "cps_race": data["cps_race"],
        "is_married": np.asarray(data["is_married"]).astype(float),
        "own_children_in_household": data["own_children_in_household"],
        "employment_income": data["employment_income"],
        "interest_dividend_income": data["interest_dividend_income"],
        "social_security_pension_income": data["social_security_pension_income"],
        "auto_loan_balance": data["auto_loan_balance"],
        "auto_loan_interest": data["auto_loan_interest"],
        "weight": data["wgt"],
        "year": np.full(len(data["age"]), 2022, dtype=np.int32),
    }
)
persons["net_worth"] = data["net_worth"] if "net_worth" in data else data["networth"]
persons["person_id"] = persons["household_id"]
persons["state_fips"] = 0
persons["tenure"] = 0
persons["income"] = pd.to_numeric(persons["employment_income"], errors="coerce").fillna(0.0)

households = persons[
    ["household_id", "state_fips", "tenure", "weight", "year"]
].rename(columns={"weight": "household_weight"})

if sample_n is not None and sample_n < len(households):
    sampled = households.sample(
        n=sample_n,
        random_state=random_seed,
        replace=False,
        weights=households["household_weight"],
    ).copy()
    keep = set(sampled["household_id"])
    households = sampled.sort_values(["household_id"]).reset_index(drop=True)
    persons = (
        persons[persons["household_id"].isin(keep)]
        .sort_values(["household_id", "person_id"])
        .reset_index(drop=True)
    )
else:
    households = households.sort_values(["household_id"]).reset_index(drop=True)
    persons = persons.sort_values(["household_id", "person_id"]).reset_index(drop=True)

with open(out_path, "wb") as handle:
    pickle.dump({"households": households, "persons": persons}, handle)
"""
)


@dataclass(frozen=True)
class DonorSurveyTables:
    """Canonical household/person tables for one donor survey block."""

    households: pd.DataFrame
    persons: pd.DataFrame


DonorSurveyTablesLoader = Callable[..., DonorSurveyTables]


def _descriptor_from_tables(
    *,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    name: str,
    shareability: Shareability,
    archetype: SourceArchetype | None,
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
        shareability=shareability,
        time_structure=TimeStructure.REPEATED_CROSS_SECTION,
        archetype=archetype,
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


def _build_static_descriptor(
    *,
    spec: PESourceImputeBlockSpec,
    shareability: Shareability,
) -> SourceDescriptor:
    return SourceDescriptor(
        name=spec.descriptor_name,
        shareability=shareability,
        time_structure=TimeStructure.REPEATED_CROSS_SECTION,
        archetype=spec.archetype,
        observations=(
            EntityObservation(
                entity=EntityType.HOUSEHOLD,
                key_column="household_id",
                variable_names=spec.household_variables,
                weight_column="household_weight",
                period_column="year",
            ),
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=spec.person_variables,
                weight_column="weight",
                period_column="year",
            ),
        ),
    )


def _ensure_person_ids(persons: pd.DataFrame) -> pd.DataFrame:
    result = persons.copy()
    if "person_id" in result.columns:
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


def _sample_households_and_persons(
    *,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    sample_n: int | None,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    households = households.reset_index(drop=True)
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
    sampled_households = households.sample(
        n=sample_n,
        random_state=random_seed,
        replace=False,
        weights=sample_weights,
    ).copy()
    keep = set(sampled_households["household_id"])
    sampled_persons = persons[persons["household_id"].isin(keep)].copy()
    return (
        sampled_households.sort_values(["household_id"]).reset_index(drop=True),
        sampled_persons.sort_values(["household_id", "person_id"]).reset_index(drop=True),
    )


def _build_observation_frame(
    *,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    source_name: str,
    shareability: Shareability,
    archetype: SourceArchetype | None,
) -> ObservationFrame:
    normalized_households = households.copy()
    normalized_persons = _ensure_person_ids(persons)
    descriptor = _descriptor_from_tables(
        households=normalized_households,
        persons=normalized_persons,
        name=source_name,
        shareability=shareability,
        archetype=archetype,
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


def _run_policyengine_dataset_loader(
    *,
    script: str,
    sample_n: int | None,
    random_seed: int,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> DonorSurveyTables:
    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    resolved_python = resolve_policyengine_us_data_python(
        policyengine_us_data_python,
        repo_root=resolved_repo,
    )
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    with tempfile.TemporaryDirectory(prefix="microplex-us-donor-") as tempdir:
        payload_path = Path(tempdir) / "tables.pkl"
        subprocess.run(
            [
                str(resolved_python),
                "-c",
                script,
                str(payload_path),
                "None" if sample_n is None else str(int(sample_n)),
                str(int(random_seed)),
            ],
            check=True,
            cwd=resolved_repo,
            env=env,
        )
        with payload_path.open("rb") as handle:
            payload = pickle.load(handle)
    return DonorSurveyTables(
        households=payload["households"],
        persons=payload["persons"],
    )


def _default_acs_tables_loader(
    *,
    year: int,
    sample_n: int | None,
    random_seed: int,
    cache_dir: Path | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> DonorSurveyTables:
    _ = cache_dir
    if int(year) != 2022:
        raise ValueError("ACS donor source provider currently supports year=2022 only")
    return _run_policyengine_dataset_loader(
        script=PE_ACS_LOADER_SCRIPT,
        sample_n=sample_n,
        random_seed=random_seed,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
    )


def _default_scf_tables_loader(
    *,
    year: int,
    sample_n: int | None,
    random_seed: int,
    cache_dir: Path | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> DonorSurveyTables:
    _ = cache_dir
    if int(year) != 2022:
        raise ValueError("SCF donor source provider currently supports year=2022 only")
    return _run_policyengine_dataset_loader(
        script=PE_SCF_LOADER_SCRIPT,
        sample_n=sample_n,
        random_seed=random_seed,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
    )


def _download_policyengine_us_data_file(
    *,
    filename: str,
    cache_dir: Path | None,
) -> Path:
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "microplex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / filename
    if destination.exists():
        return destination
    if not HF_AVAILABLE:
        raise ImportError("huggingface_hub required: pip install huggingface_hub")
    downloaded = hf_hub_download(
        repo_id="PolicyEngine/policyengine-us-data",
        filename=filename,
        repo_type="model",
        local_dir=cache_dir,
    )
    return Path(downloaded)


def _default_sipp_tips_tables_loader(
    *,
    year: int,
    sample_n: int | None,
    random_seed: int,
    cache_dir: Path | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> DonorSurveyTables:
    _ = policyengine_us_data_repo, policyengine_us_data_python
    if int(year) != 2023:
        raise ValueError("SIPP tips provider currently supports year=2023 only")
    sipp_path = _download_policyengine_us_data_file(
        filename="pu2023_slim.csv",
        cache_dir=cache_dir,
    )
    df = pd.read_csv(sipp_path)
    txamt_columns = [column for column in df.columns if "TXAMT" in column]
    df["tip_income"] = df[txamt_columns].fillna(0).sum(axis=1) * 12
    df["employment_income"] = pd.to_numeric(df["TPTOTINC"], errors="coerce").fillna(0.0) * 12
    df["age"] = pd.to_numeric(df["TAGE"], errors="coerce").fillna(0).astype(int)
    df["sex"] = pd.to_numeric(df["ESEX"], errors="coerce").fillna(0).astype(int)
    df["income"] = df["employment_income"]
    df["year"] = int(year)
    df["household_id"] = df["SSUID"].astype(str) + ":" + df["MONTHCODE"].astype(str)
    df["person_id"] = (
        df["SSUID"].astype(str)
        + ":"
        + df["MONTHCODE"].astype(str)
        + ":"
        + df["PNUM"].astype(str)
    )
    df["weight"] = pd.to_numeric(df["WPFINWGT"], errors="coerce").fillna(0.0)
    df["is_under_18"] = df["age"] < 18
    df["is_under_6"] = df["age"] < 6
    df["count_under_18"] = (
        df.groupby("household_id")["is_under_18"].transform("sum").astype(float)
    )
    df["count_under_6"] = (
        df.groupby("household_id")["is_under_6"].transform("sum").astype(float)
    )
    df["state_fips"] = 0
    df["tenure"] = 0
    households = (
        df[["household_id", "weight", "state_fips", "tenure", "year"]]
        .rename(columns={"weight": "household_weight"})
        .drop_duplicates(subset=["household_id"])
        .reset_index(drop=True)
    )
    persons = df[
        [
            "person_id",
            "household_id",
            "age",
            "sex",
            "employment_income",
            "income",
            "tip_income",
            "count_under_18",
            "count_under_6",
            "weight",
            "year",
        ]
    ].copy()
    households, persons = _sample_households_and_persons(
        households=households,
        persons=persons,
        sample_n=sample_n,
        random_seed=random_seed,
    )
    return DonorSurveyTables(households=households, persons=persons)


def _default_sipp_assets_tables_loader(
    *,
    year: int,
    sample_n: int | None,
    random_seed: int,
    cache_dir: Path | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> DonorSurveyTables:
    _ = policyengine_us_data_repo, policyengine_us_data_python
    if int(year) != 2023:
        raise ValueError("SIPP assets provider currently supports year=2023 only")
    sipp_path = _download_policyengine_us_data_file(
        filename="pu2023.csv",
        cache_dir=cache_dir,
    )
    df = pd.read_csv(
        sipp_path,
        delimiter="|",
        usecols=(
            "SSUID",
            "PNUM",
            "MONTHCODE",
            "WPFINWGT",
            "TAGE",
            "ESEX",
            "EMS",
            "TPTOTINC",
            "TVAL_BANK",
            "TVAL_STMF",
            "TVAL_BOND",
        ),
    )
    df = df[df["MONTHCODE"] == 12].copy()
    df["bank_account_assets"] = pd.to_numeric(df["TVAL_BANK"], errors="coerce").fillna(0.0)
    df["stock_assets"] = pd.to_numeric(df["TVAL_STMF"], errors="coerce").fillna(0.0)
    df["bond_assets"] = pd.to_numeric(df["TVAL_BOND"], errors="coerce").fillna(0.0)
    df["age"] = pd.to_numeric(df["TAGE"], errors="coerce").fillna(0).astype(int)
    df["sex"] = pd.to_numeric(df["ESEX"], errors="coerce").fillna(0).astype(int)
    df["is_female"] = df["sex"].eq(2).astype(float)
    df["is_married"] = pd.to_numeric(df["EMS"], errors="coerce").fillna(0).eq(1).astype(float)
    df["employment_income"] = pd.to_numeric(df["TPTOTINC"], errors="coerce").fillna(0.0) * 12
    df["income"] = df["employment_income"]
    df["year"] = int(year)
    df["household_id"] = df["SSUID"].astype(str)
    df["person_id"] = df["SSUID"].astype(str) + ":" + df["PNUM"].astype(str)
    df["weight"] = pd.to_numeric(df["WPFINWGT"], errors="coerce").fillna(0.0)
    df["is_under_18"] = df["age"] < 18
    df["count_under_18"] = (
        df.groupby("household_id")["is_under_18"].transform("sum").astype(float)
    )
    df["state_fips"] = 0
    df["tenure"] = 0
    households = (
        df[["household_id", "weight", "state_fips", "tenure", "year"]]
        .rename(columns={"weight": "household_weight"})
        .drop_duplicates(subset=["household_id"])
        .reset_index(drop=True)
    )
    persons = df[
        [
            "person_id",
            "household_id",
            "age",
            "sex",
            "is_female",
            "is_married",
            "employment_income",
            "income",
            "count_under_18",
            "bank_account_assets",
            "stock_assets",
            "bond_assets",
            "weight",
            "year",
        ]
    ].copy()
    households, persons = _sample_households_and_persons(
        households=households,
        persons=persons,
        sample_n=sample_n,
        random_seed=random_seed,
    )
    return DonorSurveyTables(households=households, persons=persons)


BLOCK_LOADERS: dict[str, DonorSurveyTablesLoader] = {
    "acs": _default_acs_tables_loader,
    "sipp_tips": _default_sipp_tips_tables_loader,
    "sipp_assets": _default_sipp_assets_tables_loader,
    "scf": _default_scf_tables_loader,
}


DonorSurveyProviderSpec = PESourceImputeBlockSpec


def _default_loader_for_spec(spec: PESourceImputeBlockSpec) -> DonorSurveyTablesLoader:
    return BLOCK_LOADERS[spec.key]


def resolve_sipp_donor_survey_spec(block: str) -> DonorSurveyProviderSpec:
    return resolve_sipp_source_impute_block_spec(block)


class DonorSurveySourceProvider:
    """Generic source provider for one donor survey block."""

    def __init__(
        self,
        *,
        spec: DonorSurveyProviderSpec,
        year: int | None = None,
        cache_dir: str | Path | None = None,
        shareability: Shareability = Shareability.PUBLIC,
        loader: DonorSurveyTablesLoader | None = None,
        policyengine_us_data_repo: str | Path | None = None,
        policyengine_us_data_python: str | Path | None = None,
    ) -> None:
        self.spec = spec
        self.year = int(spec.default_year if year is None else year)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.shareability = shareability
        self.loader = loader
        self.policyengine_us_data_repo = policyengine_us_data_repo
        self.policyengine_us_data_python = policyengine_us_data_python
        self._descriptor_cache: SourceDescriptor | None = None

    @property
    def descriptor(self) -> SourceDescriptor:
        if self._descriptor_cache is not None:
            return self._descriptor_cache
        return _build_static_descriptor(
            spec=self.spec,
            shareability=self.shareability,
        )

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        query = query or SourceQuery()
        provider_filters = query.provider_filters
        loader = self.loader or _default_loader_for_spec(self.spec)
        year = int(provider_filters.get("year", self.year))
        tables = loader(
            year=year,
            sample_n=provider_filters.get("sample_n"),
            random_seed=int(provider_filters.get("random_seed", 0)),
            cache_dir=provider_filters.get("cache_dir", self.cache_dir),
            policyengine_us_data_repo=provider_filters.get(
                "policyengine_us_data_repo",
                self.policyengine_us_data_repo,
            ),
            policyengine_us_data_python=provider_filters.get(
                "policyengine_us_data_python",
                self.policyengine_us_data_python,
            ),
        )
        frame = _build_observation_frame(
            households=tables.households,
            persons=tables.persons,
            source_name=self.spec.source_name(year),
            shareability=self.shareability,
            archetype=self.spec.archetype,
        )
        self._descriptor_cache = frame.source
        return apply_source_query(frame, query)


class ACSSourceProvider(DonorSurveySourceProvider):
    """PolicyEngine-aligned ACS donor provider."""

    def __init__(
        self,
        *,
        year: int = get_pe_source_impute_block_spec("acs").default_year,
        shareability: Shareability = Shareability.PUBLIC,
        loader: DonorSurveyTablesLoader | None = None,
        policyengine_us_data_repo: str | Path | None = None,
        policyengine_us_data_python: str | Path | None = None,
    ) -> None:
        super().__init__(
            spec=get_pe_source_impute_block_spec("acs"),
            year=year,
            shareability=shareability,
            loader=loader,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
        )


class SIPPSourceProvider(DonorSurveySourceProvider):
    """PolicyEngine-aligned SIPP donor provider with block-level specs."""

    def __init__(
        self,
        *,
        block: str,
        year: int | None = None,
        cache_dir: str | Path | None = None,
        shareability: Shareability = Shareability.PUBLIC,
        loader: DonorSurveyTablesLoader | None = None,
    ) -> None:
        self.block = block
        super().__init__(
            spec=resolve_sipp_donor_survey_spec(block),
            year=year,
            cache_dir=cache_dir,
            shareability=shareability,
            loader=loader,
        )


class SIPPTipsSourceProvider(SIPPSourceProvider):
    """Backward-compatible alias for the SIPP tips donor block."""

    def __init__(
        self,
        *,
        year: int | None = None,
        cache_dir: str | Path | None = None,
        shareability: Shareability = Shareability.PUBLIC,
        loader: DonorSurveyTablesLoader | None = None,
    ) -> None:
        super().__init__(
            block="tips",
            year=year,
            cache_dir=cache_dir,
            shareability=shareability,
            loader=loader,
        )


class SIPPAssetsSourceProvider(SIPPSourceProvider):
    """Backward-compatible alias for the SIPP asset donor block."""

    def __init__(
        self,
        *,
        year: int | None = None,
        cache_dir: str | Path | None = None,
        shareability: Shareability = Shareability.PUBLIC,
        loader: DonorSurveyTablesLoader | None = None,
    ) -> None:
        super().__init__(
            block="assets",
            year=year,
            cache_dir=cache_dir,
            shareability=shareability,
            loader=loader,
        )


class SCFSourceProvider(DonorSurveySourceProvider):
    """PolicyEngine-aligned SCF donor provider."""

    def __init__(
        self,
        *,
        year: int = get_pe_source_impute_block_spec("scf").default_year,
        shareability: Shareability = Shareability.PUBLIC,
        loader: DonorSurveyTablesLoader | None = None,
        policyengine_us_data_repo: str | Path | None = None,
        policyengine_us_data_python: str | Path | None = None,
    ) -> None:
        super().__init__(
            spec=get_pe_source_impute_block_spec("scf"),
            year=year,
            shareability=shareability,
            loader=loader,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
        )
