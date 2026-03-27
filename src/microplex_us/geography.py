"""US-specific Census block geography helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from microplex.geography import (
    AtomicGeographyCrosswalk,
    GeographyProvider,
    GeographyQuery,
    ProbabilisticAtomicGeographyAssigner,
    nearest_numeric_partition_key,
)

STATE_LEN = 2
COUNTY_LEN = 3
TRACT_LEN = 6
BLOCK_LEN = 4

STATE_GEOID_LEN = STATE_LEN
COUNTY_GEOID_LEN = STATE_LEN + COUNTY_LEN
TRACT_GEOID_LEN = STATE_LEN + COUNTY_LEN + TRACT_LEN
BLOCK_GEOID_LEN = STATE_LEN + COUNTY_LEN + TRACT_LEN + BLOCK_LEN

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR_CANDIDATES = (
    PACKAGE_ROOT / "data",
    PACKAGE_ROOT.parent / "microplex" / "data",
)
DEFAULT_DATA_DIR = next(
    (candidate for candidate in DEFAULT_DATA_DIR_CANDIDATES if candidate.exists()),
    DEFAULT_DATA_DIR_CANDIDATES[0],
)
DEFAULT_BLOCK_PROBABILITIES_PATH = DEFAULT_DATA_DIR / "block_probabilities.parquet"


def load_block_probabilities(path: str | Path | None = None) -> pd.DataFrame:
    """Load US Census block probabilities from parquet."""
    path = DEFAULT_BLOCK_PROBABILITIES_PATH if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Block probabilities file not found at {path}.\n"
            "Run the US geography preparation pipeline first."
        )
    return pd.read_parquet(path)


def normalize_us_state_fips(value: Any) -> str:
    """Normalize US state FIPS values to two-character strings."""
    return str(int(round(float(value)))).zfill(2)


def derive_geographies(
    block_geoids: list[str] | np.ndarray | pd.Series,
    include_cd: bool = False,
    include_sld: bool = False,
    block_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Derive parent geographies from Census block GEOIDs."""
    geoids = pd.Series(block_geoids).astype(str)
    result = pd.DataFrame(
        {
            "block_geoid": geoids,
            "state_fips": geoids.str[:STATE_GEOID_LEN],
            "county_fips": geoids.str[:COUNTY_GEOID_LEN],
            "tract_geoid": geoids.str[:TRACT_GEOID_LEN],
        }
    )
    if include_cd or include_sld:
        block_data = load_block_probabilities() if block_data is None else block_data
    if include_cd:
        result["cd_id"] = geoids.map(dict(zip(block_data["geoid"], block_data["cd_id"])))
    if include_sld:
        if "sldu_id" in block_data.columns:
            result["sldu_id"] = geoids.map(dict(zip(block_data["geoid"], block_data["sldu_id"])))
        if "sldl_id" in block_data.columns:
            result["sldl_id"] = geoids.map(dict(zip(block_data["geoid"], block_data["sldl_id"])))
    return result


class BlockGeography(GeographyProvider):
    """US atomic-geography provider backed by Census blocks."""

    def __init__(
        self,
        data_path: str | Path | None = None,
        lazy_load: bool = True,
    ):
        self._data_path = data_path
        self._data: pd.DataFrame | None = None
        self._cd_lookup: dict[str, str] | None = None
        self._sldu_lookup: dict[str, str] | None = None
        self._sldl_lookup: dict[str, str] | None = None
        self._state_blocks: dict[str, pd.DataFrame] | None = None
        if not lazy_load:
            self._load_data()

    @classmethod
    def from_data(cls, data: pd.DataFrame) -> BlockGeography:
        instance = cls(lazy_load=True)
        instance._data = data.copy()
        return instance

    def _load_data(self) -> None:
        if self._data is None:
            self._data = load_block_probabilities(self._data_path)

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._load_data()
        return self._data

    @staticmethod
    @lru_cache(maxsize=100000)
    def get_state(block_geoid: str) -> str:
        return block_geoid[:STATE_GEOID_LEN]

    @staticmethod
    @lru_cache(maxsize=100000)
    def get_county(block_geoid: str) -> str:
        return block_geoid[:COUNTY_GEOID_LEN]

    @staticmethod
    @lru_cache(maxsize=100000)
    def get_tract(block_geoid: str) -> str:
        return block_geoid[:TRACT_GEOID_LEN]

    def get_cd(self, block_geoid: str) -> str | None:
        if self._cd_lookup is None:
            self._build_lookups()
        return self._cd_lookup.get(block_geoid)

    def get_sldu(self, block_geoid: str) -> str | None:
        if self._sldu_lookup is None:
            self._build_lookups()
        return self._sldu_lookup.get(block_geoid)

    def get_sldl(self, block_geoid: str) -> str | None:
        if self._sldl_lookup is None:
            self._build_lookups()
        return self._sldl_lookup.get(block_geoid)

    def _build_lookups(self) -> None:
        self._cd_lookup = dict(zip(self.data["geoid"], self.data["cd_id"]))
        self._sldu_lookup = (
            dict(zip(self.data["geoid"], self.data["sldu_id"]))
            if "sldu_id" in self.data.columns
            else {}
        )
        self._sldl_lookup = (
            dict(zip(self.data["geoid"], self.data["sldl_id"]))
            if "sldl_id" in self.data.columns
            else {}
        )

    def get_all_geographies(self, block_geoid: str) -> dict[str, str | None]:
        return {
            "state_fips": self.get_state(block_geoid),
            "county_fips": self.get_county(block_geoid),
            "tract_geoid": self.get_tract(block_geoid),
            "cd_id": self.get_cd(block_geoid),
            "sldu_id": self.get_sldu(block_geoid),
            "sldl_id": self.get_sldl(block_geoid),
        }

    def to_crosswalk(self) -> AtomicGeographyCrosswalk:
        crosswalk = self.data.copy()
        if "county_fips" not in crosswalk.columns and {"state_fips", "county"}.issubset(crosswalk.columns):
            crosswalk["county_fips"] = (
                crosswalk["state_fips"].astype(str) + crosswalk["county"].astype(str)
            )
        if "tract_geoid" not in crosswalk.columns and {"state_fips", "county", "tract"}.issubset(crosswalk.columns):
            crosswalk["tract_geoid"] = (
                crosswalk["state_fips"].astype(str)
                + crosswalk["county"].astype(str)
                + crosswalk["tract"].astype(str)
            )
        geography_columns = tuple(
            column
            for column in ("state_fips", "county_fips", "tract_geoid", "cd_id", "sldu_id", "sldl_id")
            if column in crosswalk.columns
        )
        return AtomicGeographyCrosswalk(
            data=crosswalk.rename(columns={"geoid": "block_geoid"}),
            atomic_id_column="block_geoid",
            geography_columns=geography_columns,
            probability_column="prob" if "prob" in crosswalk.columns else None,
        )

    def load_crosswalk(self, query: GeographyQuery | None = None) -> AtomicGeographyCrosswalk:
        query = query or GeographyQuery()
        crosswalk = self.to_crosswalk()
        if not query.geography_columns and query.probability_column is None:
            return crosswalk
        return AtomicGeographyCrosswalk(
            data=crosswalk.data.copy(),
            atomic_id_column=crosswalk.atomic_id_column,
            geography_columns=tuple(query.geography_columns) or crosswalk.geography_columns,
            probability_column=query.probability_column or crosswalk.probability_column,
        )

    def load_assigner(
        self,
        query: GeographyQuery | None = None,
    ) -> ProbabilisticAtomicGeographyAssigner:
        query = query or GeographyQuery()
        partition_columns = tuple(query.partition_columns) or ("state_fips",)
        partition_normalizers = dict(query.partition_normalizers)
        fallback_resolver = query.fallback_resolver
        if partition_columns == ("state_fips",):
            partition_normalizers.setdefault("state_fips", normalize_us_state_fips)
            if fallback_resolver is None:
                fallback_resolver = nearest_numeric_partition_key
        return ProbabilisticAtomicGeographyAssigner(
            crosswalk=self.load_crosswalk(query),
            partition_columns=partition_columns,
            probability_column=query.probability_column,
            partition_normalizers=partition_normalizers,
            fallback_resolver=fallback_resolver,
        )

    def assign(
        self,
        frame: pd.DataFrame,
        *,
        state_column: str = "state_fips",
        atomic_id_column: str = "block_geoid",
        random_state: int | None = None,
    ) -> pd.DataFrame:
        working = frame.copy()
        if state_column != "state_fips":
            working = working.rename(columns={state_column: "state_fips"})
        assigned = self.load_assigner().assign(
            working,
            atomic_id_column=atomic_id_column,
            random_state=random_state,
        )
        if state_column != "state_fips":
            assigned = assigned.rename(columns={"state_fips": state_column})
        return assigned

    def materialize(
        self,
        frame: pd.DataFrame,
        *,
        columns: tuple[str, ...] | list[str] | None = None,
        atomic_id_column: str = "block_geoid",
    ) -> pd.DataFrame:
        return self.to_crosswalk().materialize(
            frame,
            columns=columns,
            atomic_id_column=atomic_id_column,
        )

    def sample_blocks(
        self,
        state_fips: str,
        n: int,
        replace: bool = True,
        random_state: int | None = None,
    ) -> np.ndarray:
        if self._state_blocks is None:
            self._build_state_index()
        if state_fips not in self._state_blocks:
            raise ValueError(f"State FIPS '{state_fips}' not found in block data.")
        state_df = self._state_blocks[state_fips]
        if random_state is not None:
            np.random.seed(random_state)
        sampled_indices = np.random.choice(
            len(state_df),
            size=n,
            replace=replace,
            p=state_df["prob"].values,
        )
        geoids = state_df["geoid"].astype(str).to_numpy()
        return np.asarray(geoids[sampled_indices])

    def _build_state_index(self) -> None:
        self._state_blocks = {}
        for state_fips, group in self.data.groupby("state_fips"):
            self._state_blocks[state_fips] = group[["geoid", "prob"]].copy()

    def sample_blocks_national(
        self,
        n: int,
        replace: bool = True,
        random_state: int | None = None,
    ) -> np.ndarray:
        if random_state is not None:
            np.random.seed(random_state)
        sampled_indices = np.random.choice(
            len(self.data),
            size=n,
            replace=replace,
            p=self.data["national_prob"].values,
        )
        geoids = self.data["geoid"].astype(str).to_numpy()
        return np.asarray(geoids[sampled_indices])

    def get_blocks_in_state(self, state_fips: str) -> pd.DataFrame:
        return self.data[self.data["state_fips"] == state_fips].copy()

    def get_blocks_in_county(self, county_fips: str) -> pd.DataFrame:
        state = county_fips[:STATE_GEOID_LEN]
        county = county_fips[STATE_GEOID_LEN:]
        return self.data[
            (self.data["state_fips"] == state) & (self.data["county"] == county)
        ].copy()

    def get_blocks_in_tract(self, tract_geoid: str) -> pd.DataFrame:
        return self.data[self.data["tract_geoid"] == tract_geoid].copy()

    def get_blocks_in_cd(self, cd_id: str) -> pd.DataFrame:
        return self.data[self.data["cd_id"] == cd_id].copy()

    def get_blocks_in_sldu(self, sldu_id: str) -> pd.DataFrame:
        if "sldu_id" not in self.data.columns:
            return pd.DataFrame()
        return self.data[self.data["sldu_id"] == sldu_id].copy()

    def get_blocks_in_sldl(self, sldl_id: str) -> pd.DataFrame:
        if "sldl_id" not in self.data.columns:
            return pd.DataFrame()
        return self.data[self.data["sldl_id"] == sldl_id].copy()

    @property
    def states(self) -> list[str]:
        return sorted(self.data["state_fips"].unique())

    @property
    def n_blocks(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        if self._data is None:
            return "BlockGeography(not loaded)"
        return f"BlockGeography({self.n_blocks:,} blocks, {len(self.states)} states)"


__all__ = [
    "STATE_LEN",
    "COUNTY_LEN",
    "TRACT_LEN",
    "BLOCK_LEN",
    "STATE_GEOID_LEN",
    "COUNTY_GEOID_LEN",
    "TRACT_GEOID_LEN",
    "BLOCK_GEOID_LEN",
    "DEFAULT_DATA_DIR",
    "DEFAULT_BLOCK_PROBABILITIES_PATH",
    "load_block_probabilities",
    "normalize_us_state_fips",
    "derive_geographies",
    "BlockGeography",
]
