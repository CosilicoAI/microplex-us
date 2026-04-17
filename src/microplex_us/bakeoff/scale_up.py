"""Synthesizer scale-up benchmark harness.

Stages per `docs/synthesizer-benchmark-scale-up.md`:

- stage1: 100,000 rows x 50 columns of real enhanced_cps_2024 data
- stage2: 1,000,000 rows x 50 columns (via row replication or a larger source)
- stage3: 3,373,378 rows x 155 columns (v6 seed-ready shape — requires
  regenerating the seed from donor integration; out of scope for this harness)

The harness is deliberately narrow:

- Single data source (enhanced_cps_2024).
- Fixed pool of synthesizer methods via `microplex.eval.benchmark.*Method`.
- PRDC coverage + wall time + peak RSS + rare-cell preservation.
- One result row per (method, stage, seed).

Wider comparisons (CTGAN, TVAE, external tabular models) are left to
follow-up harnesses. Multi-source fusion is NOT exercised here — the v6
pipeline's multi-source donor integration happens upstream of this eval.
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

try:
    from prdc import compute_prdc  # noqa: F401  (probed at run time)
except ImportError:  # pragma: no cover - optional dep
    compute_prdc = None

LOGGER = logging.getLogger(__name__)

DEFAULT_ENHANCED_CPS_PATH = (
    Path.home()
    / "PolicyEngine/policyengine-us-data/policyengine_us_data/storage/enhanced_cps_2024.h5"
)


# Curated default conditioning variables — demographics + household structure.
# Chosen to be numeric, low-cardinality, and genuinely shared across typical
# microsimulation use cases. Kept to 14 to leave room for 36 target variables
# under a 50-column stage-1 cap.
DEFAULT_CONDITION_COLS: tuple[str, ...] = (
    "age",
    "is_female",
    "is_hispanic",
    "cps_race",
    "is_disabled",
    "is_blind",
    "is_military",
    "is_full_time_college_student",
    "is_separated",
    "state_fips",  # broadcast from household
    "has_esi",
    "has_marketplace_health_coverage",
    "own_children_in_household",
    "pre_tax_contributions",
)


# Curated default target variables — income components, wealth, benefits.
# Chosen to span zero-inflated (most benefits, capital gains), continuous
# heavy-tailed (employment income, interest), and derived (net_worth).
DEFAULT_TARGET_COLS: tuple[str, ...] = (
    # Labor income (2)
    "employment_income_last_year",
    "self_employment_income_last_year",
    # Interest + dividends (4)
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    # Capital gains (2)
    "long_term_capital_gains",
    "short_term_capital_gains",
    # Retirement income (4)
    "taxable_pension_income",
    "tax_exempt_pension_income",
    "taxable_ira_distributions",
    "social_security",
    # Social Security split (3)
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    # Other income (5)
    "rental_income",
    "farm_income",
    "unemployment_compensation",
    "alimony_income",
    "miscellaneous_income",
    # Wealth (5)
    "bank_account_assets",
    "bond_assets",
    "stock_assets",
    "net_worth",
    "auto_loan_balance",
    # Benefits / transfers (11)
    "snap_reported",
    "housing_assistance",
    "ssi_reported",
    "tanf_reported",
    "disability_benefits",
    "workers_compensation",
    "veterans_benefits",
    "child_support_received",
    "child_support_expense",
    "real_estate_taxes",
    "health_savings_account_ald",
)


@dataclass(frozen=True)
class ScaleUpStageConfig:
    """One stage of the synthesizer scale-up protocol."""

    stage: str
    n_rows: int | None  # None means "use all available"
    methods: tuple[str, ...]
    condition_cols: tuple[str, ...] = DEFAULT_CONDITION_COLS
    target_cols: tuple[str, ...] = DEFAULT_TARGET_COLS
    holdout_frac: float = 0.2
    seed: int = 42
    k: int = 5  # PRDC nearest-neighbor k
    n_generate: int | None = None  # None => match training-set size
    data_path: Path = field(default=DEFAULT_ENHANCED_CPS_PATH)
    year: str = "2024"
    rare_cell_checks: tuple[dict[str, Any], ...] = field(
        default_factory=lambda: (
            {
                "name": "elderly_self_employed",
                "mask": lambda df: (df["age"] >= 62)
                & (df["self_employment_income_last_year"] > 0),
            },
            {
                "name": "young_dividend",
                "mask": lambda df: (df["age"] < 30)
                & (df["qualified_dividend_income"] > 0),
            },
            {
                "name": "disabled_ssdi",
                "mask": lambda df: (df["is_disabled"] == 1)
                & (df["social_security_disability"] > 0),
            },
            {
                "name": "top_1pct_employment",
                "mask": lambda df: df["employment_income_last_year"]
                >= df["employment_income_last_year"].quantile(0.99),
            },
        )
    )

    @property
    def all_cols(self) -> list[str]:
        # preserve order: conditioning first, then targets
        seen: set[str] = set()
        out: list[str] = []
        for c in list(self.condition_cols) + list(self.target_cols):
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out


@dataclass
class ScaleUpResult:
    """One (method, stage) outcome."""

    stage: str
    method: str
    seed: int
    n_train_rows: int
    n_holdout_rows: int
    n_cols: int
    fit_wall_seconds: float
    generate_wall_seconds: float
    peak_rss_gb_during_fit: float
    precision: float
    density: float
    coverage: float
    rare_cell_ratios: dict[str, float]
    zero_rate_mae: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stage1_config(methods: tuple[str, ...] = ("ZI-QRF", "ZI-MAF", "ZI-QDNN")) -> ScaleUpStageConfig:
    """Stage 1: ~100k rows x 50 cols on real enhanced_cps_2024.

    enhanced_cps_2024 has 77,006 rows — use all of them. The nominal
    100k-row target from the protocol doc isn't achievable with only this
    source; use the full dataset and note the actual row count in the
    result record.
    """
    return ScaleUpStageConfig(stage="stage1", n_rows=None, methods=methods)


def stage2_config(methods: tuple[str, ...] = ("ZI-QRF", "ZI-MAF", "ZI-QDNN")) -> ScaleUpStageConfig:
    """Stage 2: 1M rows x 50 cols.

    Requires a larger source than enhanced_cps_2024 (77k rows). Intended
    future use once the v6 seed-like 3.4M-row frame is retrievable.
    Running stage 2 against enhanced_cps_2024 replicates rows, which is
    not the same thing — not recommended.
    """
    return ScaleUpStageConfig(stage="stage2", n_rows=1_000_000, methods=methods)


def stage3_config(methods: tuple[str, ...] = ("ZI-QRF", "ZI-MAF", "ZI-QDNN")) -> ScaleUpStageConfig:
    """Stage 3: full 3.4M-row x 155-col v6 seed-ready shape."""
    return ScaleUpStageConfig(stage="stage3", n_rows=3_373_378, methods=methods)


_ENTITY_LINK_COLUMNS: tuple[tuple[str, str, str], ...] = (
    # (entity_name, entity_id_column, person_link_column)
    ("household", "household_id", "person_household_id"),
    ("spm_unit", "spm_unit_id", "person_spm_unit_id"),
    ("tax_unit", "tax_unit_id", "person_tax_unit_id"),
    ("family", "family_id", "person_family_id"),
    ("marital_unit", "marital_unit_id", "person_marital_unit_id"),
)


def _build_entity_lookups(
    f: h5py.File, year: str
) -> tuple[int, dict[str, tuple[int, np.ndarray]]]:
    """Return (person_n, {entity_name: (entity_n, person_to_entity_position)}).

    For each non-person entity, returns a length-`person_n` integer array that,
    when used to index a length-`entity_n` variable, broadcasts the entity
    value down to person level.
    """
    if "person_id" not in f or year not in f["person_id"]:
        raise KeyError(
            f"person_id/{year} missing from enhanced_cps file. Can't determine "
            "person count."
        )
    person_n = int(f["person_id"][year].shape[0])

    lookups: dict[str, tuple[int, np.ndarray]] = {}
    for ent_name, eid_col, pid_col in _ENTITY_LINK_COLUMNS:
        if eid_col not in f or year not in f[eid_col]:
            continue
        if pid_col not in f or year not in f[pid_col]:
            continue
        entity_ids = f[eid_col][year][:]
        person_ent_ids = f[pid_col][year][:]
        id_to_idx = {int(v): i for i, v in enumerate(entity_ids)}
        try:
            lookup = np.fromiter(
                (id_to_idx[int(v)] for v in person_ent_ids),
                dtype=np.int64,
                count=len(person_ent_ids),
            )
        except KeyError as exc:
            raise ValueError(
                f"entity {ent_name!r}: person's {pid_col} value {exc} not in "
                f"{eid_col} — entity table inconsistent"
            ) from exc
        lookups[ent_name] = (int(len(entity_ids)), lookup)
    return person_n, lookups


def _load_enhanced_cps(
    data_path: Path,
    year: str,
    columns: list[str],
) -> pd.DataFrame:
    """Load enhanced_cps columns, broadcasting non-person entities to person level.

    enhanced_cps_2024 stores variables at their native entity level (person,
    household, tax_unit, spm_unit, family, marital_unit). To land a flat
    person-level DataFrame, this helper uses the `person_<entity>_id` →
    `<entity>_id` linkage to project parent-entity values down.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"enhanced_cps_{year} not found at {data_path}. "
            "Set `data_path` explicitly in ScaleUpStageConfig."
        )

    with h5py.File(data_path, "r") as f:
        available = set(f.keys())
        missing = [c for c in columns if c not in available]
        if missing:
            raise KeyError(
                f"Columns not in enhanced_cps: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        person_n, entity_lookups = _build_entity_lookups(f, year)

        data: dict[str, np.ndarray] = {}
        for col in columns:
            grp = f[col]
            if year not in grp:
                raise KeyError(f"Column {col!r} has no {year!r} entry")
            arr = grp[year][:]
            if arr.shape[0] == person_n:
                data[col] = arr
                continue
            # Broadcast via entity lookup
            broadcast = None
            for ent_name, (ent_n, lookup) in entity_lookups.items():
                if arr.shape[0] == ent_n:
                    broadcast = arr[lookup]
                    break
            if broadcast is None:
                available_sizes = {e: n for e, (n, _) in entity_lookups.items()}
                available_sizes["person"] = person_n
                raise ValueError(
                    f"Column {col!r} has {arr.shape[0]} rows but no matching "
                    f"entity linkage. Sizes available: {available_sizes}"
                )
            data[col] = broadcast

    return pd.DataFrame(data)


def _peak_rss_gb() -> float:
    """Current process's max resident set size in GB.

    Unit of `ru_maxrss` is platform-dependent:
      - Linux: kilobytes
      - macOS (Darwin): bytes
      - FreeBSD: kilobytes (but verify)

    Cross-checked against psutil on macOS Python 3.14: ru_maxrss is in bytes
    (e.g., 190_873_600 raw = 0.18 GB matches `psutil.Process().memory_info().rss`).
    """
    import sys

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        bytes_rss = r
    else:
        # Linux and most BSDs: kilobytes
        bytes_rss = r * 1024
    return bytes_rss / (1024**3)


def _compute_rare_cell_ratios(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    checks: tuple[dict[str, Any], ...],
) -> dict[str, float]:
    """Per-check: synthetic count / real count in the rare cell.

    Matches the pattern in `microplex/benchmarks/results/sparse_coverage.csv`.
    1.0 means the synthetic preserves the rare cell at its real frequency;
    0.0 means the cell is annihilated.
    """
    ratios: dict[str, float] = {}
    for check in checks:
        name = check["name"]
        mask_fn = check["mask"]
        try:
            real_mask = mask_fn(real).fillna(False)
        except (KeyError, AttributeError) as exc:
            ratios[name] = float("nan")
            LOGGER.warning(
                "rare-cell check %r skipped (%s: %s)", name, type(exc).__name__, exc
            )
            continue
        try:
            synth_mask = mask_fn(synthetic).fillna(False)
        except (KeyError, AttributeError):
            ratios[name] = float("nan")
            continue
        real_count = max(int(real_mask.sum()), 1)
        synth_count = int(synth_mask.sum())
        ratios[name] = float(synth_count) / float(real_count)
    return ratios


def _compute_zero_rate_mae(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    """Mean absolute error in per-column zero-rate across the common column set."""
    cols = [c for c in real.columns if c in synthetic.columns]
    errs = []
    for c in cols:
        r_zero = float((real[c] == 0).mean())
        s_zero = float((synthetic[c] == 0).mean())
        errs.append(abs(r_zero - s_zero))
    return float(np.mean(errs)) if errs else 0.0


def _compute_prdc(
    real: pd.DataFrame, synthetic: pd.DataFrame, k: int
) -> tuple[float, float, float]:
    """Return (precision, density, coverage) via the `prdc` library."""
    if compute_prdc is None:
        raise ImportError(
            "PRDC requires the `prdc` package. "
            "Install with: uv pip install prdc"
        )

    from sklearn.preprocessing import StandardScaler

    cols = [c for c in real.columns if c in synthetic.columns]
    if not cols:
        raise ValueError("No shared columns between real and synthetic for PRDC")

    r = real[cols].to_numpy(dtype=np.float64)
    s = synthetic[cols].to_numpy(dtype=np.float64)

    if len(r) < k + 1 or len(s) < k + 1:
        return (0.0, 0.0, 0.0)

    scaler = StandardScaler()
    r_scaled = scaler.fit_transform(r)
    s_scaled = scaler.transform(s)

    metrics = compute_prdc(r_scaled, s_scaled, nearest_k=k)
    return (
        float(metrics["precision"]),
        float(metrics["density"]),
        float(metrics["coverage"]),
    )


def _build_method(method_name: str) -> Any:
    from microplex.eval.benchmark import (
        CTGANMethod,
        MAFMethod,
        QDNNMethod,
        QRFMethod,
        TVAEMethod,
        ZIMAFMethod,
        ZIQDNNMethod,
        ZIQRFMethod,
    )

    registry = {
        "QRF": QRFMethod,
        "ZI-QRF": ZIQRFMethod,
        "QDNN": QDNNMethod,
        "ZI-QDNN": ZIQDNNMethod,
        "MAF": MAFMethod,
        "ZI-MAF": ZIMAFMethod,
        "CTGAN": CTGANMethod,
        "TVAE": TVAEMethod,
    }
    if method_name not in registry:
        raise ValueError(
            f"Unknown method {method_name!r}. Known: {sorted(registry)}"
        )
    return registry[method_name]()


class ScaleUpRunner:
    """Runs one stage of the scale-up protocol."""

    def __init__(self, config: ScaleUpStageConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ScaleUpRunner")

    def load_frame(self) -> pd.DataFrame:
        df = _load_enhanced_cps(
            self.config.data_path, self.config.year, self.config.all_cols
        )
        self.logger.info(
            "loaded enhanced_cps: %d rows, %d cols", len(df), len(df.columns)
        )
        # Cast to a single dtype so downstream DataFrame.values stays
        # numeric-uniform (torch-based methods reject object arrays, which
        # is what pandas produces when columns mix bool/int32/float32).
        df = df.astype(np.float32, copy=False)
        if self.config.n_rows is not None and len(df) > self.config.n_rows:
            rng = np.random.default_rng(self.config.seed)
            idx = rng.choice(len(df), size=self.config.n_rows, replace=False)
            df = df.iloc[idx].reset_index(drop=True)
            self.logger.info("subsampled to %d rows", len(df))
        return df

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(self.config.seed)
        idx = rng.permutation(len(df))
        cut = int(len(df) * (1.0 - self.config.holdout_frac))
        train_idx, holdout_idx = idx[:cut], idx[cut:]
        train = df.iloc[train_idx].reset_index(drop=True)
        holdout = df.iloc[holdout_idx].reset_index(drop=True)
        return train, holdout

    def fit_and_generate(
        self, method_name: str, train: pd.DataFrame, n_generate: int
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Fit method on `train` and generate `n_generate` synthetic records."""
        method = _build_method(method_name)

        # The benchmark methods take a multi-source dict; pass a single source.
        sources = {"enhanced_cps_2024": train.copy()}
        shared_cols = list(self.config.condition_cols)

        before_rss = _peak_rss_gb()
        t_fit = time.perf_counter()
        method.fit(sources=sources, shared_cols=shared_cols)
        fit_wall = time.perf_counter() - t_fit
        peak_fit_rss = max(_peak_rss_gb(), before_rss)

        t_gen = time.perf_counter()
        synthetic = method.generate(n_generate, seed=self.config.seed)
        gen_wall = time.perf_counter() - t_gen

        return synthetic, {
            "fit_wall_seconds": fit_wall,
            "generate_wall_seconds": gen_wall,
            "peak_rss_gb_during_fit": peak_fit_rss,
        }

    def run(
        self,
        incremental_path: Path | None = None,
    ) -> list[ScaleUpResult]:
        """Run every configured method on the loaded frame; return results.

        If `incremental_path` is given, each method's `ScaleUpResult` is
        appended to that path as JSONL *as soon as it completes*. This
        guarantees at least partial output if a later method crashes or
        the host is interrupted.
        """
        df = self.load_frame()
        train, holdout = self.split(df)
        n_generate = self.config.n_generate or len(train)
        self.logger.info(
            "split %d train / %d holdout; will generate %d synthetic",
            len(train),
            len(holdout),
            n_generate,
        )

        if incremental_path is not None:
            incremental_path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate any prior JSONL so this run's output is self-contained.
            incremental_path.write_text("")

        results: list[ScaleUpResult] = []
        for method_name in self.config.methods:
            self.logger.info("== fitting %s ==", method_name)
            try:
                synthetic, timing = self.fit_and_generate(
                    method_name, train, n_generate
                )
            except Exception as exc:  # pragma: no cover
                self.logger.error("method %s failed: %s", method_name, exc)
                result = ScaleUpResult(
                    stage=self.config.stage,
                    method=method_name,
                    seed=self.config.seed,
                    n_train_rows=len(train),
                    n_holdout_rows=len(holdout),
                    n_cols=len(df.columns),
                    fit_wall_seconds=0.0,
                    generate_wall_seconds=0.0,
                    peak_rss_gb_during_fit=0.0,
                    precision=0.0,
                    density=0.0,
                    coverage=0.0,
                    rare_cell_ratios={},
                    zero_rate_mae=0.0,
                    notes=f"FAILED: {type(exc).__name__}: {exc}",
                )
                results.append(result)
                self._persist_incremental(incremental_path, result)
                continue

            precision, density, coverage = _compute_prdc(
                holdout, synthetic, k=self.config.k
            )
            rare = _compute_rare_cell_ratios(
                holdout, synthetic, self.config.rare_cell_checks
            )
            zero_mae = _compute_zero_rate_mae(holdout, synthetic)

            result = ScaleUpResult(
                stage=self.config.stage,
                method=method_name,
                seed=self.config.seed,
                n_train_rows=len(train),
                n_holdout_rows=len(holdout),
                n_cols=len(df.columns),
                fit_wall_seconds=timing["fit_wall_seconds"],
                generate_wall_seconds=timing["generate_wall_seconds"],
                peak_rss_gb_during_fit=timing["peak_rss_gb_during_fit"],
                precision=precision,
                density=density,
                coverage=coverage,
                rare_cell_ratios=rare,
                zero_rate_mae=zero_mae,
                notes="",
            )
            results.append(result)
            self._persist_incremental(incremental_path, result)
            self.logger.info(
                "  %s: coverage=%.3f precision=%.3f density=%.3f fit=%.1fs gen=%.1fs peak_rss=%.2fGB",
                method_name,
                coverage,
                precision,
                density,
                timing["fit_wall_seconds"],
                timing["generate_wall_seconds"],
                timing["peak_rss_gb_during_fit"],
            )
        return results

    @staticmethod
    def _persist_incremental(
        path: Path | None, result: ScaleUpResult
    ) -> None:
        """Append one `ScaleUpResult` as a JSONL row (if path is set)."""
        if path is None:
            return
        with path.open("a") as f:
            f.write(json.dumps(result.to_dict(), default=str))
            f.write("\n")


def _results_to_dataframe(results: list[ScaleUpResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in results:
        d = r.to_dict()
        rare = d.pop("rare_cell_ratios")
        for cell_name, ratio in rare.items():
            d[f"rare__{cell_name}"] = ratio
        rows.append(d)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "scale-up runner")
    parser.add_argument(
        "--stage",
        choices=["stage1", "stage2", "stage3"],
        default="stage1",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ZI-QRF", "ZI-MAF", "ZI-QDNN"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/scale_up_results.json"),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--incremental-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional path to a JSONL file where each method's result is "
            "appended as soon as it completes. Defaults to the final "
            "--output path with '.partial.jsonl' appended."
        ),
    )
    args = parser.parse_args(argv)

    if args.incremental_jsonl is None:
        args.incremental_jsonl = args.output.with_suffix(
            args.output.suffix + ".partial.jsonl"
        )

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    stage_fn = {"stage1": stage1_config, "stage2": stage2_config, "stage3": stage3_config}
    cfg = stage_fn[args.stage](methods=tuple(args.methods))
    cfg = ScaleUpStageConfig(
        stage=cfg.stage,
        n_rows=cfg.n_rows,
        methods=tuple(args.methods),
        condition_cols=cfg.condition_cols,
        target_cols=cfg.target_cols,
        holdout_frac=cfg.holdout_frac,
        seed=args.seed,
        k=cfg.k,
        n_generate=cfg.n_generate,
        data_path=cfg.data_path,
        year=cfg.year,
        rare_cell_checks=cfg.rare_cell_checks,
    )

    runner = ScaleUpRunner(cfg)
    results = runner.run(incremental_path=args.incremental_jsonl)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "stage": cfg.stage,
                "methods": list(cfg.methods),
                "seed": cfg.seed,
                "n_conditioning_cols": len(cfg.condition_cols),
                "n_target_cols": len(cfg.target_cols),
                "results": [r.to_dict() for r in results],
            },
            indent=2,
            default=str,
        )
    )
    LOGGER.info("wrote %d results to %s", len(results), args.output)

    df = _results_to_dataframe(results)
    print()
    print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
