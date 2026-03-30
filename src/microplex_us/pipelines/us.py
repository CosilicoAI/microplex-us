"""Library-first US microplex build pipeline."""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import FunctionType
from typing import Any, Literal

import numpy as np
import pandas as pd
from microplex.calibration import (
    Calibrator,
    HardConcreteCalibrator,
    LinearConstraint,
    SparseCalibrator,
)
from microplex.core import (
    EntityObservation,
    EntityRelationship,
    EntityType,
    ObservationFrame,
    RelationshipCardinality,
    Shareability,
    SourceDescriptor,
    SourceProvider,
    SourceQuery,
    TimeStructure,
)
from microplex.fusion import FusionPlan
from microplex.hierarchical import TaxUnitOptimizer
from microplex.synthesizer import Synthesizer
from microplex.targets import TargetQuery, TargetSpec
from sklearn.ensemble import RandomForestClassifier

from microplex_us.policyengine.target_profiles import (
    PolicyEngineUSTargetCell,
    resolve_policyengine_us_target_profile,
)
from microplex_us.policyengine.us import (
    PolicyEngineUSDBTargetProvider,
    PolicyEngineUSEntityTableBundle,
    PolicyEngineUSMicrosimulationAdapter,
    PolicyEngineUSQuantityTarget,
    PolicyEngineUSVariableBinding,
    build_policyengine_us_export_variable_maps,
    build_policyengine_us_time_period_arrays,
    compile_supported_policyengine_us_household_linear_constraints,
    detect_policyengine_pseudo_inputs,
    filter_supported_policyengine_us_targets,
    infer_policyengine_us_variable_bindings,
    materialize_policyengine_us_variables_safely,
    policyengine_us_variables_to_materialize,
    write_policyengine_us_time_period_dataset,
)
from microplex_us.variables import (
    DonorMatchStrategy,
    VariableSupportFamily,
    donor_imputation_block_specs,
    is_projected_condition_var_compatible,
    normalize_dividend_columns,
    prune_redundant_variables,
    score_donor_condition_var,
    variable_semantic_spec_for,
)

STATE_FIPS = {
    1: "AL",
    2: "AK",
    4: "AZ",
    5: "AR",
    6: "CA",
    8: "CO",
    9: "CT",
    10: "DE",
    11: "DC",
    12: "FL",
    13: "GA",
    15: "HI",
    16: "ID",
    17: "IL",
    18: "IN",
    19: "IA",
    20: "KS",
    21: "KY",
    22: "LA",
    23: "ME",
    24: "MD",
    25: "MA",
    26: "MI",
    27: "MN",
    28: "MS",
    29: "MO",
    30: "MT",
    31: "NE",
    32: "NV",
    33: "NH",
    34: "NJ",
    35: "NM",
    36: "NY",
    37: "NC",
    38: "ND",
    39: "OH",
    40: "OK",
    41: "OR",
    42: "PA",
    44: "RI",
    45: "SC",
    46: "SD",
    47: "TN",
    48: "TX",
    49: "UT",
    50: "VT",
    51: "VA",
    53: "WA",
    54: "WV",
    55: "WI",
    56: "WY",
}

AGE_BINS = [0, 18, 35, 55, 65, np.inf]


class ColumnwiseQRFDonorImputer:
    """Columnwise QRF donor imputer, optionally with zero-inflated support."""

    def __init__(
        self,
        *,
        condition_vars: list[str],
        target_vars: list[str],
        n_estimators: int = 100,
        zero_inflated_vars: set[str] | None = None,
        nonnegative_vars: set[str] | None = None,
        zero_threshold: float = 0.05,
        quantiles: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
    ) -> None:
        self.condition_vars = list(condition_vars)
        self.target_vars = list(target_vars)
        self.n_estimators = int(n_estimators)
        self.zero_inflated_vars = set(zero_inflated_vars or ())
        self.nonnegative_vars = set(nonnegative_vars or ())
        self.zero_threshold = float(zero_threshold)
        self.quantiles = tuple(float(value) for value in quantiles)
        self._models: dict[str, Any] = {}
        self._zero_models: dict[str, RandomForestClassifier] = {}

    def fit(
        self,
        data: pd.DataFrame,
        *,
        weight_col: str | None = "weight",
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        verbose: bool = False,
    ) -> ColumnwiseQRFDonorImputer:
        del weight_col, epochs, batch_size, learning_rate, verbose
        if importlib.util.find_spec("quantile_forest") is None:
            raise ImportError("quantile-forest is required for donor_imputer_backend='qrf'")
        from quantile_forest import RandomForestQuantileRegressor

        self._models = {}
        self._zero_models = {}
        for column in self.target_vars:
            subset = data[self.condition_vars + [column]].dropna()
            if len(subset) < 25:
                continue
            x_values = subset[self.condition_vars].to_numpy(dtype=float)
            y_values = subset[column].to_numpy(dtype=float)
            if (
                column in self.zero_inflated_vars
                and (y_values == 0).mean() >= self.zero_threshold
                and (y_values == 0).sum() >= 10
                and (y_values > 0).sum() >= 10
            ):
                zero_model = RandomForestClassifier(
                    n_estimators=max(50, self.n_estimators // 2),
                    random_state=42,
                    n_jobs=-1,
                )
                zero_model.fit(x_values, (y_values > 0).astype(int))
                self._zero_models[column] = zero_model
                x_values = x_values[y_values > 0]
                y_values = y_values[y_values > 0]
            if len(y_values) < 25:
                continue
            model = RandomForestQuantileRegressor(
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(x_values, y_values)
            self._models[column] = model
        return self

    def generate(
        self,
        conditions: pd.DataFrame,
        seed: int | None = None,
    ) -> pd.DataFrame:
        rng = np.random.RandomState(seed or 42)
        synthetic = conditions.copy().reset_index(drop=True)
        x_values = synthetic[self.condition_vars].to_numpy(dtype=float)
        for column in self.target_vars:
            model = self._models.get(column)
            if model is None:
                synthetic[column] = np.nan
                continue
            values = np.zeros(len(synthetic), dtype=float)
            target_rows = np.ones(len(synthetic), dtype=bool)
            zero_model = self._zero_models.get(column)
            if zero_model is not None:
                probabilities = zero_model.predict_proba(x_values)
                positive_probs = (
                    probabilities[:, 1]
                    if probabilities.shape[1] > 1
                    else np.zeros(len(synthetic), dtype=float)
                )
                target_rows = rng.random(len(synthetic)) < positive_probs
                values[:] = 0.0
            if target_rows.any():
                predictions = model.predict(
                    x_values[target_rows],
                    quantiles=list(self.quantiles),
                )
                quantile_choices = rng.choice(len(self.quantiles), size=target_rows.sum())
                draws = predictions[np.arange(target_rows.sum()), quantile_choices]
                if column in self.nonnegative_vars:
                    draws = np.maximum(draws, 0.0)
                values[target_rows] = draws
            synthetic[column] = values
        return synthetic
AGE_LABELS = ["0-17", "18-34", "35-54", "55-64", "65+"]
INCOME_BINS = [-np.inf, 25_000, 50_000, 100_000, np.inf]
INCOME_LABELS = ["<25k", "25-50k", "50-100k", "100k+"]
ENTITY_ID_COLUMNS = {
    EntityType.PERSON: "person_id",
    EntityType.HOUSEHOLD: "household_id",
    EntityType.TAX_UNIT: "tax_unit_id",
    EntityType.SPM_UNIT: "spm_unit_id",
    EntityType.FAMILY: "family_id",
}
TINY_WEIGHT_THRESHOLD = 1e-8
DEFAULT_POLICYENGINE_CALIBRATION_MAX_CONSTRAINTS_PER_HOUSEHOLD = 1.0
DEFAULT_POLICYENGINE_CALIBRATION_MIN_ACTIVE_HOUSEHOLDS = 5
CALIBRATION_FEASIBILITY_DROP_WARNING_THRESHOLD = 0.2
STATE_PROGRAM_SUPPORT_PROXY_VARIABLES = (
    "has_medicaid",
    "public_assistance",
    "ssi",
    "social_security",
)
STATE_PROGRAM_AUTO_CONDITION_VARIABLES = ("has_medicaid",)


def _summarize_weight_diagnostics(
    weights: pd.Series | np.ndarray | list[float],
    *,
    tiny_threshold: float = TINY_WEIGHT_THRESHOLD,
) -> dict[str, Any]:
    """Summarize whether a calibrated weight vector looks numerically healthy."""
    series = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0).astype(float)
    row_count = int(len(series))
    if row_count == 0:
        return {
            "row_count": 0,
            "positive_count": 0,
            "nonpositive_count": 0,
            "tiny_count": 0,
            "tiny_share": 0.0,
            "total_weight": 0.0,
            "min_weight": 0.0,
            "p01_weight": 0.0,
            "p50_weight": 0.0,
            "p99_weight": 0.0,
            "max_weight": 0.0,
            "effective_sample_size": 0.0,
            "collapse_suspected": True,
        }

    total_weight = float(series.sum())
    squared_weight_sum = float(np.square(series).sum())
    positive_count = int((series > 0.0).sum())
    nonpositive_count = row_count - positive_count
    tiny_count = int((series <= tiny_threshold).sum())
    tiny_share = float(tiny_count / row_count)
    effective_sample_size = (
        float((total_weight * total_weight) / squared_weight_sum)
        if squared_weight_sum > 0.0
        else 0.0
    )
    effective_sample_ratio = (
        float(effective_sample_size / positive_count) if positive_count > 0 else 0.0
    )
    collapse_suspected = bool(
        total_weight <= tiny_threshold
        or positive_count == 0
        or tiny_share >= 0.95
        or effective_sample_ratio <= 0.25
    )
    return {
        "row_count": row_count,
        "positive_count": positive_count,
        "nonpositive_count": nonpositive_count,
        "tiny_count": tiny_count,
        "tiny_share": tiny_share,
        "total_weight": total_weight,
        "min_weight": float(series.min()),
        "p01_weight": float(series.quantile(0.01)),
        "p50_weight": float(series.quantile(0.5)),
        "p99_weight": float(series.quantile(0.99)),
        "max_weight": float(series.max()),
        "effective_sample_size": effective_sample_size,
        "effective_sample_ratio": effective_sample_ratio,
        "collapse_suspected": collapse_suspected,
    }


def _state_program_support_proxy_summary(available_columns: set[str]) -> dict[str, list[str]]:
    available = sorted(
        variable for variable in STATE_PROGRAM_SUPPORT_PROXY_VARIABLES if variable in available_columns
    )
    missing = sorted(
        variable
        for variable in STATE_PROGRAM_SUPPORT_PROXY_VARIABLES
        if variable not in available_columns
    )
    return {
        "available": available,
        "missing": missing,
    }


def _subset_policyengine_linear_constraints(
    constraints: tuple[LinearConstraint, ...] | list[LinearConstraint],
    household_mask: np.ndarray,
) -> tuple[LinearConstraint, ...]:
    mask = np.asarray(household_mask, dtype=bool)
    subset: list[LinearConstraint] = []
    for constraint in constraints:
        coefficients = np.asarray(constraint.coefficients, dtype=float)
        if len(coefficients) != len(mask):
            raise ValueError(
                "PolicyEngine linear constraint coefficients do not match household mask length"
            )
        subset.append(
            LinearConstraint(
                name=constraint.name,
                coefficients=coefficients[mask],
                target=float(constraint.target),
            )
        )
    return tuple(subset)


def _subset_policyengine_tables_by_households(
    tables: PolicyEngineUSEntityTableBundle,
    household_ids: pd.Index,
) -> PolicyEngineUSEntityTableBundle:
    selected_ids = pd.Index(household_ids, name="household_id")
    household_order = pd.Series(np.arange(len(selected_ids)), index=selected_ids)

    households = tables.households.loc[
        tables.households["household_id"].isin(selected_ids)
    ].copy()
    households = (
        households.assign(
            _household_order=households["household_id"].map(household_order)
        )
        .sort_values("_household_order")
        .drop(columns="_household_order")
        .reset_index(drop=True)
    )

    def _subset_related(table: pd.DataFrame | None) -> pd.DataFrame | None:
        if table is None:
            return None
        subset = table.loc[table["household_id"].isin(selected_ids)].copy()
        return subset.reset_index(drop=True)

    return PolicyEngineUSEntityTableBundle(
        households=households,
        persons=_subset_related(tables.persons),
        tax_units=_subset_related(tables.tax_units),
        spm_units=_subset_related(tables.spm_units),
        families=_subset_related(tables.families),
        marital_units=_subset_related(tables.marital_units),
    )


def _policyengine_target_geo_priority(target: TargetSpec) -> int:
    geo_level = str(target.metadata.get("geo_level", "")).lower()
    return {
        "national": 0,
        "state": 1,
        "district": 2,
    }.get(geo_level, 99)


def _constraint_active_household_count(
    constraint: Any,
    *,
    epsilon: float = 1e-12,
) -> int:
    coefficients = np.asarray(getattr(constraint, "coefficients", ()), dtype=float)
    if coefficients.size == 0:
        return 0
    return int(np.count_nonzero(np.abs(coefficients) > epsilon))


def _select_feasible_policyengine_calibration_constraints(
    targets: list[TargetSpec],
    constraints: tuple[Any, ...],
    *,
    household_count: int,
    max_constraints: int | None,
    max_constraints_per_household: float | None,
    min_active_households: int,
) -> tuple[list[TargetSpec], tuple[Any, ...], dict[str, Any]]:
    selected_targets = list(targets)
    selected_constraints = tuple(constraints)
    requested_max_constraints = max_constraints
    if (
        requested_max_constraints is None
        and max_constraints_per_household is not None
        and household_count > 0
    ):
        requested_max_constraints = max(
            1,
            int(np.floor(max_constraints_per_household * household_count)),
        )

    records = []
    for target, constraint in zip(targets, constraints, strict=True):
        active_households = _constraint_active_household_count(constraint)
        records.append(
            {
                "target": target,
                "constraint": constraint,
                "active_households": active_households,
                "geo_priority": _policyengine_target_geo_priority(target),
                "aggregation_priority": 0 if target.aggregation.name == "COUNT" else 1,
                "coefficient_mass": float(
                    np.abs(
                        np.asarray(getattr(constraint, "coefficients", ()), dtype=float)
                    ).sum()
                ),
            }
        )

    min_required_households = max(1, int(min_active_households))
    support_filtered = [
        record
        for record in records
        if record["active_households"] >= min_required_households
    ]
    low_support_dropped = len(records) - len(support_filtered)

    support_filtered.sort(
        key=lambda record: (
            record["geo_priority"],
            record["aggregation_priority"],
            -record["active_households"],
            -record["coefficient_mass"],
            record["target"].name,
        )
    )

    over_capacity_dropped = 0
    if requested_max_constraints is not None and len(support_filtered) > requested_max_constraints:
        over_capacity_dropped = len(support_filtered) - requested_max_constraints
        support_filtered = support_filtered[:requested_max_constraints]

    selected_targets = [record["target"] for record in support_filtered]
    selected_constraints = tuple(record["constraint"] for record in support_filtered)
    dropped_total = low_support_dropped + over_capacity_dropped
    drop_share = float(dropped_total / len(records)) if records else 0.0
    warning_messages: list[str] = []
    if drop_share > CALIBRATION_FEASIBILITY_DROP_WARNING_THRESHOLD:
        warning_messages.append(
            "Calibration feasibility filter dropped "
            f"{dropped_total}/{len(records)} constraints "
            f"({drop_share:.1%}) before solving."
        )
    diagnostics = {
        "requested_max_constraints": requested_max_constraints,
        "max_constraints_per_household": max_constraints_per_household,
        "min_active_households": min_required_households,
        "n_constraints_before_feasibility_filter": len(constraints),
        "n_constraints_after_feasibility_filter": len(selected_constraints),
        "n_constraints_dropped_low_support": low_support_dropped,
        "n_constraints_dropped_over_capacity": over_capacity_dropped,
        "n_constraints_dropped_total": dropped_total,
        "constraint_drop_share": drop_share,
        "warning_messages": warning_messages,
        "feasibility_filter_applied": bool(
            low_support_dropped > 0 or over_capacity_dropped > 0
        ),
    }
    return selected_targets, selected_constraints, diagnostics


@dataclass(frozen=True)
class USMicroplexBuildConfig:
    """Configuration for the US microplex build pipeline."""

    n_synthetic: int = 100_000
    synthesis_backend: Literal["bootstrap", "synthesizer", "seed"] = "synthesizer"
    calibration_backend: Literal["entropy", "ipf", "chi2", "sparse", "hardconcrete"] = (
        "entropy"
    )
    calibration_tol: float = 1e-6
    calibration_max_iter: int = 100
    random_seed: int = 42
    target_sparsity: float = 0.9
    device: str = "cpu"
    synthesizer_condition_vars: tuple[str, ...] = (
        "age",
        "sex",
        "education",
        "employment_status",
        "state_fips",
        "tenure",
    )
    synthesizer_target_vars: tuple[str, ...] = ("income",)
    synthesizer_epochs: int = 100
    synthesizer_batch_size: int = 256
    synthesizer_learning_rate: float = 1e-3
    synthesizer_n_layers: int = 4
    synthesizer_hidden_dim: int = 64
    donor_imputer_epochs: int = 20
    donor_imputer_batch_size: int = 128
    donor_imputer_learning_rate: float = 1e-3
    donor_imputer_n_layers: int = 2
    donor_imputer_hidden_dim: int = 32
    donor_imputer_backend: Literal["maf", "qrf", "zi_qrf"] = "maf"
    donor_imputer_qrf_n_estimators: int = 100
    donor_imputer_qrf_zero_threshold: float = 0.05
    donor_imputer_condition_selection: Literal["all_shared", "top_correlated"] = (
        "top_correlated"
    )
    donor_imputer_max_condition_vars: int | None = 8
    donor_imputer_excluded_variables: tuple[str, ...] = ("filing_status_code",)
    bootstrap_strata_columns: tuple[str, ...] = ()
    prefer_cached_cps_asec_source: bool = False
    cps_asec_source_year: int = 2023
    cps_asec_cache_dir: str | None = None
    policyengine_dataset: str | None = None
    policyengine_baseline_dataset: str | None = None
    policyengine_dataset_year: int | None = None
    policyengine_direct_override_variables: tuple[str, ...] = ()
    policyengine_quantity_targets: tuple[PolicyEngineUSQuantityTarget, ...] = ()
    policyengine_targets_db: str | None = None
    policyengine_target_period: int | None = None
    policyengine_target_variables: tuple[str, ...] = ()
    policyengine_target_domains: tuple[str, ...] = ()
    policyengine_target_geo_levels: tuple[str, ...] = ()
    policyengine_target_profile: str | None = None
    policyengine_calibration_target_variables: tuple[str, ...] = ()
    policyengine_calibration_target_domains: tuple[str, ...] = ()
    policyengine_calibration_target_geo_levels: tuple[str, ...] = ()
    policyengine_calibration_target_profile: str | None = None
    policyengine_selection_household_budget: int | None = None
    policyengine_calibration_max_constraints: int | None = None
    policyengine_calibration_max_constraints_per_household: float | None = (
        DEFAULT_POLICYENGINE_CALIBRATION_MAX_CONSTRAINTS_PER_HOUSEHOLD
    )
    policyengine_calibration_min_active_households: int = (
        DEFAULT_POLICYENGINE_CALIBRATION_MIN_ACTIVE_HOUSEHOLDS
    )
    policyengine_target_reform_id: int = 0
    policyengine_simulation_cls: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return _normalize_config_value(asdict(self))


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _normalize_config_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_config_value(item) for item in value]
    if isinstance(value, type) or isinstance(value, FunctionType):
        return f"{value.__module__}.{value.__qualname__}"
    return value


@dataclass(frozen=True)
class USMicroplexTargets:
    """Calibration targets for the US microplex pipeline."""

    marginal: dict[str, dict[str, float]]
    continuous: dict[str, float]


@dataclass(frozen=True)
class USMicroplexSourceInput:
    """Normalized source-planning context for one US build."""

    frame: ObservationFrame
    fusion_plan: FusionPlan
    household_observation: EntityObservation
    person_observation: EntityObservation
    household_person_relationship: EntityRelationship
    households: pd.DataFrame
    persons: pd.DataFrame


@dataclass(frozen=True)
class USMicroplexSynthesisVariables:
    """Observed variables to use during synthesis."""

    condition_vars: tuple[str, ...]
    target_vars: tuple[str, ...]


@dataclass
class USMicroplexBuildResult:
    """Artifacts from a US microplex build."""

    config: USMicroplexBuildConfig
    seed_data: pd.DataFrame
    synthetic_data: pd.DataFrame
    calibrated_data: pd.DataFrame
    targets: USMicroplexTargets
    calibration_summary: dict[str, Any]
    synthesis_metadata: dict[str, Any] = field(default_factory=dict)
    synthesizer: Synthesizer | Any | None = None
    policyengine_tables: PolicyEngineUSEntityTableBundle | None = None
    source_frame: ObservationFrame | None = None
    source_frames: tuple[ObservationFrame, ...] = ()
    fusion_plan: FusionPlan | None = None

    @property
    def n_nonzero_weights(self) -> int:
        if "weight" not in self.calibrated_data.columns:
            return 0
        return int((self.calibrated_data["weight"] > 1e-9).sum())

    @property
    def total_weighted_population(self) -> float:
        if "weight" not in self.calibrated_data.columns:
            return 0.0
        return float(self.calibrated_data["weight"].sum())


class USMicroplexPipeline:
    """End-to-end build orchestration for a US microplex dataset."""

    def __init__(self, config: USMicroplexBuildConfig | None = None):
        self.config = config or USMicroplexBuildConfig()

    def build_from_data_dir(self, data_dir: str | Path) -> USMicroplexBuildResult:
        from microplex_us.data_sources.cps import (
            DEFAULT_CACHE_DIR,
            CPSASECParquetSourceProvider,
            CPSASECSourceProvider,
        )

        if self.config.prefer_cached_cps_asec_source:
            cache_dir = (
                Path(self.config.cps_asec_cache_dir)
                if self.config.cps_asec_cache_dir is not None
                else DEFAULT_CACHE_DIR
            )
            processed_path = (
                cache_dir / f"cps_asec_{int(self.config.cps_asec_source_year)}_processed.parquet"
            )
            if processed_path.exists():
                return self.build_from_source_provider(
                    CPSASECSourceProvider(
                        year=int(self.config.cps_asec_source_year),
                        cache_dir=cache_dir,
                        download=False,
                    )
                )

        return self.build_from_source_provider(
            CPSASECParquetSourceProvider(data_dir=data_dir)
        )

    def build_from_source_provider(
        self,
        provider: SourceProvider,
        query: SourceQuery | None = None,
    ) -> USMicroplexBuildResult:
        frame = provider.load_frame(query)
        return self.build_from_frames([frame])

    def build_from_source_providers(
        self,
        providers: list[SourceProvider],
        queries: dict[str, SourceQuery] | None = None,
    ) -> USMicroplexBuildResult:
        if not providers:
            raise ValueError("USMicroplexPipeline requires at least one source provider")

        frames: list[ObservationFrame] = []
        for provider in providers:
            frame = provider.load_frame(
                self._resolve_source_query(provider, queries or {})
            )
            frames.append(frame)
        return self.build_from_frames(frames)

    def build_from_frame(self, frame: ObservationFrame) -> USMicroplexBuildResult:
        return self.build_from_frames([frame])

    def build_from_frames(
        self,
        frames: list[ObservationFrame],
    ) -> USMicroplexBuildResult:
        if not frames:
            raise ValueError("USMicroplexPipeline requires at least one observation frame")

        source_inputs = [self.prepare_source_input(frame) for frame in frames]
        fusion_plan = FusionPlan.from_sources([frame.source for frame in frames])
        scaffold_input = self._select_scaffold_source(source_inputs)
        seed_data = self.prepare_seed_data_from_source(scaffold_input)
        donor_integration = self._integrate_donor_sources(
            seed_data,
            scaffold_input=scaffold_input,
            donor_inputs=[source for source in source_inputs if source is not scaffold_input],
        )
        seed_data = donor_integration["seed_data"]
        targets = self.build_targets(seed_data)
        synthesis_variables = self._resolve_synthesis_variables(
            scaffold_input,
            fusion_plan=fusion_plan,
            include_all_observed_targets=len(source_inputs) > 1,
            available_columns=set(seed_data.columns),
            observed_frame=seed_data,
        )
        synthetic_data, synthesizer, synthesis_metadata = self.synthesize(
            seed_data,
            synthesis_variables=synthesis_variables,
        )
        synthesis_metadata = {
            **synthesis_metadata,
            "source_names": fusion_plan.source_names,
            "condition_vars": list(synthesis_variables.condition_vars),
            "target_vars": list(synthesis_variables.target_vars),
            "scaffold_source": scaffold_input.frame.source.name,
            "donor_integrated_variables": donor_integration["integrated_variables"],
            "donor_excluded_variables": list(self.config.donor_imputer_excluded_variables),
            "state_program_support_proxies": _state_program_support_proxy_summary(
                set(seed_data.columns)
            ),
        }
        synthetic_data = self.ensure_target_support(synthetic_data, seed_data, targets)
        if self.config.policyengine_targets_db is not None:
            synthetic_tables = self.build_policyengine_entity_tables(synthetic_data)
            (
                policyengine_tables,
                calibrated_data,
                calibration_summary,
            ) = self.calibrate_policyengine_tables(synthetic_tables)
        else:
            calibrated_data, calibration_summary = self.calibrate(synthetic_data, targets)
            policyengine_tables = self.build_policyengine_entity_tables(calibrated_data)

        return USMicroplexBuildResult(
            config=self.config,
            seed_data=seed_data,
            synthetic_data=synthetic_data,
            calibrated_data=calibrated_data,
            targets=targets,
            calibration_summary=calibration_summary,
            synthesis_metadata=synthesis_metadata,
            synthesizer=synthesizer,
            policyengine_tables=policyengine_tables,
            source_frame=scaffold_input.frame,
            source_frames=tuple(frame for frame in frames),
            fusion_plan=fusion_plan,
        )

    def build(
        self,
        persons: pd.DataFrame,
        households: pd.DataFrame,
    ) -> USMicroplexBuildResult:
        return self.build_from_frame(
            self._build_direct_input_frame(
                persons=persons,
                households=households,
            )
        )

    def _resolve_source_query(
        self,
        provider: SourceProvider,
        queries: dict[str, SourceQuery],
    ) -> SourceQuery | None:
        for key in self._source_query_keys(provider):
            query = queries.get(key)
            if query is not None:
                return query
        return None

    def _source_query_keys(self, provider: SourceProvider) -> tuple[str, ...]:
        base_name = provider.descriptor.name
        keys: list[str] = [base_name]
        for attr_name in ("year", "target_year"):
            attr_value = getattr(provider, attr_name, None)
            if attr_value is None:
                continue
            keys.append(f"{base_name}_{attr_value}")
        descriptor_cache = getattr(provider, "_descriptor_cache", None)
        cached_name = getattr(descriptor_cache, "name", None)
        if cached_name is not None:
            keys.append(cached_name)
        return tuple(dict.fromkeys(keys))

    def prepare_source_input(
        self,
        frame: ObservationFrame,
    ) -> USMicroplexSourceInput:
        """Validate and extract the source-planning context for a US build."""
        frame.validate()
        households = frame.tables.get(EntityType.HOUSEHOLD)
        persons = frame.tables.get(EntityType.PERSON)
        if households is None or persons is None:
            raise ValueError(
                "USMicroplexPipeline requires household and person tables from the source provider"
            )

        fusion_plan = FusionPlan.from_sources([frame.source])
        observations_by_entity = {
            observation.entity: observation for observation in frame.source.observations
        }
        household_observation = observations_by_entity.get(EntityType.HOUSEHOLD)
        person_observation = observations_by_entity.get(EntityType.PERSON)
        if household_observation is None or person_observation is None:
            raise ValueError(
                "USMicroplexPipeline requires household and person observations in the source descriptor"
            )

        relationship = next(
            (
                candidate
                for candidate in frame.relationships
                if candidate.parent_entity == EntityType.HOUSEHOLD
                and candidate.child_entity == EntityType.PERSON
                and candidate.cardinality == RelationshipCardinality.ONE_TO_MANY
            ),
            None,
        )
        if relationship is None:
            raise ValueError(
                "USMicroplexPipeline requires a one-to-many household-to-person relationship"
            )

        return USMicroplexSourceInput(
            frame=frame,
            fusion_plan=fusion_plan,
            household_observation=household_observation,
            person_observation=person_observation,
            household_person_relationship=relationship,
            households=households,
            persons=persons,
        )

    def prepare_seed_data_from_source(
        self,
        source_input: USMicroplexSourceInput,
    ) -> pd.DataFrame:
        """Project an observation frame into the canonical US seed schema."""
        household_coverage = source_input.fusion_plan.variables_for(EntityType.HOUSEHOLD)
        person_coverage = source_input.fusion_plan.variables_for(EntityType.PERSON)
        relationship = source_input.household_person_relationship

        hh = source_input.households.copy()
        persons_df = source_input.persons.copy()

        household_renames = {
            relationship.parent_key: "household_id",
        }
        if source_input.household_observation.weight_column is not None:
            household_renames[source_input.household_observation.weight_column] = "hh_weight"
        hh = hh.rename(columns=household_renames)

        person_renames = {
            source_input.person_observation.key_column: "person_id",
            relationship.child_key: "household_id",
        }
        persons_df = persons_df.rename(columns=person_renames)

        if "household_id" not in hh.columns:
            raise ValueError(
                "USMicroplexPipeline could not resolve a canonical household_id from the source frame"
            )
        if "household_id" not in persons_df.columns or "person_id" not in persons_df.columns:
            raise ValueError(
                "USMicroplexPipeline could not resolve canonical person/household linkage columns"
            )

        if "hh_weight" not in hh.columns:
            hh["hh_weight"] = 1.0
        if "state_fips" not in household_coverage or "state_fips" not in hh.columns:
            hh["state_fips"] = 0
        if "county_fips" not in household_coverage or "county_fips" not in hh.columns:
            hh["county_fips"] = 0
        if "tenure" not in household_coverage or "tenure" not in hh.columns:
            hh["tenure"] = 0

        required_person_defaults = {
            "age": 0,
            "sex": 0,
            "education": 0,
            "employment_status": 0,
            "income": 0.0,
        }
        for column, default in required_person_defaults.items():
            if column not in person_coverage or column not in persons_df.columns:
                persons_df[column] = default

        seed_data = persons_df.merge(
            hh[["household_id", "state_fips", "county_fips", "hh_weight", "tenure"]],
            on="household_id",
            how="left",
            suffixes=("", "__household"),
        )
        for column in ("state_fips", "county_fips", "hh_weight", "tenure"):
            household_column = f"{column}__household"
            if household_column not in seed_data.columns:
                continue
            if column in seed_data.columns:
                seed_data[column] = seed_data[household_column].combine_first(
                    seed_data[column]
                )
            else:
                seed_data[column] = seed_data[household_column]
            seed_data = seed_data.drop(columns=[household_column])
        seed_data["hh_weight"] = seed_data["hh_weight"].fillna(1.0).astype(float)
        seed_data["tenure"] = seed_data["tenure"].fillna(0).astype(int)
        seed_data["state_fips"] = seed_data["state_fips"].fillna(0).astype(int)
        seed_data["county_fips"] = seed_data["county_fips"].fillna(0).astype(int)
        seed_data["income"] = pd.to_numeric(seed_data["income"], errors="coerce").fillna(0.0)

        seed_data["state"] = seed_data["state_fips"].map(STATE_FIPS).fillna("UNK")
        seed_data["age_group"] = pd.cut(
            seed_data["age"],
            bins=AGE_BINS,
            labels=AGE_LABELS,
            right=False,
        )
        seed_data["income_bracket"] = pd.cut(
            seed_data["income"],
            bins=INCOME_BINS,
            labels=INCOME_LABELS,
        )

        return seed_data.reset_index(drop=True)

    def prepare_seed_data(
        self,
        persons: pd.DataFrame,
        households: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge canonical person and household inputs into a synthesis-ready seed frame."""
        return self.prepare_seed_data_from_source(
            self.prepare_source_input(
                self._build_direct_input_frame(
                    persons=persons,
                    households=households,
                )
            )
        )

    def _build_direct_input_frame(
        self,
        *,
        persons: pd.DataFrame,
        households: pd.DataFrame,
    ) -> ObservationFrame:
        """Wrap direct person/household inputs in an observation frame."""
        household_weight_column = next(
            (
                column
                for column in ("hh_weight", "household_weight")
                if column in households.columns
            ),
            None,
        )
        person_weight_column = "weight" if "weight" in persons.columns else None
        household_columns = tuple(
            column
            for column in households.columns
            if column
            not in {
                "household_id",
                household_weight_column,
            }
        )
        person_columns = tuple(
            column
            for column in persons.columns
            if column
            not in {
                "person_id",
                "household_id",
                person_weight_column,
            }
        )
        frame = ObservationFrame(
            source=SourceDescriptor(
                name="us_microplex_direct_input",
                shareability=Shareability.PUBLIC,
                time_structure=TimeStructure.REPEATED_CROSS_SECTION,
                observations=(
                    EntityObservation(
                        entity=EntityType.HOUSEHOLD,
                        key_column="household_id",
                        variable_names=household_columns,
                        weight_column=household_weight_column,
                    ),
                    EntityObservation(
                        entity=EntityType.PERSON,
                        key_column="person_id",
                        variable_names=person_columns,
                        weight_column=person_weight_column,
                    ),
                ),
            ),
            tables={
                EntityType.HOUSEHOLD: households.copy(),
                EntityType.PERSON: persons.copy(),
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

    def build_targets(
        self,
        seed_data: pd.DataFrame,
        weight_col: str = "hh_weight",
    ) -> USMicroplexTargets:
        """Build weighted calibration targets from the seed data."""
        weights = seed_data[weight_col].astype(float).values
        marginal: dict[str, dict[str, float]] = {}

        for column in ("state", "age_group", "income_bracket"):
            marginal[column] = {}
            categories = seed_data[column].dropna().astype(str).unique()
            for category in categories:
                mask = seed_data[column].astype(str) == category
                marginal[column][category] = float(weights[mask].sum())

        continuous = {
            "income": float((weights * seed_data["income"].astype(float).values).sum())
        }

        if self.config.policyengine_quantity_targets:
            if self.config.policyengine_dataset is None:
                raise ValueError(
                    "policyengine_dataset is required when policyengine_quantity_targets are configured"
                )
            adapter = PolicyEngineUSMicrosimulationAdapter.from_dataset(
                self.config.policyengine_dataset,
                dataset_year=self.config.policyengine_dataset_year,
            )
            continuous.update(
                self.build_policyengine_continuous_targets(
                    seed_data=seed_data,
                    adapter=adapter,
                    quantity_targets=self.config.policyengine_quantity_targets,
                )
            )

        return USMicroplexTargets(marginal=marginal, continuous=continuous)

    def build_policyengine_continuous_targets(
        self,
        seed_data: pd.DataFrame,
        adapter: PolicyEngineUSMicrosimulationAdapter | Any,
        quantity_targets: tuple[PolicyEngineUSQuantityTarget, ...],
    ) -> dict[str, float]:
        """Compute PE-based continuous totals for columns present in the seed data."""
        missing_columns = sorted(
            {
                target.column
                for target in quantity_targets
                if target.column not in seed_data.columns
            }
        )
        if missing_columns:
            raise ValueError(
                f"PolicyEngine target columns not available in seed data: {missing_columns}"
            )

        computed = adapter.compute_targets(quantity_targets)
        continuous_targets: dict[str, float] = {}
        for target in quantity_targets:
            if target.name not in computed:
                raise ValueError(
                    f"PolicyEngine adapter did not return target '{target.name}'"
                )
            continuous_targets[target.column] = float(computed[target.name])
        return continuous_targets

    def ensure_target_support(
        self,
        synthetic_data: pd.DataFrame,
        seed_data: pd.DataFrame,
        targets: USMicroplexTargets,
    ) -> pd.DataFrame:
        """Ensure every marginal target category has support in the synthetic sample."""
        result = synthetic_data.copy().reset_index(drop=True)
        bool_columns = [
            column
            for column in result.columns
            if pd.api.types.is_bool_dtype(result[column].dtype)
        ]
        if bool_columns:
            result[bool_columns] = result[bool_columns].astype(float)
        replace_idx = 0

        for _ in range(sum(len(v) for v in targets.marginal.values())):
            missing: list[tuple[str, str]] = []
            for column, categories in targets.marginal.items():
                current = result[column].astype(str)
                for category in categories:
                    if not (current == str(category)).any():
                        missing.append((column, str(category)))

            if not missing:
                break

            for column, category in missing:
                exemplars = seed_data[seed_data[column].astype(str) == category]
                if exemplars.empty:
                    continue
                exemplar = exemplars.iloc[0]
                row_idx = replace_idx % len(result)
                for column_name, value in exemplar.items():
                    if column_name in result.columns and column_name not in {"person_id", "household_id", "weight"}:
                        resolved_value = value
                        destination = result[column_name]
                        if pd.api.types.is_bool_dtype(destination.dtype) and not isinstance(
                            resolved_value,
                            (bool, np.bool_),
                        ):
                            result[column_name] = destination.astype(float)
                            destination = result[column_name]
                        if pd.api.types.is_numeric_dtype(destination.dtype) and isinstance(
                            value,
                            (bool, np.bool_),
                        ):
                            resolved_value = float(value)
                        result.at[row_idx, column_name] = resolved_value
                replace_idx += 1

        initial_weight = float(result["weight"].mean()) if "weight" in result.columns else 1.0
        base = result.drop(
            columns=["person_id", "state", "age_group", "income_bracket"],
            errors="ignore",
        )
        return self._finalize_synthetic_population(base, initial_weight=initial_weight)

    def synthesize(
        self,
        seed_data: pd.DataFrame,
        synthesis_variables: USMicroplexSynthesisVariables | None = None,
    ) -> tuple[pd.DataFrame, Synthesizer | None, dict[str, Any]]:
        """Generate synthetic records from the seed data."""
        if "hh_weight" in seed_data.columns:
            initial_weight = float(seed_data["hh_weight"].sum()) / max(
                self.config.n_synthetic, 1
            )
        else:
            initial_weight = 1.0
        synthesis_variables = synthesis_variables or USMicroplexSynthesisVariables(
            condition_vars=self._resolve_synthesis_condition_vars(
                seed_data.columns,
                observed_frame=seed_data,
            ),
            target_vars=tuple(
                column
                for column in self.config.synthesizer_target_vars
                if column in seed_data.columns
            ),
        )

        if self.config.synthesis_backend == "seed":
            synthetic = seed_data.copy()
            if "hh_weight" in synthetic.columns and "weight" not in synthetic.columns:
                synthetic["weight"] = (
                    pd.to_numeric(synthetic["hh_weight"], errors="coerce")
                    .fillna(initial_weight)
                    .astype(float)
                )
            synthetic = self._finalize_synthetic_population(
                synthetic,
                initial_weight=float(
                    pd.to_numeric(
                        synthetic.get("weight", pd.Series([initial_weight])),
                        errors="coerce",
                    )
                    .fillna(initial_weight)
                    .mean()
                ),
            )
            return synthetic, None, {
                "backend": "seed",
                "n_seed_records": int(len(seed_data)),
            }

        if self.config.synthesis_backend == "bootstrap":
            bootstrap_strata_columns = self._resolve_bootstrap_strata_columns(seed_data)
            synthetic = self._synthesize_bootstrap(
                seed_data,
                initial_weight=initial_weight,
                strata_columns=bootstrap_strata_columns,
            )
            return synthetic, None, {
                "backend": "bootstrap",
                "bootstrap_strata_columns": list(bootstrap_strata_columns),
            }

        synthesizer = self._fit_synthesizer(seed_data, synthesis_variables)
        synthetic = synthesizer.sample(
            self.config.n_synthetic,
            seed=self.config.random_seed,
        )
        synthetic = self._finalize_synthetic_population(
            synthetic,
            initial_weight=initial_weight,
        )
        return synthetic, synthesizer, {"backend": "synthesizer"}

    def calibrate(
        self,
        synthetic_data: pd.DataFrame,
        targets: USMicroplexTargets,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Calibrate synthetic records to weighted targets."""
        calibrator = self._build_weight_calibrator()
        if self.config.calibration_backend in {"entropy", "ipf", "chi2"}:
            calibrated = calibrator.fit_transform(
                synthetic_data,
                targets.marginal,
                targets.continuous,
                weight_col="weight",
            )
            validation = calibrator.validate(calibrated)
            all_errors = []
            for var_errors in validation["marginal_errors"].values():
                all_errors.extend(item["relative_error"] for item in var_errors.values())
            all_errors.extend(
                item["relative_error"] for item in validation["continuous_errors"].values()
            )
            summary = {
                "backend": self.config.calibration_backend,
                "max_error": float(validation["max_error"]),
                "mean_error": float(np.mean(all_errors)) if all_errors else 0.0,
                "converged": bool(validation["converged"]),
            }
            return calibrated, summary

        calibrated = calibrator.fit_transform(
            synthetic_data,
            targets.marginal,
            targets.continuous,
            weight_col="weight",
        )
        validation = calibrator.validate(calibrated)
        summary = {
            "backend": self.config.calibration_backend,
            "max_error": float(validation["max_error"]),
            "mean_error": float(validation["mean_error"]),
            "sparsity": float(validation.get("sparsity", 0.0)),
            "converged": bool(validation.get("converged", False)),
        }
        return calibrated, summary

    def _build_weight_calibrator(
        self,
    ) -> Calibrator | SparseCalibrator | HardConcreteCalibrator:
        if self.config.calibration_backend in {"entropy", "ipf", "chi2"}:
            return Calibrator(
                method=self.config.calibration_backend,
                tol=self.config.calibration_tol,
                max_iter=self.config.calibration_max_iter,
            )
        if self.config.calibration_backend == "sparse":
            return SparseCalibrator(
                target_sparsity=self.config.target_sparsity,
                tol=self.config.calibration_tol,
                max_iter=max(self.config.calibration_max_iter, 1_000),
            )
        if self.config.calibration_backend == "hardconcrete":
            return HardConcreteCalibrator(
                lambda_l0=1e-4,
                epochs=max(self.config.calibration_max_iter, 500),
                lr=0.1,
                device=self.config.device,
                verbose=False,
            )
        raise ValueError(
            f"Unsupported calibration backend: {self.config.calibration_backend}"
        )

    def _select_policyengine_household_budget(
        self,
        tables: PolicyEngineUSEntityTableBundle,
        supported_targets: list[TargetSpec],
        constraints: tuple[LinearConstraint, ...],
    ) -> tuple[
        PolicyEngineUSEntityTableBundle,
        list[TargetSpec],
        tuple[LinearConstraint, ...],
        dict[str, Any],
    ]:
        requested_budget = self.config.policyengine_selection_household_budget
        household_count = len(tables.households)
        if requested_budget is None or requested_budget >= household_count:
            return (
                tables,
                supported_targets,
                constraints,
                {
                    "applied": False,
                    "requested_household_budget": requested_budget,
                    "input_household_count": household_count,
                },
            )
        if requested_budget <= 0:
            raise ValueError("policyengine_selection_household_budget must be positive")
        if not constraints:
            return (
                tables,
                supported_targets,
                constraints,
                {
                    "applied": False,
                    "requested_household_budget": requested_budget,
                    "input_household_count": household_count,
                    "reason": "no_constraints",
                },
            )

        target_sparsity = max(0.0, 1.0 - (requested_budget / household_count))
        selector = SparseCalibrator(
            target_sparsity=target_sparsity,
            tol=self.config.calibration_tol,
            max_iter=max(self.config.calibration_max_iter, 1_000),
        )
        selector_result = selector.fit_transform(
            tables.households.copy(),
            {},
            weight_col="household_weight",
            linear_constraints=constraints,
        )
        selector_validation = selector.validate(selector_result)
        selector_weights = (
            pd.to_numeric(selector_result["household_weight"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        household_ids = tables.households["household_id"].to_numpy(dtype=np.int64)
        ranking = np.lexsort((household_ids, -selector_weights))
        selected_positions = np.sort(ranking[:requested_budget])
        household_mask = np.zeros(household_count, dtype=bool)
        household_mask[selected_positions] = True
        selected_ids = pd.Index(household_ids[household_mask], name="household_id")

        return (
            _subset_policyengine_tables_by_households(tables, selected_ids),
            supported_targets,
            _subset_policyengine_linear_constraints(constraints, household_mask),
            {
                "applied": True,
                "backend": "sparse",
                "requested_household_budget": int(requested_budget),
                "input_household_count": int(household_count),
                "selected_household_count": int(household_mask.sum()),
                "target_sparsity": float(target_sparsity),
                "selector_converged": bool(selector_validation.get("converged", False)),
                "selector_max_error": float(selector_validation.get("max_error", 0.0)),
                "selector_mean_error": float(selector_validation.get("mean_error", 0.0)),
                "selector_sparsity": float(selector_validation.get("sparsity", 0.0)),
                "selector_nonzero_count": int((selector_weights > 0.0).sum()),
                "selector_positive_selected_count": int(
                    (selector_weights[household_mask] > 0.0).sum()
                ),
                "selector_weight_diagnostics": _summarize_weight_diagnostics(
                    selector_weights
                ),
            },
        )

    def calibrate_policyengine_tables(
        self,
        tables: PolicyEngineUSEntityTableBundle,
    ) -> tuple[PolicyEngineUSEntityTableBundle, pd.DataFrame, dict[str, Any]]:
        """Calibrate household weights using PolicyEngine US target DB constraints."""
        if self.config.policyengine_targets_db is None:
            raise ValueError("policyengine_targets_db is required for DB calibration")

        provider = PolicyEngineUSDBTargetProvider(self.config.policyengine_targets_db)
        target_period = (
            self.config.policyengine_target_period
            or self.config.policyengine_dataset_year
            or 2024
        )
        (
            tables,
            bindings,
            canonical_targets,
            supported_targets,
            unsupported_targets,
            constraints,
            feasibility_filter_summary,
            materialized_variables,
            materialization_failures,
        ) = self._resolve_policyengine_calibration_targets(
            tables,
            provider=provider,
            target_period=target_period,
        )
        if not supported_targets:
            raise ValueError("No supported PolicyEngine DB targets matched current tables")
        selection_summary: dict[str, Any] | None = None
        if self.config.policyengine_selection_household_budget is not None:
            (
                tables,
                supported_targets,
                constraints,
                selection_summary,
            ) = self._select_policyengine_household_budget(
                tables,
                supported_targets,
                tuple(constraints),
            )
            if selection_summary.get("applied"):
                (
                    supported_targets,
                    constraints,
                    post_selection_feasibility_summary,
                ) = _select_feasible_policyengine_calibration_constraints(
                    supported_targets,
                    constraints,
                    household_count=len(tables.households),
                    max_constraints=self.config.policyengine_calibration_max_constraints,
                    max_constraints_per_household=(
                        self.config.policyengine_calibration_max_constraints_per_household
                    ),
                    min_active_households=(
                        self.config.policyengine_calibration_min_active_households
                    ),
                )
                feasibility_filter_summary = {
                    **post_selection_feasibility_summary,
                    "pre_selection": feasibility_filter_summary,
                }
                if not supported_targets:
                    raise ValueError(
                        "No supported PolicyEngine DB targets remained after household-budget selection"
                    )
        calibrator = self._build_weight_calibrator()
        calibrated_households = calibrator.fit_transform(
            tables.households.copy(),
            {},
            weight_col="household_weight",
            linear_constraints=constraints,
        )
        validation = calibrator.validate(calibrated_households)

        household_weights = calibrated_households.set_index("household_id")["household_weight"]
        calibrated_persons = tables.persons.copy() if tables.persons is not None else pd.DataFrame()
        if not calibrated_persons.empty:
            calibrated_persons["weight"] = calibrated_persons["household_id"].map(
                household_weights
            ).astype(float)

        updated_tables = PolicyEngineUSEntityTableBundle(
            households=calibrated_households,
            persons=calibrated_persons if not calibrated_persons.empty else tables.persons,
            tax_units=tables.tax_units,
            spm_units=tables.spm_units,
            families=tables.families,
            marital_units=tables.marital_units,
        )
        household_weight_diagnostics = _summarize_weight_diagnostics(
            calibrated_households["household_weight"]
        )
        person_weight_diagnostics = (
            _summarize_weight_diagnostics(calibrated_persons["weight"])
            if not calibrated_persons.empty and "weight" in calibrated_persons.columns
            else None
        )
        linear_errors = list(validation.get("linear_errors", {}).values())
        summary = {
            "backend": f"policyengine_db_{self.config.calibration_backend}",
            "period": int(target_period),
            "n_loaded_targets": len(canonical_targets),
            "n_supported_targets": len(supported_targets),
            "n_unsupported_targets": len(unsupported_targets),
            "n_constraints": len(constraints),
            "feasibility_filter": feasibility_filter_summary,
            "target_variables": list(self._policyengine_target_scope(for_calibration=True)[0]),
            "target_domains": list(self._policyengine_target_scope(for_calibration=True)[1]),
            "target_geo_levels": list(self._policyengine_target_scope(for_calibration=True)[2]),
            "target_profile": self._policyengine_target_profile(for_calibration=True),
            "target_cell_count": len(self._policyengine_target_cells(for_calibration=True)),
            "materialized_variables": sorted(materialized_variables),
            "materialization_failures": materialization_failures,
            "max_error": float(validation["max_error"]),
            "mean_error": (
                float(np.mean([error["relative_error"] for error in linear_errors]))
                if linear_errors
                else 0.0
            ),
            "converged": bool(validation["converged"]),
            "sparsity": float(validation.get("sparsity", 0.0)),
            "weight_collapse_suspected": bool(
                household_weight_diagnostics["collapse_suspected"]
                or (
                    person_weight_diagnostics is not None
                    and person_weight_diagnostics["collapse_suspected"]
                )
            ),
            "household_weight_diagnostics": household_weight_diagnostics,
            "person_weight_diagnostics": person_weight_diagnostics,
        }
        if selection_summary is not None:
            summary["selection"] = selection_summary
        warning_messages = list(feasibility_filter_summary.get("warning_messages", ()))
        if not summary["converged"]:
            warning_messages.append(
                "Calibration did not converge on the selected constraint set."
            )
        summary["warnings"] = warning_messages
        for message in warning_messages:
            warnings.warn(message, stacklevel=2)
        return updated_tables, calibrated_persons, summary

    def _resolve_policyengine_calibration_targets(
        self,
        tables: PolicyEngineUSEntityTableBundle,
        *,
        provider: PolicyEngineUSDBTargetProvider,
        target_period: int,
    ) -> tuple[
        PolicyEngineUSEntityTableBundle,
        dict[str, PolicyEngineUSVariableBinding],
        list[TargetSpec],
        list[TargetSpec],
        list[TargetSpec],
        tuple[Any, ...],
        dict[str, Any],
        set[str],
        dict[str, str],
    ]:
        bindings = infer_policyengine_us_variable_bindings(tables)
        canonical_targets = self._load_policyengine_target_set(
            provider,
            bindings=bindings,
            period=target_period,
            for_calibration=True,
        ).targets
        missing_variables = policyengine_us_variables_to_materialize(
            canonical_targets,
            bindings,
        )
        materialization_failures: dict[str, str] = {}
        materialized_variables: set[str] = set()
        if missing_variables:
            materialization_result = materialize_policyengine_us_variables_safely(
                tables,
                variables=tuple(sorted(missing_variables)),
                period=target_period,
                dataset_year=self.config.policyengine_dataset_year or target_period,
                simulation_cls=self.config.policyengine_simulation_cls,
            )
            tables = materialization_result.tables
            bindings = {
                **bindings,
                **materialization_result.bindings,
            }
            materialized_variables = set(materialization_result.materialized_variables)
            materialization_failures = dict(materialization_result.failed_variables)
            canonical_targets = self._load_policyengine_target_set(
                provider,
                bindings=bindings,
                period=target_period,
                for_calibration=True,
            ).targets
        supported_targets = filter_supported_policyengine_us_targets(
            canonical_targets,
            tables,
            bindings,
        )
        supported_targets, unsupported_targets, constraints = (
            compile_supported_policyengine_us_household_linear_constraints(
                supported_targets,
                tables,
                variable_bindings=bindings,
            )
        )
        (
            supported_targets,
            constraints,
            feasibility_filter_summary,
        ) = _select_feasible_policyengine_calibration_constraints(
            supported_targets,
            constraints,
            household_count=len(tables.households),
            max_constraints=self.config.policyengine_calibration_max_constraints,
            max_constraints_per_household=(
                self.config.policyengine_calibration_max_constraints_per_household
            ),
            min_active_households=(
                self.config.policyengine_calibration_min_active_households
            ),
        )
        return (
            tables,
            bindings,
            canonical_targets,
            supported_targets,
            unsupported_targets,
            constraints,
            feasibility_filter_summary,
            materialized_variables,
            materialization_failures,
        )

    def _load_policyengine_target_set(
        self,
        provider: PolicyEngineUSDBTargetProvider,
        *,
        bindings: dict[str, PolicyEngineUSVariableBinding],
        period: int,
        for_calibration: bool,
    ):
        return provider.load_target_set(
            self._build_policyengine_target_query(
                bindings,
                period=period,
                for_calibration=for_calibration,
            )
        )

    def _policyengine_target_scope(
        self,
        *,
        for_calibration: bool,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        variables = (
            self.config.policyengine_calibration_target_variables
            if for_calibration and self.config.policyengine_calibration_target_variables
            else self.config.policyengine_target_variables
        )
        domain_variables = (
            self.config.policyengine_calibration_target_domains
            if for_calibration and self.config.policyengine_calibration_target_domains
            else self.config.policyengine_target_domains
        )
        geo_levels = (
            self.config.policyengine_calibration_target_geo_levels
            if for_calibration and self.config.policyengine_calibration_target_geo_levels
            else self.config.policyengine_target_geo_levels
        )
        return variables, domain_variables, geo_levels

    def _policyengine_target_profile(
        self,
        *,
        for_calibration: bool,
    ) -> str | None:
        return (
            self.config.policyengine_calibration_target_profile
            if for_calibration and self.config.policyengine_calibration_target_profile
            else self.config.policyengine_target_profile
        )

    def _policyengine_target_cells(
        self,
        *,
        for_calibration: bool,
    ) -> tuple[PolicyEngineUSTargetCell, ...]:
        profile_name = self._policyengine_target_profile(for_calibration=for_calibration)
        if profile_name is None:
            return ()
        return resolve_policyengine_us_target_profile(profile_name)

    def _build_policyengine_target_query(
        self,
        bindings: dict[str, PolicyEngineUSVariableBinding],
        *,
        period: int,
        for_calibration: bool = False,
    ) -> TargetQuery:
        variables, domain_variables, geo_levels = self._policyengine_target_scope(
            for_calibration=for_calibration
        )
        profile_name = self._policyengine_target_profile(for_calibration=for_calibration)
        target_cells = self._policyengine_target_cells(for_calibration=for_calibration)
        return TargetQuery(
            period=period,
            provider_filters={
                "variables": list(variables) if variables else None,
                "domain_variables": (
                    list(domain_variables) if domain_variables else None
                ),
                "geo_levels": list(geo_levels) if geo_levels else None,
                "target_profile": profile_name,
                "target_cells": (
                    [cell.to_provider_filter() for cell in target_cells]
                    if target_cells
                    else None
                ),
                "reform_id": self.config.policyengine_target_reform_id,
                "entity_overrides": {
                    variable: binding.entity for variable, binding in bindings.items()
                },
            },
        )

    def build_policyengine_entity_tables(
        self,
        population: pd.DataFrame,
    ) -> PolicyEngineUSEntityTableBundle:
        """Build a PolicyEngine-oriented multientity bundle from person rows."""
        persons = population.copy().reset_index(drop=True)
        if "person_id" not in persons.columns:
            persons["person_id"] = np.arange(len(persons), dtype=np.int64)
        if "household_id" not in persons.columns:
            persons["household_id"] = np.arange(len(persons), dtype=np.int64)
        if "weight" not in persons.columns:
            persons["weight"] = 1.0
        if "income" not in persons.columns:
            persons["income"] = 0.0
        if "age" not in persons.columns:
            persons["age"] = 0

        persons["person_id"] = persons["person_id"].astype(np.int64)
        persons["household_id"] = persons["household_id"].astype(np.int64)
        persons["weight"] = pd.to_numeric(persons["weight"], errors="coerce").fillna(0.0)
        persons["income"] = pd.to_numeric(persons["income"], errors="coerce").fillna(0.0)
        persons["age"] = pd.to_numeric(persons["age"], errors="coerce").fillna(0).astype(int)
        persons = self._augment_policyengine_person_inputs(persons)
        persons["relationship_to_head"] = self._normalize_relationship_to_head(persons)

        households = self._build_policyengine_households(persons)
        tax_units, persons = self._build_policyengine_tax_units(persons)
        persons = self._assign_family_and_spm_units(persons)
        families = self._collapse_group_table(persons, "family_id")
        spm_units = self._collapse_group_table(persons, "spm_unit_id")
        persons = self._assign_marital_units(persons)
        marital_units = self._collapse_group_table(persons, "marital_unit_id")

        return PolicyEngineUSEntityTableBundle(
            households=households,
            persons=persons,
            tax_units=tax_units,
            spm_units=spm_units,
            families=families,
            marital_units=marital_units,
        )

    def export_policyengine_dataset(
        self,
        result: USMicroplexBuildResult,
        path: str | Path,
        *,
        period: int | None = None,
        direct_override_variables: tuple[str, ...] | None = None,
    ) -> Path:
        """Export a build result as a PolicyEngine-readable HDF5 dataset."""
        export_period = (
            period
            or self.config.policyengine_dataset_year
            or result.config.policyengine_dataset_year
            or 2024
        )
        export_direct_override_variables = (
            direct_override_variables
            if direct_override_variables is not None
            else (
                self.config.policyengine_direct_override_variables
                or result.config.policyengine_direct_override_variables
            )
        )
        tables = result.policyengine_tables or self.build_policyengine_entity_tables(
            result.calibrated_data
        )
        tax_benefit_system = self._resolve_policyengine_tax_benefit_system()
        export_maps = build_policyengine_us_export_variable_maps(
            tables,
            tax_benefit_system=tax_benefit_system,
            direct_override_variables=export_direct_override_variables,
        )
        excluded_variables = detect_policyengine_pseudo_inputs(
            tax_benefit_system,
            sorted(
                {
                    target
                    for variable_map in export_maps.values()
                    for target in variable_map.values()
                }
            ),
        )
        arrays = build_policyengine_us_time_period_arrays(
            tables,
            period=export_period,
            household_variable_map=export_maps["household"],
            person_variable_map=export_maps["person"],
            tax_unit_variable_map=export_maps["tax_unit"],
            spm_unit_variable_map=export_maps["spm_unit"],
            family_variable_map=export_maps["family"],
        )
        return write_policyengine_us_time_period_dataset(
            arrays,
            path,
            excluded_variables=excluded_variables,
        )

    def _fit_synthesizer(
        self,
        seed_data: pd.DataFrame,
        synthesis_variables: USMicroplexSynthesisVariables,
    ) -> Synthesizer:
        """Fit a microplex synthesizer on the seed data."""
        condition_vars = list(synthesis_variables.condition_vars)
        target_vars = list(synthesis_variables.target_vars)
        if not target_vars:
            raise ValueError("USMicroplexPipeline requires at least one observed target variable")

        synthesizer = Synthesizer(
            target_vars=target_vars,
            condition_vars=condition_vars,
            n_layers=self.config.synthesizer_n_layers,
            hidden_dim=self.config.synthesizer_hidden_dim,
        )
        synthesizer.fit(
            seed_data[condition_vars + target_vars + ["hh_weight"]].rename(
                columns={"hh_weight": "weight"}
            ),
            weight_col="weight",
            epochs=self.config.synthesizer_epochs,
            batch_size=self.config.synthesizer_batch_size,
            learning_rate=self.config.synthesizer_learning_rate,
            verbose=False,
        )
        return synthesizer

    def _build_donor_imputer(
        self,
        *,
        condition_vars: list[str],
        target_vars: tuple[str, ...],
    ) -> Synthesizer | ColumnwiseQRFDonorImputer:
        backend = self.config.donor_imputer_backend
        if backend == "maf":
            return Synthesizer(
                target_vars=list(target_vars),
                condition_vars=condition_vars,
                n_layers=self.config.donor_imputer_n_layers,
                hidden_dim=self.config.donor_imputer_hidden_dim,
            )

        support_families = {
            variable: variable_semantic_spec_for(variable).support_family
            for variable in target_vars
        }
        zero_inflated_vars = (
            {
                variable
                for variable, support_family in support_families.items()
                if support_family is VariableSupportFamily.ZERO_INFLATED_POSITIVE
            }
            if backend == "zi_qrf"
            else set()
        )
        nonnegative_vars = {
            variable
            for variable, support_family in support_families.items()
            if support_family
            in {
                VariableSupportFamily.ZERO_INFLATED_POSITIVE,
                VariableSupportFamily.BOUNDED_SHARE,
            }
        }
        return ColumnwiseQRFDonorImputer(
            condition_vars=condition_vars,
            target_vars=list(target_vars),
            n_estimators=self.config.donor_imputer_qrf_n_estimators,
            zero_inflated_vars=zero_inflated_vars,
            nonnegative_vars=nonnegative_vars,
            zero_threshold=self.config.donor_imputer_qrf_zero_threshold,
        )

    def _resolve_synthesis_variables(
        self,
        source_input: USMicroplexSourceInput,
        *,
        fusion_plan: FusionPlan | None = None,
        include_all_observed_targets: bool = False,
        available_columns: set[str] | None = None,
        observed_frame: pd.DataFrame | None = None,
    ) -> USMicroplexSynthesisVariables:
        """Select the observed variables to feed into synthesis."""
        active_plan = fusion_plan or source_input.fusion_plan
        available_variables = prune_redundant_variables(
            active_plan.variables_for(EntityType.HOUSEHOLD)
            | active_plan.variables_for(EntityType.PERSON)
        )
        if available_columns is not None:
            available_variables = available_variables & available_columns
        condition_vars = self._resolve_synthesis_condition_vars(
            available_variables,
            observed_frame=observed_frame,
        )
        configured_targets = [
            variable
            for variable in self.config.synthesizer_target_vars
            if variable in available_variables and variable not in condition_vars
        ]
        configured_targets.extend(
            variable
            for variable in STATE_PROGRAM_SUPPORT_PROXY_VARIABLES
            if variable in available_variables
            and variable not in condition_vars
            and variable not in configured_targets
        )
        extra_targets: list[str] = []
        if include_all_observed_targets:
            excluded = {
                "person_id",
                "household_id",
                "hh_weight",
                "weight",
                "state",
                "age_group",
                "income_bracket",
            }
            extra_targets = sorted(
                variable
                for variable in available_variables
                if variable not in excluded
                and variable not in condition_vars
                and variable not in configured_targets
            )
        return USMicroplexSynthesisVariables(
            condition_vars=condition_vars,
            target_vars=tuple(configured_targets + extra_targets),
        )

    def _resolve_synthesis_condition_vars(
        self,
        available_columns: Iterable[str],
        *,
        observed_frame: pd.DataFrame | None = None,
    ) -> tuple[str, ...]:
        available = set(available_columns)
        ordered = list(self.config.synthesizer_condition_vars)
        for variable in STATE_PROGRAM_AUTO_CONDITION_VARIABLES:
            if (
                variable in available
                and variable not in ordered
                and (
                    observed_frame is None
                    or self._is_informative_state_program_proxy(
                        observed_frame,
                        variable,
                    )
                )
            ):
                ordered.append(variable)
        return tuple(variable for variable in ordered if variable in available)

    def _is_informative_state_program_proxy(
        self,
        frame: pd.DataFrame,
        variable: str,
    ) -> bool:
        if variable not in frame.columns:
            return False
        series = pd.to_numeric(frame[variable], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        series = series.dropna()
        if series.empty:
            return False
        return bool(series.nunique(dropna=True) > 1)

    def _select_scaffold_source(
        self,
        source_inputs: list[USMicroplexSourceInput],
    ) -> USMicroplexSourceInput:
        candidates = [
            source
            for source in source_inputs
            if source.household_observation is not None
            and source.household_person_relationship is not None
        ]
        if not candidates:
            raise ValueError(
                "USMicroplexPipeline requires at least one structured source with household and person observations"
            )

        def score(source: USMicroplexSourceInput) -> tuple[int, int, int, int]:
            public_score = int(source.frame.source.shareability == Shareability.PUBLIC)
            geography_score = self._household_geography_coverage(source)
            observed_variables = (
                source.fusion_plan.variables_for(EntityType.HOUSEHOLD)
                | source.fusion_plan.variables_for(EntityType.PERSON)
            )
            support_proxy_score = sum(
                variable in observed_variables
                for variable in STATE_PROGRAM_SUPPORT_PROXY_VARIABLES
            )
            observed_vars = len(observed_variables)
            household_rows = (
                len(source.households)
                if source.households is not None
                else 0
            )
            return (
                public_score,
                geography_score,
                support_proxy_score,
                observed_vars,
                household_rows,
            )

        return max(candidates, key=score)

    def _household_geography_coverage(
        self,
        source: USMicroplexSourceInput,
    ) -> int:
        households = source.households
        if households is None or "state_fips" not in households.columns:
            return 0
        state_fips = pd.to_numeric(households["state_fips"], errors="coerce").fillna(0)
        return int((state_fips > 0).sum())

    def _integrate_donor_sources(
        self,
        seed_data: pd.DataFrame,
        *,
        scaffold_input: USMicroplexSourceInput,
        donor_inputs: list[USMicroplexSourceInput],
    ) -> dict[str, Any]:
        current = seed_data.copy()
        integrated_variables: list[str] = []
        scaffold_observed = prune_redundant_variables(
            scaffold_input.fusion_plan.variables_for(EntityType.HOUSEHOLD)
            | scaffold_input.fusion_plan.variables_for(EntityType.PERSON)
        )
        excluded = {
            "person_id",
            "household_id",
            "hh_weight",
            "weight",
            "household_weight",
            "tax_unit_id",
            "family_id",
            "spm_unit_id",
            "marital_unit_id",
            "state",
            "age_group",
            "income_bracket",
            "is_head",
            "is_spouse",
            "is_dependent",
        }
        rng = np.random.default_rng(self.config.random_seed)

        for donor_input in donor_inputs:
            donor_seed = self.prepare_seed_data_from_source(donor_input)
            donor_observed = prune_redundant_variables(
                donor_input.fusion_plan.variables_for(EntityType.HOUSEHOLD)
                | donor_input.fusion_plan.variables_for(EntityType.PERSON)
            )
            numeric_current = {
                column
                for column in current.columns
                if pd.api.types.is_numeric_dtype(current[column])
            }
            numeric_donor = {
                column
                for column in donor_seed.columns
                if pd.api.types.is_numeric_dtype(donor_seed[column])
            }
            shared_vars = sorted(
                variable
                for variable in scaffold_observed & donor_observed
                if variable not in excluded
                and variable in current.columns
                and variable in donor_seed.columns
                and variable in numeric_current
                and variable in numeric_donor
                and scaffold_input.frame.source.allows_conditioning_on(variable)
                and donor_input.frame.source.allows_conditioning_on(variable)
                and self._is_compatible_donor_condition(
                    current[variable],
                    donor_seed[variable],
                )
            )
            donor_only_vars = sorted(
                variable
                for variable in donor_observed - scaffold_observed
                if variable not in excluded
                and variable not in self.config.donor_imputer_excluded_variables
                and variable in donor_seed.columns
                and variable in numeric_donor
                and donor_input.frame.source.is_authoritative_for(variable)
                and self._should_integrate_donor_variable(current, variable)
                and self._is_compatible_donor_target(donor_seed[variable])
            )
            if not shared_vars or not donor_only_vars:
                continue

            donor_block_specs = donor_imputation_block_specs(donor_only_vars)
            required_entities = {
                donor_block_spec.native_entity
                for donor_block_spec in donor_block_specs
                if donor_block_spec.native_entity is not EntityType.PERSON
            }
            if required_entities:
                current = self._ensure_seed_entity_ids(
                    current,
                    entities=required_entities,
                )
                donor_seed = self._ensure_seed_entity_ids(
                    donor_seed,
                    entities=required_entities,
                )

            for donor_block_spec in donor_block_specs:
                donor_working = donor_seed.copy()
                if donor_block_spec.prepare_frame is not None:
                    donor_working = donor_block_spec.prepare_frame(donor_working)
                shared_vars_for_block = list(shared_vars)
                donor_fit_source = donor_working
                current_generation_source = current
                entity_key = self._entity_key_column(donor_block_spec.native_entity)
                if self._can_project_donor_block_to_entity(
                    current,
                    donor_working,
                    donor_block_spec.native_entity,
                ):
                    entity_compatible_shared_vars = [
                        variable
                        for variable in shared_vars
                        if is_projected_condition_var_compatible(
                            variable,
                            projected_entity=donor_block_spec.native_entity,
                            allowed_condition_entities=donor_block_spec.condition_entities,
                        )
                    ]
                    if entity_compatible_shared_vars:
                        shared_vars_for_block = entity_compatible_shared_vars
                    donor_fit_source = self._project_frame_to_entity(
                        donor_working,
                        entity=donor_block_spec.native_entity,
                        variables=(
                            set(shared_vars_for_block)
                            | set(donor_block_spec.model_variables)
                            | {"hh_weight"}
                        ),
                    )
                    current_generation_source = self._project_frame_to_entity(
                        current,
                        entity=donor_block_spec.native_entity,
                        variables=set(shared_vars_for_block),
                    )
                donor_condition_vars = self._select_donor_condition_vars(
                    donor_fit_source,
                    shared_vars_for_block,
                    donor_block_spec.model_variables,
                )
                if not donor_condition_vars:
                    continue

                fit_frame = donor_fit_source[
                    donor_condition_vars + list(donor_block_spec.model_variables) + ["hh_weight"]
                ].copy()
                fit_frame = fit_frame.rename(columns={"hh_weight": "weight"})
                imputer = self._build_donor_imputer(
                    condition_vars=donor_condition_vars,
                    target_vars=donor_block_spec.model_variables,
                )
                imputer.fit(
                    fit_frame,
                    weight_col="weight",
                    epochs=self.config.donor_imputer_epochs,
                    batch_size=self.config.donor_imputer_batch_size,
                    learning_rate=self.config.donor_imputer_learning_rate,
                    verbose=False,
                )
                generated = imputer.generate(
                    current_generation_source[donor_condition_vars].copy(),
                    seed=self.config.random_seed,
                )
                for variable in donor_block_spec.model_variables:
                    donor_support = (
                        pd.to_numeric(donor_fit_source[variable], errors="coerce")
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                    )
                    generated_scores = pd.to_numeric(
                        generated[variable],
                        errors="coerce",
                    ).replace([np.inf, -np.inf], np.nan)
                    if donor_support.empty:
                        current[variable] = generated_scores.fillna(0.0).astype(float)
                        continue
                    donor_weights = pd.to_numeric(
                        donor_fit_source.loc[donor_support.index, "hh_weight"],
                        errors="coerce",
                    ).fillna(0.0)
                    matched_values = self._rank_match_donor_values(
                        generated_scores.fillna(float(donor_support.median())).astype(float),
                        donor_values=donor_support.astype(float),
                        donor_weights=donor_weights.astype(float),
                        rng=rng,
                        strategy=donor_block_spec.strategy_for(variable),
                    )
                    if entity_key is not None and entity_key in current_generation_source.columns:
                        entity_values = pd.Series(
                            matched_values.to_numpy(dtype=float),
                            index=current_generation_source[entity_key].to_numpy(),
                            dtype=float,
                        )
                        current[variable] = pd.to_numeric(
                            current[entity_key].map(entity_values),
                            errors="coerce",
                        ).fillna(0.0)
                    else:
                        current[variable] = matched_values
                if donor_block_spec.restore_frame is not None:
                    current = donor_block_spec.restore_frame(current)
                integrated_variables.extend(donor_block_spec.restored_variables)

        return {
            "seed_data": current,
            "integrated_variables": sorted(set(integrated_variables)),
        }

    def _select_donor_condition_vars(
        self,
        donor_frame: pd.DataFrame,
        shared_vars: list[str],
        donor_block: tuple[str, ...],
    ) -> list[str]:
        condition_vars = [
            variable for variable in shared_vars if variable in donor_frame.columns
        ]
        if len(condition_vars) <= 1:
            return condition_vars

        max_condition_vars = self.config.donor_imputer_max_condition_vars
        if (
            self.config.donor_imputer_condition_selection == "all_shared"
            or max_condition_vars is None
            or len(condition_vars) <= max_condition_vars
        ):
            return condition_vars

        scored_conditions = [
            (
                score_donor_condition_var(
                    donor_frame[variable],
                    [donor_frame[target] for target in donor_block if target in donor_frame.columns],
                    score_modes={
                        variable_semantic_spec_for(target).condition_score_mode
                        for target in donor_block
                    },
                ),
                variable,
            )
            for variable in condition_vars
        ]
        scored_conditions = [
            (score, variable)
            for score, variable in scored_conditions
            if score > 0.0
        ]
        if not scored_conditions:
            return condition_vars[:max_condition_vars]

        scored_conditions.sort(key=lambda item: (-item[0], item[1]))
        return [
            variable
            for _, variable in scored_conditions[:max_condition_vars]
        ]

    def _entity_key_column(self, entity: EntityType) -> str | None:
        return ENTITY_ID_COLUMNS.get(entity)

    def _ensure_seed_entity_ids(
        self,
        frame: pd.DataFrame,
        *,
        entities: set[EntityType],
    ) -> pd.DataFrame:
        missing_columns = [
            self._entity_key_column(entity)
            for entity in entities
            if entity is not EntityType.PERSON
            and self._entity_key_column(entity) not in frame.columns
        ]
        if not missing_columns:
            return frame
        working = frame.copy()
        original_person_ids = working["person_id"].copy()
        working["person_id"] = np.arange(len(working), dtype=np.int64)
        if "household_id" in working.columns:
            working["household_id"] = pd.factorize(working["household_id"])[0].astype(
                np.int64
            )
        persons = self.build_policyengine_entity_tables(working).persons.copy()
        persons["source_person_id"] = original_person_ids.to_numpy()
        mapping = persons[["source_person_id", *missing_columns]]
        if mapping["source_person_id"].duplicated().any():
            raise ValueError(
                "PolicyEngine entity table build produced duplicate person mappings"
            )
        return frame.merge(
            mapping,
            left_on="person_id",
            right_on="source_person_id",
            how="left",
        ).drop(columns=["source_person_id"])

    def _can_project_donor_block_to_entity(
        self,
        current_frame: pd.DataFrame,
        donor_frame: pd.DataFrame,
        entity: EntityType,
    ) -> bool:
        if entity is EntityType.PERSON:
            return False
        entity_key = self._entity_key_column(entity)
        return bool(
            entity_key
            and entity_key in current_frame.columns
            and entity_key in donor_frame.columns
            and current_frame[entity_key].notna().all()
            and donor_frame[entity_key].notna().all()
        )

    def _project_frame_to_entity(
        self,
        frame: pd.DataFrame,
        *,
        entity: EntityType,
        variables: set[str],
    ) -> pd.DataFrame:
        entity_key = self._entity_key_column(entity)
        if entity_key is None:
            raise ValueError(f"Unsupported donor projection entity: {entity}")
        columns = [
            entity_key,
            *[
                variable
                for variable in sorted(variables)
                if variable != entity_key and variable in frame.columns
            ],
        ]
        projected = frame[columns].copy()
        if entity is EntityType.PERSON:
            return projected

        sort_columns = [
            column
            for column in (entity_key, "household_id", "person_id")
            if column in projected.columns
        ]
        if sort_columns:
            projected = projected.sort_values(sort_columns, kind="mergesort")
        aggregations = {
            column: self._projection_aggregation_for(column)
            for column in projected.columns
            if column != entity_key
        }
        return projected.groupby(entity_key, as_index=False).agg(aggregations)

    def _projection_aggregation_for(self, column: str) -> str:
        if column in {"hh_weight", "household_id", "person_id", "year"}:
            return "first"
        return variable_semantic_spec_for(column).projection_aggregation.value

    def _should_integrate_donor_variable(
        self,
        current: pd.DataFrame,
        variable: str,
    ) -> bool:
        if variable not in current.columns:
            return True
        current_values = pd.to_numeric(
            current[variable],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        informative = current_values.dropna()
        if informative.empty:
            return True
        if (informative != 0).any():
            return False
        return informative.nunique() <= 1

    def _is_compatible_donor_condition(
        self,
        current_series: pd.Series,
        donor_series: pd.Series,
    ) -> bool:
        current_values = (
            pd.to_numeric(current_series, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        donor_values = (
            pd.to_numeric(donor_series, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if current_values.empty or donor_values.empty:
            return False
        if current_values.nunique() <= 1:
            return False
        if donor_values.nunique() <= 1:
            return False
        return True

    def _is_compatible_donor_target(self, series: pd.Series) -> bool:
        values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = values.dropna()
        if values.empty:
            return False
        if values.nunique() <= 1:
            return False
        return bool((values > 0).any())

    def _rank_match_donor_values(
        self,
        scores: pd.Series,
        *,
        donor_values: pd.Series,
        donor_weights: pd.Series | None,
        rng: np.random.Generator,
        strategy: DonorMatchStrategy = DonorMatchStrategy.RANK,
    ) -> pd.Series:
        """Assign donor values by rank, preserving the donor marginal distribution."""
        if donor_values.empty:
            return pd.Series(0.0, index=scores.index, dtype=float)

        donor_array = donor_values.to_numpy(dtype=float)
        donor_weight_array = None
        if donor_weights is not None and not donor_weights.empty:
            donor_weight_array = donor_weights.to_numpy(dtype=float)
            donor_weight_array = np.clip(donor_weight_array, a_min=0.0, a_max=None)

        if (
            strategy is DonorMatchStrategy.ZERO_INFLATED_POSITIVE
            or (
                strategy is DonorMatchStrategy.RANK
                and self._is_zero_inflated_positive_distribution(donor_array)
            )
        ):
            return self._rank_match_zero_inflated_positive_values(
                scores,
                donor_values=donor_array,
                donor_weights=donor_weight_array,
                rng=rng,
            )

        sampled_values = self._sample_donor_array(
            donor_array,
            size=len(scores),
            donor_weights=donor_weight_array,
            rng=rng,
        )

        sampled_values = np.sort(sampled_values.astype(float))
        order = np.argsort(scores.to_numpy(dtype=float), kind="mergesort")
        matched = np.empty(len(scores), dtype=float)
        matched[order] = sampled_values
        return pd.Series(matched, index=scores.index, dtype=float)

    def _rank_match_zero_inflated_positive_values(
        self,
        scores: pd.Series,
        *,
        donor_values: np.ndarray,
        donor_weights: np.ndarray | None,
        rng: np.random.Generator,
    ) -> pd.Series:
        matched = np.zeros(len(scores), dtype=float)
        positive_mask = donor_values > 0.0
        positive_values = donor_values[positive_mask]
        if len(positive_values) == 0:
            return pd.Series(matched, index=scores.index, dtype=float)

        positive_rate = self._weighted_positive_rate(
            donor_values,
            donor_weights=donor_weights,
        )
        n_positive = int(round(positive_rate * len(scores)))
        n_positive = min(max(n_positive, 0), len(scores))
        if n_positive == 0:
            return pd.Series(matched, index=scores.index, dtype=float)

        positive_weights = donor_weights[positive_mask] if donor_weights is not None else None
        sampled_positive = self._sample_donor_array(
            positive_values,
            size=n_positive,
            donor_weights=positive_weights,
            rng=rng,
        )
        sampled_positive = np.sort(sampled_positive.astype(float))
        order = np.argsort(scores.to_numpy(dtype=float), kind="mergesort")
        matched[order[-n_positive:]] = sampled_positive
        return pd.Series(matched, index=scores.index, dtype=float)

    def _sample_donor_array(
        self,
        donor_values: np.ndarray,
        *,
        size: int,
        donor_weights: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if len(donor_values) == size:
            return donor_values.copy()

        probabilities = None
        if donor_weights is not None and len(donor_weights) == len(donor_values):
            weight_sum = float(donor_weights.sum())
            if weight_sum > 0.0:
                probabilities = donor_weights / weight_sum
        return rng.choice(
            donor_values,
            size=size,
            replace=True,
            p=probabilities,
        )

    def _weighted_positive_rate(
        self,
        donor_values: np.ndarray,
        *,
        donor_weights: np.ndarray | None,
    ) -> float:
        positive_mask = donor_values > 0.0
        if donor_weights is None or len(donor_weights) != len(donor_values):
            return float(np.mean(positive_mask))
        weight_sum = float(donor_weights.sum())
        if weight_sum <= 0.0:
            return float(np.mean(positive_mask))
        return float(donor_weights[positive_mask].sum() / weight_sum)

    def _is_zero_inflated_positive_distribution(self, donor_values: np.ndarray) -> bool:
        return bool(
            len(donor_values) > 0
            and np.all(donor_values >= 0.0)
            and np.any(donor_values == 0.0)
            and np.any(donor_values > 0.0)
        )

    def _synthesize_bootstrap(
        self,
        seed_data: pd.DataFrame,
        initial_weight: float,
        *,
        strata_columns: tuple[str, ...] = (),
    ) -> pd.DataFrame:
        """Generate synthetic households via weighted bootstrap resampling."""
        rng = np.random.default_rng(self.config.random_seed)
        households = (
            seed_data.groupby("household_id", as_index=False)
            .agg(
                {
                    "hh_weight": "first",
                    **{
                        column: "first"
                        for column in strata_columns
                        if column in seed_data.columns
                    },
                }
            )
            .rename(columns={"hh_weight": "household_weight"})
        )
        sampled_households = self._sample_bootstrap_household_ids(
            households,
            rng=rng,
            strata_columns=strata_columns,
        )

        cloned_households: list[pd.DataFrame] = []
        for new_household_id, source_household_id in enumerate(sampled_households):
            household_persons = seed_data[
                seed_data["household_id"] == source_household_id
            ].copy()
            household_persons["household_id"] = new_household_id
            cloned_households.append(household_persons)

        synthetic = pd.concat(cloned_households, ignore_index=True)
        if "income" in synthetic.columns:
            synthetic["income"] = synthetic["income"].astype(float) * rng.lognormal(
                mean=0.0,
                sigma=0.05,
                size=len(synthetic),
            )
            synthetic["income"] = synthetic["income"].clip(lower=0.0)
        return self._finalize_synthetic_population(
            synthetic,
            initial_weight=initial_weight,
        )

    def _resolve_bootstrap_strata_columns(
        self,
        seed_data: pd.DataFrame,
    ) -> tuple[str, ...]:
        if self.config.bootstrap_strata_columns:
            missing_columns = [
                column
                for column in self.config.bootstrap_strata_columns
                if column not in seed_data.columns
            ]
            if missing_columns:
                raise ValueError(
                    "bootstrap_strata_columns are not available in seed data: "
                    f"{missing_columns}"
                )
            return self.config.bootstrap_strata_columns

        requested_geo_levels: set[str] = set()
        for scope in (False, True):
            _, _, geo_levels = self._policyengine_target_scope(for_calibration=scope)
            requested_geo_levels.update(geo_levels)

        inferred_columns: list[str] = []
        if (
            {"state", "district", "county"} & requested_geo_levels
            and "state_fips" in seed_data.columns
        ):
            inferred_columns.append("state_fips")
        if "county" in requested_geo_levels and "county_fips" in seed_data.columns:
            inferred_columns.append("county_fips")
        if (
            "district" in requested_geo_levels
            and "congressional_district_geoid" in seed_data.columns
        ):
            inferred_columns.append("congressional_district_geoid")
        return tuple(dict.fromkeys(inferred_columns))

    def _sample_bootstrap_household_ids(
        self,
        households: pd.DataFrame,
        *,
        rng: np.random.Generator,
        strata_columns: tuple[str, ...],
    ) -> np.ndarray:
        weights = households["household_weight"].astype(float).to_numpy()
        household_ids = households["household_id"].to_numpy()
        if (
            not strata_columns
            or self.config.n_synthetic <= 0
            or len(household_ids) == 0
        ):
            probabilities = weights / weights.sum()
            return rng.choice(
                household_ids,
                size=self.config.n_synthetic,
                replace=True,
                p=probabilities,
            )

        stratum_frame = households.loc[:, list(strata_columns)].copy()
        for column in stratum_frame.columns:
            values = stratum_frame[column]
            if pd.api.types.is_numeric_dtype(values):
                stratum_frame[column] = values.fillna(-1)
            else:
                stratum_frame[column] = values.astype("string").fillna("__missing__")
        stratum_keys = pd.MultiIndex.from_frame(stratum_frame)
        weighted_households = households.assign(_bootstrap_stratum_key=stratum_keys)
        stratum_weights = (
            weighted_households.groupby("_bootstrap_stratum_key", dropna=False)[
                "household_weight"
            ]
            .sum()
            .astype(float)
        )
        stratum_weights = stratum_weights[stratum_weights > 0]
        if stratum_weights.empty:
            probabilities = weights / weights.sum()
            return rng.choice(
                household_ids,
                size=self.config.n_synthetic,
                replace=True,
                p=probabilities,
            )

        n_strata = len(stratum_weights)
        base_counts = pd.Series(0, index=stratum_weights.index, dtype=int)
        remaining = self.config.n_synthetic
        if self.config.n_synthetic >= n_strata:
            base_counts += 1
            remaining -= n_strata

        probabilities = (stratum_weights / stratum_weights.sum()).to_numpy(dtype=float)
        extra_counts = (
            rng.multinomial(remaining, probabilities)
            if remaining > 0
            else np.zeros(n_strata, dtype=int)
        )

        sampled_households: list[np.ndarray] = []
        for stratum_key, sample_count in zip(
            stratum_weights.index,
            base_counts.to_numpy(dtype=int) + extra_counts,
            strict=False,
        ):
            if sample_count <= 0:
                continue
            candidates = weighted_households.loc[
                weighted_households["_bootstrap_stratum_key"] == stratum_key
            ]
            candidate_ids = candidates["household_id"].to_numpy()
            candidate_weights = candidates["household_weight"].astype(float).to_numpy()
            if candidate_weights.sum() <= 0:
                candidate_probabilities = np.full(
                    len(candidate_ids),
                    1.0 / max(len(candidate_ids), 1),
                )
            else:
                candidate_probabilities = candidate_weights / candidate_weights.sum()
            sampled_households.append(
                rng.choice(
                    candidate_ids,
                    size=int(sample_count),
                    replace=True,
                    p=candidate_probabilities,
                )
            )

        if not sampled_households:
            probabilities = weights / weights.sum()
            return rng.choice(
                household_ids,
                size=self.config.n_synthetic,
                replace=True,
                p=probabilities,
            )

        return rng.permutation(np.concatenate(sampled_households))

    def _finalize_synthetic_population(
        self,
        synthetic: pd.DataFrame,
        initial_weight: float,
    ) -> pd.DataFrame:
        """Add derived fields and canonical identifiers to synthetic output."""
        result = synthetic.copy().reset_index(drop=True)
        for column, default in {
            "state_fips": 0,
            "tenure": 0,
            "age": 0,
            "sex": 0,
            "education": 0,
            "employment_status": 0,
            "income": 0.0,
        }.items():
            if column not in result.columns:
                result[column] = default
        result["person_id"] = np.arange(len(result))
        if "household_id" in result.columns:
            result["household_id"] = pd.factorize(result["household_id"])[0].astype(
                np.int64
            )
        else:
            result["household_id"] = np.arange(len(result), dtype=np.int64)
        result["state"] = result["state_fips"].map(STATE_FIPS).fillna("UNK")
        result["age_group"] = pd.cut(
            result["age"],
            bins=AGE_BINS,
            labels=AGE_LABELS,
            right=False,
        ).astype(str)
        result["income_bracket"] = pd.cut(
            result["income"],
            bins=INCOME_BINS,
            labels=INCOME_LABELS,
        ).astype(str)
        if "weight" not in result.columns:
            result["weight"] = float(initial_weight)
        else:
            result["weight"] = (
                pd.to_numeric(result["weight"], errors="coerce")
                .fillna(float(initial_weight))
                .astype(float)
            )
        return result

    def _build_policyengine_households(self, persons: pd.DataFrame) -> pd.DataFrame:
        household_columns = [
            column
            for column in ("state_fips", "tenure", "state")
            if column in persons.columns
        ]
        aggregations = {column: "first" for column in household_columns}
        aggregations["weight"] = "mean"
        households = (
            persons.groupby("household_id", as_index=False)
            .agg(aggregations)
            .rename(columns={"weight": "household_weight"})
        )
        return households

    def _build_policyengine_tax_units(
        self,
        persons: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        optimizer = TaxUnitOptimizer()
        person_rows = persons.copy()
        tax_unit_rows: list[dict[str, Any]] = []
        person_to_tax_unit: dict[int, int] = {}
        next_tax_unit_id = 0

        for household_id in person_rows["household_id"].drop_duplicates().tolist():
            hh_persons = person_rows[person_rows["household_id"] == household_id].copy()
            if hh_persons.empty:
                continue
            optimized_units = optimizer.optimize_household(int(household_id), hh_persons)
            if not optimized_units:
                optimized_units = [
                    {
                        "tax_unit_id": 0,
                        "household_id": int(household_id),
                        "filing_status": "single",
                        "filer_ids": [int(hh_persons.iloc[0]["person_id"])],
                        "dependent_ids": [],
                        "n_dependents": 0,
                        "total_income": float(hh_persons["income"].sum()),
                        "tax_liability": 0.0,
                    }
                ]

            assigned_person_ids: set[int] = set()
            for unit in optimized_units:
                unit_person_ids = [
                    int(person_id)
                    for person_id in list(unit.get("filer_ids", []))
                    + list(unit.get("dependent_ids", []))
                ]
                if not unit_person_ids:
                    continue
                global_tax_unit_id = next_tax_unit_id
                next_tax_unit_id += 1
                for person_id in unit_person_ids:
                    person_to_tax_unit[person_id] = global_tax_unit_id
                    assigned_person_ids.add(person_id)
                tax_unit_rows.append(
                    {
                        "tax_unit_id": global_tax_unit_id,
                        "household_id": int(household_id),
                        "filing_status": self._normalize_policyengine_filing_status(
                            unit.get("filing_status", "single")
                        ),
                        "n_dependents": int(unit.get("n_dependents", 0)),
                        "total_income": float(unit.get("total_income", 0.0)),
                        "tax_liability": float(unit.get("tax_liability", 0.0)),
                    }
                )

            unassigned = [
                int(person_id)
                for person_id in hh_persons["person_id"].tolist()
                if int(person_id) not in assigned_person_ids
            ]
            for person_id in unassigned:
                global_tax_unit_id = next_tax_unit_id
                next_tax_unit_id += 1
                person_to_tax_unit[person_id] = global_tax_unit_id
                tax_unit_rows.append(
                    {
                        "tax_unit_id": global_tax_unit_id,
                        "household_id": int(household_id),
                        "filing_status": "SINGLE",
                        "n_dependents": 0,
                        "total_income": float(
                            hh_persons.loc[
                                hh_persons["person_id"] == person_id, "income"
                            ].iloc[0]
                        ),
                        "tax_liability": 0.0,
                    }
                )

        person_rows["tax_unit_id"] = person_rows["person_id"].map(person_to_tax_unit)
        tax_units = pd.DataFrame(tax_unit_rows)
        return tax_units, person_rows

    def _assign_family_and_spm_units(self, persons: pd.DataFrame) -> pd.DataFrame:
        result = persons.copy()
        family_ids: dict[int, int] = {}
        spm_unit_ids: dict[int, int] = {}
        next_family_id = 0
        next_spm_unit_id = 0

        for _, household_persons in result.groupby("household_id", sort=False):
            primary_mask = household_persons["relationship_to_head"].isin({0, 1, 2})
            if primary_mask.any():
                primary_family_id = next_family_id
                primary_spm_id = next_spm_unit_id
                next_family_id += 1
                next_spm_unit_id += 1
            else:
                primary_family_id = None
                primary_spm_id = None

            for _, row in household_persons.iterrows():
                if primary_family_id is not None and row["relationship_to_head"] in {
                    0,
                    1,
                    2,
                }:
                    family_ids[int(row.name)] = primary_family_id
                    spm_unit_ids[int(row.name)] = primary_spm_id
                    continue

                family_ids[int(row.name)] = next_family_id
                spm_unit_ids[int(row.name)] = next_spm_unit_id
                next_family_id += 1
                next_spm_unit_id += 1

        result["family_id"] = result.index.map(family_ids).astype(np.int64)
        result["spm_unit_id"] = result.index.map(spm_unit_ids).astype(np.int64)
        return result

    def _assign_marital_units(
        self,
        persons: pd.DataFrame,
    ) -> pd.DataFrame:
        result = persons.copy()
        marital_unit_by_person: dict[int, int] = {}
        next_marital_unit_id = 0

        for tax_unit_id, unit_persons in result.groupby("tax_unit_id", sort=False):
            _ = tax_unit_id
            filers = unit_persons[unit_persons["relationship_to_head"].isin({0, 1})]
            if len(filers) >= 2:
                marital_unit_id = next_marital_unit_id
                next_marital_unit_id += 1
                for person_id in filers.head(2)["person_id"].tolist():
                    marital_unit_by_person[int(person_id)] = marital_unit_id
            elif len(filers) == 1:
                marital_unit_by_person[int(filers.iloc[0]["person_id"])] = (
                    next_marital_unit_id
                )
                next_marital_unit_id += 1

            for person_id in unit_persons["person_id"].tolist():
                if int(person_id) in marital_unit_by_person:
                    continue
                marital_unit_by_person[int(person_id)] = next_marital_unit_id
                next_marital_unit_id += 1

        result["marital_unit_id"] = result["person_id"].map(marital_unit_by_person).astype(
            np.int64
        )
        return result

    def _collapse_group_table(
        self,
        persons: pd.DataFrame,
        id_column: str,
    ) -> pd.DataFrame:
        return (
            persons.groupby(id_column, as_index=False)
            .agg({"household_id": "first"})
            .astype({id_column: np.int64, "household_id": np.int64})
        )

    def _normalize_relationship_to_head(self, persons: pd.DataFrame) -> pd.Series:
        family_normalized: pd.Series | None = None
        if "family_relationship" in persons.columns:
            family_relationship = (
                pd.to_numeric(persons["family_relationship"], errors="coerce")
                .fillna(-1)
                .astype(int)
            )
            unique_values = set(family_relationship.unique().tolist())
            if unique_values.issubset({0, 1, 2, 3, 4}):
                family_normalized = pd.Series(3, index=persons.index, dtype=int)
                household_groups = (
                    persons.groupby("household_id", sort=False).groups.values()
                    if "household_id" in persons.columns
                    else [persons.index]
                )
                for member_index in household_groups:
                    member_index = list(member_index)
                    household_codes = set(family_relationship.loc[member_index].tolist())
                    if 0 in household_codes:
                        # Some sources already use the optimizer's 0-based coding.
                        mapped = family_relationship.loc[member_index].map(
                            {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
                        )
                    else:
                        # CPS A_FAMREL is 1-based: 1=head, 2=spouse, 3=child, 4=other.
                        mapped = family_relationship.loc[member_index].map(
                            {1: 0, 2: 1, 3: 2, 4: 3}
                        )
                    family_normalized.loc[member_index] = mapped.fillna(3).astype(int)

        if "relationship_to_head" not in persons.columns:
            if family_normalized is not None:
                return self._repair_relationship_to_head(persons, family_normalized)
            if "is_spouse" in persons.columns or "is_dependent" in persons.columns:
                order = persons.groupby("household_id").cumcount()
                normalized = pd.Series(3, index=persons.index, dtype=int)
                normalized.loc[order == 0] = 0
                if "is_spouse" in persons.columns:
                    spouse_mask = (
                        pd.to_numeric(persons["is_spouse"], errors="coerce")
                        .fillna(0)
                        .astype(int)
                        > 0
                    )
                    normalized.loc[spouse_mask] = 1
                if "is_dependent" in persons.columns:
                    dependent_mask = (
                        pd.to_numeric(persons["is_dependent"], errors="coerce")
                        .fillna(0)
                        .astype(int)
                        > 0
                    )
                    normalized.loc[dependent_mask & ~normalized.eq(1)] = 2
                return self._repair_relationship_to_head(persons, normalized)
            order = persons.groupby("household_id").cumcount()
            normalized = order.map(lambda idx: 0 if idx == 0 else 3).astype(int)
            return self._repair_relationship_to_head(persons, normalized)

        relationship = (
            pd.to_numeric(persons["relationship_to_head"], errors="coerce")
            .fillna(-1)
            .astype(int)
        )
        unique_values = set(relationship.unique().tolist())
        if unique_values.issubset({0, 1, 2, 3}):
            if family_normalized is not None:
                relationship_detail = set(relationship.unique().tolist()) & {1, 2}
                family_detail = set(family_normalized.unique().tolist()) & {1, 2}
                if len(family_detail) > len(relationship_detail):
                    return self._repair_relationship_to_head(persons, family_normalized)
            return self._repair_relationship_to_head(persons, relationship)

        if unique_values.issubset({1, 2, 3, 4}):
            normalized = relationship.map({1: 0, 2: 1, 3: 3, 4: 2}).fillna(3).astype(int)
            return self._repair_relationship_to_head(persons, normalized)

        order = persons.groupby("household_id").cumcount()
        normalized = pd.Series(3, index=persons.index, dtype=int)
        normalized.loc[order == 0] = 0
        normalized.loc[(order == 1) & (persons["age"] >= 18)] = 1
        normalized.loc[persons["age"] < 18] = 2
        return self._repair_relationship_to_head(persons, normalized)

    def _repair_relationship_to_head(
        self,
        persons: pd.DataFrame,
        relationship: pd.Series,
    ) -> pd.Series:
        """Repair household relationship patterns so tax-unit construction has one clear head."""
        normalized = relationship.astype(int).copy()
        if "household_id" not in persons.columns:
            return normalized

        ages = pd.to_numeric(persons.get("age", 0), errors="coerce").fillna(0.0)
        grouped = persons.groupby("household_id", sort=False).groups
        for member_index in grouped.values():
            member_index = list(member_index)
            household_relationship = normalized.loc[member_index].copy()
            household_ages = ages.loc[member_index]

            head_index = household_relationship[household_relationship.eq(0)].index.tolist()
            if not head_index:
                spouse_candidates = [
                    index
                    for index in household_relationship[household_relationship.eq(1)].index.tolist()
                    if household_ages.loc[index] >= 18
                ]
                adult_candidates = [
                    index
                    for index in household_relationship.index.tolist()
                    if household_ages.loc[index] >= 18
                ]
                candidate_pool = spouse_candidates or adult_candidates or household_relationship.index.tolist()
                head_choice = max(candidate_pool, key=lambda index: household_ages.loc[index])
                normalized.loc[head_choice] = 0
                head_index = [head_choice]
            elif len(head_index) > 1:
                keep_head = max(head_index, key=lambda index: household_ages.loc[index])
                for index in head_index:
                    if index == keep_head:
                        continue
                    normalized.loc[index] = 3 if household_ages.loc[index] >= 19 else 2
                head_index = [keep_head]

            spouse_index = normalized.loc[member_index][normalized.loc[member_index].eq(1)].index.tolist()
            if len(spouse_index) > 1:
                keep_spouse = max(spouse_index, key=lambda index: household_ages.loc[index])
                for index in spouse_index:
                    if index == keep_spouse:
                        continue
                    normalized.loc[index] = 3 if household_ages.loc[index] >= 19 else 2

        return normalized.astype(int)

    def _infer_policyengine_variable_bindings(
        self,
        tables: PolicyEngineUSEntityTableBundle,
    ) -> dict[str, PolicyEngineUSVariableBinding]:
        return infer_policyengine_us_variable_bindings(tables)

    def _filter_supported_policyengine_targets(
        self,
        targets: list[TargetSpec],
        tables: PolicyEngineUSEntityTableBundle,
        bindings: dict[str, PolicyEngineUSVariableBinding],
    ) -> list[TargetSpec]:
        return filter_supported_policyengine_us_targets(targets, tables, bindings)

    def _policyengine_variables_to_materialize(
        self,
        targets: list[TargetSpec],
        bindings: dict[str, PolicyEngineUSVariableBinding],
    ) -> set[str]:
        return policyengine_us_variables_to_materialize(targets, bindings)

    def _has_policyengine_entity_table(
        self,
        entity: EntityType,
        tables: PolicyEngineUSEntityTableBundle,
    ) -> bool:
        entity_tables = {
            EntityType.HOUSEHOLD: tables.households,
            EntityType.PERSON: tables.persons,
            EntityType.TAX_UNIT: tables.tax_units,
            EntityType.SPM_UNIT: tables.spm_units,
            EntityType.FAMILY: tables.families,
        }
        table = entity_tables.get(entity)
        return table is not None

    def _normalize_policyengine_filing_status(self, value: Any) -> str:
        normalized = str(value).strip().lower()
        mapping = {
            "single": "SINGLE",
            "married_joint": "JOINT",
            "married_filing_jointly": "JOINT",
            "joint": "JOINT",
            "married_filing_separately": "SEPARATE",
            "separate": "SEPARATE",
            "head_of_household": "HEAD_OF_HOUSEHOLD",
            "qualifying_widow": "SURVIVING_SPOUSE",
            "surviving_spouse": "SURVIVING_SPOUSE",
        }
        return mapping.get(normalized, "SINGLE")

    def _augment_policyengine_person_inputs(
        self,
        persons: pd.DataFrame,
    ) -> pd.DataFrame:
        result = normalize_dividend_columns(persons)
        zero = pd.Series(0.0, index=result.index, dtype=float)

        def first_present(*columns: str) -> pd.Series:
            for column in columns:
                if column in result.columns:
                    return pd.to_numeric(
                        result[column],
                        errors="coerce",
                    ).fillna(0.0).astype(float)
            return zero.copy()

        def has_any(*columns: str) -> bool:
            return any(column in result.columns for column in columns)

        if "is_female" in result.columns:
            result["is_female"] = result["is_female"].fillna(False).astype(bool)
        elif "sex" in result.columns:
            sex = pd.to_numeric(result["sex"], errors="coerce").fillna(0).astype(int)
            result["is_female"] = sex.eq(2)

        if "cps_race" in result.columns:
            result["cps_race"] = (
                pd.to_numeric(result["cps_race"], errors="coerce").fillna(0).astype(int)
            )
        elif "race" in result.columns:
            result["cps_race"] = (
                pd.to_numeric(result["race"], errors="coerce").fillna(0).astype(int)
            )

        if "is_hispanic" in result.columns:
            result["is_hispanic"] = result["is_hispanic"].fillna(False).astype(bool)
        elif "hispanic" in result.columns:
            hispanic = pd.to_numeric(result["hispanic"], errors="coerce")
            observed_codes = set(hispanic.dropna().astype(int).unique().tolist())
            if observed_codes and observed_codes <= {1, 2}:
                result["is_hispanic"] = hispanic.fillna(0).astype(int).eq(1)
            else:
                result["is_hispanic"] = hispanic.fillna(0).astype(int).ne(0)

        if "medicaid" in result.columns:
            result["medicaid"] = (
                pd.to_numeric(result["medicaid"], errors="coerce").fillna(0.0).astype(float)
            )
        if "medicaid_enrolled" in result.columns:
            result["medicaid_enrolled"] = (
                result["medicaid_enrolled"].fillna(False).astype(bool)
            )

        known_nonemployment = (
            first_present("self_employment_income")
            + first_present("taxable_interest_income", "interest_income")
            + first_present("ordinary_dividend_income", "dividend_income")
            + first_present("rental_income")
            + first_present("gross_social_security", "social_security")
            + first_present("ssi")
            + first_present("public_assistance")
            + first_present("taxable_pension_income", "pension_income")
            + first_present("unemployment_compensation")
        )
        fallback_employment_income = (
            pd.to_numeric(result.get("income", zero), errors="coerce")
            .fillna(0.0)
            .astype(float)
            - known_nonemployment
        ).clip(lower=0.0)

        result["employment_income_before_lsr"] = (
            first_present("employment_income_before_lsr", "employment_income", "wage_income")
            if has_any("employment_income_before_lsr", "employment_income", "wage_income")
            else fallback_employment_income
        )
        result["self_employment_income_before_lsr"] = first_present(
            "self_employment_income_before_lsr",
            "self_employment_income",
        )
        result["taxable_interest_income"] = first_present(
            "taxable_interest_income",
            "interest_income",
        )
        result["tax_exempt_interest_income"] = first_present("tax_exempt_interest_income")
        result["qualified_dividend_income"] = first_present(
            "qualified_dividend_income",
        ).clip(lower=0.0)
        result["non_qualified_dividend_income"] = first_present(
            "non_qualified_dividend_income",
        ).clip(lower=0.0)
        result["ordinary_dividend_income"] = first_present(
            "ordinary_dividend_income",
            "dividend_income",
        ).clip(lower=0.0)
        if has_any("qualified_dividend_income", "non_qualified_dividend_income"):
            dividend_total = (
                result["qualified_dividend_income"]
                + result["non_qualified_dividend_income"]
            ).clip(lower=0.0)
            result["ordinary_dividend_income"] = dividend_total
            result["dividend_income"] = dividend_total
        else:
            result = normalize_dividend_columns(result)

        result["short_term_capital_gains"] = first_present("short_term_capital_gains")
        result["long_term_capital_gains_before_response"] = (
            first_present(
                "long_term_capital_gains_before_response",
                "long_term_capital_gains",
            )
            if has_any(
                "long_term_capital_gains_before_response",
                "long_term_capital_gains",
            )
            else first_present("capital_gains")
        )
        result["partnership_s_corp_income"] = first_present("partnership_s_corp_income")
        result["farm_income"] = first_present("farm_income")
        result["rental_income"] = first_present("rental_income")
        result["taxable_private_pension_income"] = first_present(
            "taxable_private_pension_income",
            "taxable_pension_income",
            "pension_income",
        )
        result["taxable_public_pension_income"] = first_present("taxable_public_pension_income")
        result["tax_exempt_private_pension_income"] = first_present(
            "tax_exempt_private_pension_income"
        )
        result["tax_exempt_public_pension_income"] = first_present(
            "tax_exempt_public_pension_income"
        )
        result["social_security_retirement"] = first_present(
            "social_security_retirement",
            "social_security",
            "gross_social_security",
        )
        result["social_security_disability"] = first_present("social_security_disability")
        result["social_security_survivors"] = first_present("social_security_survivors")
        result["social_security_dependents"] = first_present("social_security_dependents")
        result["unemployment_compensation"] = first_present("unemployment_compensation")
        result["state_income_tax_reported"] = first_present(
            "state_income_tax_reported",
            "state_income_tax_paid",
        )
        result["student_loan_interest"] = first_present("student_loan_interest")
        return result

    def _resolve_policyengine_tax_benefit_system(self) -> Any:
        simulation_cls = self.config.policyengine_simulation_cls
        if simulation_cls is None:
            import policyengine_us

            return getattr(policyengine_us.system, "system", policyengine_us.system)

        tax_benefit_system = getattr(simulation_cls, "tax_benefit_system", None)
        if tax_benefit_system is None:
            tax_benefit_system = getattr(simulation_cls, "system", None)
        if tax_benefit_system is not None:
            return getattr(tax_benefit_system, "system", tax_benefit_system)
        raise ValueError(
            "policyengine_simulation_cls must expose a tax_benefit_system or system attribute"
        )


def build_us_microplex(
    persons: pd.DataFrame,
    households: pd.DataFrame,
    config: USMicroplexBuildConfig | None = None,
) -> USMicroplexBuildResult:
    """Convenience wrapper for the US microplex pipeline."""
    pipeline = USMicroplexPipeline(config)
    return pipeline.build(persons, households)
