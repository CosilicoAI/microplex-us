"""
Data sources for microplex.

This module provides loaders for various microdata sources:
- CPS ASEC (Census Bureau's primary income/poverty survey)
- PSID (Panel Study of Income Dynamics - longitudinal household survey)
- PUF (Public Use File - tax return data)
- CPS to Cosilico variable mappings with legal references
- Data transformation utilities
"""

from microplex_us.data_sources.cps import (
    HOUSEHOLD_VARIABLES,
    PERSON_VARIABLES,
    CPSASECParquetSourceProvider,
    CPSASECSourceProvider,
    CPSDataset,
    download_cps_asec,
    get_available_years,
)
from microplex_us.data_sources.cps import (
    load_cps_asec as load_cps_asec_polars,
)
from microplex_us.data_sources.cps_mappings import (
    CoverageGap,
    CoverageLevel,
    VariableMapping,
    coverage_summary,
    get_all_mappings,
    get_mapping_metadata,
    map_age,
    map_agi_proxy,
    map_ctc_qualifying_children,
    map_earned_income,
    map_filing_status,
    map_household_size,
    map_is_blind,
    map_is_dependent,
)
from microplex_us.data_sources.cps_transform import (
    TransformedDataset,
    transform_cps_to_cosilico,
)
from microplex_us.data_sources.donor_surveys import (
    ACSSourceProvider,
    SCFSourceProvider,
    SIPPAssetsSourceProvider,
    SIPPTipsSourceProvider,
)
from microplex_us.data_sources.family_imputation_benchmark import (
    DecomposableFamilyBenchmarkSpec,
    FamilyImputationBenchmarkResult,
    FamilyImputationMethodBenchmark,
    benchmark_decomposable_family_imputers,
    reconcile_component_predictions_to_total,
)
from microplex_us.data_sources.psid import (
    PSID_TO_MICROPLEX_VARS,
    PSIDDataset,
    PSIDSourceProvider,
    calibrate_divorce_rates,
    calibrate_marriage_rates,
    create_psid_fusion_source,
    extract_transition_rates,
    get_age_specific_rates,
    load_psid_panel,
)
from microplex_us.data_sources.puf import (
    PUF_EXCLUSIVE_VARS,
    PUF_VARIABLE_MAP,
    SHARED_VARS,
    UPRATING_FACTORS,
    PUFSourceProvider,
    download_puf,
    expand_to_persons,
    load_puf,
    map_puf_variables,
    uprate_puf,
)

__all__ = [
    # CPS loading
    "CPSDataset",
    "CPSASECSourceProvider",
    "CPSASECParquetSourceProvider",
    "download_cps_asec",
    "load_cps_asec_polars",
    "get_available_years",
    "PERSON_VARIABLES",
    "HOUSEHOLD_VARIABLES",
    # Mappings
    "CoverageLevel",
    "CoverageGap",
    "VariableMapping",
    "map_age",
    "map_earned_income",
    "map_filing_status",
    "map_is_blind",
    "map_is_dependent",
    "map_ctc_qualifying_children",
    "map_agi_proxy",
    "map_household_size",
    "get_mapping_metadata",
    "get_all_mappings",
    "coverage_summary",
    # Transform
    "TransformedDataset",
    "transform_cps_to_cosilico",
    # PE donor surveys
    "ACSSourceProvider",
    "SIPPTipsSourceProvider",
    "SIPPAssetsSourceProvider",
    "SCFSourceProvider",
    # Family imputation benchmarks
    "DecomposableFamilyBenchmarkSpec",
    "FamilyImputationMethodBenchmark",
    "FamilyImputationBenchmarkResult",
    "benchmark_decomposable_family_imputers",
    "reconcile_component_predictions_to_total",
    # PUF loading
    "load_puf",
    "PUFSourceProvider",
    "download_puf",
    "map_puf_variables",
    "uprate_puf",
    "expand_to_persons",
    "PUF_VARIABLE_MAP",
    "UPRATING_FACTORS",
    "PUF_EXCLUSIVE_VARS",
    "SHARED_VARS",
    # PSID loading
    "PSIDDataset",
    "PSIDSourceProvider",
    "load_psid_panel",
    "extract_transition_rates",
    "get_age_specific_rates",
    "calibrate_marriage_rates",
    "calibrate_divorce_rates",
    "create_psid_fusion_source",
    "PSID_TO_MICROPLEX_VARS",
]
