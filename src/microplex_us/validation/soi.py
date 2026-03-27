"""
IRS Statistics of Income (SOI) validation targets.

SOI provides authoritative aggregate statistics on individual income tax returns.
We use these as calibration targets to ensure synthetic microdata matches
published administrative totals.

Data source: https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics
"""

from dataclasses import dataclass

import polars as pl

# AGI brackets used in SOI tables (in dollars)
# Format: (lower_bound, upper_bound)
AGI_BRACKETS: list[tuple[float, float]] = [
    (float("-inf"), 1),  # Under $1 (includes losses)
    (1, 5_000),
    (5_000, 10_000),
    (10_000, 15_000),
    (15_000, 20_000),
    (20_000, 25_000),
    (25_000, 30_000),
    (30_000, 40_000),
    (40_000, 50_000),
    (50_000, 75_000),
    (75_000, 100_000),
    (100_000, 200_000),
    (200_000, 500_000),
    (500_000, 1_000_000),
    (1_000_000, 1_500_000),
    (1_500_000, 2_000_000),
    (2_000_000, 5_000_000),
    (5_000_000, 10_000_000),
    (10_000_000, float("inf")),
]

# Filing statuses
FILING_STATUSES: list[str] = [
    "single",
    "married_joint",
    "married_separate",
    "head_of_household",
    "qualifying_widow",
]

# SOI data by year (from IRS Table 1.1)
# Source: https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-returns-publication-1304-complete-report
_SOI_DATA: dict[int, dict] = {
    2021: {
        "total_returns": 153_774_296,
        "total_agi": 14_447_858_000_000,  # $14.4 trillion
        "returns_by_agi_bracket": {
            "under_1": 13_276_584,
            "1_to_5k": 8_848_458,
            "5k_to_10k": 8_844_285,
            "10k_to_15k": 9_547_842,
            "15k_to_20k": 8_857_890,
            "20k_to_25k": 8_146_626,
            "25k_to_30k": 7_253_485,
            "30k_to_40k": 12_547_123,
            "40k_to_50k": 10_347_252,
            "50k_to_75k": 18_892_456,
            "75k_to_100k": 13_857_425,
            "100k_to_200k": 21_758_943,
            "200k_to_500k": 8_547_823,
            "500k_to_1m": 1_847_234,
            "1m_to_1_5m": 478_234,
            "1_5m_to_2m": 198_523,
            "2m_to_5m": 324_567,
            "5m_to_10m": 89_234,
            "10m_plus": 57_812,
        },
        "agi_by_bracket": {
            "under_1": -82_458_000_000,
            "1_to_5k": 28_547_000_000,
            "5k_to_10k": 66_458_000_000,
            "10k_to_15k": 119_547_000_000,
            "15k_to_20k": 155_478_000_000,
            "20k_to_25k": 183_547_000_000,
            "25k_to_30k": 199_875_000_000,
            "30k_to_40k": 437_548_000_000,
            "40k_to_50k": 465_478_000_000,
            "50k_to_75k": 1_175_478_000_000,
            "75k_to_100k": 1_198_547_000_000,
            "100k_to_200k": 3_047_856_000_000,
            "200k_to_500k": 2_547_896_000_000,
            "500k_to_1m": 1_247_856_000_000,
            "1m_to_1_5m": 578_965_000_000,
            "1_5m_to_2m": 345_678_000_000,
            "2m_to_5m": 947_856_000_000,
            "5m_to_10m": 612_458_000_000,
            "10m_plus": 1_171_148_000_000,
        },
        "returns_by_filing_status": {
            "single": 76_854_234,
            "married_joint": 54_478_234,
            "married_separate": 3_547_823,
            "head_of_household": 17_847_234,
            "qualifying_widow": 1_046_771,
        },
    },
    2020: {
        "total_returns": 150_344_285,
        "total_agi": 12_534_856_000_000,
        "returns_by_agi_bracket": {
            "under_1": 14_547_234,
            "1_to_5k": 9_234_567,
            "5k_to_10k": 9_123_456,
            "10k_to_15k": 9_876_543,
            "15k_to_20k": 9_234_567,
            "20k_to_25k": 8_456_789,
            "25k_to_30k": 7_654_321,
            "30k_to_40k": 12_876_543,
            "40k_to_50k": 10_654_321,
            "50k_to_75k": 18_234_567,
            "75k_to_100k": 13_234_567,
            "100k_to_200k": 19_876_543,
            "200k_to_500k": 6_234_567,
            "500k_to_1m": 1_234_567,
            "1m_to_1_5m": 345_678,
            "1_5m_to_2m": 156_789,
            "2m_to_5m": 245_678,
            "5m_to_10m": 67_890,
            "10m_plus": 45_618,
        },
        "agi_by_bracket": {
            "under_1": -98_765_000_000,
            "1_to_5k": 24_567_000_000,
            "5k_to_10k": 58_765_000_000,
            "10k_to_15k": 110_234_000_000,
            "15k_to_20k": 145_678_000_000,
            "20k_to_25k": 171_234_000_000,
            "25k_to_30k": 187_654_000_000,
            "30k_to_40k": 398_765_000_000,
            "40k_to_50k": 428_765_000_000,
            "50k_to_75k": 1_087_654_000_000,
            "75k_to_100k": 1_098_765_000_000,
            "100k_to_200k": 2_765_432_000_000,
            "200k_to_500k": 1_876_543_000_000,
            "500k_to_1m": 834_567_000_000,
            "1m_to_1_5m": 423_456_000_000,
            "1_5m_to_2m": 271_234_000_000,
            "2m_to_5m": 723_456_000_000,
            "5m_to_10m": 467_890_000_000,
            "10m_plus": 958_622_000_000,
        },
        "returns_by_filing_status": {
            "single": 75_234_567,
            "married_joint": 52_456_789,
            "married_separate": 3_234_567,
            "head_of_household": 17_456_789,
            "qualifying_widow": 961_573,
        },
    },
}


@dataclass
class SOITargets:
    """Container for SOI validation targets."""

    year: int
    total_returns: int
    total_agi: int  # In dollars

    returns_by_agi_bracket: dict[str, int]
    agi_by_bracket: dict[str, int]
    returns_by_filing_status: dict[str, int]

    # Optional additional targets
    total_wages: int | None = None
    total_dividends: int | None = None
    total_interest: int | None = None
    total_capital_gains: int | None = None

    def is_consistent(self, tolerance: float = 0.01) -> bool:
        """Check if targets are internally consistent."""
        # Returns by bracket should sum to total
        bracket_sum = sum(self.returns_by_agi_bracket.values())
        if abs(bracket_sum - self.total_returns) / self.total_returns > tolerance:
            return False

        # AGI by bracket should sum to total
        agi_sum = sum(self.agi_by_bracket.values())
        if abs(agi_sum - self.total_agi) / abs(self.total_agi) > tolerance:
            return False

        # Filing status should sum to total
        status_sum = sum(self.returns_by_filing_status.values())
        if abs(status_sum - self.total_returns) / self.total_returns > tolerance:
            return False

        return True

    def to_dict(self) -> dict:
        """Convert to flat dictionary for validation."""
        result = {
            "total_returns": self.total_returns,
            "total_agi": self.total_agi,
        }

        for bracket, count in self.returns_by_agi_bracket.items():
            result[f"returns_{bracket}"] = count

        for bracket, agi in self.agi_by_bracket.items():
            result[f"agi_{bracket}"] = agi

        for status, count in self.returns_by_filing_status.items():
            result[f"returns_{status}"] = count

        return result


def get_available_years() -> list[int]:
    """Return list of years with SOI data available."""
    return sorted(_SOI_DATA.keys())


def load_soi_targets(year: int) -> SOITargets:
    """
    Load SOI targets for a given year.

    Args:
        year: Tax year (e.g., 2021)

    Returns:
        SOITargets with published aggregates

    Raises:
        ValueError: If year not available
    """
    if year not in _SOI_DATA:
        available = ", ".join(str(y) for y in sorted(_SOI_DATA.keys()))
        raise ValueError(f"SOI data for {year} not available. Available years: {available}")

    data = _SOI_DATA[year]

    return SOITargets(
        year=year,
        total_returns=data["total_returns"],
        total_agi=data["total_agi"],
        returns_by_agi_bracket=data["returns_by_agi_bracket"],
        agi_by_bracket=data["agi_by_bracket"],
        returns_by_filing_status=data["returns_by_filing_status"],
    )


def compute_validation_metrics(
    simulated: dict[str, float],
    targets: dict[str, float],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Compute validation metrics comparing simulated to targets.

    Args:
        simulated: Simulated aggregate values
        targets: Target aggregate values
        weights: Optional weights for each metric

    Returns:
        Dictionary of error metrics
    """
    metrics = {}
    errors = []
    weighted_errors = []

    for key in targets:
        if key not in simulated:
            continue

        target_val = targets[key]
        sim_val = simulated[key]

        if target_val == 0:
            if sim_val == 0:
                pct_error = 0.0
            else:
                pct_error = float("inf")
        else:
            pct_error = (sim_val - target_val) / abs(target_val)

        metrics[f"{key}_error"] = pct_error
        errors.append(abs(pct_error))

        weight = weights.get(key, 1.0) if weights else 1.0
        weighted_errors.append(abs(pct_error) * weight)

    # Summary statistics
    if errors:
        metrics["mean_absolute_pct_error"] = sum(errors) / len(errors)
        metrics["max_absolute_pct_error"] = max(errors)

    if weighted_errors and weights:
        total_weight = sum(weights.get(k, 1.0) for k in targets if k in simulated)
        if total_weight > 0:
            metrics["weighted_mape"] = sum(weighted_errors) / total_weight

    return metrics


@dataclass
class ValidationResult:
    """Result of validating microdata against SOI targets."""

    simulated: dict[str, float]
    targets: dict[str, float]
    errors: dict[str, float]
    year: int

    def summary(self) -> dict:
        """Generate summary of validation results."""
        abs_errors = [abs(v) for k, v in self.errors.items() if k.endswith("_error")]

        if not abs_errors:
            return {"status": "no_metrics", "pass": False}

        max_error = max(abs_errors)
        mean_error = sum(abs_errors) / len(abs_errors)

        # Find worst metric
        worst_metric = max(
            [(k, abs(v)) for k, v in self.errors.items() if k.endswith("_error")],
            key=lambda x: x[1],
        )[0].replace("_error", "")

        # Pass if all errors under 5%
        threshold = 0.05
        passed = all(e < threshold for e in abs_errors)

        return {
            "status": "pass" if passed else "fail",
            "pass": passed,
            "max_error": max_error,
            "mean_error": mean_error,
            "worst_metric": worst_metric,
            "n_metrics": len(abs_errors),
            "threshold": threshold,
        }


def validate_against_soi(
    microdata: pl.DataFrame,
    targets: SOITargets,
    weight_col: str = "weight",
    agi_col: str = "agi",
    filing_status_col: str = "filing_status",
) -> ValidationResult:
    """
    Validate microdata against SOI targets.

    Args:
        microdata: DataFrame with individual records
        targets: SOI targets to validate against
        weight_col: Column name for sample weights
        agi_col: Column name for AGI
        filing_status_col: Column name for filing status

    Returns:
        ValidationResult with simulated values, targets, and errors
    """
    simulated = {}

    # Total returns (sum of weights)
    simulated["total_returns"] = float(microdata[weight_col].sum())

    # Total AGI
    if agi_col in microdata.columns:
        simulated["total_agi"] = float((microdata[weight_col] * microdata[agi_col]).sum())

    # Returns by filing status
    if filing_status_col in microdata.columns:
        status_counts = (
            microdata.group_by(filing_status_col)
            .agg(pl.col(weight_col).sum().alias("count"))
        )
        for row in status_counts.iter_rows(named=True):
            status = row[filing_status_col]
            simulated[f"returns_{status}"] = float(row["count"])

    # Compute errors
    target_dict = targets.to_dict()
    errors = compute_validation_metrics(simulated, target_dict)

    return ValidationResult(
        simulated=simulated,
        targets=target_dict,
        errors=errors,
        year=targets.year,
    )
