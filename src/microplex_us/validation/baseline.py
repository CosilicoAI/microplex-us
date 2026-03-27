"""
Baseline comparison of CPS microdata against SOI targets.

This module quantifies the gaps BEFORE any calibration, establishing
a baseline understanding of where CPS underreports relative to
administrative tax data.

Key gaps documented:
- Capital gains: ~$1.2T in SOI, $0 in CPS
- Interest/dividends: underreported ~40%
- High-income returns: underrepresented in CPS
"""

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from microplex_us.data_sources.cps_transform import TransformedDataset
from microplex_us.validation.soi import SOITargets


@dataclass
class MetricComparison:
    """Comparison of a single metric between CPS and SOI."""

    name: str
    category: str  # "aggregate", "by_filing_status", "by_agi_bracket"
    cps_value: float
    soi_value: float
    unit: str  # "count", "dollars", "rate"
    statute_ref: str | None = None

    @property
    def pct_error(self) -> float | None:
        """Percentage error: (CPS - SOI) / SOI."""
        if self.soi_value == 0:
            return None
        return (self.cps_value - self.soi_value) / abs(self.soi_value)

    @property
    def abs_error(self) -> float:
        """Absolute error."""
        return abs(self.cps_value - self.soi_value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "cps_value": self.cps_value,
            "soi_value": self.soi_value,
            "pct_error": self.pct_error,
            "unit": self.unit,
            "statute_ref": self.statute_ref,
        }


@dataclass
class BaselineComparison:
    """Baseline comparison of CPS vs SOI before calibration."""

    cps_year: int
    soi_year: int
    metrics: dict[str, MetricComparison] = field(default_factory=dict)
    coverage_gaps: list[dict] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        errors = [
            abs(m.pct_error)
            for m in self.metrics.values()
            if m.pct_error is not None
        ]

        if not errors:
            return {
                "n_metrics": 0,
                "mean_abs_error": None,
                "max_abs_error": None,
                "worst_metric": None,
            }

        worst = max(
            [(k, abs(m.pct_error)) for k, m in self.metrics.items() if m.pct_error is not None],
            key=lambda x: x[1],
        )

        return {
            "n_metrics": len(errors),
            "mean_abs_error": sum(errors) / len(errors),
            "max_abs_error": max(errors),
            "worst_metric": worst[0],
        }


def compute_baseline_comparison(
    transformed: TransformedDataset,
    soi_targets: SOITargets,
) -> BaselineComparison:
    """
    Compute baseline comparison between CPS and SOI.

    This establishes the gap BEFORE calibration, showing:
    - How many returns CPS represents vs SOI
    - How much income CPS captures vs SOI
    - Where the major shortfalls are

    Args:
        transformed: Transformed CPS dataset with tax units
        soi_targets: SOI targets for comparison

    Returns:
        BaselineComparison with all metric comparisons
    """
    comparison = BaselineComparison(
        cps_year=transformed.year,
        soi_year=soi_targets.year,
    )

    tax_units = transformed.tax_units

    # Total returns (weighted sum of tax units)
    cps_returns = float(tax_units["weight"].sum())
    comparison.metrics["total_returns"] = MetricComparison(
        name="total_returns",
        category="aggregate",
        cps_value=cps_returns,
        soi_value=float(soi_targets.total_returns),
        unit="count",
        statute_ref="26 USC 6012 - Returns required",
    )

    # Total AGI
    if "agi_proxy" in tax_units.columns:
        cps_agi = float((tax_units["weight"] * tax_units["agi_proxy"]).sum())
        comparison.metrics["total_agi"] = MetricComparison(
            name="total_agi",
            category="aggregate",
            cps_value=cps_agi,
            soi_value=float(soi_targets.total_agi),
            unit="dollars",
            statute_ref="26 USC 62(a) - Adjusted gross income defined",
        )

    # Total earned income (only if SOI has wages data)
    if "earned_income" in tax_units.columns and soi_targets.total_wages is not None:
        cps_earned = float((tax_units["weight"] * tax_units["earned_income"]).sum())
        comparison.metrics["total_earned_income"] = MetricComparison(
            name="total_earned_income",
            category="aggregate",
            cps_value=cps_earned,
            soi_value=float(soi_targets.total_wages),
            unit="dollars",
            statute_ref="26 USC 32(c)(2) - Earned income defined",
        )

    # Returns by filing status
    if "filing_status" in tax_units.columns:
        status_counts = (
            tax_units.group_by("filing_status")
            .agg(pl.col("weight").sum().alias("count"))
        )

        for status, soi_count in soi_targets.returns_by_filing_status.items():
            status_df = status_counts.filter(pl.col("filing_status") == status)
            cps_count = float(status_df["count"][0]) if len(status_df) > 0 else 0

            comparison.metrics[f"returns_{status}"] = MetricComparison(
                name=f"returns_{status}",
                category="by_filing_status",
                cps_value=cps_count,
                soi_value=float(soi_count),
                unit="count",
                statute_ref="26 USC 1, 2 - Filing status",
            )

    # Document known coverage gaps from the transform
    if hasattr(transformed, "coverage_report") and "gaps" in transformed.coverage_report:
        comparison.coverage_gaps = transformed.coverage_report["gaps"]
    else:
        # Default gaps if not in coverage report
        comparison.coverage_gaps = _get_default_coverage_gaps()

    return comparison


def _get_default_coverage_gaps() -> list[dict]:
    """Return default coverage gaps for CPS vs SOI."""
    return [
        {
            "variable": "agi_proxy",
            "component": "capital_gains",
            "statute_ref": "26 USC 1222",
            "impact": "high",
            "notes": "CPS does not collect capital gains. SOI 2021: ~$1.2T.",
        },
        {
            "variable": "agi_proxy",
            "component": "above_line_deductions",
            "statute_ref": "26 USC 62(a)(7)",
            "impact": "medium",
            "notes": "IRA contributions, student loan interest not in CPS.",
        },
        {
            "variable": "filing_status",
            "component": "head_of_household",
            "statute_ref": "26 USC 2(b)",
            "impact": "medium",
            "notes": "CPS married status doesn't map directly to HoH eligibility.",
        },
        {
            "variable": "interest_income",
            "component": "underreporting",
            "statute_ref": "26 USC 61(a)(4)",
            "impact": "medium",
            "notes": "Survey underreporting of interest income ~40%.",
        },
        {
            "variable": "dividend_income",
            "component": "underreporting",
            "statute_ref": "26 USC 61(a)(7)",
            "impact": "medium",
            "notes": "Survey underreporting of dividend income ~40%.",
        },
    ]


def export_comparison_json(comparison: BaselineComparison) -> dict[str, Any]:
    """
    Export comparison to JSON-serializable dictionary.

    Format designed for dashboard visualization.
    """
    return {
        "cps_year": comparison.cps_year,
        "soi_year": comparison.soi_year,
        "metrics": [m.to_dict() for m in comparison.metrics.values()],
        "summary": comparison.summary(),
        "coverage_gaps": comparison.coverage_gaps,
    }
