"""Downstream tax-benefit aggregate validation (paper reviewer response B2).

Input-target validation (see ``soi.py``, ``baseline.py``) asks whether
the calibrated synthetic frame's marginal sums match administrative
totals on the *variables the calibrator was told to target*.
Downstream validation asks the different, stricter question: when the
calibrated frame is ingested by ``policyengine_us.Microsimulation``,
do the *computed policy outputs* — federal income tax, EITC, CTC,
SNAP, SSI, ACA PTC — match administrative aggregates?

This module contains:

- ``DownstreamBenchmark`` record (name, computed, benchmark, unit, source).
- ``DOWNSTREAM_BENCHMARKS_2024`` canonical 2024 benchmark set. Each
  record is sourced to an IRS / USDA / SSA / CMS / CBO publication.
- ``compute_downstream_aggregates(dataset_path, period)`` runs the
  simulation and returns a dict of variable → weighted sum.
- ``compute_downstream_comparison(aggregates, benchmarks)`` joins
  computed values to benchmarks and returns per-variable errors.

Benchmark numbers are rounded publicly-reported totals; each has a
citation. Updates should be traceable to the cited source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DownstreamBenchmark:
    """One external-benchmark comparison.

    ``benchmark`` is the published external aggregate (e.g. IRS SOI
    total EITC disbursed 2024). ``computed`` is the aggregate computed
    on the calibrated synthetic frame by ``policyengine_us``.
    """

    name: str
    computed: float
    benchmark: float
    unit: str
    source: str

    @property
    def abs_error(self) -> float:
        return self.computed - self.benchmark

    @property
    def rel_error(self) -> float | None:
        if self.benchmark == 0:
            return None
        return (self.computed - self.benchmark) / self.benchmark

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "computed": self.computed,
            "benchmark": self.benchmark,
            "unit": self.unit,
            "source": self.source,
            "abs_error": self.abs_error,
            "rel_error": self.rel_error,
        }


@dataclass(frozen=True)
class DownstreamBenchmarkSpec:
    """A benchmark definition without a computed value attached."""

    name: str
    benchmark: float
    unit: str
    source: str


DOWNSTREAM_BENCHMARKS_2024: tuple[DownstreamBenchmarkSpec, ...] = (
    DownstreamBenchmarkSpec(
        name="income_tax",
        benchmark=2_400_000_000_000.0,
        unit="USD",
        source=(
            "IRS SOI 2022 total federal individual income tax liability "
            "~$2.22T; CBO 2024 projection ~$2.4T"
        ),
    ),
    DownstreamBenchmarkSpec(
        name="eitc",
        benchmark=64_000_000_000.0,
        unit="USD",
        source="IRS SOI 2023 EITC disbursed ~$64B (Table 2.5)",
    ),
    DownstreamBenchmarkSpec(
        name="ctc",
        benchmark=115_000_000_000.0,
        unit="USD",
        source=(
            "IRS SOI 2023 CTC disbursed ~$115B (pre-OBBBA CTC of $2,000 "
            "per qualifying child)"
        ),
    ),
    DownstreamBenchmarkSpec(
        name="snap",
        benchmark=100_000_000_000.0,
        unit="USD",
        source="USDA FNS FY2024 SNAP benefits total ~$100B",
    ),
    DownstreamBenchmarkSpec(
        name="ssi",
        benchmark=66_000_000_000.0,
        unit="USD",
        source="SSA SSI Annual Statistical Report 2024 ~$66B total payments",
    ),
    DownstreamBenchmarkSpec(
        name="aca_ptc",
        benchmark=60_000_000_000.0,
        unit="USD",
        source=(
            "CMS/IRS ACA Advance Premium Tax Credit & reconciled PTC "
            "2024 ~$60B (IRA-enhanced subsidies in effect)"
        ),
    ),
)


def compute_downstream_comparison(
    aggregates: dict[str, float],
    benchmarks: Iterable[DownstreamBenchmarkSpec],
) -> dict[str, DownstreamBenchmark]:
    """Join computed aggregates to their external benchmarks.

    Variables in ``aggregates`` without a matching benchmark are
    silently omitted — they're either not in the benchmark set or the
    caller passed extra diagnostic values.
    """
    benchmark_by_name = {spec.name: spec for spec in benchmarks}
    result: dict[str, DownstreamBenchmark] = {}
    for name, computed in aggregates.items():
        spec = benchmark_by_name.get(name)
        if spec is None:
            continue
        result[name] = DownstreamBenchmark(
            name=name,
            computed=float(computed),
            benchmark=spec.benchmark,
            unit=spec.unit,
            source=spec.source,
        )
    return result


def compute_downstream_aggregates(
    dataset_path: str | Path,
    period: int = 2024,
    variables: Iterable[str] = (
        "income_tax",
        "eitc",
        "ctc",
        "snap",
        "ssi",
        "aca_ptc",
    ),
) -> dict[str, float]:
    """Load a PolicyEngine-US dataset and compute weighted sums for ``variables``.

    Returns a dict of variable → weighted aggregate (float). Requires
    ``policyengine_us`` to be installed.
    """
    # Import lazily so the rest of this module (benchmark records,
    # comparison function) stays importable in environments without PE.
    from policyengine_us import Microsimulation  # noqa: PLC0415

    simulation = Microsimulation(dataset=str(dataset_path))
    aggregates: dict[str, float] = {}
    for variable in variables:
        series = simulation.calculate(variable, period)
        aggregates[variable] = float(series.sum())
    return aggregates
