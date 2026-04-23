"""Downstream tax-benefit aggregate validation (B2).

After calibration, the synthesized microdata is ingested by
``policyengine_us.Microsimulation``. This module computes a canonical
set of downstream aggregates — federal income tax, EITC, CTC, SNAP,
SSI, ACA PTC — and compares them against external benchmarks (IRS
SOI, USDA, SSA, CMS). The comparison is the validation a tax-microsim
reviewer actually wants: not whether input targets were hit, but
whether the downstream policy outputs computed on the synthetic frame
look like the real-world outputs.

These tests drive:

1. ``DownstreamBenchmark`` is a typed record for one
   external-benchmark comparison (name, computed, benchmark, source,
   unit).
2. ``compute_downstream_comparison`` returns a dict of benchmark
   name → ``DownstreamBenchmark`` with absolute and relative errors.
3. The module's canonical benchmark set for 2024 includes the six
   required headline aggregates.
4. Relative error is signed (computed − benchmark) / benchmark.
5. A benchmark record round-trips to JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from microplex_us.validation.downstream import (
    DOWNSTREAM_BENCHMARKS_2024,
    DownstreamBenchmark,
    compute_downstream_aggregates,
    compute_downstream_comparison,
    compute_downstream_weighted_aggregate,
)


class TestDownstreamBenchmark:
    def test_benchmark_record_fields(self) -> None:
        record = DownstreamBenchmark(
            name="eitc",
            computed=65_000_000_000.0,
            benchmark=64_000_000_000.0,
            unit="USD",
            source="IRS SOI 2024",
        )
        assert record.abs_error == pytest.approx(1_000_000_000.0)
        assert record.rel_error == pytest.approx(1_000_000_000.0 / 64_000_000_000.0)

    def test_benchmark_record_serializes_to_json(self) -> None:
        record = DownstreamBenchmark(
            name="snap",
            computed=100.0,
            benchmark=110.0,
            unit="USD",
            source="USDA 2024",
        )
        as_json = json.loads(json.dumps(record.to_dict()))
        assert as_json["name"] == "snap"
        assert as_json["computed"] == 100.0
        assert as_json["benchmark"] == 110.0
        assert as_json["rel_error"] == pytest.approx(-10.0 / 110.0)

    def test_benchmark_zero_benchmark_returns_none_rel(self) -> None:
        """Guard against divide-by-zero in report generation."""
        record = DownstreamBenchmark(
            name="zero",
            computed=5.0,
            benchmark=0.0,
            unit="USD",
            source="test",
        )
        assert record.rel_error is None


class TestDownstreamBenchmarksSet:
    def test_2024_benchmark_set_covers_headline_aggregates(self) -> None:
        names = {b.name for b in DOWNSTREAM_BENCHMARKS_2024}
        assert names >= {"income_tax", "eitc", "ctc", "snap", "ssi", "aca_ptc"}

    def test_2024_benchmarks_have_sources_cited(self) -> None:
        """No magic numbers — each benchmark must declare its source."""
        for benchmark in DOWNSTREAM_BENCHMARKS_2024:
            assert benchmark.source, f"missing source on {benchmark.name}"
            assert benchmark.benchmark > 0, f"non-positive benchmark on {benchmark.name}"


class TestComputeDownstreamComparison:
    def test_compute_from_aggregates_dict(self) -> None:
        """The pure comparison step: given computed numbers, wrap them
        with their benchmarks and errors. No PE-sim needed.
        """
        computed = {
            "income_tax": 2_300_000_000_000.0,
            "eitc": 64_000_000_000.0,
            "ctc": 115_000_000_000.0,
            "snap": 98_000_000_000.0,
            "ssi": 66_000_000_000.0,
            "aca_ptc": 55_000_000_000.0,
        }
        result = compute_downstream_comparison(computed, DOWNSTREAM_BENCHMARKS_2024)

        assert set(result) == set(computed)
        eitc = result["eitc"]
        assert eitc.computed == 64_000_000_000.0
        assert eitc.benchmark > 0
        assert abs(eitc.rel_error) < 0.2, "EITC computed ~ benchmark"
        assert eitc.source

    def test_compute_skips_missing_variables(self) -> None:
        """If a variable doesn't have a benchmark, it's silently omitted."""
        computed = {"not_a_benchmark_name": 1.0, "eitc": 60_000_000_000.0}
        result = compute_downstream_comparison(computed, DOWNSTREAM_BENCHMARKS_2024)
        assert "not_a_benchmark_name" not in result
        assert "eitc" in result


class TestComputeDownstreamAggregates:
    @staticmethod
    def _fake_simulation(
        *,
        values: dict[str, list[float]],
        entities: dict[str, str],
    ):
        class FakeMicrosimulation:
            def __init__(self, dataset: str = "fake.h5") -> None:
                self.dataset = dataset
                self.tax_benefit_system = SimpleNamespace(
                    get_variable=lambda name: SimpleNamespace(
                        entity=SimpleNamespace(key=entities[name])
                    )
                )

            def calculate(self, variable: str, period: int):
                assert period == 2024
                return SimpleNamespace(values=values[variable])

        return FakeMicrosimulation()

    def test_uses_entity_weights_for_weighted_totals(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        class FakeMicrosimulation:
            def __init__(self, dataset: str) -> None:
                self.dataset = dataset
                self.tax_benefit_system = SimpleNamespace(
                    get_variable=lambda name: SimpleNamespace(
                        entity=SimpleNamespace(
                            key={
                                "eitc": "tax_unit",
                                "snap": "spm_unit",
                                "ssi": "person",
                            }[name]
                        )
                    )
                )

            def calculate(self, variable: str, period: int):
                assert period == 2024
                values = {
                    "eitc": [10.0, 20.0],
                    "tax_unit_weight": [100.0, 200.0],
                    "snap": [1.0, 2.0, 3.0],
                    "spm_unit_weight": [10.0, 20.0, 30.0],
                    "ssi": [7.0, 11.0],
                    "person_weight": [2.0, 3.0],
                }
                return SimpleNamespace(sum=lambda: sum(values[variable]), values=values[variable])

        fake_module = ModuleType("policyengine_us")
        fake_module.Microsimulation = FakeMicrosimulation
        monkeypatch.setitem(sys.modules, "policyengine_us", fake_module)

        aggregates = compute_downstream_aggregates(
            tmp_path / "fake.h5",
            period=2024,
            variables=("eitc", "snap", "ssi"),
        )

        assert aggregates["eitc"] == pytest.approx(10.0 * 100.0 + 20.0 * 200.0)
        assert aggregates["snap"] == pytest.approx(
            1.0 * 10.0 + 2.0 * 20.0 + 3.0 * 30.0
        )
        assert aggregates["ssi"] == pytest.approx(7.0 * 2.0 + 11.0 * 3.0)

    def test_weighted_aggregate_rejects_unsupported_entity(self) -> None:
        simulation = self._fake_simulation(
            values={"odd_output": [1.0, 2.0]},
            entities={"odd_output": "benefit_unit"},
        )

        with pytest.raises(ValueError, match="Unsupported entity"):
            compute_downstream_weighted_aggregate(
                simulation,
                "odd_output",
                period=2024,
            )

    def test_weighted_aggregate_rejects_value_weight_length_mismatch(self) -> None:
        simulation = self._fake_simulation(
            values={
                "eitc": [10.0, 20.0, 30.0],
                "tax_unit_weight": [100.0, 200.0],
            },
            entities={"eitc": "tax_unit"},
        )

        with pytest.raises(ValueError, match="does not match"):
            compute_downstream_weighted_aggregate(simulation, "eitc", period=2024)
