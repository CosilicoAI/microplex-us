"""Constraint-metadata precompute + lookup path.

The calibration stage previously scanned each constraint's dense
1.5M-length coefficient array three separate times during ledger +
deferred-stage-selection. That accounted for ~30 GB of transient
``np.abs(...)`` allocations at v7/v8 scale on top of the ~48 GB
baseline — a contributor to the 172 GB-compressed v7 / 197 GB v8
jetsam kills.

Fix: precompute ``active_households`` and ``coefficient_mass`` once
per constraint, then thread a ``metadata_lookup`` dict through
``_build_policyengine_constraint_records`` and
``_constraint_active_household_count`` so the dense arrays aren't
rescanned. These tests pin that contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from microplex.calibration import LinearConstraint

from microplex_us.pipelines.us import (
    _build_policyengine_constraint_records,
    _constraint_active_household_count,
    _precompute_constraint_metadata,
    _strip_constraint_coefficients,
)


def _toy_constraints(n_hh: int = 1000) -> tuple[LinearConstraint, ...]:
    """Three constraints over ``n_hh`` households with known active counts.

    - ``all_nonzero``: every household has nonzero coefficient (count n_hh)
    - ``half``: half the households have nonzero coefficient (count n_hh/2)
    - ``rare``: only 10 households have nonzero coefficient
    """
    rng = np.random.default_rng(0)
    all_nonzero = np.ones(n_hh, dtype=float)
    half = np.where(rng.random(n_hh) > 0.5, 1.0, 0.0)
    rare = np.zeros(n_hh, dtype=float)
    rare[:10] = 1.0
    return (
        LinearConstraint(name="all_nonzero", coefficients=all_nonzero, target=100.0),
        LinearConstraint(name="half", coefficients=half, target=200.0),
        LinearConstraint(name="rare", coefficients=rare, target=10.0),
    )


class TestPrecomputeMetadata:
    def test_precomputed_scalars_match_direct_computation(self) -> None:
        constraints = _toy_constraints(n_hh=1000)
        metadata = _precompute_constraint_metadata(constraints)
        for c in constraints:
            expected_count = int(np.count_nonzero(np.abs(c.coefficients) > 1e-12))
            expected_mass = float(np.abs(c.coefficients).sum())
            assert metadata[c.name]["active_households"] == expected_count
            assert metadata[c.name]["coefficient_mass"] == pytest.approx(
                expected_mass, rel=1e-12
            )

    def test_empty_constraints_produce_empty_metadata(self) -> None:
        assert _precompute_constraint_metadata(()) == {}


class TestMetadataLookupBypassesCoefficients:
    def test_active_household_count_uses_lookup(self) -> None:
        constraints = _toy_constraints(n_hh=1000)
        metadata = _precompute_constraint_metadata(constraints)
        stripped = _strip_constraint_coefficients(constraints)
        # Sanity: stripped tuple has no coefficient data to scan.
        for c in stripped:
            assert c.coefficients.size == 0
        # Without metadata_lookup, active-count on a stripped constraint is 0.
        assert _constraint_active_household_count(stripped[0]) == 0
        # With metadata_lookup, the precomputed count is returned.
        assert (
            _constraint_active_household_count(
                stripped[0], metadata_lookup=metadata
            )
            == 1000
        )

    def test_build_records_uses_lookup_when_coefficients_stripped(self) -> None:
        """Integration: records built from stripped constraints + lookup
        match records built from the full (unstripped) constraints."""

        class FakeTarget:
            def __init__(self, name: str, geo_level: str = "national"):
                self.name = name
                self.aggregation = "SUM"
                self.metadata = {"geo_level": geo_level}
                self.required_features = ()

        constraints = _toy_constraints(n_hh=1000)
        targets = [
            FakeTarget(name="all_nonzero"),
            FakeTarget(name="half"),
            FakeTarget(name="rare"),
        ]
        expected = _build_policyengine_constraint_records(targets, constraints)

        metadata = _precompute_constraint_metadata(constraints)
        stripped = _strip_constraint_coefficients(constraints)
        actual = _build_policyengine_constraint_records(
            targets, stripped, metadata_lookup=metadata
        )

        for exp, act in zip(expected, actual, strict=True):
            assert exp["active_households"] == act["active_households"]
            assert exp["coefficient_mass"] == pytest.approx(
                act["coefficient_mass"], rel=1e-12
            )


class TestBackwardCompatibility:
    def test_records_without_lookup_still_work(self) -> None:
        """Legacy callers that don't pass metadata_lookup should still get
        correct results by scanning the coefficient arrays."""

        class FakeTarget:
            def __init__(self, name: str):
                self.name = name
                self.aggregation = "SUM"
                self.metadata = {"geo_level": "national"}
                self.required_features = ()

        constraints = _toy_constraints(n_hh=500)
        targets = [FakeTarget(name=c.name) for c in constraints]
        records = _build_policyengine_constraint_records(targets, constraints)
        assert records[0]["active_households"] == 500
        assert records[1]["active_households"] > 200  # ~half
        assert records[1]["active_households"] < 300
        assert records[2]["active_households"] == 10
