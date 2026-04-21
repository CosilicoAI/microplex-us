"""Batched-materialize equivalence tests.

Covers the batched path of :func:`materialize_policyengine_us_variables`
without spinning up a real PolicyEngine Microsimulation. A fake
``simulation_cls`` mimics the per-record-scalar semantics that
calibration targets actually use (each output is a function of the
calling chunk's own data, independent of other chunks). The test then
proves that running the function with ``batch_size=None`` and with a
sub-full batch size produces identical results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest
from microplex.core import EntityType

from microplex_us.policyengine.us import (
    PolicyEngineUSEntityTableBundle,
    materialize_policyengine_us_variables,
)


@dataclass
class FakeVariable:
    """Stand-in for a PolicyEngine Variable metadata entry."""

    name: str
    entity: str  # "household" | "person" | etc.


class FakeEntity:
    def __init__(self, key: str) -> None:
        self.key = key


class FakeTaxBenefitSystem:
    """Enough of the TaxBenefitSystem interface to satisfy the materializer.

    The real resolver checks a variables registry + entity registry. The
    fake returns hardcoded entries for the test's target variables.
    """

    def __init__(self, variables: dict[str, FakeVariable]) -> None:
        self.variables = variables
        self.entities = [FakeEntity(k) for k in ("person", "household", "tax_unit")]

    def get_variable(self, name: str) -> FakeVariable:
        if name not in self.variables:
            raise KeyError(name)
        return self.variables[name]


class FakeSimulation:
    """Fake Microsimulation that computes per-record values deterministically.

    Each variable's value is a pure function of a household-level input
    column the fake reads from the provided dataset path. Writing a
    real h5 would require the full PolicyEngine dataset machinery; for
    the test we instead accept an in-memory ``dataset`` dict.
    """

    def __init__(self, dataset: str | None = None, **kwargs: Any) -> None:
        # The real code writes an h5 and points the sim at its path;
        # for this fake we pull the chunk arrays off ``_fake_chunk_data``
        # (set via the monkeypatch below).
        chunk = getattr(FakeSimulation, "_fake_chunk_data", None)
        if chunk is None:
            raise RuntimeError(
                "FakeSimulation needs _fake_chunk_data set by the test."
            )
        self._hh = chunk["households"]
        self.tax_benefit_system = FakeTaxBenefitSystem(
            {
                "doubled_base": FakeVariable(name="doubled_base", entity="household"),
                "squared_base": FakeVariable(name="squared_base", entity="household"),
            }
        )

    def calculate(self, variable: str, period: Any = None, map_to: Any = None):
        # Pure per-record scalar; returns len(households) values.
        base = self._hh["base_value"].to_numpy(dtype=float)
        if variable == "doubled_base":
            return base * 2.0
        if variable == "squared_base":
            return base**2
        raise KeyError(variable)


@pytest.fixture
def fake_sim(monkeypatch):
    """Register FakeSimulation as the simulation_cls and patch the
    materializer's internal helpers so they accept our in-memory chunk."""
    # Patch the module-level resolver the materializer uses to look up
    # the tax-benefit system. We monkey the whole pipeline rather than
    # write a real h5.
    from microplex_us.policyengine import us as us_module

    monkeypatch.setattr(
        us_module,
        "_resolve_policyengine_us_tax_benefit_system",
        lambda simulation_cls=None: FakeTaxBenefitSystem(
            {
                "doubled_base": FakeVariable("doubled_base", "household"),
                "squared_base": FakeVariable("squared_base", "household"),
            }
        ),
    )
    monkeypatch.setattr(
        us_module,
        "build_policyengine_us_export_variable_maps",
        lambda tables, **_: {
            "household": {"base_value": "base_value"},
            "person": {},
            "tax_unit": {},
            "spm_unit": {},
            "family": {},
        },
    )
    monkeypatch.setattr(
        us_module,
        "resolve_policyengine_excluded_export_variables",
        lambda *args, **kwargs: set(),
    )

    def _build_arrays(tables, **kwargs):
        # The real function produces a period-keyed dict of arrays; we
        # just stash the chunk on the fake class and ignore the output.
        FakeSimulation._fake_chunk_data = {
            "households": tables.households,
        }
        return {}

    monkeypatch.setattr(
        us_module,
        "build_policyengine_us_time_period_arrays",
        _build_arrays,
    )
    monkeypatch.setattr(
        us_module,
        "write_policyengine_us_time_period_dataset",
        lambda *args, **kwargs: None,
    )

    # Patch the adapter factory to return our fake
    from microplex_us.policyengine.us import (
        PolicyEngineUSMicrosimulationAdapter,
    )

    def _fake_from_dataset(*args, **kwargs):
        return PolicyEngineUSMicrosimulationAdapter(simulation=FakeSimulation())

    monkeypatch.setattr(
        PolicyEngineUSMicrosimulationAdapter,
        "from_dataset",
        classmethod(lambda cls, *a, **k: _fake_from_dataset(*a, **k)),
    )

    # Patch variable_entity so the attach helper routes all variables
    # to the household table.
    monkeypatch.setattr(
        PolicyEngineUSMicrosimulationAdapter,
        "variable_entity",
        lambda self, variable: EntityType.HOUSEHOLD,
    )


def _make_bundle(n: int = 50, seed: int = 0) -> PolicyEngineUSEntityTableBundle:
    rng = np.random.default_rng(seed)
    household_ids = np.arange(n) + 1
    households = pd.DataFrame(
        {
            "household_id": household_ids,
            "base_value": rng.uniform(1, 10, size=n),
        }
    )
    persons = pd.DataFrame(
        {
            "household_id": household_ids,
            "person_id": household_ids * 10,
        }
    )
    return PolicyEngineUSEntityTableBundle(
        households=households,
        persons=persons,
        tax_units=None,
        spm_units=None,
        families=None,
        marital_units=None,
    )


class TestBatchedMaterializeEquivalence:
    """Batched output must equal single-pass output element-wise."""

    def test_single_pass_vs_batched_equivalent(self, fake_sim) -> None:
        tables = _make_bundle(n=50)

        full_tables, full_bindings = materialize_policyengine_us_variables(
            tables,
            variables=["doubled_base", "squared_base"],
            period=2024,
            batch_size=None,
        )
        batched_tables, batched_bindings = materialize_policyengine_us_variables(
            tables,
            variables=["doubled_base", "squared_base"],
            period=2024,
            batch_size=10,  # 5 chunks
        )

        pd.testing.assert_frame_equal(
            full_tables.households.sort_values("household_id").reset_index(drop=True),
            batched_tables.households.sort_values("household_id").reset_index(drop=True),
        )
        assert set(full_bindings) == set(batched_bindings)

    def test_batch_size_larger_than_data_is_noop(self, fake_sim) -> None:
        tables = _make_bundle(n=10)
        full, _ = materialize_policyengine_us_variables(
            tables,
            variables=["doubled_base"],
            period=2024,
            batch_size=None,
        )
        batched, _ = materialize_policyengine_us_variables(
            tables,
            variables=["doubled_base"],
            period=2024,
            batch_size=10_000,  # > n=10
        )
        pd.testing.assert_frame_equal(full.households, batched.households)

    def test_uneven_batch_split(self, fake_sim) -> None:
        """50 records with batch_size=17 → chunks of 17, 17, 16."""
        tables = _make_bundle(n=50)
        batched, _ = materialize_policyengine_us_variables(
            tables,
            variables=["doubled_base"],
            period=2024,
            batch_size=17,
        )
        assert len(batched.households) == 50
        # Values correct (doubled_base = 2 * base_value)
        np.testing.assert_allclose(
            batched.households["doubled_base"].to_numpy(),
            2.0 * batched.households["base_value"].to_numpy(),
            rtol=0,
            atol=0,
        )
