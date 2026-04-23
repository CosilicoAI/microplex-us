from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest


def _load_run_b2_batched_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "run_b2_batched.py"
    )
    spec = importlib.util.spec_from_file_location("run_b2_batched", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRunB2BatchedEntityResolution:
    def test_prefers_policyengine_metadata_over_length_match(self) -> None:
        module = _load_run_b2_batched_module()
        arrays = {
            "household_id": np.array([1, 2, 3]),
            "tax_unit_id": np.array([10, 20, 30]),
            "some_tax_unit_var": np.array([100.0, 200.0, 300.0]),
        }

        entity = module._entity_of(
            "some_tax_unit_var",
            arrays,
            variable_entities={"some_tax_unit_var": "tax_unit"},
        )

        assert entity == "tax_unit"

    def test_ambiguous_length_match_raises_without_metadata(self) -> None:
        module = _load_run_b2_batched_module()
        arrays = {
            "household_id": np.array([1, 2, 3]),
            "tax_unit_id": np.array([10, 20, 30]),
            "ambiguous_var": np.array([100.0, 200.0, 300.0]),
        }

        with pytest.raises(ValueError, match="Ambiguous entity for variable"):
            module._entity_of("ambiguous_var", arrays)

    def test_write_chunk_h5_slices_mixed_entities(
        self,
        tmp_path: Path,
    ) -> None:
        module = _load_run_b2_batched_module()
        arrays = {
            "household_id": np.array([1, 2]),
            "household_weight": np.array([100.0, 200.0]),
            "person_id": np.array([10, 11, 20]),
            "person_household_id": np.array([1, 1, 2]),
            "tax_unit_id": np.array([100, 200]),
            "person_tax_unit_id": np.array([100, 100, 200]),
            "tax_unit_weight": np.array([100.0, 200.0]),
            "household_output": np.array([1.0, 2.0]),
            "person_output": np.array([3.0, 4.0, 5.0]),
            "tax_unit_output": np.array([6.0, 7.0]),
        }
        masks = module._build_entity_masks(arrays, np.array([1]))
        output_path = tmp_path / "chunk.h5"

        module._write_chunk_h5(
            arrays,
            masks,
            "2024",
            output_path,
            variable_entities={
                "household_output": "household",
                "person_output": "person",
                "tax_unit_output": "tax_unit",
            },
        )

        with h5py.File(output_path, "r") as handle:
            assert handle["household_id"]["2024"][:].tolist() == [1]
            assert handle["person_id"]["2024"][:].tolist() == [10, 11]
            assert handle["tax_unit_id"]["2024"][:].tolist() == [100]
            assert handle["household_output"]["2024"][:].tolist() == [1.0]
            assert handle["person_output"]["2024"][:].tolist() == [3.0, 4.0]
            assert handle["tax_unit_output"]["2024"][:].tolist() == [6.0]
            assert handle["tax_unit_weight"]["2024"][:].tolist() == [100.0]
