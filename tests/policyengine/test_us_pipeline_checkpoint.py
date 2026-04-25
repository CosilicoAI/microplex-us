"""US pipeline checkpoint save/load tests.

The pipeline takes ~11 hours to synthesize + impute + build PE tables
before calibration even starts. Then PE microsim materializes target
variables (~30 min) before calibration fits. If any later stage fails
(OOM, bad config, disk full, sparsity collapse), we want to iterate
without re-paying earlier work.

``save_us_pipeline_checkpoint`` and ``load_us_pipeline_checkpoint``
round-trip a ``PolicyEngineUSEntityTableBundle`` at a named pipeline
stage so a downstream rerun can resume from that point.

These tests drive:

1. Basic round-trip equivalence at each stage.
2. Partial bundles (some entity tables ``None``) round-trip correctly.
3. Metadata file is written alongside the parquet files and contains
   enough info to validate the bundle (row counts, column names, stage).
4. Load from a missing path raises a clear error.
5. Save with invalid stage raises.
6. Loading with ``expected_stage`` mismatch raises.
7. Saving twice to the same path replaces the earlier snapshot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microplex_us.policyengine.us import (
    PolicyEngineUSEntityTableBundle,
    load_us_pipeline_checkpoint,
    save_us_pipeline_checkpoint,
)


def _make_bundle(n: int = 50, seed: int = 0) -> PolicyEngineUSEntityTableBundle:
    rng = np.random.default_rng(seed)
    household_ids = np.arange(n) + 1
    households = pd.DataFrame(
        {
            "household_id": household_ids,
            "household_weight": rng.uniform(0.5, 2.0, size=n),
            "state_fips": rng.integers(1, 57, size=n),
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": household_ids * 10,
            "household_id": household_ids,
            "age": rng.integers(0, 85, size=n),
            "employment_income": rng.uniform(0, 200_000, size=n),
        }
    )
    tax_units = pd.DataFrame(
        {
            "tax_unit_id": household_ids * 100,
            "household_id": household_ids,
            "filing_status": rng.choice(["SINGLE", "JOINT"], size=n),
        }
    )
    return PolicyEngineUSEntityTableBundle(
        households=households,
        persons=persons,
        tax_units=tax_units,
        spm_units=None,
        families=None,
        marital_units=None,
    )


class TestUSPipelineCheckpoint:
    @pytest.mark.parametrize("stage", ["post_imputation", "post_microsim"])
    def test_full_roundtrip_equivalent(self, tmp_path: Path, stage: str) -> None:
        bundle = _make_bundle(n=100)
        save_us_pipeline_checkpoint(bundle, tmp_path / "checkpoint", stage=stage)
        loaded, metadata = load_us_pipeline_checkpoint(tmp_path / "checkpoint")

        pd.testing.assert_frame_equal(loaded.households, bundle.households)
        pd.testing.assert_frame_equal(loaded.persons, bundle.persons)
        pd.testing.assert_frame_equal(loaded.tax_units, bundle.tax_units)
        assert loaded.spm_units is None
        assert loaded.families is None
        assert loaded.marital_units is None
        assert metadata["stage"] == stage

    def test_partial_bundle_roundtrip(self, tmp_path: Path) -> None:
        """A households-only bundle (no other entity tables) round-trips."""
        households = pd.DataFrame(
            {"household_id": [1, 2, 3], "household_weight": [1.0, 2.0, 3.0]}
        )
        bundle = PolicyEngineUSEntityTableBundle(
            households=households,
            persons=None,
            tax_units=None,
            spm_units=None,
            families=None,
            marital_units=None,
        )
        save_us_pipeline_checkpoint(
            bundle, tmp_path / "checkpoint", stage="post_imputation"
        )
        loaded, _ = load_us_pipeline_checkpoint(tmp_path / "checkpoint")

        pd.testing.assert_frame_equal(loaded.households, bundle.households)
        assert loaded.persons is None
        assert loaded.tax_units is None

    def test_metadata_written_with_row_counts(self, tmp_path: Path) -> None:
        bundle = _make_bundle(n=75)
        save_us_pipeline_checkpoint(
            bundle, tmp_path / "checkpoint", stage="post_microsim"
        )

        metadata_path = tmp_path / "checkpoint" / "metadata.json"
        assert metadata_path.exists()

        import json

        metadata = json.loads(metadata_path.read_text())
        assert metadata["stage"] == "post_microsim"
        assert metadata["households"]["rows"] == 75
        assert "household_id" in metadata["households"]["columns"]
        assert metadata["persons"]["rows"] == 75
        assert metadata["tax_units"]["rows"] == 75
        assert metadata["spm_units"] is None

    def test_load_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="US pipeline checkpoint"):
            load_us_pipeline_checkpoint(tmp_path / "does_not_exist")

    def test_save_with_invalid_stage_raises(self, tmp_path: Path) -> None:
        bundle = _make_bundle(n=5)
        with pytest.raises(ValueError, match="stage must be one of"):
            save_us_pipeline_checkpoint(bundle, tmp_path / "checkpoint", stage="bogus")  # type: ignore[arg-type]

    def test_load_with_stage_mismatch_raises(self, tmp_path: Path) -> None:
        bundle = _make_bundle(n=5)
        save_us_pipeline_checkpoint(
            bundle, tmp_path / "checkpoint", stage="post_imputation"
        )
        with pytest.raises(ValueError, match="expected 'post_microsim'"):
            load_us_pipeline_checkpoint(
                tmp_path / "checkpoint", expected_stage="post_microsim"
            )

    def test_save_overwrites_existing(self, tmp_path: Path) -> None:
        first = _make_bundle(n=10, seed=0)
        second = _make_bundle(n=20, seed=1)

        save_us_pipeline_checkpoint(
            first, tmp_path / "checkpoint", stage="post_imputation"
        )
        save_us_pipeline_checkpoint(
            second, tmp_path / "checkpoint", stage="post_imputation"
        )

        loaded, _ = load_us_pipeline_checkpoint(tmp_path / "checkpoint")
        assert len(loaded.households) == 20
        pd.testing.assert_frame_equal(loaded.households, second.households)
