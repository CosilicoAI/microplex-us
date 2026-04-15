from __future__ import annotations

from pathlib import Path

import pandas as pd

from microplex_us.pipelines.summarize_child_tax_unit_agi_drift import (
    summarize_child_tax_unit_agi_drift,
)


def test_summarize_child_tax_unit_agi_drift(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text("{}")
    frame = pd.DataFrame(
        {
            "age": [5, 35, 12],
            "is_tax_unit_dependent": [1, 0, 1],
            "tax_unit_id": ["tu1", "tu2", "tu1"],
            "partnership_s_corp_income": [0.0, 100.0, 5.0],
            "taxable_interest_income": [0.0, 10.0, 0.0],
        }
    )
    frame.to_parquet(artifact_dir / "seed_data.parquet", index=False)

    payload = summarize_child_tax_unit_agi_drift(artifact_dir)

    seed = payload["stages"]["seed"]
    assert seed["row_count"] == 3
    dependents = seed["subsets"]["dependents_under_20"]["partnership_s_corp_income"]
    assert dependents["count"] == 2
    assert dependents["sum"] == 5.0
    assert dependents["nonzero_share"] == 0.5

    tax_units = seed["tax_unit_subsets"]["with_children"]["partnership_s_corp_income"]
    assert tax_units["count"] == 1
    assert tax_units["sum"] == 5.0
