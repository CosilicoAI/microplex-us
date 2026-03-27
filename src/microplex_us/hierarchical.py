"""US-specific preprocessing helpers around the generic hierarchical synthesizer."""

from __future__ import annotations

import pandas as pd


def prepare_cps_for_hierarchical(
    cps_person_data: pd.DataFrame,
    hh_id_col: str = "household_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate CPS person rows into household summaries for hierarchical synthesis."""
    persons = cps_person_data.copy()

    household_summary = persons.groupby(hh_id_col).agg(
        {
            "age": ["count", lambda values: (values >= 18).sum(), lambda values: (values < 18).sum()],
        }
    )
    household_summary.columns = ["n_persons", "n_adults", "n_children"]
    household_summary = household_summary.reset_index()

    for variable in ["state_fips", "tenure", "hh_weight"]:
        if variable in persons.columns:
            first_values = persons.groupby(hh_id_col)[variable].first()
            household_summary[variable] = household_summary[hh_id_col].map(first_values)

    return household_summary, persons


__all__ = ["prepare_cps_for_hierarchical"]
