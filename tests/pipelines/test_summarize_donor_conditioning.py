from __future__ import annotations

import json
from pathlib import Path

from microplex_us.pipelines.summarize_donor_conditioning import (
    summarize_donor_conditioning,
)


def test_summarize_donor_conditioning_filters_and_counts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "synthesis": {
                    "donor_conditioning_diagnostics": [
                        {
                            "donor_source": "irs_soi_puf_2024",
                            "model_variables": ["taxable_interest_income"],
                            "restored_variables": ["taxable_interest_income"],
                            "condition_selection": "pe_prespecified",
                            "used_condition_surface": False,
                            "raw_shared_vars": [
                                "age",
                                "employment_status",
                                "income",
                                "state_fips",
                            ],
                            "shared_vars_after_model_exclusion": [
                                "age",
                                "employment_status",
                                "income",
                                "state_fips",
                            ],
                            "projection_applied": False,
                            "entity_compatible_shared_vars": [],
                            "requested_supplemental_shared_condition_vars": [
                                "employment_status",
                                "income",
                                "state_fips",
                            ],
                            "requested_challenger_shared_condition_vars": [
                                "self_employment_income",
                                "rental_income",
                            ],
                            "raw_supplemental_shared_condition_var_status": [
                                {
                                    "variable": "employment_status",
                                    "selected": False,
                                    "in_shared_overlap": False,
                                    "reason": "incompatible_condition_support",
                                },
                                {
                                    "variable": "income",
                                    "selected": False,
                                    "in_shared_overlap": False,
                                    "reason": "excluded_from_shared_overlap",
                                },
                                {
                                    "variable": "state_fips",
                                    "selected": True,
                                    "in_shared_overlap": True,
                                    "reason": "selected",
                                },
                            ],
                            "raw_challenger_shared_condition_var_status": [
                                {
                                    "variable": "self_employment_income",
                                    "selected": True,
                                    "in_shared_overlap": True,
                                    "reason": "selected",
                                },
                                {
                                    "variable": "rental_income",
                                    "selected": False,
                                    "in_shared_overlap": False,
                                    "reason": "excluded_from_shared_overlap",
                                },
                            ],
                            "supplemental_shared_condition_var_status": [
                                {
                                    "variable": "employment_status",
                                    "selected": False,
                                    "in_shared_overlap": False,
                                    "reason": "missing_current_column",
                                },
                                {
                                    "variable": "income",
                                    "selected": True,
                                    "in_shared_overlap": True,
                                    "reason": "selected",
                                },
                                {
                                    "variable": "state_fips",
                                    "selected": True,
                                    "in_shared_overlap": True,
                                    "reason": "selected",
                                },
                            ],
                            "challenger_shared_condition_var_status": [
                                {
                                    "variable": "self_employment_income",
                                    "selected": True,
                                    "in_shared_overlap": True,
                                    "reason": "selected",
                                },
                                {
                                    "variable": "rental_income",
                                    "selected": False,
                                    "in_shared_overlap": False,
                                    "reason": "missing_current_column",
                                },
                            ],
                            "selected_condition_vars": [
                                "age",
                                "income",
                                "state_fips",
                            ],
                            "dropped_shared_vars": ["education", "tenure"],
                        },
                        {
                            "donor_source": "scf",
                            "model_variables": ["bank_account_assets"],
                            "restored_variables": ["bank_account_assets"],
                            "condition_selection": "top_correlated",
                            "used_condition_surface": False,
                            "selected_condition_vars": ["age", "income"],
                            "dropped_shared_vars": ["state_fips"],
                        },
                    ]
                }
            }
        )
    )

    payload = summarize_donor_conditioning(
        artifact_dir,
        focus_variables=("taxable_interest_income",),
    )

    assert payload["block_count"] == 1
    assert payload["focus_variables"] == ["taxable_interest_income"]
    assert payload["selected_condition_var_frequency"] == {
        "age": 1,
        "income": 1,
        "state_fips": 1,
    }
    assert payload["dropped_shared_var_frequency"] == {
        "education": 1,
        "tenure": 1,
    }
    assert payload["supplemental_shared_condition_reason_frequency"] == {
        "missing_current_column": 1,
        "selected": 2,
    }
    assert payload["raw_supplemental_shared_condition_reason_frequency"] == {
        "excluded_from_shared_overlap": 1,
        "incompatible_condition_support": 1,
        "selected": 1,
    }
    assert payload["raw_challenger_shared_condition_reason_frequency"] == {
        "excluded_from_shared_overlap": 1,
        "selected": 1,
    }
    assert payload["challenger_shared_condition_reason_frequency"] == {
        "missing_current_column": 1,
        "selected": 1,
    }
    assert payload["blocks"][0]["donor_source"] == "irs_soi_puf_2024"
    assert payload["blocks"][0]["raw_shared_vars"] == [
        "age",
        "employment_status",
        "income",
        "state_fips",
    ]
    assert payload["blocks"][0]["raw_supplemental_shared_condition_var_status"] == [
        {
            "variable": "employment_status",
            "selected": False,
            "in_shared_overlap": False,
            "reason": "incompatible_condition_support",
        },
        {
            "variable": "income",
            "selected": False,
            "in_shared_overlap": False,
            "reason": "excluded_from_shared_overlap",
        },
        {
            "variable": "state_fips",
            "selected": True,
            "in_shared_overlap": True,
            "reason": "selected",
        },
    ]
    assert payload["blocks"][0]["requested_challenger_shared_condition_vars"] == [
        "self_employment_income",
        "rental_income",
    ]
    assert payload["blocks"][0]["raw_challenger_shared_condition_var_status"] == [
        {
            "variable": "self_employment_income",
            "selected": True,
            "in_shared_overlap": True,
            "reason": "selected",
        },
        {
            "variable": "rental_income",
            "selected": False,
            "in_shared_overlap": False,
            "reason": "excluded_from_shared_overlap",
        },
    ]
