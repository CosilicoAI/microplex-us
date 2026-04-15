"""PolicyEngine-native scoring helpers for US Microplex artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

_DEFAULT_PE_US_DATA_REPO = Path.home() / "PolicyEngine" / "policyengine-us-data"
_PE_US_DATA_PYTHON_ENV = "MICROPLEX_US_POLICYENGINE_US_DATA_PYTHON"
_PE_US_DATA_REPO_ENV = "MICROPLEX_US_POLICYENGINE_US_DATA_REPO"
_PE_NATIVE_SCORE_BASE_ENV_VARS: tuple[str, ...] = (
    "HOME",
    "PATH",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "TZ",
)

_ENHANCED_CPS_BAD_TARGETS: tuple[str, ...] = (
    "nation/irs/adjusted gross income/total/AGI in 10k-15k/taxable/Head of Household",
    "nation/irs/adjusted gross income/total/AGI in 15k-20k/taxable/Head of Household",
    "nation/irs/adjusted gross income/total/AGI in 10k-15k/taxable/Married Filing Jointly/Surviving Spouse",
    "nation/irs/adjusted gross income/total/AGI in 15k-20k/taxable/Married Filing Jointly/Surviving Spouse",
    "nation/irs/count/count/AGI in 10k-15k/taxable/Head of Household",
    "nation/irs/count/count/AGI in 15k-20k/taxable/Head of Household",
    "nation/irs/count/count/AGI in 10k-15k/taxable/Married Filing Jointly/Surviving Spouse",
    "nation/irs/count/count/AGI in 15k-20k/taxable/Married Filing Jointly/Surviving Spouse",
    "state/RI/adjusted_gross_income/amount/-inf_1",
    "nation/irs/exempt interest/count/AGI in -inf-inf/taxable/All",
)

_PE_NATIVE_BROAD_SCORE_SCRIPT = """
import json
import sys
from pathlib import Path

import numpy as np
from policyengine_core.data import Dataset

REPO_ROOT = sys.argv[1]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from policyengine_us import Microsimulation
from policyengine_us_data.utils.loss import build_loss_matrix

BAD_TARGETS = tuple(json.loads(sys.argv[2]))
PERIOD = int(sys.argv[3])
CANDIDATE_DATASET = sys.argv[4]
BASELINE_DATASET = sys.argv[5]


def dataset_from_path(dataset_path: str, dataset_name: str):
    class LocalDataset(Dataset):
        name = dataset_name
        label = dataset_name
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = PERIOD

    return LocalDataset


def classify_target_family(target_name: str) -> str:
    parts = target_name.split("/")
    if target_name.startswith("state/census/age/"):
        return "state_age_distribution"
    if target_name.startswith("state/census/population_by_state/"):
        return "state_population"
    if target_name.startswith("state/census/population_under_5_by_state/"):
        return "state_population_under_5"
    if target_name.startswith("nation/irs/aca_spending/"):
        return "state_aca_spending"
    if target_name.startswith("state/irs/aca_enrollment/"):
        return "state_aca_enrollment"
    if target_name.startswith("irs/medicaid_enrollment/"):
        return "state_medicaid_enrollment"
    if target_name.endswith("/snap-cost"):
        return "state_snap_cost"
    if target_name.endswith("/snap-hhs"):
        return "state_snap_households"
    if target_name.startswith("state/real_estate_taxes/"):
        return "state_real_estate_taxes"
    if len(parts) >= 3 and parts[0] == "state" and parts[2] == "adjusted_gross_income":
        return "state_agi_distribution"
    if target_name.startswith("nation/jct/"):
        return "national_tax_expenditures"
    if target_name.startswith("nation/net_worth/"):
        return "national_net_worth"
    if target_name.startswith("nation/ssa/"):
        return "national_ssa"
    if target_name.startswith("nation/census/population_by_age/"):
        return "national_population_by_age"
    if target_name == "nation/census/infants":
        return "national_infants"
    if target_name.startswith("nation/census/agi_in_spm_threshold_decile_"):
        return "national_spm_threshold_agi"
    if target_name.startswith("nation/census/count_in_spm_threshold_decile_"):
        return "national_spm_threshold_count"
    if target_name.startswith("nation/census/"):
        return "national_census_other"
    if target_name.startswith("nation/irs/"):
        return "national_irs_other"
    return "other"


def build_family_breakdown(target_names, candidate_terms, baseline_terms, candidate_rel_error, baseline_rel_error):
    family_rows = []
    target_names = list(target_names)
    unique_families = sorted({classify_target_family(name) for name in target_names})
    n_targets_total = float(len(target_names))
    for family in unique_families:
        idx = [i for i, name in enumerate(target_names) if classify_target_family(name) == family]
        if not idx:
            continue
        candidate_slice = candidate_terms[idx]
        baseline_slice = baseline_terms[idx]
        candidate_rel_slice = candidate_rel_error[idx]
        baseline_rel_slice = baseline_rel_error[idx]
        family_rows.append(
            {
                "family": family,
                "n_targets": int(len(idx)),
                "candidate_loss_contribution": float(candidate_slice.sum() / n_targets_total),
                "baseline_loss_contribution": float(baseline_slice.sum() / n_targets_total),
                "loss_contribution_delta": float((candidate_slice.sum() - baseline_slice.sum()) / n_targets_total),
                "candidate_mean_weighted_loss": float(candidate_slice.mean()),
                "baseline_mean_weighted_loss": float(baseline_slice.mean()),
                "candidate_mean_unweighted_msre": float(candidate_rel_slice.mean()),
                "baseline_mean_unweighted_msre": float(baseline_rel_slice.mean()),
                "unweighted_msre_delta": float(candidate_rel_slice.mean() - baseline_rel_slice.mean()),
            }
        )
    family_rows.sort(key=lambda row: row["loss_contribution_delta"], reverse=True)
    return family_rows


def compute(dataset_path: str) -> dict[str, float | int]:
    dataset_cls = dataset_from_path(
        dataset_path,
        Path(dataset_path).stem.replace("-", "_"),
    )
    loss_matrix, targets_array = build_loss_matrix(dataset_cls, PERIOD)
    target_names = np.asarray(loss_matrix.columns)
    zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
    bad_mask = np.isin(target_names, BAD_TARGETS)
    keep_mask = ~(zero_mask | bad_mask)

    filtered = loss_matrix.loc[:, keep_mask]
    filtered_targets = np.asarray(targets_array[keep_mask], dtype=np.float64)
    is_national = np.asarray(filtered.columns.str.startswith("nation/"), dtype=bool)
    n_national = int(is_national.sum())
    n_state = int((~is_national).sum())
    if n_national == 0 or n_state == 0:
        raise ValueError(
            "PE-native broad loss requires both national and state targets after filtering"
        )

    normalisation_factor = np.where(
        is_national,
        1.0 / n_national,
        1.0 / n_state,
    ).astype(np.float64)
    inv_mean_normalisation = 1.0 / float(np.mean(normalisation_factor))

    sim = Microsimulation(dataset=dataset_cls)
    sim.default_calculation_period = PERIOD
    weights = sim.calculate(
        "household_weight",
        map_to="household",
        period=PERIOD,
    ).values.astype(np.float64)

    estimate = weights @ filtered.to_numpy(dtype=np.float64)
    rel_error = (((estimate - filtered_targets) + 1.0) / (filtered_targets + 1.0)) ** 2
    weighted_terms = inv_mean_normalisation * rel_error * normalisation_factor
    loss_value = float(weighted_terms.mean())
    unweighted_msre = float(rel_error.mean())

    return {
        "loss": loss_value,
        "unweighted_msre": unweighted_msre,
        "n_targets_total": int(len(target_names)),
        "n_targets_kept": int(keep_mask.sum()),
        "n_targets_zero_dropped": int(zero_mask.sum()),
        "n_targets_bad_dropped": int(bad_mask.sum()),
        "n_national_targets": n_national,
        "n_state_targets": n_state,
        "weight_sum": float(weights.sum()),
        "target_names": filtered.columns.tolist(),
        "weighted_terms": weighted_terms.tolist(),
        "rel_error": rel_error.tolist(),
    }


candidate = compute(CANDIDATE_DATASET)
baseline = compute(BASELINE_DATASET)

if candidate["n_targets_kept"] != baseline["n_targets_kept"]:
    raise ValueError(
        "Candidate and baseline produced different target counts after filtering: "
        f"{candidate['n_targets_kept']} vs {baseline['n_targets_kept']}"
    )
if candidate["target_names"] != baseline["target_names"]:
    raise ValueError("Candidate and baseline produced different target names after filtering")

payload = {
    "metric": "enhanced_cps_native_loss",
    "period": PERIOD,
    "candidate_dataset": CANDIDATE_DATASET,
    "baseline_dataset": BASELINE_DATASET,
    "candidate_enhanced_cps_native_loss": candidate["loss"],
    "baseline_enhanced_cps_native_loss": baseline["loss"],
    "enhanced_cps_native_loss_delta": candidate["loss"] - baseline["loss"],
    "candidate_unweighted_msre": candidate["unweighted_msre"],
    "baseline_unweighted_msre": baseline["unweighted_msre"],
    "unweighted_msre_delta": (
        candidate["unweighted_msre"] - baseline["unweighted_msre"]
    ),
    "n_targets_total": candidate["n_targets_total"],
    "n_targets_kept": candidate["n_targets_kept"],
    "n_targets_zero_dropped": candidate["n_targets_zero_dropped"],
    "n_targets_bad_dropped": candidate["n_targets_bad_dropped"],
    "n_national_targets": candidate["n_national_targets"],
    "n_state_targets": candidate["n_state_targets"],
    "candidate_weight_sum": candidate["weight_sum"],
    "baseline_weight_sum": baseline["weight_sum"],
    "family_breakdown": build_family_breakdown(
        candidate["target_names"],
        np.asarray(candidate["weighted_terms"], dtype=np.float64),
        np.asarray(baseline["weighted_terms"], dtype=np.float64),
        np.asarray(candidate["rel_error"], dtype=np.float64),
        np.asarray(baseline["rel_error"], dtype=np.float64),
    ),
}
print(json.dumps(payload, sort_keys=True))
""".strip()

_PE_NATIVE_BROAD_BATCH_SCORE_SCRIPT = """
import json
import sys
from pathlib import Path

import numpy as np
from policyengine_core.data import Dataset

REPO_ROOT = sys.argv[1]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from policyengine_us import Microsimulation
from policyengine_us_data.utils.loss import build_loss_matrix

BAD_TARGETS = tuple(json.loads(sys.argv[2]))
PERIOD = int(sys.argv[3])
BASELINE_DATASET = sys.argv[4]
CANDIDATE_DATASETS = tuple(json.loads(sys.argv[5]))


def dataset_from_path(dataset_path: str, dataset_name: str):
    class LocalDataset(Dataset):
        name = dataset_name
        label = dataset_name
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = PERIOD

    return LocalDataset


def classify_target_family(target_name: str) -> str:
    parts = target_name.split("/")
    if target_name.startswith("state/census/age/"):
        return "state_age_distribution"
    if target_name.startswith("state/census/population_by_state/"):
        return "state_population"
    if target_name.startswith("state/census/population_under_5_by_state/"):
        return "state_population_under_5"
    if target_name.startswith("nation/irs/aca_spending/"):
        return "state_aca_spending"
    if target_name.startswith("state/irs/aca_enrollment/"):
        return "state_aca_enrollment"
    if target_name.startswith("irs/medicaid_enrollment/"):
        return "state_medicaid_enrollment"
    if target_name.endswith("/snap-cost"):
        return "state_snap_cost"
    if target_name.endswith("/snap-hhs"):
        return "state_snap_households"
    if target_name.startswith("state/real_estate_taxes/"):
        return "state_real_estate_taxes"
    if len(parts) >= 3 and parts[0] == "state" and parts[2] == "adjusted_gross_income":
        return "state_agi_distribution"
    if target_name.startswith("nation/jct/"):
        return "national_tax_expenditures"
    if target_name.startswith("nation/net_worth/"):
        return "national_net_worth"
    if target_name.startswith("nation/ssa/"):
        return "national_ssa"
    if target_name.startswith("nation/census/population_by_age/"):
        return "national_population_by_age"
    if target_name == "nation/census/infants":
        return "national_infants"
    if target_name.startswith("nation/census/agi_in_spm_threshold_decile_"):
        return "national_spm_threshold_agi"
    if target_name.startswith("nation/census/count_in_spm_threshold_decile_"):
        return "national_spm_threshold_count"
    if target_name.startswith("nation/census/"):
        return "national_census_other"
    if target_name.startswith("nation/irs/"):
        return "national_irs_other"
    return "other"


def build_family_breakdown(target_names, candidate_terms, baseline_terms, candidate_rel_error, baseline_rel_error):
    family_rows = []
    target_names = list(target_names)
    unique_families = sorted({classify_target_family(name) for name in target_names})
    n_targets_total = float(len(target_names))
    for family in unique_families:
        idx = [i for i, name in enumerate(target_names) if classify_target_family(name) == family]
        if not idx:
            continue
        candidate_slice = candidate_terms[idx]
        baseline_slice = baseline_terms[idx]
        candidate_rel_slice = candidate_rel_error[idx]
        baseline_rel_slice = baseline_rel_error[idx]
        family_rows.append(
            {
                "family": family,
                "n_targets": int(len(idx)),
                "candidate_loss_contribution": float(candidate_slice.sum() / n_targets_total),
                "baseline_loss_contribution": float(baseline_slice.sum() / n_targets_total),
                "loss_contribution_delta": float((candidate_slice.sum() - baseline_slice.sum()) / n_targets_total),
                "candidate_mean_weighted_loss": float(candidate_slice.mean()),
                "baseline_mean_weighted_loss": float(baseline_slice.mean()),
                "candidate_mean_unweighted_msre": float(candidate_rel_slice.mean()),
                "baseline_mean_unweighted_msre": float(baseline_rel_slice.mean()),
                "unweighted_msre_delta": float(candidate_rel_slice.mean() - baseline_rel_slice.mean()),
            }
        )
    family_rows.sort(key=lambda row: row["loss_contribution_delta"], reverse=True)
    return family_rows


def compute(dataset_path: str) -> dict[str, float | int]:
    dataset_cls = dataset_from_path(
        dataset_path,
        Path(dataset_path).stem.replace("-", "_"),
    )
    loss_matrix, targets_array = build_loss_matrix(dataset_cls, PERIOD)
    target_names = np.asarray(loss_matrix.columns)
    zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
    bad_mask = np.isin(target_names, BAD_TARGETS)
    keep_mask = ~(zero_mask | bad_mask)

    filtered = loss_matrix.loc[:, keep_mask]
    filtered_targets = np.asarray(targets_array[keep_mask], dtype=np.float64)
    is_national = np.asarray(filtered.columns.str.startswith("nation/"), dtype=bool)
    n_national = int(is_national.sum())
    n_state = int((~is_national).sum())
    if n_national == 0 or n_state == 0:
        raise ValueError(
            "PE-native broad loss requires both national and state targets after filtering"
        )

    normalisation_factor = np.where(
        is_national,
        1.0 / n_national,
        1.0 / n_state,
    ).astype(np.float64)
    inv_mean_normalisation = 1.0 / float(np.mean(normalisation_factor))

    sim = Microsimulation(dataset=dataset_cls)
    sim.default_calculation_period = PERIOD
    weights = sim.calculate(
        "household_weight",
        map_to="household",
        period=PERIOD,
    ).values.astype(np.float64)

    estimate = weights @ filtered.to_numpy(dtype=np.float64)
    rel_error = (((estimate - filtered_targets) + 1.0) / (filtered_targets + 1.0)) ** 2
    weighted_terms = inv_mean_normalisation * rel_error * normalisation_factor
    loss_value = float(weighted_terms.mean())
    unweighted_msre = float(rel_error.mean())

    return {
        "dataset": dataset_path,
        "loss": loss_value,
        "unweighted_msre": unweighted_msre,
        "n_targets_total": int(len(target_names)),
        "n_targets_kept": int(keep_mask.sum()),
        "n_targets_zero_dropped": int(zero_mask.sum()),
        "n_targets_bad_dropped": int(bad_mask.sum()),
        "n_national_targets": n_national,
        "n_state_targets": n_state,
        "weight_sum": float(weights.sum()),
        "target_names": filtered.columns.tolist(),
        "weighted_terms": weighted_terms.tolist(),
        "rel_error": rel_error.tolist(),
    }


baseline = compute(BASELINE_DATASET)
payload = []
for candidate_dataset in CANDIDATE_DATASETS:
    candidate = compute(candidate_dataset)
    if candidate["n_targets_kept"] != baseline["n_targets_kept"]:
        raise ValueError(
            "Candidate and baseline produced different target counts after filtering: "
            f"{candidate['n_targets_kept']} vs {baseline['n_targets_kept']}"
        )
    if candidate["target_names"] != baseline["target_names"]:
        raise ValueError("Candidate and baseline produced different target names after filtering")
    payload.append(
        {
            "metric": "enhanced_cps_native_loss",
            "period": PERIOD,
            "candidate_dataset": candidate_dataset,
            "baseline_dataset": BASELINE_DATASET,
            "candidate_enhanced_cps_native_loss": candidate["loss"],
            "baseline_enhanced_cps_native_loss": baseline["loss"],
            "enhanced_cps_native_loss_delta": candidate["loss"] - baseline["loss"],
            "candidate_beats_baseline": candidate["loss"] < baseline["loss"],
            "candidate_unweighted_msre": candidate["unweighted_msre"],
            "baseline_unweighted_msre": baseline["unweighted_msre"],
            "unweighted_msre_delta": (
                candidate["unweighted_msre"] - baseline["unweighted_msre"]
            ),
            "n_targets_total": candidate["n_targets_total"],
            "n_targets_kept": candidate["n_targets_kept"],
            "n_targets_zero_dropped": candidate["n_targets_zero_dropped"],
            "n_targets_bad_dropped": candidate["n_targets_bad_dropped"],
            "n_national_targets": candidate["n_national_targets"],
            "n_state_targets": candidate["n_state_targets"],
            "candidate_weight_sum": candidate["weight_sum"],
            "baseline_weight_sum": baseline["weight_sum"],
            "family_breakdown": build_family_breakdown(
                candidate["target_names"],
                np.asarray(candidate["weighted_terms"], dtype=np.float64),
                np.asarray(baseline["weighted_terms"], dtype=np.float64),
                np.asarray(candidate["rel_error"], dtype=np.float64),
                np.asarray(baseline["rel_error"], dtype=np.float64),
            ),
        }
    )
print(json.dumps(payload, sort_keys=True))
""".strip()

_PE_NATIVE_TARGET_DELTA_SCRIPT = """
import json
import sys
from pathlib import Path

import numpy as np
from policyengine_core.data import Dataset

REPO_ROOT = sys.argv[1]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from policyengine_us import Microsimulation
from policyengine_us_data.utils.loss import build_loss_matrix

BAD_TARGETS = tuple(json.loads(sys.argv[2]))
PERIOD = int(sys.argv[3])
FROM_DATASET = sys.argv[4]
TO_DATASET = sys.argv[5]
TOP_K = int(sys.argv[6])


def dataset_from_path(dataset_path: str, dataset_name: str):
    class LocalDataset(Dataset):
        name = dataset_name
        label = dataset_name
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = PERIOD

    return LocalDataset


def compute(dataset_path: str):
    dataset_cls = dataset_from_path(
        dataset_path,
        Path(dataset_path).stem.replace("-", "_"),
    )
    loss_matrix, targets_array = build_loss_matrix(dataset_cls, PERIOD)
    target_names = np.asarray(loss_matrix.columns)
    zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
    bad_mask = np.isin(target_names, BAD_TARGETS)
    keep_mask = ~(zero_mask | bad_mask)

    filtered = loss_matrix.loc[:, keep_mask]
    filtered_targets = np.asarray(targets_array[keep_mask], dtype=np.float64)
    is_national = np.asarray(filtered.columns.str.startswith("nation/"), dtype=bool)
    n_national = int(is_national.sum())
    n_state = int((~is_national).sum())
    if n_national == 0 or n_state == 0:
        raise ValueError(
            "PE-native broad loss requires both national and state targets after filtering"
        )

    normalisation_factor = np.where(
        is_national,
        1.0 / n_national,
        1.0 / n_state,
    ).astype(np.float64)
    inv_mean_normalisation = 1.0 / float(np.mean(normalisation_factor))

    sim = Microsimulation(dataset=dataset_cls)
    sim.default_calculation_period = PERIOD
    weights = sim.calculate(
        "household_weight",
        map_to="household",
        period=PERIOD,
    ).values.astype(np.float64)

    estimate = weights @ filtered.to_numpy(dtype=np.float64)
    rel_error = (((estimate - filtered_targets) + 1.0) / (filtered_targets + 1.0)) ** 2
    weighted_terms = inv_mean_normalisation * rel_error * normalisation_factor
    return {
        "target_names": filtered.columns.tolist(),
        "targets": filtered_targets.tolist(),
        "estimate": estimate.tolist(),
        "rel_error": rel_error.tolist(),
        "weighted_terms": weighted_terms.tolist(),
    }


from_payload = compute(FROM_DATASET)
to_payload = compute(TO_DATASET)

if from_payload["target_names"] != to_payload["target_names"]:
    raise ValueError("Datasets produced different target names after filtering")

rows = []
for idx, name in enumerate(from_payload["target_names"]):
    from_term = float(from_payload["weighted_terms"][idx])
    to_term = float(to_payload["weighted_terms"][idx])
    rows.append(
        {
            "target_name": name,
            "weighted_term_delta": to_term - from_term,
            "from_weighted_term": from_term,
            "to_weighted_term": to_term,
            "target_value": float(from_payload["targets"][idx]),
            "from_estimate": float(from_payload["estimate"][idx]),
            "to_estimate": float(to_payload["estimate"][idx]),
            "from_rel_error": float(from_payload["rel_error"][idx]),
            "to_rel_error": float(to_payload["rel_error"][idx]),
        }
    )

rows.sort(key=lambda row: row["weighted_term_delta"], reverse=True)
payload = {
    "metric": "enhanced_cps_native_loss_target_delta",
    "period": PERIOD,
    "from_dataset": FROM_DATASET,
    "to_dataset": TO_DATASET,
    "top_regressions": rows[:TOP_K],
    "top_improvements": list(reversed(rows[-TOP_K:])),
}
print(json.dumps(payload, sort_keys=True))
""".strip()

_PE_NATIVE_TARGET_DELTA_BATCH_SCRIPT = """
import json
import sys
from pathlib import Path

import numpy as np
from policyengine_core.data import Dataset

REPO_ROOT = sys.argv[1]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from policyengine_us import Microsimulation
from policyengine_us_data.utils.loss import build_loss_matrix

BAD_TARGETS = tuple(json.loads(sys.argv[2]))
PERIOD = int(sys.argv[3])
BASELINE_DATASET = sys.argv[4]
CANDIDATE_DATASETS = json.loads(sys.argv[5])
TOP_K = int(sys.argv[6])


def dataset_from_path(dataset_path: str, dataset_name: str):
    class LocalDataset(Dataset):
        name = dataset_name
        label = dataset_name
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = PERIOD

    return LocalDataset


def compute(dataset_path: str):
    dataset_cls = dataset_from_path(
        dataset_path,
        Path(dataset_path).stem.replace("-", "_"),
    )
    loss_matrix, targets_array = build_loss_matrix(dataset_cls, PERIOD)
    target_names = np.asarray(loss_matrix.columns)
    zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
    bad_mask = np.isin(target_names, BAD_TARGETS)
    keep_mask = ~(zero_mask | bad_mask)

    filtered = loss_matrix.loc[:, keep_mask]
    filtered_targets = np.asarray(targets_array[keep_mask], dtype=np.float64)
    is_national = np.asarray(filtered.columns.str.startswith("nation/"), dtype=bool)
    n_national = int(is_national.sum())
    n_state = int((~is_national).sum())
    if n_national == 0 or n_state == 0:
        raise ValueError(
            "PE-native broad loss requires both national and state targets after filtering"
        )

    normalisation_factor = np.where(
        is_national,
        1.0 / n_national,
        1.0 / n_state,
    ).astype(np.float64)
    inv_mean_normalisation = 1.0 / float(np.mean(normalisation_factor))

    sim = Microsimulation(dataset=dataset_cls)
    sim.default_calculation_period = PERIOD
    weights = sim.calculate(
        "household_weight",
        map_to="household",
        period=PERIOD,
    ).values.astype(np.float64)

    estimate = weights @ filtered.to_numpy(dtype=np.float64)
    rel_error = (((estimate - filtered_targets) + 1.0) / (filtered_targets + 1.0)) ** 2
    weighted_terms = inv_mean_normalisation * rel_error * normalisation_factor
    return {
        "target_names": filtered.columns.tolist(),
        "targets": filtered_targets.tolist(),
        "estimate": estimate.tolist(),
        "rel_error": rel_error.tolist(),
        "weighted_terms": weighted_terms.tolist(),
    }


baseline_payload = compute(BASELINE_DATASET)
results = []
for candidate_dataset in CANDIDATE_DATASETS:
    candidate_payload = compute(candidate_dataset)
    if baseline_payload["target_names"] != candidate_payload["target_names"]:
        raise ValueError("Datasets produced different target names after filtering")

    rows = []
    for idx, name in enumerate(baseline_payload["target_names"]):
        from_term = float(baseline_payload["weighted_terms"][idx])
        to_term = float(candidate_payload["weighted_terms"][idx])
        rows.append(
            {
                "target_name": name,
                "weighted_term_delta": to_term - from_term,
                "from_weighted_term": from_term,
                "to_weighted_term": to_term,
                "target_value": float(baseline_payload["targets"][idx]),
                "from_estimate": float(baseline_payload["estimate"][idx]),
                "to_estimate": float(candidate_payload["estimate"][idx]),
                "from_rel_error": float(baseline_payload["rel_error"][idx]),
                "to_rel_error": float(candidate_payload["rel_error"][idx]),
            }
        )

    rows.sort(key=lambda row: row["weighted_term_delta"], reverse=True)
    results.append(
        {
            "metric": "enhanced_cps_native_loss_target_delta",
            "period": PERIOD,
            "from_dataset": BASELINE_DATASET,
            "to_dataset": candidate_dataset,
            "top_regressions": rows[:TOP_K],
            "top_improvements": list(reversed(rows[-TOP_K:])),
        }
    )

print(json.dumps(results, sort_keys=True))
""".strip()

_PE_NATIVE_SUPPORT_AUDIT_SCRIPT = """
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from policyengine_core.data import Dataset

REPO_ROOT = sys.argv[1]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from policyengine_us import Microsimulation

PERIOD = int(sys.argv[2])
CANDIDATE_DATASET = sys.argv[3]
BASELINE_DATASET = sys.argv[4]

STATE_FIPS_TO_ABBR = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}
CRITICAL_PERSON_VARIABLES = (
    "has_marketplace_health_coverage",
    "has_esi",
    "medicare_part_b_premiums",
    "child_support_expense",
    "self_employment_income_before_lsr",
    "rental_income",
    "non_sch_d_capital_gains",
)
HIGH_SIGNAL_MFS_AGI_BINS = (
    ("75k_to_100k", 75_000.0, 100_000.0),
    ("100k_to_200k", 100_000.0, 200_000.0),
    ("200k_to_500k", 200_000.0, 500_000.0),
    ("500k_plus", 500_000.0, np.inf),
)
AGE_BUCKETS = (
    ("0_to_4", 0, 5),
    ("5_to_17", 5, 18),
    ("18_to_29", 18, 30),
    ("30_to_44", 30, 45),
    ("45_to_64", 45, 65),
    ("65_plus", 65, np.inf),
)


def dataset_from_path(dataset_path: str, dataset_name: str):
    class LocalDataset(Dataset):
        name = dataset_name
        label = dataset_name
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = PERIOD

    return LocalDataset


def stored_variables_for(dataset_path: str) -> set[str]:
    with h5py.File(dataset_path, "r") as handle:
        return set(handle.keys())


def state_abbr(value) -> str:
    if value is None:
        return "NA"
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return str(value)
    return STATE_FIPS_TO_ABBR.get(numeric, str(numeric))


def normalize_status(value) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    normalized = text.strip().upper().replace(" ", "_")
    if normalized in {
        "SINGLE",
        "JOINT",
        "SEPARATE",
        "HEAD_OF_HOUSEHOLD",
        "SURVIVING_SPOUSE",
    }:
        return normalized
    return normalized


def summarize_numeric(values, weights, *, stored: bool) -> dict[str, float | int | bool]:
    arr = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    w = np.asarray(weights, dtype=np.float64)
    positive = arr > 0.0
    negative = arr < 0.0
    nonzero = arr != 0.0
    return {
        "stored": bool(stored),
        "nonzero_count": int(nonzero.sum()),
        "positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
        "weighted_nonzero": float(w[nonzero].sum()),
        "weighted_positive": float(w[positive].sum()),
        "weighted_negative": float(w[negative].sum()),
        "value_sum": float((arr * w).sum()),
    }


def summarize_bool(values, weights, *, stored: bool) -> dict[str, float | int | bool]:
    arr = np.asarray(values).astype(bool)
    w = np.asarray(weights, dtype=np.float64)
    return {
        "stored": bool(stored),
        "true_count": int(arr.sum()),
        "false_count": int((~arr).sum()),
        "weighted_true": float(w[arr].sum()),
        "weighted_false": float(w[~arr].sum()),
    }


def build_snapshot(dataset_path: str) -> dict:
    dataset_cls = dataset_from_path(
        dataset_path,
        Path(dataset_path).stem.replace("-", "_"),
    )
    stored_variables = stored_variables_for(dataset_path)
    sim = Microsimulation(dataset=dataset_cls)
    sim.default_calculation_period = PERIOD

    person_weights = sim.calculate("person_weight", period=PERIOD).values.astype(np.float64)
    tax_unit_weights = sim.calculate("tax_unit_weight", period=PERIOD).values.astype(np.float64)
    person_state = sim.calculate("state_fips", map_to="person", period=PERIOD).values
    person_age = sim.calculate("age", period=PERIOD).values.astype(np.float64)
    marketplace = sim.calculate("has_marketplace_health_coverage", period=PERIOD).values
    filing_status = sim.calculate("filing_status", period=PERIOD).values
    adjusted_gross_income = sim.calculate("adjusted_gross_income", period=PERIOD).values.astype(np.float64)

    critical_support = {}
    for variable in CRITICAL_PERSON_VARIABLES:
        values = sim.calculate(variable, period=PERIOD).values
        if np.asarray(values).dtype == np.bool_:
            critical_support[variable] = summarize_bool(
                values,
                person_weights,
                stored=variable in stored_variables,
            )
        else:
            critical_support[variable] = summarize_numeric(
                values,
                person_weights,
                stored=variable in stored_variables,
            )

    filing_status_counts = {}
    for status in ("SINGLE", "JOINT", "SEPARATE", "HEAD_OF_HOUSEHOLD", "SURVIVING_SPOUSE"):
        mask = np.asarray([normalize_status(value) == status for value in filing_status], dtype=bool)
        filing_status_counts[status] = {
            "count": int(mask.sum()),
            "weighted_count": float(tax_unit_weights[mask].sum()),
        }

    mfs_mask = np.asarray([normalize_status(value) == "SEPARATE" for value in filing_status], dtype=bool)
    mfs_agi_support = []
    for label, lower, upper in HIGH_SIGNAL_MFS_AGI_BINS:
        mask = mfs_mask & (adjusted_gross_income >= lower) & (adjusted_gross_income < upper)
        mfs_agi_support.append(
            {
                "agi_bin": label,
                "count": int(mask.sum()),
                "weighted_count": float(tax_unit_weights[mask].sum()),
                "weighted_agi": float((adjusted_gross_income[mask] * tax_unit_weights[mask]).sum()),
            }
        )

    states = sorted({state_abbr(value) for value in person_state})
    state_marketplace = {}
    state_age_bucket = {}
    marketplace_bool = np.asarray(marketplace).astype(bool)
    for state in states:
        state_mask = np.asarray([state_abbr(value) == state for value in person_state], dtype=bool)
        enrolled = state_mask & marketplace_bool
        state_marketplace[state] = {
            "weighted_people": float(person_weights[state_mask].sum()),
            "weighted_marketplace_enrollment": float(person_weights[enrolled].sum()),
        }
        bucket_weights = {}
        nonempty = 0
        for label, lower, upper in AGE_BUCKETS:
            mask = state_mask & (person_age >= lower) & (person_age < upper)
            weight = float(person_weights[mask].sum())
            bucket_weights[label] = weight
            if weight > 0.0:
                nonempty += 1
        state_age_bucket[state] = {
            "nonempty_buckets": int(nonempty),
            "bucket_weights": bucket_weights,
        }

    return {
        "dataset": dataset_path,
        "stored_variable_count": int(len(stored_variables)),
        "stored_variables": sorted(stored_variables),
        "critical_input_support": critical_support,
        "filing_status_weighted_counts": filing_status_counts,
        "mfs_high_agi_support": mfs_agi_support,
        "state_marketplace_enrollment": state_marketplace,
        "state_age_bucket_support": state_age_bucket,
    }


def compare_snapshots(candidate: dict, baseline: dict) -> dict:
    critical_rows = []
    for variable in CRITICAL_PERSON_VARIABLES:
        candidate_row = candidate["critical_input_support"][variable]
        baseline_row = baseline["critical_input_support"][variable]
        candidate_weighted = candidate_row.get("weighted_nonzero", candidate_row.get("weighted_true", 0.0))
        baseline_weighted = baseline_row.get("weighted_nonzero", baseline_row.get("weighted_true", 0.0))
        critical_rows.append(
            {
                "variable": variable,
                "candidate_stored": bool(candidate_row.get("stored", False)),
                "baseline_stored": bool(baseline_row.get("stored", False)),
                "candidate_weighted_nonzero": float(candidate_weighted),
                "baseline_weighted_nonzero": float(baseline_weighted),
                "weighted_nonzero_delta": float(candidate_weighted - baseline_weighted),
            }
        )

    filing_status_rows = []
    for status in ("SINGLE", "JOINT", "SEPARATE", "HEAD_OF_HOUSEHOLD", "SURVIVING_SPOUSE"):
        candidate_row = candidate["filing_status_weighted_counts"][status]
        baseline_row = baseline["filing_status_weighted_counts"][status]
        filing_status_rows.append(
            {
                "filing_status": status,
                "candidate_weighted_count": float(candidate_row["weighted_count"]),
                "baseline_weighted_count": float(baseline_row["weighted_count"]),
                "weighted_count_delta": float(candidate_row["weighted_count"] - baseline_row["weighted_count"]),
            }
        )

    baseline_bins = {row["agi_bin"]: row for row in baseline["mfs_high_agi_support"]}
    mfs_rows = []
    for row in candidate["mfs_high_agi_support"]:
        other = baseline_bins[row["agi_bin"]]
        mfs_rows.append(
            {
                "agi_bin": row["agi_bin"],
                "candidate_weighted_count": float(row["weighted_count"]),
                "baseline_weighted_count": float(other["weighted_count"]),
                "weighted_count_delta": float(row["weighted_count"] - other["weighted_count"]),
                "candidate_weighted_agi": float(row["weighted_agi"]),
                "baseline_weighted_agi": float(other["weighted_agi"]),
                "weighted_agi_delta": float(row["weighted_agi"] - other["weighted_agi"]),
            }
        )

    all_states = sorted(
        set(candidate["state_marketplace_enrollment"])
        | set(baseline["state_marketplace_enrollment"])
    )
    state_marketplace_rows = []
    for state in all_states:
        candidate_row = candidate["state_marketplace_enrollment"].get(
            state,
            {"weighted_marketplace_enrollment": 0.0},
        )
        baseline_row = baseline["state_marketplace_enrollment"].get(
            state,
            {"weighted_marketplace_enrollment": 0.0},
        )
        state_marketplace_rows.append(
            {
                "state": state,
                "candidate_weighted_marketplace_enrollment": float(candidate_row["weighted_marketplace_enrollment"]),
                "baseline_weighted_marketplace_enrollment": float(baseline_row["weighted_marketplace_enrollment"]),
                "weighted_marketplace_enrollment_delta": float(
                    candidate_row["weighted_marketplace_enrollment"]
                    - baseline_row["weighted_marketplace_enrollment"]
                ),
            }
        )
    state_marketplace_rows.sort(
        key=lambda row: abs(row["weighted_marketplace_enrollment_delta"]),
        reverse=True,
    )

    all_states = sorted(
        set(candidate["state_age_bucket_support"])
        | set(baseline["state_age_bucket_support"])
    )
    state_age_rows = []
    for state in all_states:
        candidate_row = candidate["state_age_bucket_support"].get(
            state,
            {"bucket_weights": {}},
        )
        baseline_row = baseline["state_age_bucket_support"].get(
            state,
            {"bucket_weights": {}},
        )
        for label, _lower, _upper in AGE_BUCKETS:
            candidate_weight = float(candidate_row["bucket_weights"].get(label, 0.0))
            baseline_weight = float(baseline_row["bucket_weights"].get(label, 0.0))
            state_age_rows.append(
                {
                    "state": state,
                    "age_bucket": label,
                    "candidate_weight": candidate_weight,
                    "baseline_weight": baseline_weight,
                    "weight_delta": candidate_weight - baseline_weight,
                }
            )
    state_age_rows.sort(key=lambda row: abs(row["weight_delta"]), reverse=True)

    return {
        "critical_input_support": critical_rows,
        "filing_status_weighted_delta": filing_status_rows,
        "mfs_high_agi_delta": mfs_rows,
        "state_marketplace_enrollment_top_gaps": state_marketplace_rows[:15],
        "state_age_bucket_top_gaps": state_age_rows[:20],
    }


candidate = build_snapshot(CANDIDATE_DATASET)
baseline = build_snapshot(BASELINE_DATASET)
payload = {
    "metric": "enhanced_cps_support_audit",
    "period": PERIOD,
    "candidate_dataset": CANDIDATE_DATASET,
    "baseline_dataset": BASELINE_DATASET,
    "candidate": candidate,
    "baseline": baseline,
    "comparisons": compare_snapshots(candidate, baseline),
}
print(json.dumps(payload, sort_keys=True))
""".strip()

_PE_NATIVE_SUPPORT_AUDIT_BATCH_SCRIPT = """
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from policyengine_core.data import Dataset

REPO_ROOT = sys.argv[1]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from policyengine_us import Microsimulation

PERIOD = int(sys.argv[2])
BASELINE_DATASET = sys.argv[3]
CANDIDATE_DATASETS = json.loads(sys.argv[4])

STATE_FIPS_TO_ABBR = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}
CRITICAL_PERSON_VARIABLES = (
    "has_marketplace_health_coverage",
    "has_esi",
    "medicare_part_b_premiums",
    "child_support_expense",
    "self_employment_income_before_lsr",
    "rental_income",
    "non_sch_d_capital_gains",
)
HIGH_SIGNAL_MFS_AGI_BINS = (
    ("75k_to_100k", 75_000.0, 100_000.0),
    ("100k_to_200k", 100_000.0, 200_000.0),
    ("200k_to_500k", 200_000.0, 500_000.0),
    ("500k_plus", 500_000.0, np.inf),
)
AGE_BUCKETS = (
    ("0_to_4", 0, 5),
    ("5_to_17", 5, 18),
    ("18_to_29", 18, 30),
    ("30_to_44", 30, 45),
    ("45_to_64", 45, 65),
    ("65_plus", 65, np.inf),
)


def dataset_from_path(dataset_path: str, dataset_name: str):
    class LocalDataset(Dataset):
        name = dataset_name
        label = dataset_name
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = PERIOD

    return LocalDataset


def stored_variables_for(dataset_path: str) -> set[str]:
    with h5py.File(dataset_path, "r") as handle:
        return set(handle.keys())


def state_abbr(value) -> str:
    if value is None:
        return "NA"
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return str(value)
    return STATE_FIPS_TO_ABBR.get(numeric, str(numeric))


def normalize_status(value) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    normalized = text.strip().upper().replace(" ", "_")
    if normalized in {
        "SINGLE",
        "JOINT",
        "SEPARATE",
        "HEAD_OF_HOUSEHOLD",
        "SURVIVING_SPOUSE",
    }:
        return normalized
    return normalized


def summarize_numeric(values, weights, *, stored: bool) -> dict[str, float | int | bool]:
    arr = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    w = np.asarray(weights, dtype=np.float64)
    positive = arr > 0.0
    negative = arr < 0.0
    nonzero = arr != 0.0
    return {
        "stored": bool(stored),
        "nonzero_count": int(nonzero.sum()),
        "positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
        "weighted_nonzero": float(w[nonzero].sum()),
        "weighted_positive": float(w[positive].sum()),
        "weighted_negative": float(w[negative].sum()),
        "value_sum": float((arr * w).sum()),
    }


def summarize_bool(values, weights, *, stored: bool) -> dict[str, float | int | bool]:
    arr = np.asarray(values).astype(bool)
    w = np.asarray(weights, dtype=np.float64)
    return {
        "stored": bool(stored),
        "true_count": int(arr.sum()),
        "false_count": int((~arr).sum()),
        "weighted_true": float(w[arr].sum()),
        "weighted_false": float(w[~arr].sum()),
    }


def build_snapshot(dataset_path: str) -> dict:
    dataset_cls = dataset_from_path(
        dataset_path,
        Path(dataset_path).stem.replace("-", "_"),
    )
    stored_variables = stored_variables_for(dataset_path)
    sim = Microsimulation(dataset=dataset_cls)
    sim.default_calculation_period = PERIOD

    person_weights = sim.calculate("person_weight", period=PERIOD).values.astype(np.float64)
    tax_unit_weights = sim.calculate("tax_unit_weight", period=PERIOD).values.astype(np.float64)
    person_state = sim.calculate("state_fips", map_to="person", period=PERIOD).values
    person_age = sim.calculate("age", period=PERIOD).values.astype(np.float64)
    marketplace = sim.calculate("has_marketplace_health_coverage", period=PERIOD).values
    filing_status = sim.calculate("filing_status", period=PERIOD).values
    adjusted_gross_income = sim.calculate("adjusted_gross_income", period=PERIOD).values.astype(np.float64)

    critical_support = {}
    for variable in CRITICAL_PERSON_VARIABLES:
        values = sim.calculate(variable, period=PERIOD).values
        if np.asarray(values).dtype == np.bool_:
            critical_support[variable] = summarize_bool(
                values,
                person_weights,
                stored=variable in stored_variables,
            )
        else:
            critical_support[variable] = summarize_numeric(
                values,
                person_weights,
                stored=variable in stored_variables,
            )

    filing_status_counts = {}
    for status in ("SINGLE", "JOINT", "SEPARATE", "HEAD_OF_HOUSEHOLD", "SURVIVING_SPOUSE"):
        mask = np.asarray([normalize_status(value) == status for value in filing_status], dtype=bool)
        filing_status_counts[status] = {
            "count": int(mask.sum()),
            "weighted_count": float(tax_unit_weights[mask].sum()),
        }

    mfs_mask = np.asarray([normalize_status(value) == "SEPARATE" for value in filing_status], dtype=bool)
    mfs_agi_support = []
    for label, lower, upper in HIGH_SIGNAL_MFS_AGI_BINS:
        mask = mfs_mask & (adjusted_gross_income >= lower) & (adjusted_gross_income < upper)
        mfs_agi_support.append(
            {
                "agi_bin": label,
                "count": int(mask.sum()),
                "weighted_count": float(tax_unit_weights[mask].sum()),
                "weighted_agi": float((adjusted_gross_income[mask] * tax_unit_weights[mask]).sum()),
            }
        )

    states = sorted({state_abbr(value) for value in person_state})
    state_marketplace = {}
    state_age_bucket = {}
    marketplace_bool = np.asarray(marketplace).astype(bool)
    for state in states:
        state_mask = np.asarray([state_abbr(value) == state for value in person_state], dtype=bool)
        enrolled = state_mask & marketplace_bool
        state_marketplace[state] = {
            "weighted_people": float(person_weights[state_mask].sum()),
            "weighted_marketplace_enrollment": float(person_weights[enrolled].sum()),
        }
        bucket_weights = {}
        nonempty = 0
        for label, lower, upper in AGE_BUCKETS:
            mask = state_mask & (person_age >= lower) & (person_age < upper)
            weight = float(person_weights[mask].sum())
            bucket_weights[label] = weight
            if weight > 0.0:
                nonempty += 1
        state_age_bucket[state] = {
            "nonempty_buckets": int(nonempty),
            "bucket_weights": bucket_weights,
        }

    return {
        "dataset": dataset_path,
        "stored_variable_count": int(len(stored_variables)),
        "stored_variables": sorted(stored_variables),
        "critical_input_support": critical_support,
        "filing_status_weighted_counts": filing_status_counts,
        "mfs_high_agi_support": mfs_agi_support,
        "state_marketplace_enrollment": state_marketplace,
        "state_age_bucket_support": state_age_bucket,
    }


def compare_snapshots(candidate: dict, baseline: dict) -> dict:
    critical_rows = []
    for variable in CRITICAL_PERSON_VARIABLES:
        candidate_row = candidate["critical_input_support"][variable]
        baseline_row = baseline["critical_input_support"][variable]
        candidate_weighted = candidate_row.get("weighted_nonzero", candidate_row.get("weighted_true", 0.0))
        baseline_weighted = baseline_row.get("weighted_nonzero", baseline_row.get("weighted_true", 0.0))
        critical_rows.append(
            {
                "variable": variable,
                "candidate_stored": bool(candidate_row.get("stored", False)),
                "baseline_stored": bool(baseline_row.get("stored", False)),
                "candidate_weighted_nonzero": float(candidate_weighted),
                "baseline_weighted_nonzero": float(baseline_weighted),
                "weighted_nonzero_delta": float(candidate_weighted - baseline_weighted),
            }
        )

    filing_status_rows = []
    for status in ("SINGLE", "JOINT", "SEPARATE", "HEAD_OF_HOUSEHOLD", "SURVIVING_SPOUSE"):
        candidate_row = candidate["filing_status_weighted_counts"][status]
        baseline_row = baseline["filing_status_weighted_counts"][status]
        filing_status_rows.append(
            {
                "filing_status": status,
                "candidate_weighted_count": float(candidate_row["weighted_count"]),
                "baseline_weighted_count": float(baseline_row["weighted_count"]),
                "weighted_count_delta": float(candidate_row["weighted_count"] - baseline_row["weighted_count"]),
            }
        )

    baseline_bins = {row["agi_bin"]: row for row in baseline["mfs_high_agi_support"]}
    mfs_rows = []
    for row in candidate["mfs_high_agi_support"]:
        other = baseline_bins[row["agi_bin"]]
        mfs_rows.append(
            {
                "agi_bin": row["agi_bin"],
                "candidate_weighted_count": float(row["weighted_count"]),
                "baseline_weighted_count": float(other["weighted_count"]),
                "weighted_count_delta": float(row["weighted_count"] - other["weighted_count"]),
                "candidate_weighted_agi": float(row["weighted_agi"]),
                "baseline_weighted_agi": float(other["weighted_agi"]),
                "weighted_agi_delta": float(row["weighted_agi"] - other["weighted_agi"]),
            }
        )

    all_states = sorted(
        set(candidate["state_marketplace_enrollment"])
        | set(baseline["state_marketplace_enrollment"])
    )
    state_marketplace_rows = []
    for state in all_states:
        candidate_row = candidate["state_marketplace_enrollment"].get(
            state,
            {"weighted_marketplace_enrollment": 0.0},
        )
        baseline_row = baseline["state_marketplace_enrollment"].get(
            state,
            {"weighted_marketplace_enrollment": 0.0},
        )
        state_marketplace_rows.append(
            {
                "state": state,
                "candidate_weighted_marketplace_enrollment": float(candidate_row["weighted_marketplace_enrollment"]),
                "baseline_weighted_marketplace_enrollment": float(baseline_row["weighted_marketplace_enrollment"]),
                "weighted_marketplace_enrollment_delta": float(
                    candidate_row["weighted_marketplace_enrollment"]
                    - baseline_row["weighted_marketplace_enrollment"]
                ),
            }
        )
    state_marketplace_rows.sort(
        key=lambda row: abs(row["weighted_marketplace_enrollment_delta"]),
        reverse=True,
    )

    all_states = sorted(
        set(candidate["state_age_bucket_support"])
        | set(baseline["state_age_bucket_support"])
    )
    state_age_rows = []
    for state in all_states:
        candidate_row = candidate["state_age_bucket_support"].get(
            state,
            {"bucket_weights": {}},
        )
        baseline_row = baseline["state_age_bucket_support"].get(
            state,
            {"bucket_weights": {}},
        )
        for label, _lower, _upper in AGE_BUCKETS:
            candidate_weight = float(candidate_row["bucket_weights"].get(label, 0.0))
            baseline_weight = float(baseline_row["bucket_weights"].get(label, 0.0))
            state_age_rows.append(
                {
                    "state": state,
                    "age_bucket": label,
                    "candidate_weight": candidate_weight,
                    "baseline_weight": baseline_weight,
                    "weight_delta": candidate_weight - baseline_weight,
                }
            )
    state_age_rows.sort(key=lambda row: abs(row["weight_delta"]), reverse=True)

    return {
        "critical_input_support": critical_rows,
        "filing_status_weighted_delta": filing_status_rows,
        "mfs_high_agi_delta": mfs_rows,
        "state_marketplace_enrollment_top_gaps": state_marketplace_rows[:15],
        "state_age_bucket_top_gaps": state_age_rows[:20],
    }


baseline = build_snapshot(BASELINE_DATASET)
results = []
for candidate_dataset in CANDIDATE_DATASETS:
    candidate = build_snapshot(candidate_dataset)
    results.append(
        {
            "candidate_dataset": candidate_dataset,
            "candidate": candidate,
            "comparisons": compare_snapshots(candidate, baseline),
        }
    )

payload = {
    "metric": "enhanced_cps_support_audit_batch",
    "period": PERIOD,
    "baseline_dataset": BASELINE_DATASET,
    "baseline": baseline,
    "results": results,
}
print(json.dumps(payload, sort_keys=True))
""".strip()


@dataclass(frozen=True)
class PolicyEngineUSEnhancedCPSNativeScores:
    """Exact enhanced-CPS native-loss comparison for one candidate/baseline pair."""

    metric: str
    period: int
    candidate_dataset: str
    baseline_dataset: str
    candidate_enhanced_cps_native_loss: float
    baseline_enhanced_cps_native_loss: float
    enhanced_cps_native_loss_delta: float
    candidate_unweighted_msre: float
    baseline_unweighted_msre: float
    unweighted_msre_delta: float
    n_targets_total: int
    n_targets_kept: int
    n_targets_zero_dropped: int
    n_targets_bad_dropped: int
    n_national_targets: int
    n_state_targets: int
    candidate_weight_sum: float
    baseline_weight_sum: float
    family_breakdown: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "period": self.period,
            "candidate_dataset": self.candidate_dataset,
            "baseline_dataset": self.baseline_dataset,
            "candidate_enhanced_cps_native_loss": (
                self.candidate_enhanced_cps_native_loss
            ),
            "baseline_enhanced_cps_native_loss": (
                self.baseline_enhanced_cps_native_loss
            ),
            "enhanced_cps_native_loss_delta": self.enhanced_cps_native_loss_delta,
            "candidate_unweighted_msre": self.candidate_unweighted_msre,
            "baseline_unweighted_msre": self.baseline_unweighted_msre,
            "unweighted_msre_delta": self.unweighted_msre_delta,
            "n_targets_total": self.n_targets_total,
            "n_targets_kept": self.n_targets_kept,
            "n_targets_zero_dropped": self.n_targets_zero_dropped,
            "n_targets_bad_dropped": self.n_targets_bad_dropped,
            "n_national_targets": self.n_national_targets,
            "n_state_targets": self.n_state_targets,
            "candidate_weight_sum": self.candidate_weight_sum,
            "baseline_weight_sum": self.baseline_weight_sum,
            "family_breakdown": list(self.family_breakdown),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicyEngineUSEnhancedCPSNativeScores:
        return cls(
            metric=str(payload["metric"]),
            period=int(payload["period"]),
            candidate_dataset=str(payload["candidate_dataset"]),
            baseline_dataset=str(payload["baseline_dataset"]),
            candidate_enhanced_cps_native_loss=float(
                payload["candidate_enhanced_cps_native_loss"]
            ),
            baseline_enhanced_cps_native_loss=float(
                payload["baseline_enhanced_cps_native_loss"]
            ),
            enhanced_cps_native_loss_delta=float(
                payload["enhanced_cps_native_loss_delta"]
            ),
            candidate_unweighted_msre=float(payload["candidate_unweighted_msre"]),
            baseline_unweighted_msre=float(payload["baseline_unweighted_msre"]),
            unweighted_msre_delta=float(payload["unweighted_msre_delta"]),
            n_targets_total=int(payload["n_targets_total"]),
            n_targets_kept=int(payload["n_targets_kept"]),
            n_targets_zero_dropped=int(payload["n_targets_zero_dropped"]),
            n_targets_bad_dropped=int(payload["n_targets_bad_dropped"]),
            n_national_targets=int(payload["n_national_targets"]),
            n_state_targets=int(payload["n_state_targets"]),
            candidate_weight_sum=float(payload["candidate_weight_sum"]),
            baseline_weight_sum=float(payload["baseline_weight_sum"]),
            family_breakdown=tuple(payload.get("family_breakdown", ())),
        )


PolicyEngineUSNativeBroadLossScore = PolicyEngineUSEnhancedCPSNativeScores


def resolve_policyengine_us_data_repo_root(
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve the local policyengine-us-data checkout used for native scoring."""

    candidates: list[Path] = []
    if repo_root is not None:
        candidates.append(Path(repo_root))
    env_repo = os.environ.get(_PE_US_DATA_REPO_ENV)
    if env_repo:
        candidates.append(Path(env_repo))
    candidates.append(_DEFAULT_PE_US_DATA_REPO)

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "policyengine_us_data").exists():
            return resolved
    searched = ", ".join(str(path.expanduser()) for path in candidates)
    raise FileNotFoundError(
        "Could not resolve policyengine-us-data repo root. "
        f"Searched: {searched}"
    )


def resolve_policyengine_us_data_python(
    python_executable: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve a Python executable with policyengine-us-data installed."""

    candidates: list[Path] = []
    if python_executable is not None:
        candidates.append(Path(python_executable))
    env_python = os.environ.get(_PE_US_DATA_PYTHON_ENV)
    if env_python:
        candidates.append(Path(env_python))
    resolved_repo = resolve_policyengine_us_data_repo_root(repo_root)
    candidates.extend(
        (
            resolved_repo / ".venv" / "bin" / "python",
            resolved_repo / "venv" / "bin" / "python",
        )
    )

    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.exists() and os.access(expanded, os.X_OK):
            return expanded
    searched = ", ".join(str(path.expanduser()) for path in candidates)
    raise FileNotFoundError(
        "Could not resolve a usable policyengine-us-data Python executable. "
        f"Searched: {searched}"
    )


def build_policyengine_us_data_pythonpath(
    repo_root: str | Path | None = None,
    *,
    existing_pythonpath: str | None = None,
) -> str:
    """Build the native-scoring PYTHONPATH for local PE-US-data checkouts."""

    resolved_repo = resolve_policyengine_us_data_repo_root(repo_root)
    path_entries: list[str] = [str(resolved_repo)]

    sibling_microimpute = resolved_repo.parent / "microimpute"
    if (sibling_microimpute / "microimpute").exists():
        path_entries.append(str(sibling_microimpute))

    if existing_pythonpath:
        path_entries.extend(
            entry for entry in existing_pythonpath.split(os.pathsep) if entry
        )
    return os.pathsep.join(path_entries)


def build_policyengine_us_data_subprocess_env(
    repo_root: str | Path | None = None,
    *,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a clean subprocess env for PE-native scoring helpers."""

    source_env = dict(os.environ if base_env is None else base_env)
    env = {
        key: source_env[key]
        for key in _PE_NATIVE_SCORE_BASE_ENV_VARS
        if key in source_env and source_env[key]
    }
    env["PYTHONPATH"] = build_policyengine_us_data_pythonpath(
        repo_root,
        existing_pythonpath=source_env.get("PYTHONPATH"),
    )
    return env


def compute_policyengine_us_enhanced_cps_native_scores(
    candidate_dataset: str | Path,
    baseline_dataset: str | Path,
    *,
    period: int = 2024,
    policyengine_us_data_python: str | Path | None = None,
    policyengine_us_data_repo: str | Path | None = None,
) -> PolicyEngineUSEnhancedCPSNativeScores:
    """Score one candidate and baseline under the exact enhanced-CPS loss."""
    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    if policyengine_us_data_python is not None:
        command = [str(Path(policyengine_us_data_python).expanduser())]
    else:
        command = ["uv", "run", "--project", str(resolved_repo), "python"]
    completed = subprocess.run(
        [
            *command,
            "-c",
            _PE_NATIVE_BROAD_SCORE_SCRIPT,
            str(resolved_repo),
            json.dumps(_ENHANCED_CPS_BAD_TARGETS),
            str(int(period)),
            str(Path(candidate_dataset).expanduser().resolve()),
            str(Path(baseline_dataset).expanduser().resolve()),
        ],
        cwd=resolved_repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"PE-native broad loss scoring failed: {detail}")
    payload = json.loads(completed.stdout)
    return PolicyEngineUSEnhancedCPSNativeScores.from_dict(payload)


def score_policyengine_us_native_broad_loss(
    candidate_dataset: str | Path,
    baseline_dataset: str | Path,
    *,
    period: int = 2024,
    python_executable: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> PolicyEngineUSEnhancedCPSNativeScores:
    """Backward-compatible alias for the exact enhanced-CPS loss scorer."""
    return compute_policyengine_us_enhanced_cps_native_scores(
        candidate_dataset,
        baseline_dataset,
        period=period,
        policyengine_us_data_python=python_executable,
        policyengine_us_data_repo=repo_root,
    )


def compute_us_pe_native_scores(
    *,
    candidate_dataset_path: str | Path,
    baseline_dataset_path: str | Path,
    period: int = 2024,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> dict[str, Any]:
    """Build the saved manifest payload for PE-native broad scoring."""

    score = compute_policyengine_us_enhanced_cps_native_scores(
        candidate_dataset_path,
        baseline_dataset_path,
        period=period,
        policyengine_us_data_python=policyengine_us_data_python,
        policyengine_us_data_repo=policyengine_us_data_repo,
    )
    return {
        "metric": score.metric,
        "period": score.period,
        "summary": {
            "candidate_enhanced_cps_native_loss": (
                score.candidate_enhanced_cps_native_loss
            ),
            "baseline_enhanced_cps_native_loss": (
                score.baseline_enhanced_cps_native_loss
            ),
            "enhanced_cps_native_loss_delta": score.enhanced_cps_native_loss_delta,
            "candidate_beats_baseline": score.enhanced_cps_native_loss_delta < 0.0,
            "candidate_unweighted_msre": score.candidate_unweighted_msre,
            "baseline_unweighted_msre": score.baseline_unweighted_msre,
            "unweighted_msre_delta": score.unweighted_msre_delta,
            "n_targets_total": score.n_targets_total,
            "n_targets_kept": score.n_targets_kept,
            "n_targets_zero_dropped": score.n_targets_zero_dropped,
            "n_targets_bad_dropped": score.n_targets_bad_dropped,
            "n_national_targets": score.n_national_targets,
            "n_state_targets": score.n_state_targets,
        },
        "broad_loss": score.to_dict(),
        "family_breakdown": list(score.family_breakdown),
    }


def compute_batch_us_pe_native_scores(
    *,
    candidate_dataset_paths: list[str | Path] | tuple[str | Path, ...],
    baseline_dataset_path: str | Path,
    period: int = 2024,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Score multiple candidates against one baseline in a single PE-native subprocess."""

    if not candidate_dataset_paths:
        return []
    started_at = perf_counter()
    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    if policyengine_us_data_python is not None:
        command = [str(Path(policyengine_us_data_python).expanduser())]
    else:
        command = ["uv", "run", "--project", str(resolved_repo), "python"]
    completed = subprocess.run(
        [
            *command,
            "-c",
            _PE_NATIVE_BROAD_BATCH_SCORE_SCRIPT,
            str(resolved_repo),
            json.dumps(_ENHANCED_CPS_BAD_TARGETS),
            str(int(period)),
            str(Path(baseline_dataset_path).expanduser().resolve()),
            json.dumps(
                [
                    str(Path(candidate_path).expanduser().resolve())
                    for candidate_path in candidate_dataset_paths
                ]
            ),
        ],
        cwd=resolved_repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"PE-native batch broad loss scoring failed: {detail}")
    payload = json.loads(completed.stdout)
    elapsed_seconds = perf_counter() - started_at
    results = [
        {
            "metric": item["metric"],
            "period": int(item["period"]),
            "summary": {
                "candidate_enhanced_cps_native_loss": float(
                    item["candidate_enhanced_cps_native_loss"]
                ),
                "baseline_enhanced_cps_native_loss": float(
                    item["baseline_enhanced_cps_native_loss"]
                ),
                "enhanced_cps_native_loss_delta": float(
                    item["enhanced_cps_native_loss_delta"]
                ),
                "candidate_beats_baseline": bool(
                    item["candidate_beats_baseline"]
                ),
                "candidate_unweighted_msre": float(item["candidate_unweighted_msre"]),
                "baseline_unweighted_msre": float(item["baseline_unweighted_msre"]),
                "unweighted_msre_delta": float(item["unweighted_msre_delta"]),
                "n_targets_total": int(item["n_targets_total"]),
                "n_targets_kept": int(item["n_targets_kept"]),
                "n_targets_zero_dropped": int(item["n_targets_zero_dropped"]),
                "n_targets_bad_dropped": int(item["n_targets_bad_dropped"]),
                "n_national_targets": int(item["n_national_targets"]),
                "n_state_targets": int(item["n_state_targets"]),
            },
            "broad_loss": {
                "metric": item["metric"],
                "period": int(item["period"]),
                "candidate_dataset": str(item["candidate_dataset"]),
                "baseline_dataset": str(item["baseline_dataset"]),
                "candidate_enhanced_cps_native_loss": float(
                    item["candidate_enhanced_cps_native_loss"]
                ),
                "baseline_enhanced_cps_native_loss": float(
                    item["baseline_enhanced_cps_native_loss"]
                ),
                "enhanced_cps_native_loss_delta": float(
                    item["enhanced_cps_native_loss_delta"]
                ),
                "candidate_beats_baseline": bool(
                    item["candidate_beats_baseline"]
                ),
                "candidate_unweighted_msre": float(item["candidate_unweighted_msre"]),
                "baseline_unweighted_msre": float(item["baseline_unweighted_msre"]),
                "unweighted_msre_delta": float(item["unweighted_msre_delta"]),
                "n_targets_total": int(item["n_targets_total"]),
                "n_targets_kept": int(item["n_targets_kept"]),
                "n_targets_zero_dropped": int(item["n_targets_zero_dropped"]),
                "n_targets_bad_dropped": int(item["n_targets_bad_dropped"]),
                "n_national_targets": int(item["n_national_targets"]),
                "n_state_targets": int(item["n_state_targets"]),
                "candidate_weight_sum": float(item["candidate_weight_sum"]),
                "baseline_weight_sum": float(item["baseline_weight_sum"]),
                "family_breakdown": list(item.get("family_breakdown", [])),
            },
            "family_breakdown": list(item.get("family_breakdown", [])),
        }
        for item in payload
    ]
    for item in results:
        item["timing"] = {
            "batch_elapsed_seconds": float(elapsed_seconds),
            "batch_candidate_count": len(candidate_dataset_paths),
        }
    return results


def compare_us_pe_native_target_deltas(
    *,
    from_dataset_path: str | Path,
    to_dataset_path: str | Path,
    period: int = 2024,
    top_k: int = 25,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> dict[str, Any]:
    """Compare per-target PE-native weighted-loss terms between two datasets."""

    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    if policyengine_us_data_python is not None:
        command = [str(Path(policyengine_us_data_python).expanduser())]
    else:
        command = ["uv", "run", "--project", str(resolved_repo), "python"]
    completed = subprocess.run(
        [
            *command,
            "-c",
            _PE_NATIVE_TARGET_DELTA_SCRIPT,
            str(resolved_repo),
            json.dumps(_ENHANCED_CPS_BAD_TARGETS),
            str(int(period)),
            str(Path(from_dataset_path).expanduser().resolve()),
            str(Path(to_dataset_path).expanduser().resolve()),
            str(int(top_k)),
        ],
        cwd=resolved_repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"PE-native target delta comparison failed: {detail}")
    return json.loads(completed.stdout)


def compute_batch_us_pe_native_target_deltas(
    *,
    candidate_dataset_paths: list[str | Path] | tuple[str | Path, ...],
    baseline_dataset_path: str | Path,
    period: int = 2024,
    top_k: int = 25,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Compare PE-native weighted-loss targets for many candidates against one baseline."""

    if not candidate_dataset_paths:
        return []
    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    if policyengine_us_data_python is not None:
        command = [str(Path(policyengine_us_data_python).expanduser())]
    else:
        command = ["uv", "run", "--project", str(resolved_repo), "python"]
    completed = subprocess.run(
        [
            *command,
            "-c",
            _PE_NATIVE_TARGET_DELTA_BATCH_SCRIPT,
            str(resolved_repo),
            json.dumps(_ENHANCED_CPS_BAD_TARGETS),
            str(int(period)),
            str(Path(baseline_dataset_path).expanduser().resolve()),
            json.dumps(
                [
                    str(Path(candidate_path).expanduser().resolve())
                    for candidate_path in candidate_dataset_paths
                ]
            ),
            str(int(top_k)),
        ],
        cwd=resolved_repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"PE-native batch target delta comparison failed: {detail}")
    return list(json.loads(completed.stdout))


def compute_us_pe_native_support_audit(
    *,
    candidate_dataset_path: str | Path,
    baseline_dataset_path: str | Path,
    period: int = 2024,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> dict[str, Any]:
    """Compare candidate vs baseline structural support on selected PE surfaces."""

    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    if policyengine_us_data_python is not None:
        command = [str(Path(policyengine_us_data_python).expanduser())]
    else:
        command = ["uv", "run", "--project", str(resolved_repo), "python"]
    completed = subprocess.run(
        [
            *command,
            "-c",
            _PE_NATIVE_SUPPORT_AUDIT_SCRIPT,
            str(resolved_repo),
            str(int(period)),
            str(Path(candidate_dataset_path).expanduser().resolve()),
            str(Path(baseline_dataset_path).expanduser().resolve()),
        ],
        cwd=resolved_repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"PE-native support audit failed: {detail}")
    return json.loads(completed.stdout)


def compute_batch_us_pe_native_support_audits(
    *,
    candidate_dataset_paths: list[str | Path] | tuple[str | Path, ...],
    baseline_dataset_path: str | Path,
    period: int = 2024,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Compare PE support structure for many candidates against one baseline."""

    if not candidate_dataset_paths:
        return []
    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    if policyengine_us_data_python is not None:
        command = [str(Path(policyengine_us_data_python).expanduser())]
    else:
        command = ["uv", "run", "--project", str(resolved_repo), "python"]
    completed = subprocess.run(
        [
            *command,
            "-c",
            _PE_NATIVE_SUPPORT_AUDIT_BATCH_SCRIPT,
            str(resolved_repo),
            str(int(period)),
            str(Path(baseline_dataset_path).expanduser().resolve()),
            json.dumps(
                [
                    str(Path(candidate_path).expanduser().resolve())
                    for candidate_path in candidate_dataset_paths
                ]
            ),
        ],
        cwd=resolved_repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"PE-native batch support audit failed: {detail}")

    payload = json.loads(completed.stdout)
    baseline_dataset = str(payload["baseline_dataset"])
    baseline_snapshot = payload["baseline"]
    period_value = int(payload["period"])
    return [
        {
            "metric": "enhanced_cps_support_audit",
            "period": period_value,
            "candidate_dataset": str(item["candidate_dataset"]),
            "baseline_dataset": baseline_dataset,
            "candidate": item["candidate"],
            "baseline": baseline_snapshot,
            "comparisons": item["comparisons"],
        }
        for item in payload.get("results", ())
    ]


def write_us_pe_native_scores(
    output_path: str | Path,
    *,
    candidate_dataset_path: str | Path,
    baseline_dataset_path: str | Path,
    period: int = 2024,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> Path:
    """Write PE-native broad scoring payload to disk."""

    payload = compute_us_pe_native_scores(
        candidate_dataset_path=candidate_dataset_path,
        baseline_dataset_path=baseline_dataset_path,
        period=period,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return destination


def main(argv: list[str] | None = None) -> int:
    """CLI for exact broad PE-native loss scoring."""

    parser = argparse.ArgumentParser(
        description="Score a candidate and baseline under PE-US's enhanced-CPS native loss."
    )
    parser.add_argument("--candidate-dataset", required=True)
    parser.add_argument("--baseline-dataset", required=True)
    parser.add_argument("--period", type=int, default=2024)
    parser.add_argument("--policyengine-us-data-python")
    parser.add_argument("--policyengine-us-data-repo")
    args = parser.parse_args(argv)

    score = compute_policyengine_us_enhanced_cps_native_scores(
        args.candidate_dataset,
        args.baseline_dataset,
        period=args.period,
        policyengine_us_data_python=args.policyengine_us_data_python,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
    )
    print(json.dumps(score.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
