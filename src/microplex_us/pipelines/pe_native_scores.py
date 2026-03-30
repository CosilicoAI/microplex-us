"""PolicyEngine-native scoring helpers for US Microplex artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DEFAULT_PE_US_DATA_REPO = Path.home() / "PolicyEngine" / "policyengine-us-data"
_PE_US_DATA_PYTHON_ENV = "MICROPLEX_US_POLICYENGINE_US_DATA_PYTHON"
_PE_US_DATA_REPO_ENV = "MICROPLEX_US_POLICYENGINE_US_DATA_REPO"

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
        resolved = candidate.expanduser().resolve()
        if resolved.exists() and os.access(resolved, os.X_OK):
            return resolved
    searched = ", ".join(str(path.expanduser()) for path in candidates)
    raise FileNotFoundError(
        "Could not resolve a usable policyengine-us-data Python executable. "
        f"Searched: {searched}"
    )


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
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{resolved_repo}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(resolved_repo)
    )
    if policyengine_us_data_python is not None:
        command = [str(Path(policyengine_us_data_python).expanduser().resolve())]
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
    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{resolved_repo}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(resolved_repo)
    )
    if policyengine_us_data_python is not None:
        command = [str(Path(policyengine_us_data_python).expanduser().resolve())]
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
    return [
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
