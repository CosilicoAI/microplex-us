"""Targeted PE-native focus checks for ACA spending/enrollment and state AGI bins."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from microplex_us.pipelines.pe_native_scores import (
    build_policyengine_us_data_subprocess_env,
    resolve_policyengine_us_data_repo_root,
)

_PE_FOCUS_TARGETS_SCRIPT = r"""
import json
import sys
from pathlib import Path
from typing import Any

import sqlite3
import numpy as np
from policyengine_core.data import Dataset
from policyengine_us import Microsimulation
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.census import STATE_ABBREV_TO_FIPS

PERIOD = int(sys.argv[1])
BASELINE_PATH = sys.argv[2]
CANDIDATE_PATH = sys.argv[3]

def dataset_from_path(dataset_path: str, dataset_name: str):
    class LocalDataset(Dataset):
        name = dataset_name
        label = dataset_name
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = PERIOD

    return LocalDataset


def get_agi_band_label(lower: float, upper: float) -> str:
    if lower == -np.inf:
        return f"-inf_{int(upper)}"
    if upper == np.inf:
        return f"{int(lower)}_inf"
    return f"{int(lower)}_{int(upper)}"


def _load_focus_targets_from_db(period: int) -> list[dict[str, Any]]:
    db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            '''
            SELECT
                t.target_id,
                t.variable,
                t.value,
                t.period,
                t.stratum_id,
                sc.constraint_variable,
                sc.operation,
                sc.value AS constraint_value
            FROM targets t
            JOIN strata s ON t.stratum_id = s.stratum_id
            LEFT JOIN stratum_constraints sc
                ON s.stratum_id = sc.stratum_id
            WHERE t.active = 1
              AND t.reform_id = 0
              AND t.period <= ?
              AND t.variable IN ('aca_ptc', 'person_count', 'adjusted_gross_income', 'tax_unit_count')
            ''',
            (period,),
        ).fetchall()
    finally:
        conn.close()

    targets_by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        target_id = int(row["target_id"])
        target = targets_by_id.setdefault(
            target_id,
            {
                "target_id": target_id,
                "variable": row["variable"],
                "value": float(row["value"]),
                "period": int(row["period"]),
                "stratum_id": int(row["stratum_id"]),
                "constraints": [],
            },
        )
        if row["constraint_variable"] is not None:
            target["constraints"].append(
                {
                    "variable": row["constraint_variable"],
                    "operation": row["operation"],
                    "value": row["constraint_value"],
                }
            )

    best_targets: dict[tuple[int, str], dict[str, Any]] = {}
    for target in targets_by_id.values():
        key = (target["stratum_id"], target["variable"])
        existing = best_targets.get(key)
        if existing is None or target["period"] > existing["period"]:
            best_targets[key] = target

    fips_to_state = {v: k for k, v in STATE_ABBREV_TO_FIPS.items()}

    focus_targets: list[dict[str, Any]] = []
    for target in best_targets.values():
        constraints = target["constraints"]
        if not constraints:
            continue

        constraint_vars = {c["variable"] for c in constraints}
        if "state_fips" not in constraint_vars:
            continue
        if "congressional_district_geoid" in constraint_vars:
            continue

        state_value = next(
            (c["value"] for c in constraints if c["variable"] == "state_fips"),
            None,
        )
        if state_value is None:
            continue
        state_fips = f"{int(state_value):02d}"
        state_abbrev = fips_to_state.get(state_fips)
        if state_abbrev is None:
            continue

        if target["variable"] == "aca_ptc":
            focus_targets.append(
                {
                    "target_name": f"nation/irs/aca_spending/{state_abbrev.lower()}",
                    "target": target["value"],
                    "state_abbrev": state_abbrev,
                    "kind": "aca_spending",
                }
            )
            continue

        if target["variable"] == "person_count" and (
            "aca_ptc" in constraint_vars or "is_aca_ptc_eligible" in constraint_vars
        ):
            focus_targets.append(
                {
                    "target_name": f"state/irs/aca_enrollment/{state_abbrev.lower()}",
                    "target": target["value"],
                    "state_abbrev": state_abbrev,
                    "kind": "aca_enrollment",
                }
            )
            continue

        if target["variable"] not in {"adjusted_gross_income", "tax_unit_count"}:
            continue

        if "adjusted_gross_income" not in constraint_vars:
            continue

        lower = float("-inf")
        upper = float("inf")
        for c in constraints:
            if c["variable"] != "adjusted_gross_income":
                continue
            value = float(c["value"])
            if c["operation"] in (">=", ">"):
                lower = max(lower, value)
            elif c["operation"] in ("<=", "<"):
                upper = min(upper, value)

        band = get_agi_band_label(lower, upper)
        focus_targets.append(
            {
                "target_name": f"state/{state_abbrev}/{target['variable']}/{band}",
                "target": target["value"],
                "state_abbrev": state_abbrev,
                "kind": "agi",
                "agi_lower": lower,
                "agi_upper": upper,
                "is_count": target["variable"] == "tax_unit_count",
            }
        )

    return focus_targets


def compute_focus(dataset_path: str) -> list[dict[str, float | str]]:
    dataset_cls = dataset_from_path(dataset_path, Path(dataset_path).stem)
    sim = Microsimulation(dataset=dataset_cls)
    sim.default_calculation_period = PERIOD
    weights = sim.calculate(
        "household_weight",
        map_to="household",
        period=PERIOD,
    ).values.astype(np.float64)

    rows: list[dict[str, float | str]] = []

    focus_targets = _load_focus_targets_from_db(PERIOD)
    aca_value = sim.calculate("aca_ptc", map_to="household", period=2025).values
    state_household = sim.calculate("state_code", map_to="household").values
    state_person = sim.calculate("state_code", map_to="person").values
    in_tax_unit_with_aca = (
        sim.calculate("aca_ptc", map_to="person", period=2025).values > 0
    )
    is_aca_eligible = sim.calculate(
        "is_aca_ptc_eligible", map_to="person", period=2025
    ).values
    is_enrolled = in_tax_unit_with_aca & is_aca_eligible
    agi = sim.calculate("adjusted_gross_income").values
    state_tax_unit = sim.map_result(
        state_person, "person", "tax_unit", how="value_from_first_person"
    )

    for target in focus_targets:
        kind = target["kind"]
        state_abbrev = target["state_abbrev"]

        if kind == "aca_spending":
            in_state = state_household == state_abbrev
            metric = aca_value * in_state
        elif kind == "aca_enrollment":
            in_state = state_person == state_abbrev
            metric = sim.map_result(in_state & is_enrolled, "person", "household")
        else:
            lower = float(target.get("agi_lower", float("-inf")))
            upper = float(target.get("agi_upper", float("inf")))
            in_state = state_tax_unit == state_abbrev
            in_band = (agi > lower) & (agi <= upper)
            if target.get("is_count"):
                metric = (in_state & in_band & (agi > 0)).astype(float)
            else:
                metric = np.where(in_state & in_band, agi, 0.0)
            metric = sim.map_result(metric, "tax_unit", "household")

        estimate = float(np.sum(metric * weights))
        rows.append(
            {
                "target_name": target["target_name"],
                "target": float(target["target"]),
                "estimate": estimate,
            }
        )

    return rows


baseline_rows = compute_focus(BASELINE_PATH)
candidate_rows = compute_focus(CANDIDATE_PATH)

baseline_map = {row["target_name"]: row for row in baseline_rows}
candidate_map = {row["target_name"]: row for row in candidate_rows}

rows = []
for name in sorted(set(baseline_map) | set(candidate_map)):
    base = baseline_map.get(name, {})
    cand = candidate_map.get(name, {})
    target = base.get("target", cand.get("target"))
    baseline_est = base.get("estimate")
    candidate_est = cand.get("estimate")
    rows.append(
        {
            "target_name": name,
            "target": target,
            "baseline_estimate": baseline_est,
            "candidate_estimate": candidate_est,
            "candidate_over_target": (
                candidate_est / target if target not in (None, 0) else None
            ),
            "candidate_over_baseline": (
                candidate_est / baseline_est
                if baseline_est not in (None, 0)
                else None
            ),
        }
    )

print(json.dumps({"rows": rows}, sort_keys=True))
""".strip()


def compare_us_pe_focus_targets(
    *,
    baseline_dataset_path: str | Path,
    candidate_dataset_path: str | Path,
    period: int = 2024,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> dict[str, Any]:
    """Compare ACA/AGI focus targets between a baseline and candidate dataset."""

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
            _PE_FOCUS_TARGETS_SCRIPT,
            str(int(period)),
            str(Path(baseline_dataset_path).expanduser().resolve()),
            str(Path(candidate_dataset_path).expanduser().resolve()),
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
        raise RuntimeError(f"PE focus target comparison failed: {detail}")
    return json.loads(completed.stdout)


__all__ = ["compare_us_pe_focus_targets"]
