"""Summarize donor-conditioning diagnostics recorded in artifact manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _resolve_artifact_dir(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        if (candidate / "manifest.json").exists():
            return candidate
        manifest = next(candidate.glob("**/manifest.json"), None)
        if manifest is None:
            raise FileNotFoundError(f"No manifest.json found under {candidate}")
        return manifest.parent
    if candidate.name == "manifest.json":
        return candidate.parent
    raise FileNotFoundError(f"Expected an artifact directory or manifest.json, got {candidate}")


def summarize_donor_conditioning(
    artifact_path: str | Path,
    *,
    focus_variables: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Summarize donor-conditioning diagnostics for one artifact bundle."""
    artifact_dir = _resolve_artifact_dir(artifact_path)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    diagnostics = list(
        manifest.get("synthesis", {}).get("donor_conditioning_diagnostics", [])
    )
    focus = set(focus_variables or ())
    if focus:
        diagnostics = [
            entry
            for entry in diagnostics
            if focus
            & {
                *entry.get("model_variables", []),
                *entry.get("restored_variables", []),
            }
        ]

    selected_counter: Counter[str] = Counter()
    dropped_counter: Counter[str] = Counter()
    raw_supplemental_reason_counter: Counter[str] = Counter()
    supplemental_reason_counter: Counter[str] = Counter()
    raw_challenger_reason_counter: Counter[str] = Counter()
    challenger_reason_counter: Counter[str] = Counter()
    block_summaries: list[dict[str, Any]] = []
    for entry in diagnostics:
        selected = list(entry.get("selected_condition_vars", []))
        dropped = list(entry.get("dropped_shared_vars", []))
        raw_supplemental_status = list(
            entry.get("raw_supplemental_shared_condition_var_status", [])
        )
        supplemental_status = list(
            entry.get("supplemental_shared_condition_var_status", [])
        )
        raw_challenger_status = list(
            entry.get("raw_challenger_shared_condition_var_status", [])
        )
        challenger_status = list(
            entry.get("challenger_shared_condition_var_status", [])
        )
        selected_counter.update(selected)
        dropped_counter.update(dropped)
        raw_supplemental_reason_counter.update(
            status.get("reason")
            for status in raw_supplemental_status
            if status.get("reason") is not None
        )
        supplemental_reason_counter.update(
            status.get("reason")
            for status in supplemental_status
            if status.get("reason") is not None
        )
        raw_challenger_reason_counter.update(
            status.get("reason")
            for status in raw_challenger_status
            if status.get("reason") is not None
        )
        challenger_reason_counter.update(
            status.get("reason")
            for status in challenger_status
            if status.get("reason") is not None
        )
        block_summaries.append(
            {
                "donor_source": entry.get("donor_source"),
                "model_variables": list(entry.get("model_variables", [])),
                "restored_variables": list(entry.get("restored_variables", [])),
                "condition_selection": entry.get("condition_selection"),
                "used_condition_surface": bool(
                    entry.get("used_condition_surface", False)
                ),
                "raw_shared_vars": list(entry.get("raw_shared_vars", [])),
                "shared_vars_after_model_exclusion": list(
                    entry.get("shared_vars_after_model_exclusion", [])
                ),
                "projection_applied": bool(entry.get("projection_applied", False)),
                "entity_compatible_shared_vars": list(
                    entry.get("entity_compatible_shared_vars", [])
                ),
                "requested_supplemental_shared_condition_vars": list(
                    entry.get("requested_supplemental_shared_condition_vars", [])
                ),
                "requested_challenger_shared_condition_vars": list(
                    entry.get("requested_challenger_shared_condition_vars", [])
                ),
                "raw_supplemental_shared_condition_var_status": raw_supplemental_status,
                "raw_challenger_shared_condition_var_status": raw_challenger_status,
                "supplemental_shared_condition_var_status": supplemental_status,
                "challenger_shared_condition_var_status": challenger_status,
                "selected_condition_vars": selected,
                "dropped_shared_vars": dropped,
            }
        )

    return {
        "artifact_path": str(artifact_dir),
        "block_count": len(block_summaries),
        "focus_variables": sorted(focus),
        "selected_condition_var_frequency": dict(sorted(selected_counter.items())),
        "dropped_shared_var_frequency": dict(sorted(dropped_counter.items())),
        "raw_supplemental_shared_condition_reason_frequency": dict(
            sorted(raw_supplemental_reason_counter.items())
        ),
        "raw_challenger_shared_condition_reason_frequency": dict(
            sorted(raw_challenger_reason_counter.items())
        ),
        "supplemental_shared_condition_reason_frequency": dict(
            sorted(supplemental_reason_counter.items())
        ),
        "challenger_shared_condition_reason_frequency": dict(
            sorted(challenger_reason_counter.items())
        ),
        "blocks": block_summaries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize donor-conditioning diagnostics for one artifact."
    )
    parser.add_argument("artifact", help="Artifact directory or manifest.json path.")
    parser.add_argument("--out", help="Optional JSON output path.")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Optional model/restored variables to focus on.",
    )
    args = parser.parse_args(argv)

    payload = summarize_donor_conditioning(
        args.artifact,
        focus_variables=tuple(args.variables) if args.variables else None,
    )
    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).expanduser().write_text(output)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
