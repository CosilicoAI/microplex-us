"""Validate that the committed US site snapshot matches its source artifact."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from microplex_us.pipelines.site_snapshot import build_us_microplex_site_snapshot


def check_us_microplex_site_snapshot(
    snapshot_path: str | Path = "artifacts/site_snapshot_us.json",
) -> Path:
    """Raise if the saved US site snapshot is stale or inconsistent."""
    snapshot_file = Path(snapshot_path)
    snapshot = json.loads(snapshot_file.read_text())
    artifact_dir = snapshot["sourceArtifact"]["artifactDir"]
    regenerated = build_us_microplex_site_snapshot(artifact_dir)
    if snapshot != regenerated:
        raise SystemExit(_snapshot_diff(snapshot, regenerated, snapshot_file))
    return snapshot_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check that the canonical US site snapshot is up to date."
    )
    parser.add_argument(
        "snapshot_path",
        nargs="?",
        default="artifacts/site_snapshot_us.json",
        help="Path to the saved US site snapshot JSON.",
    )
    args = parser.parse_args(argv)
    checked = check_us_microplex_site_snapshot(args.snapshot_path)
    print(checked)
    return 0


def _snapshot_diff(
    saved: dict,
    regenerated: dict,
    snapshot_file: Path,
) -> str:
    saved_text = json.dumps(saved, indent=2, sort_keys=True).splitlines()
    regenerated_text = json.dumps(regenerated, indent=2, sort_keys=True).splitlines()
    diff = "\n".join(
        difflib.unified_diff(
            saved_text,
            regenerated_text,
            fromfile=str(snapshot_file),
            tofile=f"{snapshot_file} (regenerated)",
            lineterm="",
        )
    )
    return f"US site snapshot is stale or inconsistent:\n{diff}"


if __name__ == "__main__":
    raise SystemExit(main())
