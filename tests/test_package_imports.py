"""Package import contract tests."""

from __future__ import annotations

import subprocess
import sys


def test_root_import_leaves_pipeline_exports_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import microplex_us; print('build_us_microplex' in vars(microplex_us))"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_data_sources_import_leaves_family_benchmark_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import microplex_us.data_sources; "
                "print('microplex_us.data_sources.family_imputation_benchmark' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
