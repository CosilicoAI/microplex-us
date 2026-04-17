"""Scale-up benchmark harness for synthesizer comparison.

Implements the stage-1/2/3 scale-up protocol from
`docs/synthesizer-benchmark-scale-up.md`: load real enhanced_cps_2024,
sub-sample to the stage's row count, fit each specified synthesizer on the
conditioning + target column set, and report PRDC coverage, training wall
time, peak RSS, and rare-cell preservation.

Use from the CLI:

    uv run python -m microplex_us.bakeoff.scale_up \\
        --stage stage1 \\
        --methods ZI-QRF ZI-MAF ZI-QDNN \\
        --output artifacts/scale_up_stage1.json

or programmatically:

    from microplex_us.bakeoff import ScaleUpRunner, stage1_config
    runner = ScaleUpRunner(stage1_config())
    results = runner.run()
"""

from microplex_us.bakeoff.scale_up import (
    ScaleUpResult,
    ScaleUpRunner,
    ScaleUpStageConfig,
    DEFAULT_CONDITION_COLS,
    DEFAULT_TARGET_COLS,
    stage1_config,
    stage2_config,
    stage3_config,
)

__all__ = [
    "ScaleUpResult",
    "ScaleUpRunner",
    "ScaleUpStageConfig",
    "DEFAULT_CONDITION_COLS",
    "DEFAULT_TARGET_COLS",
    "stage1_config",
    "stage2_config",
    "stage3_config",
]
