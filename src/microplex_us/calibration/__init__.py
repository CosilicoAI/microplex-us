"""Calibration backends for microplex-us.

The mainline production calibrator is `MicrocalibrateAdapter`, which wraps
the `microcalibrate` gradient-descent chi-squared solver in the same
interface the rest of microplex-us expects from the legacy
`microplex.calibration.Calibrator`.

See `docs/calibrator-decision.md` for the rationale.
"""

from microplex_us.calibration.microcalibrate_adapter import (
    MicrocalibrateAdapter,
    MicrocalibrateAdapterConfig,
)

__all__ = [
    "MicrocalibrateAdapter",
    "MicrocalibrateAdapterConfig",
]
