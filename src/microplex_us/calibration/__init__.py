"""Calibration backends for microplex-us.

The mainline production calibrator is `MicrocalibrateAdapter`, which
wraps `microcalibrate`'s gradient-descent chi-squared solver. It is now
country-agnostic and lives in upstream `microplex.calibration` so every
country package (microplex-us, microplex-uk, etc.) shares one
identity-preserving calibrator. This module re-exports the adapter so
existing `from microplex_us.calibration import MicrocalibrateAdapter`
imports keep working.

See `docs/calibrator-decision.md` for the rationale.
"""

from microplex.calibration import (
    MicrocalibrateAdapter,
    MicrocalibrateAdapterConfig,
)

__all__ = [
    "MicrocalibrateAdapter",
    "MicrocalibrateAdapterConfig",
]
