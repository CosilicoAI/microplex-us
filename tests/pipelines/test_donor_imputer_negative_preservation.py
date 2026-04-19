"""Donor imputer must preserve negative values in zero-inflated-sign-mixed columns.

v7 bug (`us.py:235`, pre-fix): `ColumnwiseQRFDonorImputer` applies
`y_values > 0` as its nonzero filter. For columns that can be negative
(short-term capital gains, partnership/S-corp income, farm income,
rental income), this drops all negative training rows — the QRF only
sees positives and therefore produces zero-or-positive predictions.
The entire negative tail disappears from the synthetic frame.

v9 fix: swap the ad-hoc gate for `microimpute.models.ZeroInflatedImputer`,
which auto-detects the three-sign regime and routes negative-gated
records to a negative-only QRF.

These tests pin the post-fix contract by fitting on a column that
genuinely spans neg/0/pos and asserting negatives survive to the
synthetic output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("quantile_forest")
pytest.importorskip("microimpute")


def _three_sign_frame(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Training frame with a three-sign target.

    ~40% negative, ~20% zero, ~40% positive. Positive regime has
    distinct distribution from negative regime, so the sign is
    predictable from the conditioning variables.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, size=n).astype(float)
    is_female = rng.integers(0, 2, size=n).astype(float)

    # Regime assignment driven by (age, is_female).
    logit_pos = -0.5 + 0.05 * (age - 50)  # older → more likely positive
    logit_neg = 0.5 - 0.05 * (age - 50)  # younger → more likely negative
    logit_zero = 1.0 - 0.02 * age

    logits = np.stack([logit_neg, logit_zero, logit_pos], axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)

    u = rng.random(n)
    cum = np.cumsum(probs, axis=1)
    regime_idx = (cum >= u[:, None]).argmax(axis=1)

    y = np.zeros(n)
    pos_mask = regime_idx == 2
    neg_mask = regime_idx == 0
    y[pos_mask] = 100 + rng.exponential(200, size=pos_mask.sum())
    y[neg_mask] = -(100 + rng.exponential(200, size=neg_mask.sum()))

    return pd.DataFrame(
        {
            "age": age,
            "is_female": is_female,
            "short_term_capital_gains": y,
        }
    )


class TestDonorImputerPreservesNegatives:
    """The donor imputer must emit negatives for three-sign training columns."""

    def test_fit_generate_preserves_negative_predictions(self) -> None:
        """The current v7 imputer (`y > 0` gate) should NOT pass this.
        The v9 imputer (ZeroInflatedImputer-based) should.
        """
        from microplex_us.pipelines.us import ColumnwiseQRFDonorImputer

        train = _three_sign_frame(n=800, seed=0)
        # Preconditions on the fixture: genuinely three-sign.
        y = train["short_term_capital_gains"].to_numpy()
        assert (y > 0).sum() > 50, "fixture should have meaningful positive mass"
        assert (y < 0).sum() > 50, "fixture should have meaningful negative mass"
        assert (y == 0).sum() > 50, "fixture should have meaningful zero mass"

        imputer = ColumnwiseQRFDonorImputer(
            condition_vars=["age", "is_female"],
            target_vars=["short_term_capital_gains"],
            n_estimators=30,
            zero_inflated_vars={"short_term_capital_gains"},
            zero_threshold=0.05,
        )
        imputer.fit(train)

        rng = np.random.default_rng(42)
        n_gen = 2000
        conditions = pd.DataFrame(
            {
                "age": rng.integers(18, 80, size=n_gen).astype(float),
                "is_female": rng.integers(0, 2, size=n_gen).astype(float),
            }
        )
        synthetic = imputer.generate(conditions, seed=42)
        synth_y = synthetic["short_term_capital_gains"].to_numpy()

        # The core contract: the synthetic output must contain some
        # negative values. Under the v7 `y > 0` bug this would be 0.
        n_negative = int((synth_y < 0).sum())
        assert n_negative > 0, (
            f"Donor imputer produced no negative values despite training "
            f"data having {(y < 0).sum()} negatives. This is the v7 "
            "drop-negatives bug."
        )
        # Loose sanity: the negative fraction should be materially
        # above zero (not just a single fp-edge-case).
        assert n_negative / n_gen > 0.05, (
            f"Negative fraction in synthetic = {n_negative / n_gen:.3f}; "
            "expected > 5% given the training distribution has ~40% negatives."
        )
