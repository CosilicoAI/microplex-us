"""Local synthesizer methods for the bakeoff harness.

These extend the `microplex.eval.benchmark` set without modifying the
upstream library. Methods defined here follow the same `_MultiSourceBase`
protocol so they slot into `ScaleUpRunner.fit_and_generate` unchanged.

Current contents:

- `CARTMethod`: synthpop-style CART per-column imputation. Each target
  column gets a decision tree fit on the shared conditioning variables;
  at generation time, the tree routes each synthetic record to a leaf,
  and the predicted value is drawn uniformly from the training-set
  values that landed in that leaf. This matches the default draw in
  `synthpop`'s `syn.cart` (Nowok, Raab, and Dibben, 2016).

- `ZICARTMethod`: zero-inflated variant that uses a random-forest
  classifier for P(y > 0 | x) on columns where the training-set zero
  fraction exceeds 10 %, then applies `CARTMethod` on the non-zero
  subset. Mirrors `ZIQRFMethod`'s structure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from microplex.eval.benchmark import _MultiSourceBase
from sklearn.tree import DecisionTreeRegressor


class CARTMethod(_MultiSourceBase):
    """Synthpop-style CART per-column synthesis.

    Each column gets a `DecisionTreeRegressor` fit on the shared
    conditioning variables. At generation time, each record is routed
    to a leaf via `tree.apply`, and the synthetic value is sampled
    uniformly from the training-set outcomes that landed in that leaf.
    This reproduces `synthpop`'s default CART draw.
    """

    name = "CART"

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(zero_inflated=False)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _fit_column(self, col: str, X: np.ndarray, y: np.ndarray) -> None:
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        tree.fit(X, y)
        leaf_ids = tree.apply(X)
        leaf_to_values: dict[int, np.ndarray] = {}
        for lid, val in zip(leaf_ids.tolist(), y.tolist(), strict=False):
            leaf_to_values.setdefault(lid, []).append(val)
        for lid, vals in leaf_to_values.items():
            leaf_to_values[lid] = np.asarray(vals, dtype=float)
        self._col_models[col] = {
            "tree": tree,
            "leaf_to_values": leaf_to_values,
            "fallback_value": float(np.median(y)) if len(y) > 0 else 0.0,
        }

    def _generate_column(
        self,
        col: str,
        X: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        model = self._col_models[col]
        tree = model["tree"]
        leaf_to_values = model["leaf_to_values"]
        fallback = model["fallback_value"]
        leaf_ids = tree.apply(X)
        out = np.empty(len(X), dtype=float)
        for i, lid in enumerate(leaf_ids.tolist()):
            vals = leaf_to_values.get(lid)
            if vals is None or len(vals) == 0:
                out[i] = fallback
            else:
                out[i] = float(vals[rng.randint(len(vals))])
        return out


class ZICARTMethod(CARTMethod):
    """Zero-Inflated CART: random-forest zero classifier + CART leaf draw."""

    name = "ZI-CART"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.zero_inflated = True


__all__ = ["CARTMethod", "ZICARTMethod"]
