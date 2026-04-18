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


# --- Alternative zero-inflation classifiers (QDNN family) ----------------

def _patch_zi_classifier(method_instance: Any, classifier_factory: Any) -> None:
    """Monkey-patch a ZI method's fit so the zero-classifier is a custom one.

    The upstream `_MultiSourceBase.fit` hardcodes
    `RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)`.
    This helper re-wraps `fit` so the zero-classifier is built by
    `classifier_factory()` instead. All other fit/generate behavior is
    preserved.
    """
    import numpy as np
    import pandas as pd

    original_fit = method_instance.fit.__func__

    def patched_fit(self, sources, shared_cols):
        self.shared_cols_ = list(shared_cols)
        all_cols = set(shared_cols)
        for survey_name, df in sources.items():
            for col in df.columns:
                if col not in all_cols:
                    all_cols.add(col)
                    self.col_to_survey_[col] = survey_name
        self.all_cols_ = list(all_cols)

        shared_dfs = []
        for survey_name, df in sources.items():
            available = [c for c in shared_cols if c in df.columns]
            if len(available) == len(shared_cols):
                shared_dfs.append(df[shared_cols].copy())
        self.shared_data_ = (
            pd.concat(shared_dfs, ignore_index=True)
            if shared_dfs
            else list(sources.values())[0][shared_cols].copy()
        )

        for col in self.all_cols_:
            if col in shared_cols:
                continue
            survey_name = self.col_to_survey_[col]
            survey_df = sources[survey_name]
            available_shared = [c for c in shared_cols if c in survey_df.columns]
            X = survey_df[available_shared].values
            y = survey_df[col].values

            min_val = float(np.nanmin(y))
            at_min = np.isclose(y, min_val, atol=1e-6)
            zero_frac = at_min.sum() / len(y)
            self._col_stats[col] = {"min": min_val, "zero_frac": zero_frac}

            if (
                self.zero_inflated
                and zero_frac >= self.zero_threshold
                and at_min.sum() >= 10
            ):
                labels = (~at_min).astype(int)
                unique_labels = np.unique(labels)
                if len(unique_labels) < 2:
                    # Degenerate column — all zeros or all non-zeros in
                    # training. Fall back to a constant classifier to avoid
                    # sklearn's single-class error.
                    constant_prob = float(unique_labels[0])

                    class _Constant:
                        classes_ = np.array([0, 1])

                        def predict_proba(self, X):
                            n = len(X)
                            return np.column_stack(
                                [np.full(n, 1.0 - constant_prob),
                                 np.full(n, constant_prob)]
                            )

                    self._zero_classifiers[col] = _Constant()
                else:
                    clf = classifier_factory()
                    clf.fit(X, labels)
                    self._zero_classifiers[col] = clf
                if (~at_min).sum() >= 10:
                    self._fit_column(col, X[~at_min], y[~at_min])
            else:
                self._fit_column(col, X, y)
        return self

    method_instance.fit = patched_fit.__get__(method_instance, type(method_instance))


def _make_zi_variant(base_name: str, classifier_factory: Any):
    """Create a method class that uses a custom zero-classifier."""
    from microplex.eval.benchmark import ZIQDNNMethod

    base_classes = {"ZI-QDNN": ZIQDNNMethod}
    if base_name not in base_classes:
        raise ValueError(f"Unsupported base method for ZI variant: {base_name}")
    base_cls = base_classes[base_name]

    class _Variant(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            _patch_zi_classifier(self, classifier_factory)

    return _Variant


def _rf_calibrated_factory():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=50, random_state=42, n_jobs=-1
    )
    return CalibratedClassifierCV(rf, method="isotonic", cv=3)


def _logistic_factory():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=500, n_jobs=-1)


def _hgb_factory():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(random_state=42)


def _dnn_factory():
    """A small-MLP zero-classifier for parity with the ZI-QDNN draw network.

    Uses sklearn's MLPClassifier (hidden: 64, 32; ReLU; Adam; max_iter=100).
    Probabilities are via softmax on the output head. Not pre-calibrated;
    combine with isotonic wrapping if calibration matters.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=100,
                random_state=42,
                early_stopping=True,
            ),
        ),
    ])


class ZIQDNNLogisticMethod:
    """Placeholder; actual class built by _make_zi_variant at registry time."""

    name = "ZI-QDNN-logistic"


class ZIQDNNHGBMethod:
    name = "ZI-QDNN-hgb"


class ZIQDNNCalibratedMethod:
    name = "ZI-QDNN-calibrated"


def zi_qdnn_variant_factory(variant: str):
    """Return a ZIQDNNMethod subclass with a swapped zero-classifier."""
    if variant == "logistic":
        return _make_zi_variant("ZI-QDNN", _logistic_factory)
    if variant == "hgb":
        return _make_zi_variant("ZI-QDNN", _hgb_factory)
    if variant == "calibrated":
        return _make_zi_variant("ZI-QDNN", _rf_calibrated_factory)
    if variant == "dnn":
        return _make_zi_variant("ZI-QDNN", _dnn_factory)
    raise ValueError(f"Unknown ZI variant: {variant}")


__all__ = [
    "CARTMethod",
    "ZICARTMethod",
    "zi_qdnn_variant_factory",
]
