"""Shared sampling helpers for checkpoint-scale source queries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_frame_without_replacement(
    frame: pd.DataFrame,
    *,
    sample_n: int | None,
    random_seed: int,
    weight_col: str | None = None,
    positive_only_when_weighted: bool = False,
) -> pd.DataFrame:
    """Sample rows without replacement, preserving existing weighting behavior."""

    result = frame.copy()
    if sample_n is None or sample_n >= len(result):
        return result

    sample_source = result
    sample_weights: pd.Series | None = None
    if weight_col is not None and weight_col in result.columns:
        candidate_weights = (
            pd.to_numeric(result[weight_col], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
        positive_mask = candidate_weights > 0.0
        if candidate_weights.sum() > 0.0 and int(positive_mask.sum()) >= sample_n:
            if positive_only_when_weighted:
                sample_source = result.loc[positive_mask].copy()
                sample_weights = candidate_weights.loc[positive_mask]
            else:
                sample_weights = candidate_weights

    if sample_n >= len(sample_source):
        return sample_source.copy()

    try:
        return sample_source.sample(
            n=sample_n,
            random_state=random_seed,
            replace=False,
            weights=sample_weights,
        ).copy()
    except ValueError:
        return sample_source.sample(
            n=sample_n,
            random_state=random_seed,
            replace=False,
            weights=None,
        ).copy()


def sample_frame_with_state_floor(
    frame: pd.DataFrame,
    *,
    sample_n: int | None,
    random_seed: int,
    weight_col: str | None = None,
    state_col: str = "state_fips",
    state_floor: int | None = None,
    positive_only_when_weighted: bool = False,
) -> pd.DataFrame:
    """Sample rows while guaranteeing a small minimum from each observed state."""

    result = frame.copy()
    if sample_n is None or sample_n >= len(result):
        return result
    resolved_floor = int(state_floor or 0)
    if resolved_floor <= 0 or state_col not in result.columns:
        return sample_frame_without_replacement(
            result,
            sample_n=sample_n,
            random_seed=random_seed,
            weight_col=weight_col,
            positive_only_when_weighted=positive_only_when_weighted,
        )

    state_values = pd.to_numeric(result[state_col], errors="coerce")
    eligible = result.loc[state_values.notna()].copy()
    if eligible.empty:
        return sample_frame_without_replacement(
            result,
            sample_n=sample_n,
            random_seed=random_seed,
            weight_col=weight_col,
            positive_only_when_weighted=positive_only_when_weighted,
        )

    eligible["_sampling_state_key"] = state_values.loc[eligible.index].astype(int)
    groups = [
        group.copy()
        for _, group in eligible.groupby("_sampling_state_key", sort=True, dropna=False)
    ]
    minimum_required = sum(min(len(group), resolved_floor) for group in groups)
    if minimum_required > sample_n:
        raise ValueError(
            "state_floor requires more rows than sample_n allows: "
            f"floor={resolved_floor}, required={minimum_required}, sample_n={sample_n}"
        )

    rng = np.random.default_rng(random_seed)
    floor_samples: list[pd.DataFrame] = []
    for group in groups:
        group_floor = min(len(group), resolved_floor)
        if group_floor <= 0:
            continue
        floor_samples.append(
            sample_frame_without_replacement(
                group.drop(columns="_sampling_state_key"),
                sample_n=group_floor,
                random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                weight_col=weight_col,
                positive_only_when_weighted=positive_only_when_weighted,
            )
        )
    floor_sample = (
        pd.concat(floor_samples, axis=0, ignore_index=False)
        if floor_samples
        else result.iloc[0:0].copy()
    )
    selected_index = pd.Index(floor_sample.index.unique())
    remaining_n = int(sample_n) - len(selected_index)
    if remaining_n <= 0:
        return floor_sample.copy()

    remainder = result.drop(index=selected_index, errors="ignore")
    if remainder.empty:
        return floor_sample.copy()
    remainder_sample = sample_frame_without_replacement(
        remainder,
        sample_n=remaining_n,
        random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        weight_col=weight_col,
        positive_only_when_weighted=positive_only_when_weighted,
    )
    return pd.concat([floor_sample, remainder_sample], axis=0, ignore_index=False)
