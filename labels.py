"""Forward labels for event studies."""

from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd


def forward_return(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Calculate forward close-to-close return for a horizon."""

    return close.shift(-horizon).div(close).sub(1.0)


def forward_mfe(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Calculate maximum favorable excursion over the next horizon days."""

    future_max = close.shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1))
    return future_max / close - 1.0


def forward_mae(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Calculate maximum adverse excursion over the next horizon days."""

    future_min = close.shift(-1).rolling(horizon, min_periods=1).min().shift(-(horizon - 1))
    return future_min / close - 1.0


def build_forward_labels(
    close: pd.DataFrame,
    horizons: Sequence[int],
    label_types: Sequence[str] = ("ret",),
) -> Dict[str, pd.DataFrame]:
    """Build only the requested forward-label types for all horizons."""

    supported = {"ret", "mfe", "mae"}
    requested = tuple(dict.fromkeys(str(label_type).lower() for label_type in label_types))
    unknown = set(requested) - supported
    if unknown:
        raise ValueError(f"Unsupported label types: {sorted(unknown)}")
    if "ret" not in requested:
        raise ValueError("label_types must include 'ret' for signal evaluation")
    labels: Dict[str, pd.DataFrame] = {}
    for horizon in horizons:
        if "ret" in requested:
            labels[f"ret_{horizon}d"] = forward_return(close, horizon)
        if "mfe" in requested:
            labels[f"mfe_{horizon}d"] = forward_mfe(close, horizon)
        if "mae" in requested:
            labels[f"mae_{horizon}d"] = forward_mae(close, horizon)
    return labels
