"""Statistical helpers for event studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap summary for an event effect."""

    observed: float
    p_value: float
    ci_low: float
    ci_high: float


def safe_sharpe(returns: pd.Series, annualization: float = 252.0) -> float:
    """Calculate Sharpe ratio with NaN protection."""

    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return np.nan
    std = clean.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(clean.mean() / std * np.sqrt(annualization))


def cvar(values: pd.Series, alpha: float = 0.05) -> float:
    """Calculate lower-tail conditional value at risk."""

    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    threshold = clean.quantile(alpha)
    tail = clean[clean <= threshold]
    return float(tail.mean()) if not tail.empty else np.nan


def summarize_values(values: pd.Series, control: pd.Series) -> Dict[str, float]:
    """Summarize event-label values versus a control distribution."""

    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    control_clean = control.replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "events": float(len(clean)),
        "mean": float(clean.mean()) if len(clean) else np.nan,
        "median": float(clean.median()) if len(clean) else np.nan,
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else np.nan,
        "sharpe": safe_sharpe(clean),
        "hit_rate_positive": float((clean > 0).mean()) if len(clean) else np.nan,
        "hit_rate_negative": float((clean < 0).mean()) if len(clean) else np.nan,
        "cvar_5": cvar(clean, alpha=0.05),
        "control_mean": float(control_clean.mean()) if len(control_clean) else np.nan,
        "excess_mean": (
            float(clean.mean() - control_clean.mean()) if len(clean) and len(control_clean) else np.nan
        ),
    }


def bootstrap_mean_effect(
    event_values: pd.Series,
    control_values: pd.Series,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap a two-sample mean difference against the zero-effect null."""

    event = event_values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    control = control_values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(event) == 0 or len(control) == 0:
        return BootstrapResult(np.nan, np.nan, np.nan, np.nan)
    observed = float(np.mean(event) - np.mean(control))
    if n_bootstrap <= 0:
        return BootstrapResult(observed, np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    null_boot = np.empty(n_bootstrap)
    centered_event = event - event.mean()
    centered_control = control - control.mean()
    max_batch_elements = 2_000_000
    batch_size = max(1, min(128, max_batch_elements // max(len(event), len(control), 1)))
    for start in range(0, n_bootstrap, batch_size):
        current = min(batch_size, n_bootstrap - start)
        event_indices = rng.integers(0, len(event), size=(current, len(event)))
        control_indices = rng.integers(0, len(control), size=(current, len(control)))
        boot[start : start + current] = (
            event[event_indices].mean(axis=1) - control[control_indices].mean(axis=1)
        )
        null_boot[start : start + current] = (
            centered_event[event_indices].mean(axis=1)
            - centered_control[control_indices].mean(axis=1)
        )
    exceedances = int((np.abs(null_boot) >= abs(observed)).sum())
    p_value = float((exceedances + 1) / (n_bootstrap + 1))
    return BootstrapResult(
        observed=observed,
        p_value=p_value,
        ci_low=float(np.quantile(boot, 0.025)),
        ci_high=float(np.quantile(boot, 0.975)),
    )


def quantile_groups(score: pd.DataFrame, group_count: int = 5) -> Dict[str, pd.DataFrame]:
    """Create time-series quantile group masks for a continuous event score."""

    groups: Dict[str, pd.DataFrame] = {}
    pct = score.rank(axis=0, pct=True)
    edges = np.linspace(0.0, 1.0, group_count + 1)
    for idx in range(group_count):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == 0:
            mask = pct <= upper
        else:
            mask = (pct > lower) & (pct <= upper)
        groups[f"Q{idx + 1}"] = mask.fillna(False)
    return groups


def group_label_summary(
    score: pd.DataFrame,
    label: pd.DataFrame,
    group_count: int = 5,
    groups: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Summarize future labels by score quantile groups.

    Precomputed ``groups`` can be reused across horizons for the same score.
    """

    rows: List[Dict[str, float]] = []
    group_masks = groups if groups is not None else quantile_groups(score, group_count=group_count)
    unconditional = label.stack().dropna()
    for name, mask in group_masks.items():
        values = label.where(mask).stack().dropna()
        row = summarize_values(values, unconditional)
        row["group"] = name
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_event_paths(
    close: pd.DataFrame,
    events: pd.DataFrame,
    event_name: str,
    window_before: int = 10,
    window_after: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build event paths and matched same-date non-event control paths.

    Every control path uses stocks that did not trigger ``events`` on the
    corresponding event date. Both event and control prices are normalized to
    their own event-date close, so offset zero is comparable and equals zero.
    The implementation extracts all event windows in one NumPy operation and
    calculates each unique event date's control path only once.
    """

    aligned_events = events.reindex(index=close.index, columns=close.columns).fillna(False)
    event_matrix = aligned_events.to_numpy(dtype=bool, copy=False)
    date_indices, code_indices = np.nonzero(event_matrix)
    offsets = np.arange(-window_before, window_after + 1, dtype=int)
    observation_columns = [
        "event",
        "date",
        "code",
        "offset",
        "value",
        "control_mean",
        "control_count",
    ]
    if len(date_indices) == 0:
        return pd.DataFrame(), pd.DataFrame(columns=observation_columns)

    in_bounds = (date_indices >= window_before) & (
        date_indices + window_after < len(close.index)
    )
    date_indices = date_indices[in_bounds]
    code_indices = code_indices[in_bounds]
    if len(date_indices) == 0:
        return pd.DataFrame(), pd.DataFrame(columns=observation_columns)

    close_values = close.to_numpy(dtype=float, copy=False)
    anchors = close_values[date_indices, code_indices]
    valid_anchors = np.isfinite(anchors) & (anchors != 0)
    date_indices = date_indices[valid_anchors]
    code_indices = code_indices[valid_anchors]
    anchors = anchors[valid_anchors]
    if len(date_indices) == 0:
        return pd.DataFrame(), pd.DataFrame(columns=observation_columns)

    window_rows = date_indices[:, None] + offsets[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        event_paths = close_values[window_rows, code_indices[:, None]] / anchors[:, None] - 1.0
    event_paths[~np.isfinite(event_paths)] = np.nan

    unique_dates, date_inverse = np.unique(date_indices, return_inverse=True)
    control_means = np.full((len(unique_dates), len(offsets)), np.nan, dtype=float)
    control_counts = np.zeros((len(unique_dates), len(offsets)), dtype=int)
    for unique_idx, date_idx in enumerate(unique_dates):
        control_codes = np.flatnonzero(~event_matrix[date_idx])
        if len(control_codes) == 0:
            continue
        control_anchors = close_values[date_idx, control_codes]
        valid_controls = np.isfinite(control_anchors) & (control_anchors != 0)
        control_codes = control_codes[valid_controls]
        control_anchors = control_anchors[valid_controls]
        if len(control_codes) == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = (
                close_values[date_idx + offsets[:, None], control_codes]
                / control_anchors[None, :]
                - 1.0
            )
        finite = np.isfinite(normalized)
        counts = finite.sum(axis=1)
        totals = np.where(finite, normalized, 0.0).sum(axis=1)
        control_counts[unique_idx] = counts
        np.divide(
            totals,
            counts,
            out=control_means[unique_idx],
            where=counts > 0,
        )

    matched_control_means = control_means[date_inverse]
    matched_control_counts = control_counts[date_inverse]
    event_path_frame = pd.DataFrame(event_paths, columns=offsets)
    matched_control_summary = np.where(
        np.isfinite(event_paths),
        matched_control_means,
        np.nan,
    )
    control_path_frame = pd.DataFrame(matched_control_summary, columns=offsets)
    summary = pd.DataFrame(
        {
            "mean": event_path_frame.mean(axis=0),
            "median": event_path_frame.median(axis=0),
            "p25": event_path_frame.quantile(0.25, axis=0),
            "p75": event_path_frame.quantile(0.75, axis=0),
            "control_mean": control_path_frame.mean(axis=0),
        }
    )
    summary["excess_mean"] = summary["mean"] - summary["control_mean"]
    summary.index.name = "offset"

    path_size = len(offsets)
    flat_values = event_paths.reshape(-1)
    keep = np.isfinite(flat_values)
    observations = pd.DataFrame(
        {
            "event": np.repeat(event_name, len(flat_values))[keep],
            "date": np.repeat(close.index.to_numpy()[date_indices], path_size)[keep],
            "code": np.repeat(close.columns.to_numpy()[code_indices], path_size)[keep],
            "offset": np.tile(offsets, len(date_indices))[keep],
            "value": flat_values[keep],
            "control_mean": matched_control_means.reshape(-1)[keep],
            "control_count": matched_control_counts.reshape(-1)[keep],
        },
        columns=observation_columns,
    )
    return summary, observations


def cumulative_event_path(
    close: pd.DataFrame,
    events: pd.DataFrame,
    window_before: int = 10,
    window_after: int = 20,
) -> pd.DataFrame:
    """Return the aggregate normalized event and non-event control paths."""

    summary, _ = analyze_event_paths(
        close,
        events,
        event_name="",
        window_before=window_before,
        window_after=window_after,
    )
    return summary


def event_path_observations(
    close: pd.DataFrame,
    events: pd.DataFrame,
    event_name: str,
    window_before: int = 10,
    window_after: int = 20,
) -> pd.DataFrame:
    """Build long event paths with matched same-date non-event controls."""

    _, observations = analyze_event_paths(
        close,
        events,
        event_name=event_name,
        window_before=window_before,
        window_after=window_after,
    )
    return observations
