"""Control-sample and purification tools for event-effect estimation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ControlConfig:
    """Configuration for event-effect purification."""

    mode: str = "same_date"
    style_name: Optional[str] = "lncap"
    style_bins: int = 5
    purge_days: int = 0
    min_controls: int = 3
    residualize_styles: bool = False


def align_event_matrix(events: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Validate and align an event 0/1 wide table to a reference pivot table."""

    aligned = events.reindex(index=reference.index, columns=reference.columns)
    invalid = ~(aligned.isna() | aligned.eq(0) | aligned.eq(1))
    if invalid.to_numpy().any():
        examples = pd.unique(aligned.where(invalid).stack())[:5].tolist()
        raise ValueError(f"event matrix values must be 0/1 or boolean; invalid values: {examples}")
    return aligned.fillna(0).astype(int)


def purge_event_clusters(events: pd.DataFrame, purge_days: int) -> pd.DataFrame:
    """Remove repeated triggers for each stock inside a purge window."""

    if purge_days <= 0:
        return events.fillna(0).astype(int)
    values = events.fillna(0).to_numpy(dtype=bool, copy=False)
    clean_values = np.zeros(values.shape, dtype=np.int8)
    last_kept = np.full(values.shape[1], -purge_days - 1, dtype=int)
    for row_idx in range(values.shape[0]):
        keep = values[row_idx] & ((row_idx - last_kept) > purge_days)
        clean_values[row_idx, keep] = 1
        last_kept[keep] = row_idx
    return pd.DataFrame(clean_values, index=events.index, columns=events.columns, dtype=int)


def event_observation_index(events: pd.DataFrame) -> pd.MultiIndex:
    """Return a MultiIndex of ``(date, code)`` event observations."""

    row_indices, column_indices = np.nonzero(events.fillna(0).to_numpy(dtype=bool, copy=False))
    return pd.MultiIndex.from_arrays(
        [events.index.take(row_indices), events.columns.take(column_indices)],
        names=[events.index.name, events.columns.name],
    )


def build_style_buckets(style: pd.DataFrame, bins: int) -> pd.DataFrame:
    """Build cross-sectional style buckets for all dates in one operation."""

    if bins <= 0:
        raise ValueError("bins must be positive")
    percentile = style.rank(axis=1, pct=True)
    return np.ceil(percentile * bins).clip(lower=1, upper=bins)


def residualize_by_date(label: pd.DataFrame, exposures: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Regress label values on style exposures cross-sectionally by date and return residuals."""

    if not exposures:
        return label
    residual = pd.DataFrame(np.nan, index=label.index, columns=label.columns)
    aligned_exposures = {name: frame.reindex_like(label) for name, frame in exposures.items() if not frame.empty}
    if not aligned_exposures:
        return label
    for date in label.index:
        y = label.loc[date]
        x_cols = []
        for frame in aligned_exposures.values():
            x_cols.append(frame.loc[date])
        x = pd.concat(x_cols, axis=1)
        x.columns = list(aligned_exposures.keys())
        valid = y.notna() & x.notna().all(axis=1)
        if valid.sum() < len(x.columns) + 3:
            residual.loc[date, valid] = y.loc[valid]
            continue
        x_mat = np.column_stack([np.ones(valid.sum()), x.loc[valid].to_numpy(dtype=float)])
        y_vec = y.loc[valid].to_numpy(dtype=float)
        try:
            beta = np.linalg.lstsq(x_mat, y_vec, rcond=None)[0]
            residual.loc[date, valid] = y_vec - x_mat @ beta
        except np.linalg.LinAlgError:
            residual.loc[date, valid] = y_vec
    return residual


def _same_date_control_mask(events: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Build a same-date non-event control mask."""

    return events.loc[date].fillna(0).astype(int) == 0


def _style_bucket_mask(
    style: pd.DataFrame,
    events: pd.DataFrame,
    date: pd.Timestamp,
    code: str,
    bins: int,
) -> pd.Series:
    """Build a same-date non-event control mask in the same style bucket."""

    style_row = style.loc[date]
    if code not in style_row.index or pd.isna(style_row.loc[code]):
        return _same_date_control_mask(events, date)
    pct = style_row.rank(pct=True)
    event_pct = pct.loc[code]
    bucket = min(max(int(np.ceil(event_pct * bins)), 1), bins)
    lower = (bucket - 1) / bins
    upper = bucket / bins
    same_bucket = (pct > lower) & (pct <= upper)
    return same_bucket & _same_date_control_mask(events, date)


def _control_stats(values: pd.Series) -> Tuple[float, float, int]:
    """Return mean, median, and count for a control-value sample."""

    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan, np.nan, 0
    return float(clean.mean()), float(clean.median()), int(len(clean))


def _observation_row(
    date: pd.Timestamp,
    code: str,
    event_value: float,
    control_mean: float,
    control_median: float,
    control_count: int,
    mode: str,
) -> Dict[str, object]:
    """Create one long-format event observation row."""

    return {
        "date": date,
        "code": code,
        "event_value": float(event_value),
        "control_mean": control_mean,
        "control_median": control_median,
        "excess_mean": float(event_value - control_mean) if pd.notna(control_mean) else np.nan,
        "control_count": int(control_count),
        "mode": mode,
    }


def _same_date_stats(
    label: pd.DataFrame,
    events: pd.DataFrame,
    date: pd.Timestamp,
    cache: Dict[pd.Timestamp, Tuple[float, float, int]],
) -> Tuple[float, float, int]:
    """Return cached same-date non-event control statistics."""

    if date not in cache:
        cache[date] = _control_stats(label.loc[date, _same_date_control_mask(events, date)])
    return cache[date]


def _event_positions(events: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return row and column integer positions for event observations."""

    return np.nonzero(events.to_numpy(dtype=bool, copy=False))


def _same_date_control_arrays(
    label_values: np.ndarray,
    event_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate same-date non-event mean, median, and count arrays."""

    finite = np.isfinite(label_values)
    control_mask = (~event_values) & finite
    counts = control_mask.sum(axis=1).astype(int)
    totals = np.where(control_mask, label_values, 0.0).sum(axis=1)
    means = np.full(label_values.shape[0], np.nan, dtype=float)
    np.divide(totals, counts, out=means, where=counts > 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(np.where(control_mask, label_values, np.nan), axis=1)
    return means, medians, counts


def _build_observation_frame(
    label: pd.DataFrame,
    label_values: np.ndarray,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    control_means: np.ndarray,
    control_medians: np.ndarray,
    control_counts: np.ndarray,
    mode: str,
) -> pd.DataFrame:
    """Build event-control observations from aligned NumPy arrays."""

    selected_values = label_values[row_indices, column_indices]
    valid = ~np.isnan(selected_values)
    selected_values = selected_values[valid]
    selected_means = control_means[valid]
    excess = selected_values - selected_means
    return pd.DataFrame(
        {
            "date": label.index.take(row_indices[valid]),
            "code": label.columns.take(column_indices[valid]),
            "event_value": selected_values.astype(float, copy=False),
            "control_mean": selected_means,
            "control_median": control_medians[valid],
            "excess_mean": excess,
            "control_count": control_counts[valid].astype(int, copy=False),
            "mode": mode,
        }
    )


def _collect_same_date_values(
    label: pd.DataFrame,
    events: pd.DataFrame,
    event_index: pd.MultiIndex,
    mode: str,
) -> pd.DataFrame:
    """Collect event observations against vectorized same-date controls."""

    del event_index
    row_indices, column_indices = _event_positions(events)
    label_values = label.to_numpy(dtype=float, copy=False)
    event_values = events.to_numpy(dtype=bool, copy=False)
    means_by_date, medians_by_date, counts_by_date = _same_date_control_arrays(
        label_values,
        event_values,
    )
    return _build_observation_frame(
        label,
        label_values,
        row_indices,
        column_indices,
        means_by_date[row_indices],
        medians_by_date[row_indices],
        counts_by_date[row_indices],
        mode,
    )


def _style_buckets_for_date(style: pd.DataFrame, date: pd.Timestamp, bins: int) -> pd.Series:
    """Calculate same-date style buckets from cross-sectional percentile ranks."""

    pct = style.loc[date].rank(pct=True)
    return np.ceil(pct * bins).clip(lower=1, upper=bins)


def _collect_same_date_style_values(
    label: pd.DataFrame,
    events: pd.DataFrame,
    event_index: pd.MultiIndex,
    style: pd.DataFrame,
    config: ControlConfig,
    style_buckets: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Collect event observations against vectorized same-date style controls."""

    del event_index
    row_indices, column_indices = _event_positions(events)
    label_values = label.to_numpy(dtype=float, copy=False)
    event_values = events.to_numpy(dtype=bool, copy=False)
    buckets = style_buckets if style_buckets is not None else build_style_buckets(style, config.style_bins)
    bucket_values = buckets.reindex_like(label).to_numpy(dtype=float, copy=False)
    same_date_means, same_date_medians, same_date_counts = _same_date_control_arrays(
        label_values,
        event_values,
    )

    date_count = label_values.shape[0]
    style_means = np.full((date_count, config.style_bins), np.nan, dtype=float)
    style_medians = np.full((date_count, config.style_bins), np.nan, dtype=float)
    style_counts = np.zeros((date_count, config.style_bins), dtype=int)
    finite_labels = np.isfinite(label_values)
    for bucket in range(1, config.style_bins + 1):
        mask = (~event_values) & finite_labels & (bucket_values == bucket)
        counts = mask.sum(axis=1).astype(int)
        totals = np.where(mask, label_values, 0.0).sum(axis=1)
        means = np.full(date_count, np.nan, dtype=float)
        np.divide(totals, counts, out=means, where=counts > 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            medians = np.nanmedian(np.where(mask, label_values, np.nan), axis=1)
        style_means[:, bucket - 1] = means
        style_medians[:, bucket - 1] = medians
        style_counts[:, bucket - 1] = counts

    event_buckets = bucket_values[row_indices, column_indices]
    valid_buckets = np.isfinite(event_buckets)
    bucket_indices = np.clip(
        np.nan_to_num(event_buckets, nan=1.0).astype(int) - 1,
        0,
        config.style_bins - 1,
    )
    selected_means = style_means[row_indices, bucket_indices]
    selected_medians = style_medians[row_indices, bucket_indices]
    selected_counts = style_counts[row_indices, bucket_indices]
    use_fallback = (~valid_buckets) | (selected_counts < config.min_controls)
    selected_means[use_fallback] = same_date_means[row_indices[use_fallback]]
    selected_medians[use_fallback] = same_date_medians[row_indices[use_fallback]]
    selected_counts[use_fallback] = same_date_counts[row_indices[use_fallback]]
    return _build_observation_frame(
        label,
        label_values,
        row_indices,
        column_indices,
        selected_means,
        selected_medians,
        selected_counts,
        config.mode,
    )


def collect_event_and_control_values(
    label: pd.DataFrame,
    events: pd.DataFrame,
    config: ControlConfig,
    styles: Optional[Dict[str, pd.DataFrame]] = None,
    style_buckets: Optional[pd.DataFrame] = None,
    events_aligned: bool = False,
) -> pd.DataFrame:
    """Collect event values and matched control values in long format.

    ``style_buckets`` can be precomputed with :func:`build_style_buckets` and
    reused across all labels for the same run.
    """

    styles = styles or {}
    aligned_events = events if events_aligned else align_event_matrix(events, label)
    if config.purge_days > 0:
        aligned_events = purge_event_clusters(aligned_events, config.purge_days)
    if config.residualize_styles:
        label_for_eval = residualize_by_date(label, styles)
    else:
        label_for_eval = label

    style = styles.get(config.style_name or "", pd.DataFrame()).reindex_like(label) if styles else pd.DataFrame()
    event_index = event_observation_index(aligned_events)
    if len(event_index) == 0:
        return pd.DataFrame()
    if config.mode == "same_date":
        return _collect_same_date_values(label_for_eval, aligned_events, event_index, config.mode)
    if config.mode == "same_date_style" and not style.empty:
        return _collect_same_date_style_values(
            label_for_eval,
            aligned_events,
            event_index,
            style,
            config,
            style_buckets=style_buckets,
        )
    if config.mode == "same_date_style":
        return _collect_same_date_values(label_for_eval, aligned_events, event_index, config.mode)

    row_indices, column_indices = _event_positions(aligned_events)
    label_values = label_for_eval.to_numpy(dtype=float, copy=False)
    if config.mode == "unconditional":
        event_values = aligned_events.to_numpy(dtype=bool, copy=False)
        valid_controls = (~event_values) & np.isfinite(label_values)
        control_sample = label_values[valid_controls]
        if len(control_sample) == 0:
            control_mean, control_median, control_count = np.nan, np.nan, 0
        else:
            control_mean = float(control_sample.mean())
            control_median = float(np.median(control_sample))
            control_count = int(len(control_sample))
        event_count = len(row_indices)
        return _build_observation_frame(
            label_for_eval,
            label_values,
            row_indices,
            column_indices,
            np.full(event_count, control_mean, dtype=float),
            np.full(event_count, control_median, dtype=float),
            np.full(event_count, control_count, dtype=int),
            config.mode,
        )

    rows: List[Dict[str, object]] = []
    stock_cache: Dict[str, pd.Series] = {}
    for date, code in event_index:
        event_value = label_for_eval.loc[date, code]
        if pd.isna(event_value):
            continue
        if config.mode == "event_stock_history":
            if code not in stock_cache:
                stock_cache[code] = label_for_eval[code].dropna()
            stock_values = stock_cache[code]
            control_values = stock_values.loc[stock_values.index != date]
        else:
            raise ValueError(f"Unsupported control mode: {config.mode}")

        current_mean, current_median, current_count = _control_stats(control_values)
        rows.append(_observation_row(date, code, float(event_value), current_mean, current_median, current_count, config.mode))
    return pd.DataFrame(rows)


def summarize_observation_rows(rows: pd.DataFrame) -> Dict[str, float]:
    """Summarize long event-control observations."""

    if rows.empty:
        return {
            "events": 0.0,
            "event_dates": 0.0,
            "mean": np.nan,
            "median": np.nan,
            "control_mean": np.nan,
            "excess_mean": np.nan,
            "excess_median": np.nan,
            "hit_rate_positive": np.nan,
            "hit_rate_negative": np.nan,
            "excess_hit_rate_positive": np.nan,
            "excess_hit_rate_negative": np.nan,
        }
    event_values = rows["event_value"].replace([np.inf, -np.inf], np.nan).dropna()
    excess = rows["excess_mean"].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "events": float(len(event_values)),
        "event_dates": float(rows["date"].nunique()),
        "mean": float(event_values.mean()) if len(event_values) else np.nan,
        "median": float(event_values.median()) if len(event_values) else np.nan,
        "control_mean": float(rows["control_mean"].mean()) if rows["control_mean"].notna().any() else np.nan,
        "excess_mean": float(excess.mean()) if len(excess) else np.nan,
        "excess_median": float(excess.median()) if len(excess) else np.nan,
        "hit_rate_positive": float((event_values > 0).mean()) if len(event_values) else np.nan,
        "hit_rate_negative": float((event_values < 0).mean()) if len(event_values) else np.nan,
        "excess_hit_rate_positive": float((excess > 0).mean()) if len(excess) else np.nan,
        "excess_hit_rate_negative": float((excess < 0).mean()) if len(excess) else np.nan,
    }


def _cluster_bootstrap_gpu(
    cluster_sums: np.ndarray,
    cluster_counts: np.ndarray,
    observed: float,
    n_bootstrap: int,
    seed: int,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap original and null-centered cluster means on a GPU."""

    import cupy as cp  # pylint: disable=import-outside-toplevel

    rng = cp.random.default_rng(seed)
    sums_gpu = cp.asarray(cluster_sums, dtype=cp.float64)
    counts_gpu = cp.asarray(cluster_counts, dtype=cp.float64)
    centered_sums_gpu = sums_gpu - observed * counts_gpu
    original_means = []
    null_means = []
    cluster_count = len(cluster_sums)
    for start in range(0, n_bootstrap, batch_size):
        current = min(batch_size, n_bootstrap - start)
        indices = rng.integers(0, cluster_count, size=(current, cluster_count))
        sampled_counts = counts_gpu[indices].sum(axis=1)
        original_means.append(sums_gpu[indices].sum(axis=1) / sampled_counts)
        null_means.append(centered_sums_gpu[indices].sum(axis=1) / sampled_counts)
    return (
        cp.asnumpy(cp.concatenate(original_means)),
        cp.asnumpy(cp.concatenate(null_means)),
    )


def _cluster_bootstrap_cpu(
    cluster_sums: np.ndarray,
    cluster_counts: np.ndarray,
    observed: float,
    n_bootstrap: int,
    seed: int,
    max_batch_elements: int = 2_000_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap original and null-centered cluster means in CPU batches."""

    rng = np.random.default_rng(seed)
    cluster_count = len(cluster_sums)
    batch_size = max(1, min(128, max_batch_elements // max(cluster_count, 1)))
    original_means = np.empty(n_bootstrap, dtype=float)
    null_means = np.empty(n_bootstrap, dtype=float)
    centered_sums = cluster_sums - observed * cluster_counts
    for start in range(0, n_bootstrap, batch_size):
        current = min(batch_size, n_bootstrap - start)
        indices = rng.integers(0, cluster_count, size=(current, cluster_count))
        sampled_counts = cluster_counts[indices].sum(axis=1)
        original_means[start : start + current] = (
            cluster_sums[indices].sum(axis=1) / sampled_counts
        )
        null_means[start : start + current] = (
            centered_sums[indices].sum(axis=1) / sampled_counts
        )
    return original_means, null_means


def bootstrap_rows_effect(
    rows: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
    use_gpu: bool = False,
) -> Dict[str, float]:
    """Test zero mean excess with an event-date cluster bootstrap.

    Event observations on the same date are resampled as one cluster. Confidence
    intervals use the original clustered samples, while the p-value uses samples
    centered on the zero-effect null. The add-one correction avoids zero p-values.
    """

    empty_result = {
        "bootstrap_p": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "bootstrap_clusters": 0.0,
    }
    if rows.empty or "date" not in rows or "excess_mean" not in rows:
        return empty_result
    clean = rows[["date", "excess_mean"]].copy()
    clean["excess_mean"] = clean["excess_mean"].replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=["date", "excess_mean"])
    if clean.empty:
        return empty_result
    clustered = clean.groupby("date", sort=False)["excess_mean"].agg(["sum", "count"])
    cluster_sums = clustered["sum"].to_numpy(dtype=float)
    cluster_counts = clustered["count"].to_numpy(dtype=float)
    cluster_count = len(cluster_sums)
    observed = float(cluster_sums.sum() / cluster_counts.sum())
    result = {
        "bootstrap_p": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "bootstrap_clusters": float(cluster_count),
    }
    if n_bootstrap <= 0 or cluster_count < 2:
        return result
    if use_gpu:
        try:
            boot, null_boot = _cluster_bootstrap_gpu(
                cluster_sums,
                cluster_counts,
                observed,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        except (ImportError, ModuleNotFoundError):
            boot, null_boot = _cluster_bootstrap_cpu(
                cluster_sums,
                cluster_counts,
                observed,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
    else:
        boot, null_boot = _cluster_bootstrap_cpu(
            cluster_sums,
            cluster_counts,
            observed,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    exceedances = int((np.abs(null_boot) >= abs(observed)).sum())
    result.update(
        {
            "bootstrap_p": float((exceedances + 1) / (n_bootstrap + 1)),
            "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
        }
    )
    return result
