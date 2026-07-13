"""Vectorized event-driven portfolio backtests."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from event_study_framework.data import MarketPanel


def _aligned_backtest_inputs(
    panel: MarketPanel,
    events: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return aligned signals, adjusted close prices, and eligibility."""

    close = panel["close"].apply(pd.to_numeric, errors="coerce")
    signals = events.reindex(index=close.index, columns=close.columns).fillna(0).gt(0)
    execution_price = close.where(close.gt(0))
    trade_status = panel.fields.get("trade_status")
    if trade_status is None:
        eligible = execution_price.notna()
    else:
        status = trade_status.reindex_like(close).apply(pd.to_numeric, errors="coerce")
        eligible = status.gt(0) & execution_price.notna()
    return signals, execution_price, eligible


def build_target_weights(
    signals: pd.DataFrame,
    eligible: pd.DataFrame,
    holding_days: int,
) -> pd.DataFrame:
    """Build equal-weight targets from prior-day signals without Python loops.

    A signal becomes tradable one row later.  The target remains active for
    ``holding_days`` close-to-close return intervals; another signal received
    while holding naturally extends the exit date.
    """

    if holding_days <= 0:
        raise ValueError("holding_days must be positive")
    active = (
        signals.shift(1, fill_value=False)
        .rolling(window=int(holding_days), min_periods=1)
        .max()
        .astype(bool)
        & eligible
    )
    counts = active.sum(axis=1).replace(0, np.nan)
    return active.astype(float).div(counts, axis=0).fillna(0.0)


def calculate_performance_metrics(
    daily_returns: pd.Series,
    annualization: int = 252,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Calculate compounded portfolio performance statistics."""

    returns = pd.to_numeric(daily_returns, errors="coerce").fillna(0.0)
    if returns.empty:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "calmar_ratio": np.nan,
        }
    wealth = (1.0 + returns).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)
    years = len(returns) / float(annualization)
    annualized_return = float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and wealth.iloc[-1] > 0 else np.nan
    volatility = float(returns.std(ddof=1) * np.sqrt(annualization)) if len(returns) > 1 else np.nan
    daily_risk_free = (1.0 + float(risk_free_rate)) ** (1.0 / annualization) - 1.0
    standard_deviation = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
    sharpe = (
        float((returns.mean() - daily_risk_free) / standard_deviation * np.sqrt(annualization))
        if np.isfinite(standard_deviation) and standard_deviation > 0
        else np.nan
    )
    drawdown = wealth.div(wealth.cummax()).sub(1.0)
    max_drawdown = float(-drawdown.min())
    calmar = annualized_return / max_drawdown if max_drawdown > 0 and np.isfinite(annualized_return) else np.nan
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar) if np.isfinite(calmar) else np.nan,
    }


def run_event_backtest(
    panel: MarketPanel,
    events: pd.DataFrame,
    holding_days: int,
    fee_rate: float = 0.0,
    annualization: int = 252,
    risk_free_rate: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Backtest one event/holding-period/fee combination.

    Portfolio return is the equal-weight mean return of positions held during
    each close-to-close interval.  Transaction cost is charged on one-way target-weight
    turnover, so a complete round trip costs twice ``fee_rate``.
    """

    if fee_rate < 0:
        raise ValueError("fee_rate must be non-negative")
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    signals, execution_price, eligible = _aligned_backtest_inputs(panel, events)
    target_weights = build_target_weights(signals, eligible, holding_days)
    previous_weights = target_weights.shift(1, fill_value=0.0)
    asset_returns = execution_price.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    gross_return = previous_weights.mul(asset_returns.fillna(0.0)).sum(axis=1)
    turnover = target_weights.sub(previous_weights).abs().sum(axis=1)
    transaction_cost = turnover.mul(float(fee_rate)).clip(upper=1.0)
    strategy_return = (1.0 + gross_return).mul(1.0 - transaction_cost).sub(1.0)

    benchmark_mask = eligible & eligible.shift(1, fill_value=False)
    benchmark_return = asset_returns.where(benchmark_mask).mean(axis=1).fillna(0.0)
    strategy_wealth = (1.0 + strategy_return).cumprod()
    benchmark_wealth = (1.0 + benchmark_return).cumprod()
    path = pd.DataFrame(
        {
            "strategy_return": strategy_return,
            "benchmark_return": benchmark_return,
            "strategy_cumulative": strategy_wealth - 1.0,
            "benchmark_cumulative": benchmark_wealth - 1.0,
            "excess_cumulative": strategy_wealth.div(benchmark_wealth).sub(1.0),
            "holdings": target_weights.gt(0).sum(axis=1).astype(int),
            "turnover": turnover,
            "transaction_cost": transaction_cost,
        },
        index=execution_price.index,
    )

    metrics = calculate_performance_metrics(strategy_return, annualization, risk_free_rate)
    benchmark_metrics = calculate_performance_metrics(benchmark_return, annualization, risk_free_rate)
    excess_metrics = calculate_performance_metrics(
        (1.0 + strategy_return).div(1.0 + benchmark_return).sub(1.0),
        annualization,
        risk_free_rate=0.0,
    )
    entries = target_weights.gt(0) & previous_weights.eq(0)
    exits = target_weights.eq(0) & previous_weights.gt(0)
    metrics.update(
        {
            "benchmark_total_return": benchmark_metrics["total_return"],
            "benchmark_annualized_return": benchmark_metrics["annualized_return"],
            "excess_total_return": excess_metrics["total_return"],
            "excess_annualized_return": excess_metrics["annualized_return"],
            "active_day_ratio": float(path["holdings"].gt(0).mean()),
            "average_holdings": float(path.loc[path["holdings"].gt(0), "holdings"].mean()) if path["holdings"].gt(0).any() else 0.0,
            "total_turnover": float(turnover.sum()),
            "entries": int(entries.to_numpy().sum()),
            "exits": int(exits.to_numpy().sum()),
        }
    )
    return path, metrics


def run_backtest_grid(
    panel: MarketPanel,
    events: pd.DataFrame,
    horizons: Sequence[int],
    fee_rates: Sequence[float],
    annualization: int = 252,
    risk_free_rate: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all report dropdown combinations for one event matrix."""

    path_frames = []
    metric_rows = []
    for holding_days in horizons:
        for fee_rate in fee_rates:
            path, metrics = run_event_backtest(
                panel=panel,
                events=events,
                holding_days=int(holding_days),
                fee_rate=float(fee_rate),
                annualization=annualization,
                risk_free_rate=risk_free_rate,
            )
            frame = path.reset_index(names="date")
            frame["horizon"] = int(holding_days)
            frame["fee_rate"] = float(fee_rate)
            path_frames.append(frame)
            metric_rows.append({"horizon": int(holding_days), "fee_rate": float(fee_rate), **metrics})
    return pd.concat(path_frames, ignore_index=True), pd.DataFrame(metric_rows)
