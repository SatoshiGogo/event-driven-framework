"""Built-in event detectors and event feature builders."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from event_study_framework.data import MarketPanel
from event_study_framework.event import Event, EventDefinition, EventFunction


def _positive_int(value: int, parameter_name: str) -> int:
    """Validate and return a strictly positive integer parameter."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{parameter_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{parameter_name} must be positive")
    return value


def _positive_float(value: float, parameter_name: str) -> float:
    """Validate and return a strictly positive numeric parameter."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{parameter_name} must be numeric")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{parameter_name} must be a finite positive number")
    return numeric


def rolling_zscore(frame: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Calculate rolling time-series z-score for every column."""

    validated_window = _positive_int(window, "window")
    mean = frame.rolling(validated_window, min_periods=max(5, validated_window // 3)).mean()
    std = frame.rolling(validated_window, min_periods=max(5, validated_window // 3)).std(ddof=0)
    return (frame - mean) / std.replace(0, np.nan)


def rolling_percentile(frame: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Calculate rolling percentile rank of the latest value for every column."""

    validated_window = _positive_int(window, "window")
    min_periods = max(20, validated_window // 4)
    return frame.rolling(validated_window, min_periods=min_periods).rank(
        method="max",
        pct=True,
    )


def deduplicate_event_matrix(events: pd.DataFrame, cooldown_days: int) -> pd.DataFrame:
    """Keep the first event per stock inside each cooldown window."""

    if isinstance(cooldown_days, bool) or not isinstance(cooldown_days, int):
        raise TypeError("cooldown_days must be an integer")
    if cooldown_days < 0:
        raise ValueError("cooldown_days must be non-negative")
    values = events.fillna(False).astype(bool).to_numpy()
    kept = np.zeros_like(values, dtype=bool)
    last_kept = np.full(values.shape[1], -cooldown_days - 1, dtype=np.int64)
    for row_idx in range(values.shape[0]):
        eligible = values[row_idx] & ((row_idx - last_kept) > cooldown_days)
        kept[row_idx, eligible] = True
        last_kept[eligible] = row_idx
    return pd.DataFrame(kept, index=events.index, columns=events.columns)


class MaBreakEvent(Event):
    """Moving-average breakdown after a strong prior trend."""

    required_fields = ("close",)
    required_factors = ()

    def __init__(
        self,
        ma_window: int = 20,
        trend_window: int = 60,
        trend_return_threshold: float = 0.25,
        event_name: Optional[str] = None,
    ) -> None:
        """Initialize a moving-average breakdown detector."""

        self.ma_window = _positive_int(ma_window, "ma_window")
        self.trend_window = _positive_int(trend_window, "trend_window")
        if isinstance(trend_return_threshold, bool) or not isinstance(
            trend_return_threshold, (int, float)
        ):
            raise TypeError("trend_return_threshold must be numeric")
        self.trend_return_threshold = float(trend_return_threshold)
        if not np.isfinite(self.trend_return_threshold):
            raise ValueError("trend_return_threshold must be finite")
        super().__init__(
            event_name=event_name or f"ma{self.ma_window}_break_after_trend",
            direction="exit",
            cooldown_days=10,
            description=f"强趋势后收盘跌破 MA{self.ma_window}",
        )

    def compute(
        self,
        panel: MarketPanel,
        factors: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Compute moving-average breakdown triggers."""

        close = panel["close"]
        moving_average = close.rolling(self.ma_window, min_periods=self.ma_window).mean()
        trend_return = close / close.shift(self.trend_window) - 1.0
        was_above = close.shift(1) >= moving_average.shift(1)
        break_down = close < moving_average
        trend_context = trend_return.shift(1) > self.trend_return_threshold
        return was_above & break_down & trend_context


class VolumeStallEvent(Event):
    """High-volume price stall close to a recent high."""

    required_fields = ("close", "high", "volume")
    required_factors = ()

    def __init__(
        self,
        trend_window: int = 60,
        trend_return_threshold: float = 0.25,
        volume_multiple: float = 1.8,
        near_high_ratio: float = 0.97,
        event_name: Optional[str] = None,
    ) -> None:
        """Initialize a volume-stall detector."""

        self.trend_window = _positive_int(trend_window, "trend_window")
        if isinstance(trend_return_threshold, bool) or not isinstance(
            trend_return_threshold, (int, float)
        ):
            raise TypeError("trend_return_threshold must be numeric")
        self.trend_return_threshold = float(trend_return_threshold)
        if not np.isfinite(self.trend_return_threshold):
            raise ValueError("trend_return_threshold must be finite")
        self.volume_multiple = _positive_float(volume_multiple, "volume_multiple")
        self.near_high_ratio = _positive_float(near_high_ratio, "near_high_ratio")
        super().__init__(
            event_name=event_name or "volume_stall_near_high",
            direction="exit",
            cooldown_days=10,
            description="高位放量但价格滞涨",
        )

    def compute(
        self,
        panel: MarketPanel,
        factors: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Compute high-volume price-stall triggers."""

        close = panel["close"]
        high = panel["high"]
        volume = panel["volume"]
        volume_mean = volume.rolling(20, min_periods=20).mean()
        trend_return = close / close.shift(self.trend_window) - 1.0
        recent_high = close.rolling(
            self.trend_window,
            min_periods=self.trend_window,
        ).max().shift(1)
        near_high = high >= recent_high * self.near_high_ratio
        return (
            (trend_return > self.trend_return_threshold)
            & near_high
            & (volume > volume_mean * self.volume_multiple)
            & (close <= close.shift(1) * 1.01)
        )


class BreakoutVolumeEvent(Event):
    """Volume-confirmed breakout above a recent closing high."""

    required_fields = ("close", "volume")
    required_factors = ()

    def __init__(
        self,
        breakout_window: int = 60,
        volume_multiple: float = 1.5,
        ma_window: int = 20,
        event_name: Optional[str] = None,
    ) -> None:
        """Initialize a volume-confirmed breakout detector."""

        self.breakout_window = _positive_int(breakout_window, "breakout_window")
        self.volume_multiple = _positive_float(volume_multiple, "volume_multiple")
        self.ma_window = _positive_int(ma_window, "ma_window")
        super().__init__(
            event_name=event_name or "volume_breakout",
            direction="entry",
            cooldown_days=10,
            description="放量突破近端高点",
        )

    def compute(
        self,
        panel: MarketPanel,
        factors: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Compute volume-confirmed breakout triggers."""

        close = panel["close"]
        volume = panel["volume"]
        prior_high = close.rolling(
            self.breakout_window,
            min_periods=self.breakout_window,
        ).max().shift(1)
        volume_mean = volume.rolling(20, min_periods=20).mean()
        moving_average = close.rolling(self.ma_window, min_periods=self.ma_window).mean()
        return (
            (close > prior_high)
            & (volume > volume_mean * self.volume_multiple)
            & (close > moving_average)
        )


class FactorExtremeEvent(Event):
    """Extreme time-series percentile event for a named factor."""

    required_fields = ("close",)

    def __init__(
        self,
        factor_name: str,
        side: str = "high",
        percentile: float = 0.8,
        window: int = 252,
        direction: str = "entry",
        event_name: Optional[str] = None,
    ) -> None:
        """Initialize a factor-percentile extreme detector."""

        if not isinstance(factor_name, str) or not factor_name:
            raise ValueError("factor_name must be a non-empty string")
        if side not in {"high", "low"}:
            raise ValueError("side must be 'high' or 'low'")
        if isinstance(percentile, bool) or not isinstance(percentile, (int, float)):
            raise TypeError("percentile must be numeric")
        numeric_percentile = float(percentile)
        if not np.isfinite(numeric_percentile) or not 0.0 < numeric_percentile < 1.0:
            raise ValueError("percentile must be strictly between 0 and 1")
        self.factor_name = factor_name
        self.required_factors = (factor_name,)
        self.side = side
        self.percentile = numeric_percentile
        self.window = _positive_int(window, "window")
        super().__init__(
            event_name=event_name or f"{factor_name}_{side}_ts_pct_{numeric_percentile:.2f}",
            direction=direction,
            cooldown_days=20,
            description=f"{factor_name} 时序分位数处于 {side} 极端区间",
        )

    def compute(
        self,
        panel: MarketPanel,
        factors: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Compute factor-percentile extreme triggers."""

        if not factors or self.factor_name not in factors:
            raise ValueError(f"Factor {self.factor_name} is required for this event")
        score = rolling_percentile(
            factors[self.factor_name].reindex_like(panel["close"]),
            window=self.window,
        )
        if self.side == "high":
            return score >= self.percentile
        return score <= 1.0 - self.percentile


def _to_event_definition(event: Event) -> EventDefinition:
    """Wrap a class-based event in the legacy callable event container."""

    return EventDefinition(
        name=event.name,
        func=event.compute,
        direction=event.direction,
        cooldown_days=event.cooldown_days,
        description=event.description,
        required_fields=event.required_fields,
        required_factors=event.required_factors,
    )


def ma_break_event(
    ma_window: int = 20,
    trend_window: int = 60,
    trend_return_threshold: float = 0.25,
) -> EventDefinition:
    """Create a backward-compatible moving-average breakdown event."""

    return _to_event_definition(
        MaBreakEvent(
            ma_window=ma_window,
            trend_window=trend_window,
            trend_return_threshold=trend_return_threshold,
        )
    )


def volume_stall_event(
    trend_window: int = 60,
    trend_return_threshold: float = 0.25,
    volume_multiple: float = 1.8,
    near_high_ratio: float = 0.97,
) -> EventDefinition:
    """Create a backward-compatible high-level volume-stall event."""

    return _to_event_definition(
        VolumeStallEvent(
            trend_window=trend_window,
            trend_return_threshold=trend_return_threshold,
            volume_multiple=volume_multiple,
            near_high_ratio=near_high_ratio,
        )
    )


def breakout_volume_event(
    breakout_window: int = 60,
    volume_multiple: float = 1.5,
    ma_window: int = 20,
) -> EventDefinition:
    """Create a backward-compatible volume-confirmed breakout event."""

    return _to_event_definition(
        BreakoutVolumeEvent(
            breakout_window=breakout_window,
            volume_multiple=volume_multiple,
            ma_window=ma_window,
        )
    )


def factor_extreme_event(
    factor_name: str,
    side: str = "high",
    percentile: float = 0.8,
    window: int = 252,
    direction: str = "entry",
) -> EventDefinition:
    """Create a backward-compatible factor-percentile extreme event."""

    return _to_event_definition(
        FactorExtremeEvent(
            factor_name=factor_name,
            side=side,
            percentile=percentile,
            window=window,
            direction=direction,
        )
    )


__all__ = [
    "BreakoutVolumeEvent",
    "Event",
    "EventDefinition",
    "EventFunction",
    "FactorExtremeEvent",
    "MaBreakEvent",
    "VolumeStallEvent",
    "breakout_volume_event",
    "deduplicate_event_matrix",
    "factor_extreme_event",
    "ma_break_event",
    "rolling_percentile",
    "rolling_zscore",
    "volume_stall_event",
]
