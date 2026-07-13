"""Tests for event-driven strategy backtests."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from event_study_framework.backtest import (
    build_target_weights,
    calculate_performance_metrics,
    run_event_backtest,
)
from event_study_framework.data import MarketPanel


class EventBacktestTest(unittest.TestCase):
    """Verify timing, costs, benchmark, and pool-removal behavior."""

    def setUp(self) -> None:
        """Create a small deterministic adjusted-close panel."""

        index = pd.date_range("2024-01-02", periods=5, freq="B")
        close = pd.DataFrame(
            {"A": [100.0, 100.0, 110.0, 121.0, 121.0], "B": [100.0] * 5},
            index=index,
        )
        status = pd.DataFrame(1.0, index=index, columns=close.columns)
        self.panel = MarketPanel({"close": close, "trade_status": status})
        self.events = pd.DataFrame(0, index=index, columns=close.columns)
        self.events.loc[index[0], "A"] = 1

    def test_signal_enters_next_day_and_holds_one_return_interval(self) -> None:
        """A one-day hold should earn only the first post-entry close return."""

        path, metrics = run_event_backtest(self.panel, self.events, holding_days=1, fee_rate=0.0)

        self.assertEqual(path["holdings"].tolist(), [0, 1, 0, 0, 0])
        self.assertAlmostEqual(path["strategy_return"].iloc[2], 0.10)
        self.assertAlmostEqual(path["strategy_cumulative"].iloc[-1], 0.10)
        self.assertAlmostEqual(path["benchmark_return"].iloc[2], 0.05)
        self.assertEqual(metrics["entries"], 1)
        self.assertEqual(metrics["exits"], 1)

    def test_fee_is_charged_on_entry_and_exit_turnover(self) -> None:
        """A round trip should pay the configured one-way fee twice."""

        path, _ = run_event_backtest(self.panel, self.events, holding_days=1, fee_rate=0.002)

        self.assertAlmostEqual(path["transaction_cost"].iloc[1], 0.002)
        self.assertAlmostEqual(path["transaction_cost"].iloc[2], 0.002)
        self.assertAlmostEqual(path["strategy_cumulative"].iloc[-1], 0.998 * 1.0978 - 1.0)

    def test_stock_is_removed_when_it_leaves_trade_pool(self) -> None:
        """An ineligible held stock should receive a zero target immediately."""

        status = self.panel["trade_status"].copy()
        status.loc[status.index[2]:, "A"] = 0.0
        panel = MarketPanel({**self.panel.fields, "trade_status": status})
        path, _ = run_event_backtest(panel, self.events, holding_days=3, fee_rate=0.002)

        self.assertEqual(path["holdings"].tolist()[:3], [0, 1, 0])
        self.assertAlmostEqual(path["turnover"].iloc[2], 1.0)

    def test_overlapping_signal_extends_target_vectorized(self) -> None:
        """A repeated signal should extend an existing position."""

        signals = self.events.astype(bool)
        signals.loc[signals.index[1], "A"] = True
        eligible = pd.DataFrame(True, index=signals.index, columns=signals.columns)
        weights = build_target_weights(signals, eligible, holding_days=2)

        self.assertEqual(weights["A"].gt(0).tolist(), [False, True, True, True, False])

    def test_metrics_report_positive_drawdown_magnitude(self) -> None:
        """Maximum drawdown should be exposed as a positive loss magnitude."""

        metrics = calculate_performance_metrics(pd.Series([0.10, -0.20, 0.05]), annualization=3)

        self.assertAlmostEqual(metrics["max_drawdown"], 0.20)
        self.assertTrue(np.isfinite(metrics["annualized_return"]))


if __name__ == "__main__":
    unittest.main()
