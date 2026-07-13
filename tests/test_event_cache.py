"""Tests for persistent incremental event-result caching."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from event_study_framework.cache import EventResultCache
from event_study_framework.data import MarketPanel
from event_study_framework.events import BreakoutVolumeEvent, MaBreakEvent
from event_study_framework.run_event_research import run_incremental_event_study
from event_study_framework.study import EventStudyConfig


def make_cache_panel() -> MarketPanel:
    """Create a deterministic panel large enough for built-in event detectors."""

    index = pd.date_range("2022-01-01", periods=80)
    close = pd.DataFrame(
        {
            "A": 10.0 * np.exp(np.cumsum(np.sin(np.arange(80) / 4.0) * 0.01)),
            "B": 20.0 * np.exp(np.cumsum(np.cos(np.arange(80) / 5.0) * 0.008)),
        },
        index=index,
    )
    volume = pd.DataFrame(1_000_000.0, index=index, columns=close.columns)
    return MarketPanel({"close": close, "volume": volume})


class EventCacheTest(unittest.TestCase):
    """Verify cache hits, misses, hidden events, and parameter invalidation."""

    def test_incremental_cache_reuses_only_exact_event_configuration(self) -> None:
        """Removed events should stay cached and changed parameters should miss."""

        panel = make_cache_panel()
        config = EventStudyConfig(
            horizons=(1,),
            n_bootstrap=0,
            control_modes=("same_date",),
            purge_days=0,
            show_progress=False,
        )
        original_ma = MaBreakEvent(
            ma_window=3,
            trend_window=5,
            trend_return_threshold=0.0,
            event_name="ma_cached",
        )
        changed_ma = MaBreakEvent(
            ma_window=4,
            trend_window=5,
            trend_return_threshold=0.0,
            event_name="ma_cached",
        )
        breakout = BreakoutVolumeEvent(
            breakout_window=5,
            ma_window=3,
            event_name="breakout_cached",
        )
        with TemporaryDirectory() as directory:
            cache = EventResultCache(Path(directory))
            first, first_hits, first_misses = run_incremental_event_study(
                panel,
                [original_ma],
                factors={},
                style_controls={},
                config=config,
                cache=cache,
                use_cache=True,
                refresh_cache=False,
                universe=["A", "B"],
                start="20220101",
                end=None,
            )
            added, added_hits, added_misses = run_incremental_event_study(
                panel,
                [original_ma, breakout],
                factors={},
                style_controls={},
                config=config,
                cache=cache,
                use_cache=True,
                refresh_cache=False,
                universe=["A", "B"],
                start="20220101",
                end=None,
            )
            hidden, hidden_hits, hidden_misses = run_incremental_event_study(
                panel,
                [original_ma],
                factors={},
                style_controls={},
                config=config,
                cache=cache,
                use_cache=True,
                refresh_cache=False,
                universe=["A", "B"],
                start="20220101",
                end=None,
            )
            changed, changed_hits, changed_misses = run_incremental_event_study(
                panel,
                [changed_ma, breakout],
                factors={},
                style_controls={},
                config=config,
                cache=cache,
                use_cache=True,
                refresh_cache=False,
                universe=["A", "B"],
                start="20220101",
                end=None,
            )
            restored, restored_hits, restored_misses = run_incremental_event_study(
                panel,
                [original_ma, breakout],
                factors={},
                style_controls={},
                config=config,
                cache=cache,
                use_cache=True,
                refresh_cache=False,
                universe=["A", "B"],
                start="20220101",
                end=None,
            )

        self.assertEqual(first_hits, [])
        self.assertEqual(first_misses, ["ma_cached"])
        self.assertEqual(added_hits, ["ma_cached"])
        self.assertEqual(added_misses, ["breakout_cached"])
        self.assertEqual(set(added.summary["event"]), {"ma_cached", "breakout_cached"})
        self.assertEqual(hidden_hits, ["ma_cached"])
        self.assertEqual(hidden_misses, [])
        self.assertEqual(set(hidden.summary["event"]), {"ma_cached"})
        self.assertEqual(changed_hits, ["breakout_cached"])
        self.assertEqual(changed_misses, ["ma_cached"])
        self.assertEqual(set(changed.summary["event"]), {"ma_cached", "breakout_cached"})
        self.assertEqual(set(restored_hits), {"ma_cached", "breakout_cached"})
        self.assertEqual(restored_misses, [])
        self.assertEqual(set(restored.summary["event"]), {"ma_cached", "breakout_cached"})
        self.assertEqual(set(first.summary["event"]), {"ma_cached"})


if __name__ == "__main__":
    unittest.main()
