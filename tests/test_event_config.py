"""Tests for configurable class-based events."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from event_study_framework.data import MarketPanel
from event_study_framework.event import Event, EventMeta
from event_study_framework.event_config import (
    BUILTIN_EVENT_CLASSES,
    EventConfigError,
    load_events_from_config,
)
from event_study_framework.events import MaBreakEvent
from event_study_framework.run_event_research import (
    default_events,
    load_event_factors,
    parse_args,
    required_factor_sources,
    required_market_collections,
)
from event_study_framework.study import EventMeta as StudyEventMeta


class EventConfigurationTest(unittest.TestCase):
    """Validate event class loading and compatibility APIs."""

    @staticmethod
    def _write_config(directory: str, payload: Dict[str, Any]) -> Path:
        """Write one UTF-8 YAML event configuration for a test."""

        path = Path(directory) / "events.yaml"
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def test_loads_only_enabled_classes_with_parameters(self) -> None:
        """Configuration should select classes and pass constructor parameters."""

        payload = {
            "version": 1,
            "events": [
                {
                    "class": "ma_break",
                    "params": {
                        "ma_window": 5,
                        "trend_window": 12,
                        "trend_return_threshold": 0.1,
                    },
                },
                {"class": "volume_stall", "enabled": False},
            ],
        }
        with TemporaryDirectory() as directory:
            events = load_events_from_config(self._write_config(directory, payload))
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], MaBreakEvent)
        self.assertEqual(events[0].ma_window, 5)
        self.assertEqual(events[0].trend_window, 12)
        self.assertEqual(events[0].name, "ma5_break_after_trend")

    def test_loads_trusted_external_event_class(self) -> None:
        """A module:Class reference should load an external Event subclass."""

        module_source = '''
from __future__ import annotations
from typing import Dict, Optional
import pandas as pd
from event_study_framework.data import MarketPanel
from event_study_framework.event import Event

class ThresholdEvent(Event):
    """Synthetic configurable event used by the unit test."""

    def __init__(self, threshold: float = 0.0) -> None:
        """Initialize the threshold event."""
        self.threshold = float(threshold)
        super().__init__(event_name="external_threshold", direction="entry")

    def compute(
        self,
        panel: MarketPanel,
        factors: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Trigger when close is above the configured threshold."""
        return panel["close"] > self.threshold
'''
        payload = {
            "version": 1,
            "events": [
                {
                    "class": "event_config_test_module:ThresholdEvent",
                    "event_name": "configured_external",
                    "params": {"threshold": 10.0},
                }
            ],
        }
        with TemporaryDirectory() as directory:
            module_path = Path(directory) / "event_config_test_module.py"
            module_path.write_text(module_source, encoding="utf-8")
            config_path = self._write_config(directory, payload)
            sys.path.insert(0, directory)
            try:
                events = load_events_from_config(config_path)
            finally:
                sys.path.remove(directory)
                sys.modules.pop("event_config_test_module", None)

        self.assertEqual(events[0].name, "configured_external")
        self.assertIsInstance(events[0], Event)
        close = pd.DataFrame({"A": [9.0, 11.0]})
        panel = MarketPanel({"close": close})
        self.assertEqual(events[0].evaluate(panel)["A"].tolist(), [False, True])

    def test_rejects_duplicate_and_unknown_event_configuration(self) -> None:
        """Invalid class names and duplicate output names should fail clearly."""

        duplicate_payload = {
            "version": 1,
            "events": [
                {"class": "ma_break", "event_name": "duplicate"},
                {"class": "breakout_volume", "event_name": "DUPLICATE"},
            ],
        }
        unknown_payload = {
            "version": 1,
            "events": [{"class": "missing_event"}],
        }
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(EventConfigError, "duplicate event_name"):
                load_events_from_config(self._write_config(directory, duplicate_payload))
            with self.assertRaisesRegex(EventConfigError, "unknown built-in event"):
                load_events_from_config(self._write_config(directory, unknown_payload))

    def test_study_reexports_unified_event_meta(self) -> None:
        """Old notebook imports should resolve to the unified metadata class."""

        self.assertIs(StudyEventMeta, EventMeta)

    def test_selected_events_limit_market_data_fields(self) -> None:
        """Built-in events should load only the market fields they actually use."""

        collections = required_market_collections(default_events())
        self.assertEqual(
            collections,
            ("adj_high", "adj_close", "volume"),
        )

    def test_yaml_run_defaults_are_overridden_by_cli(self) -> None:
        """The YAML run mapping should provide defaults while explicit CLI values win."""

        payload = {
            "version": 1,
            "run": {
                "codes": ["300308"],
                "start": "20200101",
                "horizons": [1, 5],
                "bootstrap": 321,
                "show_progress": False,
                "use_cache": False,
            },
            "events": [{"class": "ma_break"}],
        }
        with TemporaryDirectory() as directory:
            path = self._write_config(directory, payload)
            args = parse_args(
                [
                    "--event-config",
                    str(path),
                    "--bootstrap",
                    "55",
                    "--horizons",
                    "2",
                    "4",
                    "--use-cache",
                ]
            )
        self.assertEqual(args.codes, ["300308"])
        self.assertEqual(args.start, "20200101")
        self.assertEqual(args.bootstrap, 55)
        self.assertEqual(args.horizons, [2, 4])
        self.assertFalse(args.show_progress)
        self.assertTrue(args.use_cache)

    def test_builtin_event_classes_are_discovered_without_factories(self) -> None:
        """Classes in events.py should receive class-name and snake-case aliases."""

        import event_study_framework.events as event_module

        self.assertIs(BUILTIN_EVENT_CLASSES["MaBreakEvent"], MaBreakEvent)
        self.assertIs(BUILTIN_EVENT_CLASSES["ma_break"], MaBreakEvent)
        self.assertFalse(hasattr(event_module, "ma_break_event"))
        self.assertFalse(hasattr(event_module, "breakout_volume_event"))

    def test_event_factor_dependencies_are_deduplicated_and_loaded(self) -> None:
        """The main loader should infer shared factor sources from event classes."""

        class FactorEvent(Event):
            """Small event declaring its own Mongo factor dependency."""

            required_fields = ("close",)
            required_factors = {"quality": "FactorDB/quality_score"}

            def __init__(self, event_name: str) -> None:
                """Initialize a uniquely named test event."""

                super().__init__(event_name=event_name)

            def compute(
                self,
                panel: MarketPanel,
                factors: Optional[Dict[str, pd.DataFrame]] = None,
            ) -> pd.DataFrame:
                """Return a thresholded factor matrix."""

                if factors is None:
                    raise ValueError("factors are required")
                return factors["quality"] > 0

        class FakeSource:
            """Record factor-loading calls without a Mongo dependency."""

            def __init__(self) -> None:
                """Initialize an empty call list."""

                self.calls: List[Dict[str, Any]] = []

            def load_factor(self, **kwargs: Any) -> pd.DataFrame:
                """Return one synthetic factor and retain loader arguments."""

                self.calls.append(kwargs)
                return pd.DataFrame({"A": [1.0]})

        events = [FactorEvent("factor_one"), FactorEvent("factor_two")]
        source = FakeSource()
        factors = load_event_factors(source, events, ["A"], "20200101", "20200131")  # type: ignore[arg-type]

        self.assertEqual(required_factor_sources(events), {"quality": "FactorDB/quality_score"})
        self.assertEqual(list(factors), ["quality"])
        self.assertEqual(len(source.calls), 1)
        self.assertEqual(source.calls[0]["database"], "FactorDB")
        self.assertEqual(source.calls[0]["collection"], "quality_score")

    def test_conflicting_factor_sources_fail_before_data_loading(self) -> None:
        """One factor alias must not silently refer to two Mongo collections."""

        class ConflictingEvent(Event):
            """Event with a configurable conflicting source."""

            def __init__(self, event_name: str, source: str) -> None:
                """Initialize the event and instance-local dependency mapping."""

                self.required_factors = {"quality": source}
                super().__init__(event_name=event_name)

            def compute(
                self,
                panel: MarketPanel,
                factors: Optional[Dict[str, pd.DataFrame]] = None,
            ) -> pd.DataFrame:
                """Return an empty trigger matrix."""

                return panel["close"].notna() & False

        events = [
            ConflictingEvent("factor_a", "DB_A/quality"),
            ConflictingEvent("factor_b", "DB_B/quality"),
        ]
        with self.assertRaisesRegex(ValueError, "declared as both"):
            required_factor_sources(events)


if __name__ == "__main__":
    unittest.main()
