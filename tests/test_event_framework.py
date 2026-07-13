"""Unit tests for the event-study framework."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from event_study_framework.data import MarketPanel, normalize_wind_code
from event_study_framework.events import deduplicate_event_matrix, ma_break_event
from event_study_framework.labels import forward_mae, forward_mfe, forward_return
from event_study_framework.report import render_event_report
from event_study_framework.statistics import (
    analyze_event_paths,
    bootstrap_mean_effect,
    cumulative_event_path,
    event_path_observations,
    group_label_summary,
)
from event_study_framework.study import EventMeta, EventStudyConfig, EventStudyResult, EventStudyRunner, event_definitions_to_matrices
from event_study_framework.controls import (
    ControlConfig,
    align_event_matrix,
    bootstrap_rows_effect,
    collect_event_and_control_values,
    residualize_by_date,
)


def make_panel() -> MarketPanel:
    """Create a small synthetic market panel."""

    idx = pd.date_range("2024-01-01", periods=90)
    columns = ["300308.SZ", "300274.SZ"]
    close = pd.DataFrame(
        {
            "300308.SZ": np.r_[np.linspace(10, 18, 70), np.linspace(17, 14, 20)],
            "300274.SZ": np.linspace(20, 30, 90),
        },
        index=idx,
    )
    panel = MarketPanel(
        {
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": pd.DataFrame(1_000_000.0, index=idx, columns=columns),
            "amount": close * 1_000_000.0,
        }
    )
    return panel


class EventFrameworkTest(unittest.TestCase):
    """Tests for event-study components."""

    def test_normalize_wind_code(self) -> None:
        """Normalize common A-share symbols."""

        self.assertEqual(normalize_wind_code("300308"), "300308.SZ")
        self.assertEqual(normalize_wind_code("sh600000"), "600000.SH")
        self.assertEqual(normalize_wind_code("300274.SZ"), "300274.SZ")

    def test_forward_labels(self) -> None:
        """Forward labels should use future prices only."""

        close = pd.DataFrame({"A": [10.0, 11.0, 9.0, 12.0]})
        self.assertAlmostEqual(forward_return(close, 2).iloc[0, 0], -0.1)
        self.assertAlmostEqual(forward_mfe(close, 2).iloc[0, 0], 0.1)
        self.assertAlmostEqual(forward_mae(close, 2).iloc[0, 0], -0.1)

    def test_optional_excursion_labels(self) -> None:
        """MFE and MAE should only be built when explicitly requested."""

        panel = make_panel()
        runner = EventStudyRunner(
            panel=panel,
            event_matrices={},
            config=EventStudyConfig(
                horizons=(3,),
                label_types=("ret", "mfe", "mae"),
                show_progress=False,
            ),
        )
        self.assertEqual(set(runner.labels), {"ret_3d", "mfe_3d", "mae_3d"})

    def test_deduplicate_event_matrix(self) -> None:
        """Event de-duplication should operate per column."""

        events = pd.DataFrame({"A": [False, True, True, False, True], "B": [True, False, True, False, False]})
        deduped = deduplicate_event_matrix(events, cooldown_days=2)
        self.assertEqual(deduped["A"].tolist(), [False, True, False, False, True])
        self.assertEqual(deduped["B"].tolist(), [True, False, False, False, False])

    def test_event_matrix_rejects_non_binary_values(self) -> None:
        """Externally supplied event matrices must contain only zero and one."""

        reference = pd.DataFrame({"A": [1.0, 2.0]})
        events = pd.DataFrame({"A": [0, 2]})
        with self.assertRaisesRegex(ValueError, "must be 0/1"):
            align_event_matrix(events, reference)

    def test_event_study_runner(self) -> None:
        """Runner should accept decoupled 0/1 event matrices."""

        panel = make_panel()
        event_matrix = pd.DataFrame(0, index=panel["close"].index, columns=panel["close"].columns)
        event_matrix.iloc[50, 0] = 1
        runner = EventStudyRunner(
            panel=panel,
            event_matrices={"manual_event": event_matrix},
            event_meta={"manual_event": EventMeta(name="manual_event", direction="exit", cooldown_days=3)},
            config=EventStudyConfig(horizons=(5,), n_bootstrap=20, control_modes=("same_date",), show_progress=False),
        )
        result = runner.run()
        self.assertFalse(result.summary.empty)
        self.assertIn("manual_event", result.event_matrices)
        self.assertFalse(result.event_observations.empty)
        self.assertFalse(result.event_path_observations.empty)
        self.assertEqual(set(runner.labels), {"ret_5d"})

    def test_direction_aware_signal_assessment(self) -> None:
        """Strong positive excess should support entry and oppose exit use."""

        index = pd.date_range("2020-01-01", periods=160)
        event_positions = np.arange(10, 150, 7)
        returns = np.zeros(len(index))
        returns[event_positions + 1] = 0.02
        close = pd.DataFrame(
            {
                "EVENT": 100.0 * np.cumprod(1.0 + returns),
                "CONTROL": 100.0,
            },
            index=index,
        )
        panel = MarketPanel({"close": close})
        event_matrix = pd.DataFrame(0, index=index, columns=close.columns)
        event_matrix.iloc[event_positions, 0] = 1
        result = EventStudyRunner(
            panel=panel,
            event_matrices={"entry_signal": event_matrix, "exit_signal": event_matrix},
            event_meta={
                "entry_signal": EventMeta(name="entry_signal", direction="entry", cooldown_days=0),
                "exit_signal": EventMeta(name="exit_signal", direction="exit", cooldown_days=0),
            },
            config=EventStudyConfig(
                horizons=(1,),
                n_bootstrap=199,
                control_modes=("same_date",),
                purge_days=0,
                show_progress=False,
            ),
        ).run()
        assessments = result.summary.set_index("event")
        self.assertEqual(assessments.loc["entry_signal", "assessment"], "支持开仓")
        self.assertEqual(assessments.loc["exit_signal", "assessment"], "方向相反")
        self.assertGreater(assessments.loc["entry_signal", "signal_edge"], 0.0)
        self.assertLess(assessments.loc["exit_signal", "signal_edge"], 0.0)
        self.assertLessEqual(assessments.loc["entry_signal", "q_value"], 0.05)
        self.assertIn("开仓触发", assessments.loc["entry_signal", "action_hint"])

    def test_event_study_runner_parallel(self) -> None:
        """Runner should support event-level multiprocessing."""

        panel = make_panel()
        first_event = pd.DataFrame(0, index=panel["close"].index, columns=panel["close"].columns)
        second_event = first_event.copy()
        first_event.iloc[50, 0] = 1
        second_event.iloc[55, 1] = 1
        runner = EventStudyRunner(
            panel=panel,
            event_matrices={"first_event": first_event, "second_event": second_event},
            config=EventStudyConfig(
                horizons=(3,),
                n_bootstrap=5,
                control_modes=("same_date",),
                n_jobs=2,
                show_progress=False,
            ),
        )
        result = runner.run()
        self.assertEqual(set(result.event_matrices), {"first_event", "second_event"})
        self.assertEqual(set(result.summary["event"]), {"first_event", "second_event"})
        serial_result = EventStudyRunner(
            panel=panel,
            event_matrices={"first_event": first_event, "second_event": second_event},
            config=EventStudyConfig(
                horizons=(3,),
                n_bootstrap=5,
                control_modes=("same_date",),
                n_jobs=1,
                show_progress=False,
            ),
        ).run()
        sort_columns = ["event", "label", "control_mode"]
        pd.testing.assert_frame_equal(
            result.summary.sort_values(sort_columns).reset_index(drop=True),
            serial_result.summary.sort_values(sort_columns).reset_index(drop=True),
        )

    def test_residualized_labels_are_cached_per_label(self) -> None:
        """Residualization should run once per label, not once per event or mode."""

        panel = make_panel()
        event_matrix = pd.DataFrame(0, index=panel["close"].index, columns=panel["close"].columns)
        with patch("event_study_framework.study.residualize_by_date", wraps=residualize_by_date) as mocked:
            EventStudyRunner(
                panel=panel,
                event_matrices={"first": event_matrix, "second": event_matrix},
                config=EventStudyConfig(
                    horizons=(1, 3),
                    n_bootstrap=0,
                    control_modes=("same_date", "same_date_style"),
                    residualize_styles=True,
                    show_progress=False,
                ),
            )
        self.assertEqual(mocked.call_count, 2)

    def test_event_definitions_to_matrices(self) -> None:
        """Event generators should be separable from the evaluator."""

        panel = make_panel()
        matrices = event_definitions_to_matrices(
            panel,
            [ma_break_event(ma_window=5, trend_window=20, trend_return_threshold=0.1)],
        )
        self.assertIn("ma5_break_after_trend", matrices)
        self.assertEqual(set(matrices["ma5_break_after_trend"].stack().dropna().unique()).issubset({0, 1, False, True}), True)

    def test_collect_same_date_style_controls(self) -> None:
        """Control collection should support same-date style matching."""

        panel = make_panel()
        label = forward_return(panel["close"], 5)
        events = pd.DataFrame(0, index=panel["close"].index, columns=panel["close"].columns)
        events.iloc[50, 0] = 1
        lncap = pd.DataFrame(
            {
                "300308.SZ": np.linspace(10, 12, len(panel["close"])),
                "300274.SZ": np.linspace(10.1, 12.1, len(panel["close"])),
            },
            index=panel["close"].index,
        )
        rows = collect_event_and_control_values(
            label,
            events,
            ControlConfig(mode="same_date_style", style_name="lncap", min_controls=1),
            styles={"lncap": lncap},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]["mode"], "same_date_style")

    def test_group_label_summary(self) -> None:
        """Factor quantile summaries should include all groups."""

        idx = pd.date_range("2024-01-01", periods=30)
        score = pd.DataFrame({"A": np.arange(30), "B": np.arange(30)[::-1]}, index=idx)
        label = score.pct_change().shift(-1)
        table = group_label_summary(score, label, group_count=5)
        self.assertEqual(set(table["group"]), {"Q1", "Q2", "Q3", "Q4", "Q5"})

    def test_bootstrap_mean_effect(self) -> None:
        """Bootstrap should return finite statistics for non-empty samples."""

        event = pd.Series([0.1, 0.2, 0.15])
        control = pd.Series([0.0, -0.01, 0.02, 0.01])
        result = bootstrap_mean_effect(event, control, n_bootstrap=20)
        self.assertTrue(np.isfinite(result.observed))

    def test_bootstrap_gpu_fallback(self) -> None:
        """GPU bootstrap option should fall back cleanly when CuPy is unavailable."""

        rows = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "excess_mean": [0.01, -0.02, 0.03],
            }
        )
        result = bootstrap_rows_effect(rows, n_bootstrap=5, use_gpu=True)
        self.assertTrue(np.isfinite(result["bootstrap_p"]))

    def test_cluster_bootstrap_tests_zero_effect_null(self) -> None:
        """A consistently positive clustered effect should have a small p-value."""

        rows = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=20),
                "excess_mean": [0.02] * 20,
            }
        )
        result = bootstrap_rows_effect(rows, n_bootstrap=199, seed=7)
        self.assertAlmostEqual(result["bootstrap_p"], 1.0 / 200.0)
        self.assertAlmostEqual(result["ci_low"], 0.02)
        self.assertEqual(result["bootstrap_clusters"], 20.0)

    def test_zero_bootstrap_disables_resampling(self) -> None:
        """Zero repetitions should skip bootstrap without failing the study."""

        rows = pd.DataFrame({"excess_mean": [0.01, -0.02, 0.03]})
        result = bootstrap_rows_effect(rows, n_bootstrap=0)
        self.assertTrue(np.isnan(result["bootstrap_p"]))

    def test_cumulative_event_path(self) -> None:
        """Average event path should contain relative event-day coordinates."""

        panel = make_panel()
        events = pd.DataFrame(False, index=panel["close"].index, columns=panel["close"].columns)
        events.iloc[40, 0] = True
        path = cumulative_event_path(panel["close"], events, window_before=3, window_after=4)
        self.assertEqual(path.index.tolist(), [-3, -2, -1, 0, 1, 2, 3, 4])

    def test_event_path_observations(self) -> None:
        """Long event paths should preserve event dates and relative offsets."""

        panel = make_panel()
        events = pd.DataFrame(False, index=panel["close"].index, columns=panel["close"].columns)
        events.iloc[40, 0] = True
        paths = event_path_observations(panel["close"], events, "manual_event", window_before=2, window_after=2)
        self.assertEqual(paths["offset"].tolist(), [-2, -1, 0, 1, 2])
        self.assertEqual(paths["event"].unique().tolist(), ["manual_event"])
        self.assertAlmostEqual(paths.loc[paths["offset"] == 0, "value"].iloc[0], 0.0)

    def test_event_path_includes_same_date_non_event_control(self) -> None:
        """Path controls should use stocks not triggered on the event date."""

        index = pd.date_range("2024-01-01", periods=5)
        close = pd.DataFrame(
            {
                "EVENT": [9.0, 10.0, 10.0, 12.0, 12.0],
                "CONTROL_UP": [18.0, 18.0, 20.0, 22.0, 22.0],
                "CONTROL_DOWN": [33.0, 33.0, 30.0, 27.0, 27.0],
            },
            index=index,
        )
        events = pd.DataFrame(False, index=index, columns=close.columns)
        events.loc[index[2], "EVENT"] = True
        summary, observations = analyze_event_paths(
            close,
            events,
            event_name="manual_event",
            window_before=1,
            window_after=1,
        )
        event_day = observations.loc[observations["offset"] == 0].iloc[0]
        next_day = observations.loc[observations["offset"] == 1].iloc[0]
        self.assertAlmostEqual(event_day["value"], 0.0)
        self.assertAlmostEqual(event_day["control_mean"], 0.0)
        self.assertEqual(event_day["control_count"], 2)
        self.assertAlmostEqual(next_day["value"], 0.2)
        self.assertAlmostEqual(next_day["control_mean"], 0.0)
        self.assertAlmostEqual(summary.loc[1, "control_mean"], 0.0)

    def test_report_interactive_row_cap(self) -> None:
        """HTML reports should cap embedded observations without dropping summary tables."""

        observations = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "code": [f"A{i}" for i in range(5)],
                "event": ["manual_event"] * 5,
                "direction": ["entry"] * 5,
                "label": ["ret_1d"] * 5,
                "horizon": [1] * 5,
                "event_value": np.linspace(-0.02, 0.02, 5),
                "control_mean": [0.0] * 5,
                "control_median": [0.0] * 5,
                "excess_mean": np.linspace(-0.02, 0.02, 5),
                "control_count": [10] * 5,
                "mode": ["same_date"] * 5,
            }
        )
        result = EventStudyResult(
            summary=pd.DataFrame(
                {
                    "event": ["manual_event"],
                    "direction": ["entry"],
                    "label": ["ret_1d"],
                    "control_mode": ["same_date"],
                    "events": [5],
                    "event_dates": [5],
                    "mean": [0.0],
                    "control_mean": [0.0],
                    "excess_mean": [0.0],
                    "hit_rate_negative": [0.4],
                    "bootstrap_p": [0.5],
                    "q_value": [0.6],
                    "bootstrap_clusters": [5],
                    "signal_role": ["开仓"],
                    "signal_edge": [0.0],
                    "directional_hit_rate": [0.5],
                    "signal_ci_low": [-0.01],
                    "signal_ci_high": [0.01],
                    "assessment": ["证据不足"],
                }
            ),
            event_paths={"manual_event": pd.DataFrame({"mean": [0.0], "median": [0.0], "p25": [0.0], "p75": [0.0]}, index=[0])},
            event_matrices={},
            event_observations=observations,
            group_summaries={},
            recent_events=pd.DataFrame(),
            horizons=(1,),
        )
        with TemporaryDirectory() as tmpdir:
            output = render_event_report(result, Path(tmpdir) / "report.html", max_interactive_rows=2)
            html = output.read_text(encoding="utf-8")
        self.assertIn("event_observations.csv", html)
        self.assertIn("manual_event", html)
        self.assertIn("dateWindow", html)
        self.assertIn("controlChart", html)
        self.assertIn("cumulativeExcessChart", html)
        self.assertIn("const bins = 50", html)
        self.assertIn('id="horizonSelect"', html)
        self.assertIn('id="feeSelect"', html)
        self.assertIn('id="strategyBacktestChart"', html)
        self.assertNotIn("horizonSlider", html)
        self.assertIn("同日未触发对照均值", html)
        self.assertIn("relativeDayLabel", html)
        self.assertIn("PATH_INDEX", html)
        self.assertIn("信号决策摘要", html)
        self.assertIn("drawHorizonEvidence", html)
        self.assertIn("FDR q", html)
        self.assertNotIn('<option value="mfe">', html)
        self.assertNotIn(".chart-grid { grid-template-columns: 1fr; }", html)


if __name__ == "__main__":
    unittest.main()
