"""Event-study engine for decoupled 0/1 wide-table events."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import pandas as pd

from event_study_framework.controls import (
    ControlConfig,
    align_event_matrix,
    bootstrap_rows_effect,
    build_style_buckets,
    collect_event_and_control_values,
    event_observation_index,
    purge_event_clusters,
    residualize_by_date,
    summarize_observation_rows,
)
from event_study_framework.data import MarketPanel
from event_study_framework.event import Event, EventMeta
from event_study_framework.labels import build_forward_labels
from event_study_framework.statistics import analyze_event_paths, group_label_summary, quantile_groups


@dataclass(frozen=True)
class EventStudyConfig:
    """Configuration for an event-study run."""

    horizons: Sequence[int] = (1, 3, 5, 10, 20)
    label_types: Sequence[str] = ("ret",)
    window_before: int = 10
    window_after: int = 20
    n_bootstrap: int = 2000
    min_events: int = 5
    significance_level: float = 0.05
    control_modes: Sequence[str] = ("same_date", "same_date_style")
    style_name: str = "lncap"
    style_bins: int = 5
    purge_days: int = 10
    residualize_styles: bool = False
    n_jobs: int = 1
    show_progress: bool = True
    use_gpu: bool = False
    backtest_fees: Sequence[float] = (0.0, 0.002)
    annualization: int = 252
    risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate study settings before allocating large label matrices."""

        if not self.horizons or any(int(horizon) <= 0 for horizon in self.horizons):
            raise ValueError("horizons must contain positive integers")
        label_types = {str(label_type).lower() for label_type in self.label_types}
        if "ret" not in label_types or not label_types.issubset({"ret", "mfe", "mae"}):
            raise ValueError("label_types must include ret and may only contain ret/mfe/mae")
        if self.n_bootstrap < 0:
            raise ValueError("n_bootstrap must be non-negative")
        if not 0.0 < self.significance_level < 1.0:
            raise ValueError("significance_level must be between zero and one")
        if not self.backtest_fees or any(float(fee) < 0 for fee in self.backtest_fees):
            raise ValueError("backtest_fees must contain non-negative values")
        if self.annualization <= 0:
            raise ValueError("annualization must be positive")


@dataclass
class EventStudyResult:
    """Container holding all computed event-study outputs."""

    summary: pd.DataFrame
    event_paths: Dict[str, pd.DataFrame]
    event_matrices: Dict[str, pd.DataFrame]
    event_observations: pd.DataFrame
    group_summaries: Dict[str, pd.DataFrame]
    recent_events: pd.DataFrame
    horizons: Sequence[int] = field(default_factory=tuple)
    event_path_observations: pd.DataFrame = field(default_factory=pd.DataFrame)
    strategy_backtests: Dict[str, pd.DataFrame] = field(default_factory=dict)
    strategy_metrics: Dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class _EventRunOutput:
    """Internal result for one event computation."""

    event_name: str
    summary_rows: List[Dict[str, object]]
    event_path: pd.DataFrame
    event_matrix: pd.DataFrame
    path_observations: pd.DataFrame
    observations: pd.DataFrame
    recent_rows: List[Dict[str, object]]


T = TypeVar("T")


def _maybe_progress(iterable: Iterable[T], enabled: bool, total: Optional[int], desc: str, unit: str) -> Iterable[T]:
    """Wrap an iterable with tqdm when progress output is enabled and available."""

    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit)


def _label_horizon(label_name: str) -> int:
    """Extract horizon from label names such as ``ret_20d``."""

    try:
        return int(label_name.split("_", 1)[1].rstrip("d"))
    except (IndexError, ValueError):
        return -1


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Adjust p-values with the Benjamini-Hochberg FDR procedure."""

    adjusted = pd.Series(float("nan"), index=p_values.index, dtype=float)
    valid = pd.to_numeric(p_values, errors="coerce").dropna().clip(lower=0.0, upper=1.0)
    if valid.empty:
        return adjusted
    ordered = valid.sort_values()
    count = len(ordered)
    ranks = pd.Series(range(1, count + 1), index=ordered.index, dtype=float)
    raw_adjusted = ordered * count / ranks
    monotone = raw_adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    adjusted.loc[monotone.index] = monotone
    return adjusted


def add_signal_assessment(
    summary: pd.DataFrame,
    significance_level: float,
    min_events: int,
) -> pd.DataFrame:
    """Add direction-aware evidence fields and multiple-test adjustment."""

    if summary.empty:
        return summary
    assessed = summary.copy()
    assessed["q_value"] = _benjamini_hochberg(assessed["bootstrap_p"])
    assessments: List[str] = []
    action_hints: List[str] = []
    for _, row in assessed.iterrows():
        role = "开仓" if row.get("direction") == "entry" else "平仓"
        events = float(row.get("events", 0.0))
        clusters = float(row.get("bootstrap_clusters", 0.0))
        q_value = row.get("q_value", float("nan"))
        ci_low = row.get("signal_ci_low", float("nan"))
        ci_high = row.get("signal_ci_high", float("nan"))
        if events < min_events or clusters < 5:
            assessment = "样本不足"
            action_hint = "扩大事件日期样本后再判断"
        elif pd.notna(q_value) and q_value < significance_level and pd.notna(ci_low) and ci_low > 0:
            assessment = f"支持{role}"
            action_hint = f"可作为{role}触发或过滤条件候选"
        elif pd.notna(q_value) and q_value < significance_level and pd.notna(ci_high) and ci_high < 0:
            assessment = "方向相反"
            action_hint = "不宜按配置方向使用；检查反向用途"
        else:
            assessment = "证据不足"
            action_hint = "不宜单独作为交易条件"
        assessments.append(assessment)
        action_hints.append(action_hint)
    assessed["assessment"] = assessments
    assessed["action_hint"] = action_hints
    assessed["significant_fdr"] = (
        assessed["q_value"].lt(significance_level)
        & assessed["signal_ci_low"].gt(0)
    )
    return assessed


def _directional_stability(rows: pd.DataFrame, direction_sign: float) -> Dict[str, float]:
    """Summarize whether the direction-aware edge persists across years and recency."""

    empty = {
        "positive_year_ratio": float("nan"),
        "evaluated_years": 0.0,
        "recent_signal_edge": float("nan"),
        "prior_signal_edge": float("nan"),
    }
    if rows.empty or "date" not in rows or "excess_mean" not in rows:
        return empty
    clean = rows[["date", "excess_mean"]].copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["excess_mean"] = pd.to_numeric(clean["excess_mean"], errors="coerce")
    clean = clean.dropna()
    if clean.empty:
        return empty
    clean["signal_edge"] = clean["excess_mean"] * direction_sign
    annual = clean.groupby(clean["date"].dt.year)["signal_edge"].mean()
    unique_dates = pd.Index(clean["date"].drop_duplicates().sort_values())
    split_index = max(1, int(len(unique_dates) * 2 / 3))
    split_date = unique_dates[min(split_index, len(unique_dates) - 1)]
    prior = clean.loc[clean["date"] < split_date, "signal_edge"]
    recent = clean.loc[clean["date"] >= split_date, "signal_edge"]
    return {
        "positive_year_ratio": float((annual > 0).mean()) if len(annual) else float("nan"),
        "evaluated_years": float(len(annual)),
        "recent_signal_edge": float(recent.mean()) if len(recent) else float("nan"),
        "prior_signal_edge": float(prior.mean()) if len(prior) else float("nan"),
    }


def _recent_event_rows(
    event_name: str,
    events: pd.DataFrame,
    close: pd.DataFrame,
    labels: Dict[str, pd.DataFrame],
    horizons: Sequence[int],
    limit: int = 30,
) -> List[Dict[str, object]]:
    """Build rows for the most recent events table."""

    rows: List[Dict[str, object]] = []
    event_index = event_observation_index(events)[-limit:]
    for date, code in event_index:
        row: Dict[str, object] = {
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "code": code,
            "event": event_name,
            "close": float(close.loc[date, code]),
        }
        for horizon in horizons:
            label_name = f"ret_{horizon}d"
            if date in labels[label_name].index and code in labels[label_name].columns:
                row[label_name] = labels[label_name].loc[date, code]
        rows.append(row)
    return rows


def _compute_event_outputs(
    event_name: str,
    raw_events: pd.DataFrame,
    meta: EventMeta,
    close: pd.DataFrame,
    labels: Dict[str, pd.DataFrame],
    recent_labels: Dict[str, pd.DataFrame],
    style_controls: Dict[str, pd.DataFrame],
    style_buckets: Optional[pd.DataFrame],
    config: EventStudyConfig,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> _EventRunOutput:
    """Compute all labels and control modes for one event matrix."""

    summary_rows: List[Dict[str, object]] = []
    observation_frames: List[pd.DataFrame] = []
    purge_days = max(config.purge_days, meta.cooldown_days)
    events = purge_event_clusters(raw_events, purge_days=purge_days)
    event_path, path_observations = analyze_event_paths(
        close,
        events,
        event_name=event_name,
        window_before=config.window_before,
        window_after=config.window_after,
    )

    for label_name, label in labels.items():
        for control_mode in config.control_modes:
            control_config = ControlConfig(
                mode=control_mode,
                style_name=config.style_name,
                style_bins=config.style_bins,
                purge_days=0,
                min_controls=3,
                residualize_styles=False,
            )
            rows = collect_event_and_control_values(
                label=label,
                events=events,
                config=control_config,
                styles=style_controls,
                style_buckets=style_buckets,
                events_aligned=True,
            )
            if not rows.empty:
                rows["event"] = event_name
                rows["direction"] = meta.direction
                rows["label"] = label_name
                rows["horizon"] = _label_horizon(label_name)
                rows["description"] = meta.description
                observation_frames.append(rows)
            summary = summarize_observation_rows(rows)
            summary.update(bootstrap_rows_effect(rows, n_bootstrap=config.n_bootstrap, use_gpu=config.use_gpu))
            direction_sign = 1.0 if meta.direction == "entry" else -1.0
            directional_hit_rate = (
                summary.get("excess_hit_rate_positive")
                if meta.direction == "entry"
                else summary.get("excess_hit_rate_negative")
            )
            ci_low = summary.get("ci_low", float("nan"))
            ci_high = summary.get("ci_high", float("nan"))
            summary.update(_directional_stability(rows, direction_sign))
            summary.update(
                {
                    "event": event_name,
                    "direction": meta.direction,
                    "label": label_name,
                    "horizon": _label_horizon(label_name),
                    "control_mode": control_mode,
                    "residualized": config.residualize_styles,
                    "description": meta.description,
                    "signal_role": "开仓" if meta.direction == "entry" else "平仓",
                    "signal_edge": direction_sign * summary.get("excess_mean", float("nan")),
                    "directional_hit_rate": directional_hit_rate,
                    "signal_ci_low": ci_low if meta.direction == "entry" else -ci_high,
                    "signal_ci_high": ci_high if meta.direction == "entry" else -ci_low,
                    "bootstrap_method": "event_date_cluster_centered_null",
                }
            )
            summary_rows.append(summary)
            if progress_callback is not None:
                progress_callback(1)

    observations = pd.concat(observation_frames, ignore_index=True) if observation_frames else pd.DataFrame()
    recent_rows = _recent_event_rows(event_name, events, close, recent_labels, config.horizons)
    return _EventRunOutput(
        event_name=event_name,
        summary_rows=summary_rows,
        event_path=event_path,
        event_matrix=events,
        path_observations=path_observations,
        observations=observations,
        recent_rows=recent_rows,
    )


def event_definitions_to_matrices(
    panel: MarketPanel,
    definitions: Iterable[Event],
    factors: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """Generate decoupled 0/1 event matrices from event definitions."""

    matrices: Dict[str, pd.DataFrame] = {}
    for definition in definitions:
        matrices[definition.name] = definition.evaluate(panel, factors).astype(int)
    return matrices


def event_definitions_to_meta(definitions: Iterable[Event]) -> Dict[str, EventMeta]:
    """Convert event definitions to metadata objects."""

    return {
        definition.name: EventMeta(
            name=definition.name,
            direction=definition.direction,
            cooldown_days=definition.cooldown_days,
            description=definition.description,
        )
        for definition in definitions
    }


class EventStudyRunner:
    """Run event studies over market data, external event matrices, and optional factors."""

    def __init__(
        self,
        panel: MarketPanel,
        event_matrices: Optional[Dict[str, pd.DataFrame]] = None,
        event_meta: Optional[Dict[str, EventMeta]] = None,
        events: Optional[Iterable[Event]] = None,
        factors: Optional[Dict[str, pd.DataFrame]] = None,
        style_controls: Optional[Dict[str, pd.DataFrame]] = None,
        config: EventStudyConfig = EventStudyConfig(),
    ) -> None:
        """Initialize the runner.

        ``event_matrices`` is the preferred input: each value is a 0/1 date-by-code
        pivot table. ``events`` is kept as a convenience generator layer.
        """

        self.panel = panel.align()
        self.factors = factors or {}
        self.style_controls = {
            key: frame.reindex_like(self.panel["close"])
            for key, frame in (style_controls or {}).items()
            if not frame.empty
        }
        self.config = config
        self.labels = build_forward_labels(
            self.panel["close"],
            self.config.horizons,
            label_types=self.config.label_types,
        )
        self.event_labels = (
            {
                label_name: residualize_by_date(label, self.style_controls)
                for label_name, label in self.labels.items()
            }
            if self.config.residualize_styles
            else self.labels
        )
        style = self.style_controls.get(self.config.style_name, pd.DataFrame())
        self.style_buckets = (
            build_style_buckets(style, self.config.style_bins)
            if "same_date_style" in self.config.control_modes and not style.empty
            else None
        )

        generated_matrices: Dict[str, pd.DataFrame] = {}
        generated_meta: Dict[str, EventMeta] = {}
        if events is not None:
            event_list = list(events)
            generated_matrices = event_definitions_to_matrices(self.panel, event_list, self.factors)
            generated_meta = event_definitions_to_meta(event_list)

        merged_matrices = {**generated_matrices, **(event_matrices or {})}
        self.event_matrices = {
            name: align_event_matrix(matrix, self.panel["close"])
            for name, matrix in merged_matrices.items()
        }
        self.event_meta = {
            **{name: EventMeta(name=name) for name in self.event_matrices},
            **generated_meta,
            **(event_meta or {}),
        }

    def run(self) -> EventStudyResult:
        """Run the full event study."""

        summary_rows: List[Dict[str, object]] = []
        event_paths: Dict[str, pd.DataFrame] = {}
        clean_event_matrices: Dict[str, pd.DataFrame] = {}
        observation_frames: List[pd.DataFrame] = []
        path_observation_frames: List[pd.DataFrame] = []
        recent_rows: List[Dict[str, object]] = []
        outputs = self._run_event_outputs()
        for event_name in self.event_matrices:
            output = outputs[event_name]
            summary_rows.extend(output.summary_rows)
            event_paths[event_name] = output.event_path
            clean_event_matrices[event_name] = output.event_matrix
            if not output.observations.empty:
                observation_frames.append(output.observations)
            if not output.path_observations.empty:
                path_observation_frames.append(output.path_observations)
            recent_rows.extend(output.recent_rows)

        group_summaries = self._run_factor_group_summaries()
        summary_df = add_signal_assessment(
            pd.DataFrame(summary_rows),
            significance_level=self.config.significance_level,
            min_events=self.config.min_events,
        )
        observations = pd.concat(observation_frames, ignore_index=True) if observation_frames else pd.DataFrame()
        path_observations = pd.concat(path_observation_frames, ignore_index=True) if path_observation_frames else pd.DataFrame()
        recent_events = pd.DataFrame(recent_rows)
        if not recent_events.empty:
            recent_events = recent_events.sort_values(["date", "event"], ascending=[False, True])
        return EventStudyResult(
            summary=summary_df,
            event_paths=event_paths,
            event_matrices=clean_event_matrices,
            event_observations=observations,
            group_summaries=group_summaries,
            recent_events=recent_events,
            horizons=tuple(self.config.horizons),
            event_path_observations=path_observations,
        )

    def _run_event_outputs(self) -> Dict[str, _EventRunOutput]:
        """Run event computations serially or with process-level parallelism."""

        event_items = list(self.event_matrices.items())
        if not event_items:
            return {}
        n_jobs = max(1, int(self.config.n_jobs))
        if n_jobs == 1 or len(event_items) == 1:
            return self._run_event_outputs_serial(event_items)
        return self._run_event_outputs_parallel(event_items, n_jobs=min(n_jobs, len(event_items)))

    def _run_event_outputs_serial(self, event_items: List[Tuple[str, pd.DataFrame]]) -> Dict[str, _EventRunOutput]:
        """Run event computations in the current process with fine-grained progress."""

        outputs: Dict[str, _EventRunOutput] = {}
        total = len(event_items) * len(self.labels) * len(self.config.control_modes)
        bar = None
        if self.config.show_progress:
            try:
                from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

                bar = tqdm(total=total, desc="EventStudyRunner.run", unit="task")
            except ImportError:
                bar = None
        try:
            for event_name, raw_events in event_items:
                outputs[event_name] = _compute_event_outputs(
                    event_name=event_name,
                    raw_events=raw_events,
                    meta=self.event_meta.get(event_name, EventMeta(name=event_name)),
                    close=self.panel["close"],
                    labels=self.event_labels,
                    recent_labels=self.labels,
                    style_controls=self.style_controls,
                    style_buckets=self.style_buckets,
                    config=self.config,
                    progress_callback=bar.update if bar is not None else None,
                )
        finally:
            if bar is not None:
                bar.close()
        return outputs

    def _run_event_outputs_parallel(
        self,
        event_items: List[Tuple[str, pd.DataFrame]],
        n_jobs: int,
    ) -> Dict[str, _EventRunOutput]:
        """Run one process per event up to ``n_jobs`` workers."""

        outputs: Dict[str, _EventRunOutput] = {}
        close = self.panel["close"]
        max_workers = min(n_jobs, os.cpu_count() or n_jobs)
        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for event_name, raw_events in event_items:
                futures[
                    executor.submit(
                        _compute_event_outputs,
                        event_name,
                        raw_events,
                        self.event_meta.get(event_name, EventMeta(name=event_name)),
                        close,
                        self.event_labels,
                        self.labels,
                        self.style_controls,
                        self.style_buckets,
                        self.config,
                    )
                ] = event_name
            progress_iter = _maybe_progress(
                as_completed(futures),
                self.config.show_progress,
                total=len(futures),
                desc=f"EventStudyRunner.run ({max_workers} workers)",
                unit="event",
            )
            for future in progress_iter:
                output = future.result()
                outputs[output.event_name] = output
        return outputs

    def _run_factor_group_summaries(self) -> Dict[str, pd.DataFrame]:
        """Run factor-like quantile group studies for provided factor panels."""

        results: Dict[str, pd.DataFrame] = {}
        for factor_name, factor in self.factors.items():
            aligned_factor = factor.reindex_like(self.panel["close"])
            groups = quantile_groups(aligned_factor, group_count=5)
            frames: List[pd.DataFrame] = []
            for horizon in self.config.horizons:
                label_name = f"ret_{horizon}d"
                table = group_label_summary(
                    aligned_factor,
                    self.labels[label_name],
                    group_count=5,
                    groups=groups,
                )
                table["factor"] = factor_name
                table["label"] = label_name
                table["horizon"] = horizon
                frames.append(table)
            results[factor_name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return results
