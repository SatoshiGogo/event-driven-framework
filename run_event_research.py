"""Command-line entry point for event-study reports."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from event_study_framework.backtest import run_backtest_grid
from event_study_framework.cache import (
    EventResultCache,
    build_event_cache_key,
    combine_event_results,
    event_data_signature,
    split_event_result,
)
from event_study_framework.data import MarketPanel, MassimMongoDataSource
from event_study_framework.event import Event
from event_study_framework.event_config import (
    EventConfigError,
    load_events_from_config,
    load_run_defaults,
)
from event_study_framework.events import (
    breakout_volume_event,
    factor_extreme_event,
    ma_break_event,
    volume_stall_event,
)
from event_study_framework.report import render_event_report
from event_study_framework.study import (
    EventStudyConfig,
    EventStudyResult,
    EventStudyRunner,
    add_signal_assessment,
    event_definitions_to_matrices,
    event_definitions_to_meta,
)


DEFAULT_EVENT_CONFIG_PATH = Path(__file__).with_name("event_config.yaml")


def _add_boolean_switch(
    parser: argparse.ArgumentParser,
    enable_flag: str,
    disable_flag: str,
    destination: str,
    default: bool,
    help_text: str,
) -> None:
    """Add paired enable/disable flags with a YAML-provided default."""

    group = parser.add_mutually_exclusive_group()
    group.add_argument(enable_flag, dest=destination, action="store_true", help=help_text)
    group.add_argument(disable_flag, dest=destination, action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(**{destination: bool(default)})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI overrides on top of defaults loaded from the YAML config."""

    effective_argv = argv
    if effective_argv is None:
        try:
            from IPython import get_ipython

            if get_ipython() is not None:
                effective_argv = []
        except ImportError:
            effective_argv = None

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--event-config", type=Path, default=DEFAULT_EVENT_CONFIG_PATH)
    preliminary, _ = pre_parser.parse_known_args(effective_argv)
    try:
        defaults = load_run_defaults(preliminary.event_config)
    except EventConfigError as exc:
        raise SystemExit(f"Invalid event configuration: {exc}") from exc

    parser = argparse.ArgumentParser(description="Run rule-event research with massim Mongo data.")
    parser.add_argument(
        "--codes",
        nargs="+",
        default=defaults.get("codes"),
        help="Optional A-share stock codes. If omitted, load the full market from Mongo.",
    )
    parser.add_argument("--start", default=defaults.get("start", "20130101"), help="Start date in YYYYMMDD format.")
    parser.add_argument("--end", default=defaults.get("end"), help="End date in YYYYMMDD format.")
    parser.add_argument("--output-dir", default=defaults.get("output_dir", "event_study_framework/outputs"), help="Output directory.")
    parser.add_argument(
        "--event-config",
        type=Path,
        default=preliminary.event_config,
        help="Strict YAML file selecting event classes and constructor parameters.",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=defaults.get("horizons", [1, 3, 5, 10, 20]), help="Forward horizons.")
    parser.add_argument(
        "--label-types",
        nargs="+",
        default=defaults.get("label_types", ["ret"]),
        choices=["ret", "mfe", "mae"],
        help="Labels to calculate. ret is required; MFE/MAE are optional and disabled by default.",
    )
    parser.add_argument("--factor", action="append", default=defaults.get("factors", []), help="Factor spec: name=database/collection.")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=defaults.get("bootstrap", 2000),
        help="Event-date cluster bootstrap repetitions. Use 0 to disable inference.",
    )
    parser.add_argument(
        "--significance-level",
        type=float,
        default=defaults.get("significance_level", 0.05),
        help="FDR-adjusted significance level used by signal assessments.",
    )
    parser.add_argument(
        "--control-modes",
        nargs="+",
        default=defaults.get("control_modes", ["same_date", "same_date_style"]),
        choices=["unconditional", "same_date", "same_date_style", "event_stock_history"],
        help="Control-sample modes.",
    )
    parser.add_argument("--style-name", default=defaults.get("style_name", "lncap"), help="Style control name used by same_date_style.")
    parser.add_argument("--style-bins", type=int, default=defaults.get("style_bins", 5), help="Number of style buckets.")
    parser.add_argument("--purge-days", type=int, default=defaults.get("purge_days", 10), help="Days to suppress repeated events per stock.")
    _add_boolean_switch(
        parser,
        "--residualize-styles",
        "--no-residualize-styles",
        "residualize_styles",
        defaults.get("residualize_styles", False),
        "Evaluate labels after cross-sectional style residualization.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=defaults.get("n_jobs", 1),
        help="Number of worker processes for event-level parallelism. Use 1 to disable multiprocessing.",
    )
    _add_boolean_switch(
        parser,
        "--show-progress",
        "--no-progress",
        "show_progress",
        defaults.get("show_progress", True),
        "Show EventStudyRunner progress bars.",
    )
    _add_boolean_switch(
        parser,
        "--use-gpu",
        "--no-gpu",
        "use_gpu",
        defaults.get("use_gpu", False),
        "Use CuPy for bootstrap sampling when available; falls back to CPU if CuPy is not installed.",
    )
    parser.add_argument(
        "--report-max-rows",
        type=int,
        default=defaults.get("report_max_rows", 100_000),
        help="Maximum event-observation rows embedded in the interactive HTML report. Use 0 for no cap.",
    )
    parser.add_argument(
        "--cache-dir",
        default=defaults.get("cache_dir", "event_study_framework/cache"),
        help="Persistent per-event result cache directory.",
    )
    _add_boolean_switch(
        parser,
        "--use-cache",
        "--no-cache",
        "use_cache",
        defaults.get("use_cache", True),
        "Reuse matching event results from the persistent cache.",
    )
    _add_boolean_switch(
        parser,
        "--refresh-cache",
        "--no-refresh-cache",
        "refresh_cache",
        defaults.get("refresh_cache", False),
        "Recompute current events and replace matching cache entries.",
    )
    parser.add_argument(
        "--backtest-fees",
        nargs="+",
        type=float,
        default=defaults.get("backtest_fees", [0.0, 0.002]),
        help="One-way fee rates precomputed for the report dropdown.",
    )
    parser.add_argument("--annualization", type=int, default=defaults.get("annualization", 252), help="Trading days per year.")
    parser.add_argument("--risk-free-rate", type=float, default=defaults.get("risk_free_rate", 0.0), help="Annual risk-free rate for Sharpe.")
    return parser.parse_args(effective_argv)


def parse_factor_specs(specs: List[str]) -> Dict[str, str]:
    """Parse ``name=database/collection`` factor specifications."""

    parsed: Dict[str, str] = {}
    for spec in specs:
        if "=" not in spec or "/" not in spec:
            raise ValueError(f"Invalid factor spec: {spec}. Use name=database/collection.")
        name, path = spec.split("=", 1)
        parsed[name.strip()] = path.strip()
    return parsed


def load_factors(
    data_source: MassimMongoDataSource,
    factor_specs: Dict[str, str],
    universe: Optional[Iterable[str]],
    start: Optional[str],
    end: Optional[str],
) -> Dict[str, pd.DataFrame]:
    """Load optional factor panels from Mongo."""

    factors: Dict[str, pd.DataFrame] = {}
    for factor_name, path in factor_specs.items():
        database, collection = path.split("/", 1)
        factors[factor_name] = data_source.load_factor(
            database=database,
            collection=collection,
            universe=universe,
            start_date=start,
            end_date=end,
        )
    return factors


def default_events(factor_names: List[str]) -> List[Event]:
    """Build a default event library for demonstration and extension."""

    events = [
        ma_break_event(ma_window=20, trend_window=60, trend_return_threshold=0.25),
        volume_stall_event(trend_window=60, trend_return_threshold=0.25, volume_multiple=1.8),
        breakout_volume_event(breakout_window=60, volume_multiple=1.5, ma_window=20),
    ]
    for factor_name in factor_names:
        events.append(
            factor_extreme_event(
                factor_name=factor_name,
                side="high",
                percentile=0.8,
                window=252,
                direction="entry",
            )
        )
        events.append(
            factor_extreme_event(
                factor_name=factor_name,
                side="low",
                percentile=0.8,
                window=252,
                direction="exit",
            )
        )
    return events


def select_events(event_config: Optional[Path], factor_names: List[str]) -> List[Event]:
    """Load only configured events, or preserve the legacy default library."""

    if event_config is None:
        return default_events(factor_names)
    return load_events_from_config(event_config)


def required_market_collections(events: Sequence[Event]) -> Tuple[str, ...]:
    """Return the minimal Mongo market collections required by selected events."""

    collection_by_field = {
        "open": "adj_open",
        "high": "adj_high",
        "low": "adj_low",
        "close": "adj_close",
        "volume": "volume",
        "amount": "amount",
        "trade_status": "trade_status",
    }
    required = {"close"}
    for event in events:
        required.update(event.required_fields)
    unknown = required - set(collection_by_field)
    if unknown:
        raise ValueError(f"Events require unsupported market fields: {sorted(unknown)}")
    return tuple(
        collection
        for field, collection in collection_by_field.items()
        if field in required
    )


def build_study_config(args: argparse.Namespace) -> EventStudyConfig:
    """Build numerical study settings from merged YAML defaults and CLI overrides."""

    return EventStudyConfig(
        horizons=tuple(args.horizons),
        label_types=tuple(args.label_types),
        n_bootstrap=args.bootstrap,
        significance_level=args.significance_level,
        control_modes=tuple(args.control_modes),
        style_name=args.style_name,
        style_bins=args.style_bins,
        purge_days=args.purge_days,
        residualize_styles=args.residualize_styles,
        n_jobs=args.n_jobs,
        show_progress=args.show_progress,
        use_gpu=args.use_gpu,
        backtest_fees=tuple(args.backtest_fees),
        annualization=args.annualization,
        risk_free_rate=args.risk_free_rate,
    )


def run_incremental_event_study(
    panel: MarketPanel,
    events: Sequence[Event],
    factors: Mapping[str, pd.DataFrame],
    style_controls: Mapping[str, pd.DataFrame],
    config: EventStudyConfig,
    cache: EventResultCache,
    use_cache: bool,
    refresh_cache: bool,
    universe: Optional[Iterable[str]],
    start: Optional[str],
    end: Optional[str],
) -> Tuple[EventStudyResult, List[str], List[str]]:
    """Reuse matching event bundles and evaluate only cache misses."""

    event_results: Dict[str, EventStudyResult] = {}
    event_keys: Dict[str, str] = {}
    hits: List[str] = []
    misses: List[Event] = []
    for event in events:
        data_signature = event_data_signature(
            event,
            panel,
            factors,
            style_controls,
            style_name=config.style_name,
            universe=universe,
            start=start,
            end=end,
        )
        cache_key = build_event_cache_key(event, config, data_signature)
        event_keys[event.name] = cache_key
        cached = None
        if use_cache and not refresh_cache:
            cached = cache.load(cache_key, event.name)
        if cached is None:
            misses.append(event)
        else:
            event_results[event.name] = cached
            hits.append(event.name)

    group_summaries: Dict[str, pd.DataFrame] = {}
    if misses:
        matrices = event_definitions_to_matrices(panel, misses, dict(factors))
        metadata = event_definitions_to_meta(misses)
        fresh = EventStudyRunner(
            panel=panel,
            event_matrices=matrices,
            event_meta=metadata,
            factors=dict(factors),
            style_controls=dict(style_controls),
            config=config,
        ).run()
        for event in misses:
            backtest, metrics = run_backtest_grid(
                panel=panel,
                events=fresh.event_matrices[event.name],
                horizons=config.horizons,
                fee_rates=config.backtest_fees,
                annualization=config.annualization,
                risk_free_rate=config.risk_free_rate,
            )
            fresh.strategy_backtests[event.name] = backtest
            fresh.strategy_metrics[event.name] = metrics
        group_summaries = fresh.group_summaries
        for event in misses:
            event_result = split_event_result(fresh, event.name)
            event_results[event.name] = event_result
            if use_cache:
                cache.save(
                    event_keys[event.name],
                    event.name,
                    event_result,
                    overwrite=refresh_cache,
                )
    elif factors:
        group_summaries = EventStudyRunner(
            panel=panel,
            event_matrices={},
            factors=dict(factors),
            style_controls=dict(style_controls),
            config=config,
        ).run().group_summaries

    event_order = [event.name for event in events]
    combined = combine_event_results(
        event_results,
        event_order=event_order,
        horizons=config.horizons,
        group_summaries=group_summaries,
    )
    combined.summary = add_signal_assessment(
        combined.summary,
        significance_level=config.significance_level,
        min_events=config.min_events,
    )
    return combined, hits, [event.name for event in misses]


def main() -> None:
    """Run an event study and write HTML/CSV outputs."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    factor_specs = parse_factor_specs(args.factor)
    try:
        event_definitions = select_events(args.event_config, list(factor_specs.keys()))
    except EventConfigError as exc:
        raise SystemExit(f"Invalid event configuration: {exc}") from exc
    market_fields = required_market_collections(event_definitions)

    data_source = MassimMongoDataSource()
    universe = args.codes
    if universe is None:
        print("Universe: full market from Mongo")
    else:
        print(f"Universe: {len(universe)} selected codes")
    if args.n_jobs > 1:
        print(f"EventStudyRunner multiprocessing: {args.n_jobs} event-level workers")
    if args.use_gpu and importlib.util.find_spec("cupy") is None:
        print("GPU requested but CuPy is not installed; bootstrap calculations will fall back to CPU.")
    if args.event_config is not None:
        print(f"Event config: {args.event_config.resolve()}")
    print(f"Events: {', '.join(event.name for event in event_definitions)}")
    print(f"Market fields: {', '.join(market_fields)}")
    panel = data_source.load_market_panel(
        universe,
        start_date=args.start,
        end_date=args.end,
        fields=market_fields,
    )
    factors = load_factors(data_source, factor_specs, universe, args.start, args.end)
    style_controls = data_source.load_style_controls(universe, args.start, args.end, include_lncap=True)
    study_config = build_study_config(args)
    result, cache_hits, cache_misses = run_incremental_event_study(
        panel=panel,
        events=event_definitions,
        factors=factors,
        style_controls=style_controls,
        config=study_config,
        cache=EventResultCache(Path(args.cache_dir)),
        use_cache=args.use_cache,
        refresh_cache=args.refresh_cache,
        universe=universe,
        start=args.start,
        end=args.end,
    )
    print(f"Event cache hits ({len(cache_hits)}): {', '.join(cache_hits) if cache_hits else '-'}")
    print(f"Event cache misses ({len(cache_misses)}): {', '.join(cache_misses) if cache_misses else '-'}")

    result.summary.to_csv(output_dir / "event_summary.csv", index=False, encoding="utf-8-sig")
    result.event_observations.to_csv(output_dir / "event_observations.csv", index=False, encoding="utf-8-sig")
    result.event_path_observations.to_csv(output_dir / "event_path_observations.csv", index=False, encoding="utf-8-sig")
    result.recent_events.to_csv(output_dir / "recent_events.csv", index=False, encoding="utf-8-sig")
    current_path_files = {f"path_{event_name}.csv" for event_name in result.event_paths}
    for stale_path in output_dir.glob("path_*.csv"):
        if stale_path.name not in current_path_files:
            stale_path.unlink()
    for event_name, path in result.event_paths.items():
        path.to_csv(output_dir / f"path_{event_name}.csv", encoding="utf-8-sig")
    current_backtest_files = {f"strategy_backtest_{event_name}.csv" for event_name in result.strategy_backtests}
    current_metric_files = {f"strategy_metrics_{event_name}.csv" for event_name in result.strategy_metrics}
    for stale_path in list(output_dir.glob("strategy_backtest_*.csv")) + list(output_dir.glob("strategy_metrics_*.csv")):
        if stale_path.name not in current_backtest_files | current_metric_files:
            stale_path.unlink()
    for event_name, table in result.strategy_backtests.items():
        table.to_csv(output_dir / f"strategy_backtest_{event_name}.csv", index=False, encoding="utf-8-sig")
    for event_name, table in result.strategy_metrics.items():
        table.to_csv(output_dir / f"strategy_metrics_{event_name}.csv", index=False, encoding="utf-8-sig")
    for factor_name, table in result.group_summaries.items():
        table.to_csv(output_dir / f"factor_group_{factor_name}.csv", index=False, encoding="utf-8-sig")

    report_cap = None if args.report_max_rows <= 0 else args.report_max_rows
    report_path = render_event_report(result, output_dir / "event_study_report.html", max_interactive_rows=report_cap)
    print(f"Report written to: {report_path.resolve()}")
    if not result.summary.empty:
        display_cols = [
            "event",
            "label",
            "control_mode",
            "events",
            "signal_role",
            "signal_edge",
            "directional_hit_rate",
            "bootstrap_p",
            "q_value",
            "assessment",
        ]
        print(result.summary[display_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
