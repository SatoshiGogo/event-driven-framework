"""Persistent content-addressed cache for per-event study results."""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from event_study_framework.data import MarketPanel
from event_study_framework.event import Event
from event_study_framework.study import EventStudyConfig, EventStudyResult


CACHE_SCHEMA_VERSION = 2
_ENGINE_FILES = ("event.py", "labels.py", "controls.py", "statistics.py", "study.py", "backtest.py")


def _canonical_json(value: Any) -> str:
    """Serialize nested cache metadata deterministically."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(value: bytes) -> str:
    """Return a hexadecimal SHA-256 digest."""

    return hashlib.sha256(value).hexdigest()


def _frame_signature(frame: pd.DataFrame) -> Dict[str, Any]:
    """Build a lightweight signature that changes with shape, dates, codes, and sampled values."""

    if frame.empty:
        return {"rows": 0, "columns": 0, "first": None, "last": None, "sample_hash": None}
    sample_count = min(17, len(frame))
    positions = np.unique(np.linspace(0, len(frame) - 1, sample_count, dtype=int))
    sample = frame.iloc[positions]
    sample_hash = pd.util.hash_pandas_object(sample, index=True).to_numpy(dtype=np.uint64).tobytes()
    column_hash = _sha256_bytes("\0".join(map(str, frame.columns)).encode("utf-8"))
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "first": str(frame.index[0]),
        "last": str(frame.index[-1]),
        "column_hash": column_hash,
        "sample_hash": _sha256_bytes(sample_hash),
    }


def _event_source_signature(event: Event) -> str:
    """Hash one event class and module-level helper functions it references."""

    source_event = getattr(event, "source", event)
    try:
        event_class = type(source_event)
        module = inspect.getmodule(event_class)
        digest = hashlib.sha256(inspect.getsource(event_class).encode("utf-8"))
        pending = [getattr(event_class, "__init__", None), getattr(event_class, "compute", None)]
        visited = set()
        while pending:
            callable_object = pending.pop()
            code = getattr(callable_object, "__code__", None)
            if code is None or module is None:
                continue
            for name in code.co_names:
                helper = getattr(module, name, None)
                if not inspect.isfunction(helper) or helper in visited:
                    continue
                if getattr(helper, "__module__", None) != event_class.__module__:
                    continue
                visited.add(helper)
                digest.update(inspect.getsource(helper).encode("utf-8"))
                pending.append(helper)
        return digest.hexdigest()
    except (TypeError, OSError):
        return f"{type(source_event).__module__}:{type(source_event).__qualname__}"


def engine_signature(event: Event) -> str:
    """Hash core evaluator code and the selected event class source."""

    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for file_name in _ENGINE_FILES:
        path = package_dir / file_name
        digest.update(file_name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(_event_source_signature(event).encode("ascii", errors="ignore"))
    return digest.hexdigest()


def event_specification(event: Event) -> Dict[str, Any]:
    """Return normalized event class, parameter, and metadata configuration."""

    configured = getattr(event, "_config_spec", None)
    if configured is None:
        source_event = getattr(event, "source", event)
        parameters = {
            key: value
            for key, value in vars(source_event).items()
            if not key.startswith("_") and key not in {"func", "source"}
        }
        configured = {
            "class": f"{type(source_event).__module__}:{type(source_event).__qualname__}",
            "event_name": event.name,
            "params": parameters,
        }
    return {
        "configured": configured,
        "name": event.name,
        "direction": event.direction,
        "cooldown_days": event.cooldown_days,
        "required_fields": list(event.required_fields),
        "required_factors": None if event.required_factors is None else list(event.required_factors),
    }


def study_signature(config: EventStudyConfig) -> Dict[str, Any]:
    """Return only study settings that can change numerical event outputs."""

    return {
        "horizons": list(config.horizons),
        "label_types": list(config.label_types),
        "window_before": config.window_before,
        "window_after": config.window_after,
        "n_bootstrap": config.n_bootstrap,
        "min_events": config.min_events,
        "significance_level": config.significance_level,
        "control_modes": list(config.control_modes),
        "style_name": config.style_name,
        "style_bins": config.style_bins,
        "purge_days": config.purge_days,
        "residualize_styles": config.residualize_styles,
        "use_gpu": config.use_gpu,
        "backtest_fees": list(config.backtest_fees),
        "annualization": config.annualization,
        "risk_free_rate": config.risk_free_rate,
    }


def event_data_signature(
    event: Event,
    panel: MarketPanel,
    factors: Mapping[str, pd.DataFrame],
    style_controls: Mapping[str, pd.DataFrame],
    style_name: str,
    universe: Optional[Iterable[str]],
    start: Optional[str],
    end: Optional[str],
) -> Dict[str, Any]:
    """Describe only market, factor, and style inputs used by one event evaluation."""

    required_fields = set(event.required_fields) | {"close", "trade_status"}
    panel_signature = {
        field: _frame_signature(panel[field])
        for field in sorted(required_fields)
        if field in panel.fields
    }
    required_factors = event.required_factors
    factor_names = sorted(factors) if required_factors is None else sorted(set(required_factors))
    factor_signature = {
        name: _frame_signature(factors[name])
        for name in factor_names
        if name in factors
    }
    style = style_controls.get(style_name, pd.DataFrame())
    return {
        "requested_universe": None if universe is None else sorted(map(str, universe)),
        "requested_start": start,
        "requested_end": end,
        "panel": panel_signature,
        "factors": factor_signature,
        "style": {style_name: _frame_signature(style)},
    }


def build_event_cache_key(
    event: Event,
    config: EventStudyConfig,
    data_signature: Mapping[str, Any],
) -> str:
    """Build a content-addressed key for one event's complete numerical result."""

    payload = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "engine": engine_signature(event),
        "event": event_specification(event),
        "study": study_signature(config),
        "data": data_signature,
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def split_event_result(result: EventStudyResult, event_name: str) -> EventStudyResult:
    """Extract one event from a multi-event result for persistent caching."""

    summary = result.summary
    if not summary.empty and "event" in summary:
        summary = summary[summary["event"] == event_name].copy()
    else:
        summary = pd.DataFrame()
    observations = result.event_observations
    if not observations.empty and "event" in observations:
        observations = observations[observations["event"] == event_name].copy()
    path_observations = result.event_path_observations
    if not path_observations.empty and "event" in path_observations:
        path_observations = path_observations[path_observations["event"] == event_name].copy()
    recent = result.recent_events
    if not recent.empty and "event" in recent:
        recent = recent[recent["event"] == event_name].copy()
    return EventStudyResult(
        summary=summary,
        event_paths={event_name: result.event_paths.get(event_name, pd.DataFrame()).copy()},
        event_matrices={},
        event_observations=observations,
        group_summaries={},
        recent_events=recent,
        horizons=tuple(result.horizons),
        event_path_observations=path_observations,
        strategy_backtests={event_name: result.strategy_backtests.get(event_name, pd.DataFrame()).copy()},
        strategy_metrics={event_name: result.strategy_metrics.get(event_name, pd.DataFrame()).copy()},
    )


def combine_event_results(
    results: Mapping[str, EventStudyResult],
    event_order: Sequence[str],
    horizons: Sequence[int],
    group_summaries: Optional[Dict[str, pd.DataFrame]] = None,
) -> EventStudyResult:
    """Combine cached and newly computed event results in current-config order."""

    ordered = [results[name] for name in event_order if name in results]
    summary_frames = [result.summary for result in ordered if not result.summary.empty]
    observation_frames = [result.event_observations for result in ordered if not result.event_observations.empty]
    path_frames = [result.event_path_observations for result in ordered if not result.event_path_observations.empty]
    recent_frames = [result.recent_events for result in ordered if not result.recent_events.empty]
    event_paths = {
        name: results[name].event_paths.get(name, pd.DataFrame())
        for name in event_order
        if name in results
    }
    strategy_backtests = {
        name: results[name].strategy_backtests.get(name, pd.DataFrame())
        for name in event_order
        if name in results
    }
    strategy_metrics = {
        name: results[name].strategy_metrics.get(name, pd.DataFrame())
        for name in event_order
        if name in results
    }
    recent = pd.concat(recent_frames, ignore_index=True) if recent_frames else pd.DataFrame()
    if not recent.empty and {"date", "event"}.issubset(recent.columns):
        recent = recent.sort_values(["date", "event"], ascending=[False, True])
    return EventStudyResult(
        summary=pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(),
        event_paths=event_paths,
        event_matrices={},
        event_observations=(
            pd.concat(observation_frames, ignore_index=True) if observation_frames else pd.DataFrame()
        ),
        group_summaries=group_summaries or {},
        recent_events=recent,
        horizons=tuple(horizons),
        event_path_observations=(
            pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()
        ),
        strategy_backtests=strategy_backtests,
        strategy_metrics=strategy_metrics,
    )


class EventResultCache:
    """Store and retrieve trusted local per-event result bundles."""

    def __init__(self, root: Path) -> None:
        """Initialize a cache rooted at ``root``."""

        self.root = Path(root)

    def _entry(self, cache_key: str) -> Path:
        """Return the directory for one content-addressed entry."""

        return self.root / cache_key

    def load(self, cache_key: str, event_name: str) -> Optional[EventStudyResult]:
        """Load a complete entry, returning ``None`` for a miss or corruption."""

        entry = self._entry(cache_key)
        manifest_path = entry / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != CACHE_SCHEMA_VERSION or manifest.get("event") != event_name:
                return None
            return EventStudyResult(
                summary=pd.read_pickle(entry / "summary.pkl.gz", compression="gzip"),
                event_paths={
                    event_name: pd.read_pickle(entry / "event_path.pkl.gz", compression="gzip")
                },
                event_matrices={},
                event_observations=pd.read_pickle(
                    entry / "event_observations.pkl.gz", compression="gzip"
                ),
                group_summaries={},
                recent_events=pd.read_pickle(entry / "recent_events.pkl.gz", compression="gzip"),
                horizons=tuple(manifest.get("horizons", [])),
                event_path_observations=pd.read_pickle(
                    entry / "path_observations.pkl.gz", compression="gzip"
                ),
                strategy_backtests={
                    event_name: pd.read_pickle(entry / "strategy_backtest.pkl.gz", compression="gzip")
                },
                strategy_metrics={
                    event_name: pd.read_pickle(entry / "strategy_metrics.pkl.gz", compression="gzip")
                },
            )
        except (OSError, ValueError, TypeError, EOFError, json.JSONDecodeError):
            return None

    def save(
        self,
        cache_key: str,
        event_name: str,
        result: EventStudyResult,
        overwrite: bool = False,
    ) -> Path:
        """Atomically save one event result and return its entry directory."""

        self.root.mkdir(parents=True, exist_ok=True)
        entry = self._entry(cache_key)
        if entry.is_dir() and not overwrite:
            return entry
        temporary = self.root / f".{cache_key}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            result.summary.to_pickle(temporary / "summary.pkl.gz", compression="gzip", protocol=5)
            result.event_paths.get(event_name, pd.DataFrame()).to_pickle(
                temporary / "event_path.pkl.gz", compression="gzip", protocol=5
            )
            result.event_observations.to_pickle(
                temporary / "event_observations.pkl.gz", compression="gzip", protocol=5
            )
            result.event_path_observations.to_pickle(
                temporary / "path_observations.pkl.gz", compression="gzip", protocol=5
            )
            result.recent_events.to_pickle(
                temporary / "recent_events.pkl.gz", compression="gzip", protocol=5
            )
            result.strategy_backtests.get(event_name, pd.DataFrame()).to_pickle(
                temporary / "strategy_backtest.pkl.gz", compression="gzip", protocol=5
            )
            result.strategy_metrics.get(event_name, pd.DataFrame()).to_pickle(
                temporary / "strategy_metrics.pkl.gz", compression="gzip", protocol=5
            )
            manifest = {
                "schema": CACHE_SCHEMA_VERSION,
                "cache_key": cache_key,
                "event": event_name,
                "horizons": list(result.horizons),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if entry.exists():
                shutil.rmtree(entry)
            temporary.replace(entry)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return entry
