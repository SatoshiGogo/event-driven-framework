"""Strict YAML configuration loader for class-based events."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Type, Union

import pandas as pd
import yaml

from event_study_framework.data import MarketPanel
from event_study_framework.event import Event, validate_event_name
from event_study_framework.events import (
    BreakoutVolumeEvent,
    FactorExtremeEvent,
    MaBreakEvent,
    VolumeStallEvent,
)


CONFIG_VERSION = 1
RUN_CONFIG_KEYS = {
    "codes",
    "start",
    "end",
    "output_dir",
    "horizons",
    "label_types",
    "factors",
    "bootstrap",
    "significance_level",
    "control_modes",
    "style_name",
    "style_bins",
    "purge_days",
    "residualize_styles",
    "n_jobs",
    "show_progress",
    "use_gpu",
    "report_max_rows",
    "cache_dir",
    "use_cache",
    "refresh_cache",
    "backtest_fees",
    "annualization",
    "risk_free_rate",
}
BUILTIN_EVENT_CLASSES: Mapping[str, Type[Event]] = {
    "ma_break": MaBreakEvent,
    "volume_stall": VolumeStallEvent,
    "breakout_volume": BreakoutVolumeEvent,
    "factor_extreme": FactorExtremeEvent,
}


class EventConfigError(ValueError):
    """Raised when an event configuration file is invalid or unsafe."""


class _RenamedEvent(Event):
    """Delegate computation while applying a configured event identifier."""

    def __init__(self, source: Event, event_name: str) -> None:
        """Initialize an event-name override around an existing event."""

        super().__init__(
            event_name=event_name,
            direction=source.direction,
            cooldown_days=source.cooldown_days,
            description=source.description,
        )
        self.source = source
        self.required_fields = source.required_fields
        self.required_factors = source.required_factors

    def compute(
        self,
        panel: MarketPanel,
        factors: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Delegate raw event computation to the configured source event."""

        return self.source.compute(panel, factors)


def _reject_nonfinite_values(value: Any, location: str = "event config") -> None:
    """Reject non-finite YAML numbers recursively."""

    if isinstance(value, float) and not math.isfinite(value):
        raise EventConfigError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite_values(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite_values(child, f"{location}[{index}]")


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Safely read a UTF-8 YAML file and require a mapping at its root."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EventConfigError(f"cannot read event config '{path}': {exc}") from exc
    except (UnicodeError, yaml.YAMLError) as exc:
        raise EventConfigError(f"invalid YAML in event config '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise EventConfigError("event config root must be a YAML mapping")
    _reject_nonfinite_values(payload)
    return payload


def _check_keys(
    payload: Mapping[str, Any],
    allowed: Set[str],
    required: Set[str],
    location: str,
) -> None:
    """Reject missing and unknown YAML mapping keys."""

    present = set(payload)
    missing = required - present
    unknown = present - allowed
    if missing:
        raise EventConfigError(f"{location} is missing required keys: {sorted(missing)}")
    if unknown:
        raise EventConfigError(f"{location} contains unknown keys: {sorted(unknown)}")


def _resolve_dynamic_class(reference: str) -> Type[Event]:
    """Import a trusted ``module.path:ClassName`` Event subclass."""

    if reference.count(":") != 1:
        raise EventConfigError(
            f"unknown built-in event '{reference}'; external classes must use module.path:ClassName"
        )
    module_name, class_name = reference.split(":", 1)
    if not module_name or not class_name or module_name.startswith("."):
        raise EventConfigError(f"invalid external event class reference: {reference}")
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ValueError) as exc:
        raise EventConfigError(
            f"cannot import event module '{module_name}' for '{reference}': {exc}"
        ) from exc
    try:
        event_class = getattr(module, class_name)
    except AttributeError as exc:
        raise EventConfigError(
            f"event class '{class_name}' was not found in module '{module_name}'"
        ) from exc
    if not isinstance(event_class, type) or not issubclass(event_class, Event):
        raise EventConfigError(f"configured class '{reference}' must inherit from Event")
    return event_class


def _resolve_event_class(reference: str) -> Type[Event]:
    """Resolve a built-in short name or a trusted dynamic Event class."""

    if reference in BUILTIN_EVENT_CLASSES:
        return BUILTIN_EVENT_CLASSES[reference]
    return _resolve_dynamic_class(reference)


def _build_event(spec: Mapping[str, Any], index: int) -> Optional[Event]:
    """Validate and instantiate one event configuration entry."""

    location = f"events[{index}]"
    _check_keys(
        spec,
        allowed={"class", "enabled", "params", "event_name"},
        required={"class"},
        location=location,
    )
    class_reference = spec["class"]
    if not isinstance(class_reference, str) or not class_reference.strip():
        raise EventConfigError(f"{location}.class must be a non-empty string")
    if class_reference != class_reference.strip():
        raise EventConfigError(f"{location}.class must not contain surrounding whitespace")

    enabled = spec.get("enabled", True)
    if not isinstance(enabled, bool):
        raise EventConfigError(f"{location}.enabled must be true or false")
    params = spec.get("params", {})
    if not isinstance(params, dict):
        raise EventConfigError(f"{location}.params must be a YAML mapping")
    if "event_name" in params:
        raise EventConfigError(
            f"{location}.params.event_name is reserved; use top-level event_name instead"
        )

    configured_name = spec.get("event_name")
    if configured_name is not None:
        if not isinstance(configured_name, str):
            raise EventConfigError(f"{location}.event_name must be a string")
        try:
            validate_event_name(configured_name)
        except ValueError as exc:
            raise EventConfigError(f"{location}.event_name is unsafe: {exc}") from exc
    if not enabled:
        return None

    event_class = _resolve_event_class(class_reference)
    try:
        event = event_class(**params)
    except Exception as exc:
        raise EventConfigError(
            f"cannot construct {location} using '{class_reference}': {exc}"
        ) from exc
    if not isinstance(event, Event):
        raise EventConfigError(f"{location} did not construct an Event instance")
    try:
        source_name = event.name
        source_meta = event.meta
    except Exception as exc:
        raise EventConfigError(
            f"{location} Event subclass did not initialize the Event base metadata: {exc}"
        ) from exc
    if source_meta.name != source_name:
        raise EventConfigError(f"{location} Event metadata name is inconsistent")
    if configured_name is not None and configured_name != source_name:
        event = _RenamedEvent(source=event, event_name=configured_name)
    try:
        validate_event_name(event.name)
    except ValueError as exc:
        raise EventConfigError(f"{location} produced an unsafe event name: {exc}") from exc
    event._config_spec = {  # type: ignore[attr-defined]  # pylint: disable=protected-access
        "class": class_reference,
        "event_name": event.name,
        "params": dict(params),
    }
    return event


def load_events_from_config(path: Union[str, Path]) -> List[Event]:
    """Load enabled, uniquely named Event instances from a strict YAML file."""

    config_path = Path(path).expanduser()
    payload = _load_yaml(config_path)
    _check_keys(
        payload,
        allowed={"version", "run", "events"},
        required={"version", "events"},
        location="event config",
    )
    version = payload["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise EventConfigError("event config version must be an integer")
    if version != CONFIG_VERSION:
        raise EventConfigError(
            f"unsupported event config version {version}; expected {CONFIG_VERSION}"
        )
    event_specs = payload["events"]
    if not isinstance(event_specs, list):
        raise EventConfigError("event config events must be a YAML sequence")
    if not event_specs:
        raise EventConfigError("event config events must not be empty")

    events: List[Event] = []
    seen_names: Dict[str, str] = {}
    for index, spec in enumerate(event_specs):
        if not isinstance(spec, dict):
            raise EventConfigError(f"events[{index}] must be a YAML mapping")
        event = _build_event(spec, index)
        if event is None:
            continue
        casefolded_name = event.name.casefold()
        if casefolded_name in seen_names:
            raise EventConfigError(
                f"duplicate event_name '{event.name}' conflicts with '{seen_names[casefolded_name]}'"
            )
        seen_names[casefolded_name] = event.name
        events.append(event)
    if not events:
        raise EventConfigError("event config does not contain any enabled events")
    return events


def load_run_defaults(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and strictly validate the optional YAML ``run`` defaults mapping."""

    config_path = Path(path).expanduser()
    payload = _load_yaml(config_path)
    _check_keys(
        payload,
        allowed={"version", "run", "events"},
        required={"version", "events"},
        location="event config",
    )
    run_defaults = payload.get("run", {})
    if not isinstance(run_defaults, dict):
        raise EventConfigError("event config run must be a YAML mapping")
    _check_keys(
        run_defaults,
        allowed=RUN_CONFIG_KEYS,
        required=set(),
        location="event config.run",
    )
    _validate_run_defaults(run_defaults)
    return dict(run_defaults)


def _validate_run_defaults(defaults: Mapping[str, Any]) -> None:
    """Validate YAML run-default value types and constrained choices."""

    list_keys = {"codes", "horizons", "label_types", "factors", "control_modes", "backtest_fees"}
    for key in list_keys:
        value = defaults.get(key)
        if value is not None and not isinstance(value, list):
            raise EventConfigError(f"event config.run.{key} must be a YAML sequence or null")
    for key in {"residualize_styles", "show_progress", "use_gpu", "use_cache", "refresh_cache"}:
        value = defaults.get(key)
        if value is not None and not isinstance(value, bool):
            raise EventConfigError(f"event config.run.{key} must be true or false")
    for key in {"bootstrap", "style_bins", "purge_days", "n_jobs", "report_max_rows", "annualization"}:
        value = defaults.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise EventConfigError(f"event config.run.{key} must be an integer")
    for key in {"start", "end", "output_dir", "style_name", "cache_dir"}:
        value = defaults.get(key)
        if value is not None and not isinstance(value, str):
            raise EventConfigError(f"event config.run.{key} must be a string or null")
    alpha = defaults.get("significance_level")
    if alpha is not None and (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0.0 < float(alpha) < 1.0
    ):
        raise EventConfigError("event config.run.significance_level must be between zero and one")
    risk_free_rate = defaults.get("risk_free_rate")
    if risk_free_rate is not None and (isinstance(risk_free_rate, bool) or not isinstance(risk_free_rate, (int, float))):
        raise EventConfigError("event config.run.risk_free_rate must be numeric")
    fees = defaults.get("backtest_fees", [])
    if fees and any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in fees):
        raise EventConfigError("event config.run.backtest_fees must contain non-negative numbers")
    horizons = defaults.get("horizons", [])
    if horizons and any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in horizons):
        raise EventConfigError("event config.run.horizons must contain positive integers")
    labels = set(defaults.get("label_types", []))
    if labels and ("ret" not in labels or not labels.issubset({"ret", "mfe", "mae"})):
        raise EventConfigError("event config.run.label_types must include ret and only use ret/mfe/mae")
    modes = set(defaults.get("control_modes", []))
    allowed_modes = {"unconditional", "same_date", "same_date_style", "event_stock_history"}
    if not modes.issubset(allowed_modes):
        raise EventConfigError(f"event config.run.control_modes contains unsupported values: {sorted(modes - allowed_modes)}")


def load_event_config(path: Union[str, Path]) -> List[Event]:
    """Backward-friendly alias function for loading configured events."""

    return load_events_from_config(path)


__all__ = [
    "BUILTIN_EVENT_CLASSES",
    "CONFIG_VERSION",
    "EventConfigError",
    "RUN_CONFIG_KEYS",
    "load_event_config",
    "load_events_from_config",
    "load_run_defaults",
]
