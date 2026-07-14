"""Event-study framework for rule-based A-share timing signals."""

from event_study_framework.event import Event, EventDefinition, EventMeta
from event_study_framework.event_config import (
    BUILTIN_EVENT_CLASSES,
    EventConfigError,
    load_event_config,
    load_events_from_config,
    load_run_defaults,
)
from event_study_framework.events import (
    BreakoutVolumeEvent,
    FactorExtremeEvent,
    MaBreakEvent,
    VolumeStallEvent,
)
from event_study_framework.study import EventStudyConfig, EventStudyResult, EventStudyRunner
from event_study_framework.cache import EventResultCache


__all__ = [
    "BUILTIN_EVENT_CLASSES",
    "BreakoutVolumeEvent",
    "Event",
    "EventConfigError",
    "EventDefinition",
    "EventMeta",
    "EventResultCache",
    "EventStudyConfig",
    "EventStudyResult",
    "EventStudyRunner",
    "FactorExtremeEvent",
    "MaBreakEvent",
    "VolumeStallEvent",
    "load_event_config",
    "load_events_from_config",
    "load_run_defaults",
]
