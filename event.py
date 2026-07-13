"""Core event abstractions shared by detectors and the study engine."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import pandas as pd

from event_study_framework.data import MarketPanel


EventFunction = Callable[[MarketPanel, Optional[Dict[str, pd.DataFrame]]], pd.DataFrame]

_EVENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
_VALID_DIRECTIONS = {"entry", "exit"}


def validate_event_name(event_name: str) -> str:
    """Validate and return an event name that is safe for output filenames."""

    if not isinstance(event_name, str) or not event_name:
        raise ValueError("event_name must be a non-empty string")
    if ".." in event_name or _EVENT_NAME_PATTERN.fullmatch(event_name) is None:
        raise ValueError(
            "event_name must start with an ASCII letter or digit, contain only "
            "letters, digits, '_', '-' and '.', and must not contain '..'"
        )
    return event_name


@dataclass(frozen=True)
class EventMeta:
    """Metadata attached to an event detector or external event matrix."""

    name: str
    direction: str = "entry"
    cooldown_days: int = 10
    description: str = ""

    def __post_init__(self) -> None:
        """Validate metadata immediately so invalid events fail before evaluation."""

        validate_event_name(self.name)
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError("direction must be either 'entry' or 'exit'")
        if isinstance(self.cooldown_days, bool) or not isinstance(self.cooldown_days, int):
            raise TypeError("cooldown_days must be an integer")
        if self.cooldown_days < 0:
            raise ValueError("cooldown_days must be non-negative")
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")


class Event(ABC):
    """Abstract base class for a configurable event detector."""

    required_fields: Tuple[str, ...] = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    )
    required_factors: Optional[Tuple[str, ...]] = None

    def __init__(
        self,
        event_name: str,
        direction: str = "entry",
        cooldown_days: int = 10,
        description: str = "",
    ) -> None:
        """Initialize the common event metadata."""

        self._meta = EventMeta(
            name=event_name,
            direction=direction,
            cooldown_days=cooldown_days,
            description=description,
        )

    @property
    def name(self) -> str:
        """Return the stable event identifier used in reports and output files."""

        return self._meta.name

    @property
    def event_name(self) -> str:
        """Return the explicit event identifier used by external configuration."""

        return self._meta.name

    @property
    def direction(self) -> str:
        """Return whether the event represents an entry or an exit."""

        return self._meta.direction

    @property
    def cooldown_days(self) -> int:
        """Return the minimum number of rows between repeated triggers."""

        return self._meta.cooldown_days

    @property
    def description(self) -> str:
        """Return the human-readable event description."""

        return self._meta.description

    @property
    def meta(self) -> EventMeta:
        """Return the event metadata as an immutable value object."""

        return self._meta

    @abstractmethod
    def compute(
        self,
        panel: MarketPanel,
        factors: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Compute a raw date-by-code event matrix."""

        raise NotImplementedError

    def evaluate(
        self,
        panel: MarketPanel,
        factors: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Evaluate, validate, align, and normalize an event matrix to booleans."""

        raw = self.compute(panel, factors)
        if not isinstance(raw, pd.DataFrame):
            raise TypeError(
                f"Event '{self.name}' must return pandas.DataFrame, got {type(raw).__name__}"
            )
        return raw.reindex_like(panel["close"]).fillna(False).astype(bool)


class EventDefinition(Event):
    """Backward-compatible event backed by a plain callable."""

    def __init__(
        self,
        name: str,
        func: EventFunction,
        direction: str = "exit",
        cooldown_days: int = 10,
        description: str = "",
        required_fields: Optional[Sequence[str]] = None,
        required_factors: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a callable event using the legacy constructor signature."""

        if not callable(func):
            raise TypeError("func must be callable")
        super().__init__(
            event_name=name,
            direction=direction,
            cooldown_days=cooldown_days,
            description=description,
        )
        self.func = func
        if required_fields is not None:
            self.required_fields = tuple(required_fields)
        if required_factors is not None:
            self.required_factors = tuple(required_factors)

    def compute(
        self,
        panel: MarketPanel,
        factors: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Delegate event computation to the legacy callable."""

        return self.func(panel, factors)
