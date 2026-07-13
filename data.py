"""Data access helpers backed by massim and MongoDB pivot tables."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd


DEFAULT_MASSIM_CONFIG_PATH = r"C:\Users\Administrator\.massim\config"
DEFAULT_PYMASSIM_PATH = r"D:\workspace\pymassim"


def _ensure_massim_importable(pymassim_path: str = DEFAULT_PYMASSIM_PATH) -> None:
    """Add local pymassim source path to ``sys.path`` when needed."""

    try:
        import massim  # noqa: F401
    except ImportError:
        path = Path(pymassim_path)
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def normalize_wind_code(code: str) -> str:
    """Normalize a stock code to Wind format, e.g. ``300308.SZ``."""

    raw = code.strip().upper()
    if raw.endswith((".SZ", ".SH", ".BJ")):
        return raw
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) != 6:
        raise ValueError(f"Invalid A-share code: {code}")
    if digits.startswith(("0", "2", "3")):
        return f"{digits}.SZ"
    if digits.startswith("6"):
        return f"{digits}.SH"
    if digits.startswith(("4", "8", "9")):
        return f"{digits}.BJ"
    raise ValueError(f"Unsupported A-share code prefix: {code}")


def to_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert a massim pivot-table index to ``DatetimeIndex`` when possible."""

    out = frame.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index.astype(str), format="%Y%m%d", errors="coerce")
    return out.sort_index()


@dataclass(frozen=True)
class MarketPanel:
    """Market panel where every field is a date-by-code pivot table."""

    fields: Dict[str, pd.DataFrame]

    def align(self) -> "MarketPanel":
        """Align all field DataFrames to the common dates and codes."""

        if not self.fields:
            return self
        common_index = None
        common_columns = None
        for frame in self.fields.values():
            common_index = frame.index if common_index is None else common_index.intersection(frame.index)
            common_columns = frame.columns if common_columns is None else common_columns.intersection(frame.columns)
        aligned = {
            key: frame.reindex(index=common_index, columns=common_columns).sort_index()
            for key, frame in self.fields.items()
        }
        return MarketPanel(aligned)

    def __getitem__(self, field: str) -> pd.DataFrame:
        """Return one field from the panel."""

        return self.fields[field]


class MassimMongoDataSource:
    """Load standard date-by-code pivot tables through ``ms.load_mongo``."""

    def __init__(
        self,
        config_path: str = DEFAULT_MASSIM_CONFIG_PATH,
        pymassim_path: str = DEFAULT_PYMASSIM_PATH,
    ) -> None:
        """Initialize massim with the configured local config directory."""

        _ensure_massim_importable(pymassim_path)
        import massim as ms  # pylint: disable=import-outside-toplevel

        ms.C.set(config_path)
        self.ms = ms

    def load_pivot(
        self,
        database: str,
        collection: str,
        universe: Optional[Iterable[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        numeric: bool = True,
    ) -> pd.DataFrame:
        """Load one Mongo collection as the framework's standard pivot table."""

        wind_universe = None if universe is None else [normalize_wind_code(code) for code in universe]
        frame = self.ms.load_mongo(
            database=database,
            collection=collection,
            start_date=start_date,
            end_date=end_date,
            universe=wind_universe,
            usedf=True,
        )
        frame = to_datetime_index(frame)
        if numeric:
            frame = frame.apply(pd.to_numeric, errors="coerce")
        return frame

    def load_market_panel(
        self,
        universe: Optional[Iterable[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Sequence[str] = (
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "volume",
            "amount",
        ),
        database: str = "MKT_Ashare",
    ) -> MarketPanel:
        """Load adjusted OHLCV fields as a ``MarketPanel``.

        Passing ``None`` for ``universe`` loads all codes available from Mongo.
        """

        frames: Dict[str, pd.DataFrame] = {}
        for field in fields:
            frame = self.load_pivot(
                database=database,
                collection=field,
                universe=universe,
                start_date=start_date,
                end_date=end_date,
            )
            renamed = {
                "adj_open": "open",
                "adj_high": "high",
                "adj_low": "low",
                "adj_close": "close",
                "adj_vwap": "vwap",
            }.get(field, field)
            frames[renamed] = frame
        return MarketPanel(frames).align()

    def load_factor(
        self,
        database: str,
        collection: str,
        universe: Optional[Iterable[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load a factor as a date-by-code pivot table."""

        return self.load_pivot(
            database=database,
            collection=collection,
            universe=universe,
            start_date=start_date,
            end_date=end_date,
        )

    def load_style_controls(
        self,
        universe: Optional[Iterable[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_lncap: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Load optional style-control pivot tables used for event purification."""

        controls: Dict[str, pd.DataFrame] = {}
        if include_lncap:
            try:
                controls["lncap"] = self.load_factor(
                    database="DFJG",
                    collection="Lncap",
                    universe=universe,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception:
                controls["lncap"] = pd.DataFrame()
        return controls
