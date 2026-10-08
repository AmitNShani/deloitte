"""BTS Reporting Carrier On-Time Performance (monthly zip files on transtats.bts.gov).

Flight-level records for the larger US carriers' domestic flights. Each month (~30 MB zipped) is
downloaded once and reduced to per-airport counts; only the counts are cached.

Congestion uses the BTS cause codes. "NAS" (National Airspace System) delays and cancellations are
the ones BTS attributes to airport operations, heavy traffic volume, air traffic control and
non-extreme weather: the causes that reflect airport capacity. They are counted on *arrivals*,
because capacity limits are applied to inbound flights (FAA ground delay programs hold flights at
their origin until the destination can take them). Carrier, extreme-weather, security and
late-aircraft causes are excluded.
"""
from __future__ import annotations

import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from src import config
from src.data import http, periods
from src.data.cache import CacheResult, cached

log = logging.getLogger(__name__)

_COLUMNS = ["Origin", "Dest", "Cancelled", "CancellationCode", "Diverted",
            "DepDel15", "DepDelayMinutes", "NASDelay"]
_NAS_CANCELLATION_CODE = "C"  # A=carrier, B=weather, C=NAS, D=security


def _fetch_month(month: periods.Month) -> dict[str, dict]:
    url = config.ONTIME_ZIP_URL.format(year=month[0], month=month[1])
    payload = http.get(url, timeout=300).content
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(csv_name) as f:
            df = pd.read_csv(f, usecols=_COLUMNS, low_memory=False)

    departed = df[df["Cancelled"] == 0]
    arrived = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)]
    nas_cancelled = df[(df["Cancelled"] == 1) & (df["CancellationCode"] == _NAS_CANCELLATION_CODE)]
    agg = pd.DataFrame({
        # Departures from the airport, all causes: shown for context, not scored.
        "scheduled": df.groupby("Origin").size(),
        "cancelled": df.groupby("Origin")["Cancelled"].sum(),
        "delayed_15": departed.groupby("Origin")["DepDel15"].sum(),
        "delay_minutes": departed.groupby("Origin")["DepDelayMinutes"].sum(),
        # Arrivals into the airport, NAS cause only: the congestion signal.
        "arr_scheduled": df.groupby("Dest").size(),
        "arr_flown": arrived.groupby("Dest").size(),
        "nas_delay_minutes": arrived.groupby("Dest")["NASDelay"].sum(),  # blank for on-time flights
        "nas_cancelled": nas_cancelled.groupby("Dest").size(),
    }).fillna(0)
    return {airport: {k: float(v) for k, v in row.items()} for airport, row in agg.iterrows()}


def month_counts(month: periods.Month) -> CacheResult:
    """Per-airport counts for one month (see _fetch_month for the fields)."""
    key = f"ontime_v2_{periods.label(month)}"
    return cached(key, config.TTL_FOREVER, lambda: _fetch_month(month))


def _month_or_none(month: periods.Month) -> CacheResult | None:
    try:
        return month_counts(month)
    except Exception:
        log.warning("on-time data for %s unavailable", periods.label(month), exc_info=True)
        return None


def window_counts(months: list[periods.Month]) -> dict[periods.Month, CacheResult | None]:
    """Counts for every month in `months` (None where unavailable), downloading in parallel."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(zip(months, pool.map(_month_or_none, months)))
