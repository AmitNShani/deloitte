"""Joins the public sources into one AirportMetrics per US airport, with provenance.

Built once per process (the underlying BTS data is monthly), except live FAA status and route data,
which are fetched on demand.
"""
from __future__ import annotations

import datetime as dt
import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from src import config
from src.data import periods
from src.data.cache import CacheResult
from src.data.sources import bts_ontime, bts_segment, bts_t100, faa_nasstatus, ourairports
from src.scoring import metrics as m
from src.scoring.score import build_reference

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    airports: dict[str, m.AirportMetrics]
    reference: dict[str, list[float]]
    reference_airports: list[str]
    ttm: list[periods.Month]
    prior: list[periods.Month]
    sources: list[dict]

    @property
    def window_label(self) -> str:
        return f"{periods.label(self.ttm[0])} to {periods.label(self.ttm[-1])}"


def _provenance(name: str, url: str, result: CacheResult, note: str = "") -> dict:
    return {
        "source": name,
        "url": url,
        "fetched_at": dt.datetime.fromtimestamp(result.fetched_at, dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "stale_copy": result.stale,
        "note": note,
    }


@lru_cache(maxsize=1)
def dataset() -> Dataset:
    latest = bts_t100.latest_month()
    ttm = periods.window(latest, config.TTM_MONTHS)
    prior = periods.window(periods.shift(latest, -config.TTM_MONTHS), config.TTM_MONTHS)

    meta = ourairports.airports()
    t100 = bts_t100.monthly_rows(prior[0], ttm[-1])
    ontime = bts_ontime.window_counts(ttm)

    rows_by_airport = defaultdict(list)
    for r in t100.value:
        rows_by_airport[r["airport"]].append(r)

    airports = {}
    for a in meta.value:
        rows = rows_by_airport.get(a["iata"])
        if not rows:
            continue  # no reported traffic in the window
        metrics = m.AirportMetrics(iata=a["iata"], name=a["name"], city=a["city"], state=a["state"])
        for k, v in m.t100_metrics(rows, ttm, prior).items():
            setattr(metrics, k, v)
        monthly = [ontime[mo].value.get(a["iata"]) if ontime[mo] else None for mo in ttm]
        for k, v in m.ontime_metrics(monthly).items():
            setattr(metrics, k, v)
        if metrics.nas_delay_min_per_arrival is None:
            metrics.notes.append("no on-time data: not served by a BTS reporting carrier in the window")
        elif metrics.ontime_months_covered < len(ttm):
            metrics.notes.append(f"on-time data covers {metrics.ontime_months_covered} of {len(ttm)} months")
        if metrics.passengers > 0:
            airports[a["iata"]] = metrics

    reference = build_reference(airports.values())
    reference_airports = [a.iata for a in sorted(airports.values(), key=lambda x: -x.passengers)][
        : config.REFERENCE_SET_SIZE]
    sources = [
        _provenance("OurAirports airport metadata", config.OURAIRPORTS_URL, meta),
        _provenance("BTS T-100 by origin airport (Socrata)", config.T100_ORIGIN_URL, t100,
                    f"monthly data {periods.label(prior[0])} to {periods.label(ttm[-1])}"),
    ]
    loaded = [r for r in ontime.values() if r]
    missing = [periods.label(mo) for mo, r in ontime.items() if not r]
    if loaded:
        note = (f"monthly files {periods.label(ttm[0])} to {periods.label(ttm[-1])}; "
                "larger US carriers, domestic flights only")
        if missing:
            note += f"; MISSING months: {', '.join(missing)}"
        oldest = min(loaded, key=lambda r: r.fetched_at)
        sources.append({**_provenance("BTS Reporting Carrier On-Time Performance",
                                      "https://transtats.bts.gov/PREZIP/", oldest, note),
                        "stale_copy": any(r.stale for r in loaded)})
    return Dataset(airports, reference, reference_airports, ttm, prior, sources)


def airport(iata: str) -> m.AirportMetrics | None:
    return dataset().airports.get(iata.upper())


def all_airport_meta() -> list[dict]:
    return ourairports.airports().value


@lru_cache(maxsize=1)
def _routes_by_origin(months: tuple[periods.Month, ...]) -> tuple[dict[str, list[dict]], CacheResult]:
    """Route rows grouped by origin, parsed once per process (the cached file is ~50 MB)."""
    result = bts_segment.route_rows(list(months))
    by_origin: dict[str, list[dict]] = defaultdict(list)
    for r in result.value:
        by_origin[r["origin"]].append(r)
    return dict(by_origin), result


def route_profile(iata: str, long_haul_miles: float = config.LONG_HAUL_MILES) -> dict:
    """Route/distance profile for one origin airport from T-100 segment data."""
    ds = dataset()
    by_origin, result = _routes_by_origin(tuple(ds.ttm))
    rows = by_origin.get(iata.upper(), [])
    profile = m.route_profile(rows, long_haul_miles, config.CARGO_SERVICE_CLASSES)
    profile["window"] = ds.window_label
    profile["source"] = _provenance("BTS T-100 Segment, all carriers (TranStats table FMG)",
                                    config.SEGMENT_FORM_URL, result,
                                    "domestic and international flights, US and foreign carriers")
    return profile


def live_status(iata: str | None = None) -> dict:
    result = faa_nasstatus.status()
    events = result.value["events"]
    if iata:
        events = [e for e in events if e["airport"] == iata.upper()]
    return {
        "feed_update_time": result.value["update_time"],
        "events": events,
        "source": _provenance("FAA NAS Status", config.FAA_STATUS_URL, result),
    }
