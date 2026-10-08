"""BTS T-100 Segment Summary by Origin Airport (Socrata API, dataset r495-tyji).

Monthly per-airport departures, passengers, seats, international split and freight, for all
carriers. Drives utilization, growth and scale.
"""
from __future__ import annotations

from src import config
from src.data import http, periods
from src.data.cache import CacheResult, cached

_FIELDS = [
    "origin_airport_code", "reporting_month",
    "total_departures", "total_passengers", "total_seats",
    "domestic_departures", "domestic_passengers",
    "total_freight_lbs", "total_payload_lbs",
]
_PAGE = 50_000


def _fetch_latest_month() -> list[int]:
    rows = http.get(config.T100_ORIGIN_URL, params={"$select": "max(reporting_month) AS latest"}).json()
    latest = rows[0]["latest"]  # e.g. "2026-04-01T00:00:00.000"
    return [int(latest[:4]), int(latest[5:7])]


def latest_month() -> periods.Month:
    """Most recent month published in T-100 (the dataset lags roughly five months)."""
    return tuple(cached("t100_latest_month", config.TTL_MONTHLY, _fetch_latest_month).value)


def _fetch_window(start: periods.Month, end: periods.Month) -> list[dict]:
    where = (
        f"reporting_month >= '{periods.label(start)}-01T00:00:00' "
        f"AND reporting_month <= '{periods.label(end)}-01T00:00:00'"
    )
    rows, offset = [], 0
    while True:
        page = http.get(config.T100_ORIGIN_URL, params={
            "$select": ",".join(_FIELDS), "$where": where,
            "$order": "origin_airport_code,reporting_month",
            "$limit": _PAGE, "$offset": offset,
        }).json()
        rows.extend(page)
        if len(page) < _PAGE:
            break
        offset += _PAGE

    out = []
    for r in rows:
        rm = r["reporting_month"]
        out.append({
            "airport": r["origin_airport_code"],
            "month": [int(rm[:4]), int(rm[5:7])],
            **{f: float(r.get(f) or 0) for f in _FIELDS[2:]},
        })
    return out


def monthly_rows(start: periods.Month, end: periods.Month) -> CacheResult:
    """All airports' monthly rows between `start` and `end` inclusive."""
    key = f"t100_origin_{periods.label(start)}_{periods.label(end)}"
    return cached(key, config.TTL_FOREVER, lambda: _fetch_window(start, end))
