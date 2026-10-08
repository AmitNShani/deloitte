"""OurAirports: airport metadata (IATA code, name, city, state, coordinates).

Used to resolve names and regions ("New England" -> airports) without hard-coded airport lists.
"""
from __future__ import annotations

import csv
import io

from src import config
from src.data import http
from src.data.cache import CacheResult, cached

_TYPES = {"large_airport", "medium_airport", "small_airport"}


def _fetch() -> list[dict]:
    text = http.get(config.OURAIRPORTS_URL).text
    airports = []
    for row in csv.DictReader(io.StringIO(text)):
        if row["iso_country"] != "US" or not row["iata_code"] or row["type"] not in _TYPES:
            continue
        if row["scheduled_service"] != "yes":
            continue
        airports.append({
            "iata": row["iata_code"],
            "name": row["name"],
            "city": row["municipality"],
            "state": row["iso_region"].removeprefix("US-"),
            "type": row["type"],
            "lat": float(row["latitude_deg"]),
            "lon": float(row["longitude_deg"]),
        })
    return airports


def airports() -> CacheResult:
    """US airports with scheduled passenger service."""
    return cached("ourairports_us", config.TTL_STATIC, _fetch)
