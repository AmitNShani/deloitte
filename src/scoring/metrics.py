"""Pure functions that turn raw monthly rows into per-airport KPIs. No I/O here."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

from src.data import periods


@dataclass
class AirportMetrics:
    iata: str
    name: str = ""
    city: str = ""
    state: str = ""
    # T-100, trailing twelve months (TTM) and the twelve months before
    passengers: float = 0.0
    prior_passengers: float = 0.0
    departures: float = 0.0
    prior_departures: float = 0.0
    seats: float = 0.0
    load_factor_pct: float | None = None
    passenger_growth_pct: float | None = None
    departure_growth_pct: float | None = None
    intl_departure_share_pct: float | None = None
    freight_share_of_payload_pct: float | None = None
    # On-time performance (larger US carriers, domestic flights only), TTM.
    # Congestion signal: NAS-caused (airport/ATC capacity) delay and cancellations on arrivals.
    nas_delay_min_per_arrival: float | None = None
    nas_cancellation_pct: float | None = None
    # Departures, all causes: context only (includes airline and weather problems).
    ontime_flights: float | None = None
    delayed_15_pct: float | None = None
    avg_departure_delay_min: float | None = None
    cancellation_pct: float | None = None
    ontime_months_covered: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _pct(part: float, whole: float) -> float | None:
    return 100.0 * part / whole if whole else None


def _growth(current: float, prior: float) -> float | None:
    return 100.0 * (current - prior) / prior if prior else None


def t100_metrics(rows: Iterable[dict], ttm: list[periods.Month], prior: list[periods.Month]) -> dict:
    """KPIs from one airport's T-100 monthly rows (`month`, `total_*`, `domestic_*` fields)."""
    by_month: dict[periods.Month, dict] = {tuple(r["month"]): r for r in rows}

    def total(months, f):
        return sum(by_month[m][f] for m in months if m in by_month)

    pax, prior_pax = total(ttm, "total_passengers"), total(prior, "total_passengers")
    deps, prior_deps = total(ttm, "total_departures"), total(prior, "total_departures")
    seats = total(ttm, "total_seats")
    has_prior = all(m in by_month for m in prior)

    return {
        "passengers": pax,
        "prior_passengers": prior_pax,
        "departures": deps,
        "prior_departures": prior_deps,
        "seats": seats,
        "load_factor_pct": _pct(pax, seats),
        "passenger_growth_pct": _growth(pax, prior_pax) if has_prior else None,
        "departure_growth_pct": _growth(deps, prior_deps) if has_prior else None,
        "intl_departure_share_pct": _pct(deps - total(ttm, "domestic_departures"), deps),
        "freight_share_of_payload_pct": _pct(total(ttm, "total_freight_lbs"), total(ttm, "total_payload_lbs")),
    }


def ontime_metrics(months: list[dict | None]) -> dict:
    """KPIs from one airport's monthly on-time counts (None for months with no record)."""
    present = [m for m in months if m]
    out: dict = {"ontime_months_covered": len(present)}

    def total(field):
        return sum(m.get(field, 0.0) for m in present)

    scheduled, cancelled = total("scheduled"), total("cancelled")
    if scheduled - cancelled > 0:
        flown = scheduled - cancelled
        out.update({
            "ontime_flights": scheduled,
            "delayed_15_pct": 100.0 * total("delayed_15") / flown,
            "avg_departure_delay_min": total("delay_minutes") / flown,
            "cancellation_pct": 100.0 * cancelled / scheduled,
        })
    if total("arr_flown") > 0:
        out["nas_delay_min_per_arrival"] = total("nas_delay_minutes") / total("arr_flown")
        out["nas_cancellation_pct"] = 100.0 * total("nas_cancelled") / total("arr_scheduled")
    return out


def route_profile(routes: Iterable[dict], long_haul_miles: float, cargo_classes: set[str]) -> dict:
    """Distance mix of an airport's departures, split into passenger-service and all-cargo flights."""
    groups = {"all": [], "passenger": [], "cargo": []}
    destinations: dict[tuple, dict] = defaultdict(lambda: {"passenger": 0.0, "cargo": 0.0})
    intl = 0.0
    for r in routes:
        deps = r["departures_performed"]
        kind = "cargo" if r["class"] in cargo_classes else "passenger"
        for g in ("all", kind):
            groups[g].append((r["distance"], deps))
        destinations[(r["dest"], r["dest_country"], r["distance"])][kind] += deps
        if r["dest_country"] not in ("US", None):
            intl += deps

    def share(items, threshold):
        total = sum(d for _, d in items)
        return round(_pct(sum(d for dist, d in items if dist > threshold), total), 1) if total else None

    def dest_row(key, split):
        return {"dest": key[0], "country": key[1], "miles": key[2],
                "departures": split["passenger"] + split["cargo"],
                "passenger_departures": split["passenger"], "cargo_departures": split["cargo"]}

    total_all = sum(d for _, d in groups["all"])
    top = [dest_row(k, v) for k, v in
           sorted(destinations.items(), key=lambda kv: -(kv[1]["passenger"] + kv[1]["cargo"]))]
    return {
        "departures": total_all,
        "passenger_service_departures": sum(d for _, d in groups["passenger"]),
        "all_cargo_departures": sum(d for _, d in groups["cargo"]),
        "all_cargo_share_pct": round(_pct(sum(d for _, d in groups["cargo"]), total_all) or 0, 1),
        "international_departure_share_pct": round(_pct(intl, total_all) or 0, 1),
        "long_haul_threshold_miles": long_haul_miles,
        "long_haul_share_pct": {g: share(items, long_haul_miles) for g, items in groups.items()},
        "share_by_threshold_pct": {
            f">{t} mi": {g: share(items, t) for g, items in groups.items()} for t in (1500, 2200, 3000, 4000)
        },
        "top_long_haul_destinations": [d for d in top if d["miles"] > long_haul_miles][:10],
        "top_destinations": top[:10],
    }
