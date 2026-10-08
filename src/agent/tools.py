"""Tools the agent can call. Each wraps deterministic code and returns JSON-serializable data.

Definitions are provider-neutral (name, description, JSON schema, function); each LLM provider
translates them into its own tool format.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src import config
from src.data import repository
from src.scoring.metrics import AirportMetrics
from src.scoring.score import rank_airports, score_airport

log = logging.getLogger(__name__)


class ToolInputError(ValueError):
    """Bad arguments from the model; reported back to it as a tool error."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., Any]


# --- helpers -------------------------------------------------------------------------------

def _round(d: dict, ndigits: int = 1) -> dict:
    return {k: round(v, ndigits) if isinstance(v, float) else v for k, v in d.items()}


def _context() -> dict:
    ds = repository.dataset()
    return {
        "data_window": f"trailing 12 months {ds.window_label}, compared with the 12 months before",
        "sources": ds.sources,
    }


def _summary(m: AirportMetrics) -> dict:
    d = _round(m.to_dict())
    d["passengers"] = int(m.passengers)
    d["departures"] = int(m.departures)
    for k in ("prior_passengers", "prior_departures", "seats", "ontime_flights"):
        d.pop(k, None)
    return d


def _lookup(iata: str) -> AirportMetrics:
    if not isinstance(iata, str) or not iata.strip():
        raise ToolInputError("iata must be a non-empty airport code such as 'SFO'")
    m = repository.airport(iata.strip())
    if m is None:
        raise ToolInputError(f"no traffic data for airport code '{iata}'. Use search_airports to find the "
                             "right IATA code; very small airports have no BTS data.")
    return m


def _region_states(region: str | None, states: list[str] | None) -> set[str] | None:
    if states:
        return {s.strip().upper() for s in states}
    if region:
        key = region.strip().lower()
        found = config.CENSUS_DIVISIONS.get(key) or config.REGION_ALIASES.get(key)
        if not found:
            known = sorted(config.CENSUS_DIVISIONS) + sorted(config.REGION_ALIASES)
            raise ToolInputError(f"unknown region '{region}'. Known regions: {', '.join(known)}. "
                                 "Alternatively pass two-letter state codes in `states`.")
        return set(found)
    return None


def _pfc(m: AirportMetrics) -> dict:
    growth = m.passengers - m.prior_passengers
    return {
        "annual_pfc_base_usd": round(m.passengers * config.PFC_PER_PASSENGER),
        "incremental_pfc_from_growth_usd": round(growth * config.PFC_PER_PASSENGER),
        "note": (f"Illustrative only: enplaned passengers x ${config.PFC_PER_PASSENGER:.2f} Passenger Facility "
                 "Charge cap. Ignores exemptions and whether this airport levies the full charge. Not scored."),
    }


# --- tools ---------------------------------------------------------------------------------

def search_airports(query: str = "", region: str = "", states: list[str] | None = None,
                    min_passengers: int = 0, limit: int = 30) -> dict:
    """Find US airports by name/city/code and/or region."""
    wanted_states = _region_states(region or None, states)
    q = (query or "").strip().lower()
    ds = repository.dataset()
    results = []
    for a in repository.all_airport_meta():
        if wanted_states and a["state"] not in wanted_states:
            continue
        if q and q not in a["iata"].lower() and q not in a["name"].lower() and q not in a["city"].lower():
            continue
        m = ds.airports.get(a["iata"])
        pax = int(m.passengers) if m else 0
        if pax < min_passengers:
            continue
        results.append({"iata": a["iata"], "name": a["name"], "city": a["city"], "state": a["state"],
                        "ttm_passengers": pax, "has_traffic_data": m is not None})
    results.sort(key=lambda r: -r["ttm_passengers"])
    return {"count": len(results), "airports": results[:limit],
            "truncated": len(results) > limit, "data_window": ds.window_label}


def get_airport_profile(iata: str) -> dict:
    """All metrics, score breakdown, gate result and unmet-demand assessment for one airport."""
    m = _lookup(iata)
    ds = repository.dataset()
    scored = score_airport(m, ds.reference)
    return {"airport": _summary(m), "score": scored.to_dict(), "pfc_illustration": _pfc(m),
            "percentiles_relative_to": f"top {config.REFERENCE_SET_SIZE} US airports by passengers",
            **_context()}


def compare_airports(iatas: list[str]) -> dict:
    """Side-by-side metrics and scores for 2-10 airports."""
    if not isinstance(iatas, list) or not 2 <= len(iatas) <= 10:
        raise ToolInputError("iatas must be a list of 2 to 10 airport codes")
    ds = repository.dataset()
    rows = []
    for code in iatas:
        m = _lookup(code)
        s = score_airport(m, ds.reference)
        rows.append({"airport": _summary(m), "score": s.score, "components": s.components,
                     "qualified": s.qualified, "gate_failures": s.gate_failures,
                     "unmet_demand": s.unmet_demand, "missing_components": s.missing_components,
                     "assumed_components": s.assumed_components})
    return {"airports": rows,
            "percentiles_relative_to": f"top {config.REFERENCE_SET_SIZE} US airports by passengers",
            **_context()}


def score_airports(iatas: list[str] | None = None, region: str = "", states: list[str] | None = None,
                   min_passengers: int = 50_000, weights: dict | None = None) -> dict:
    """Rank airports with the Investment Opportunity Score (gate, then weighted rank)."""
    ds = repository.dataset()
    if weights is not None:
        if not isinstance(weights, dict) or set(weights) - set(config.WEIGHTS):
            raise ToolInputError(f"weights keys must be a subset of {sorted(config.WEIGHTS)}")
        weights = {k: float(weights.get(k, 0)) for k in config.WEIGHTS}
        if sum(weights.values()) <= 0 or min(weights.values()) < 0:
            raise ToolInputError("weights must be non-negative and not all zero")

    if iatas:
        candidates = [_lookup(c) for c in iatas]
        skipped_small = []
    else:
        wanted = _region_states(region or None, states)
        if wanted is None:
            raise ToolInputError("give either `iatas`, a `region`, or `states`")
        in_area = [m for m in ds.airports.values() if m.state in wanted]
        candidates = [m for m in in_area if m.passengers >= min_passengers]
        skipped_small = sorted(m.iata for m in in_area if m.passengers < min_passengers)

    ranked = rank_airports(candidates, ds.reference, weights)
    by_code = {m.iata: m for m in candidates}
    results = []
    for s in ranked:
        m = by_code[s.iata]
        results.append({
            "rank": s.rank, "iata": s.iata, "name": m.name, "state": m.state, "score": s.score,
            "qualified": s.qualified, "gate_failures": s.gate_failures, "components": s.components,
            "weights_used": s.weights_used, "missing_components": s.missing_components,
            "assumed_components": s.assumed_components,
            "unmet_demand": s.unmet_demand, "unmet_demand_evidence": s.unmet_demand_evidence,
            "key_metrics": _round({k: getattr(m, k) for k in (
                "passengers", "passenger_growth_pct", "departure_growth_pct", "load_factor_pct",
                "nas_delay_min_per_arrival", "nas_cancellation_pct",
                "delayed_15_pct", "cancellation_pct")}, 2),
            "notes": m.notes,
        })
    return {
        "ranked": results,
        "skipped_below_min_passengers": {"min_passengers": min_passengers, "airports": skipped_small},
        "method": {
            "gate": {"min_passenger_growth_pct": config.GATE_MIN_PASSENGER_GROWTH_PCT,
                     "min_load_factor_pct": config.GATE_MIN_LOAD_FACTOR_PCT},
            "weights": weights or config.WEIGHTS,
            "percentiles_relative_to": f"top {config.REFERENCE_SET_SIZE} US airports by passengers",
        },
        **_context(),
    }


def get_route_profile(iata: str, long_haul_miles: float | None = None) -> dict:
    """Destination and distance mix of departures, split passenger-service vs all-cargo."""
    threshold = config.LONG_HAUL_MILES if long_haul_miles is None else float(long_haul_miles)
    if threshold <= 0:
        raise ToolInputError("long_haul_miles must be positive")
    m = _lookup(iata)
    try:
        return repository.route_profile(m.iata, threshold)
    except Exception as exc:  # the TranStats form download is the most fragile source
        log.warning("route profile failed for %s", iata, exc_info=True)
        return {
            "error": f"route-level data unavailable ({type(exc).__name__}); answer is an APPROXIMATION",
            "approximation_from_airport_totals": {
                "international_departure_share_pct": m.intl_departure_share_pct,
                "freight_share_of_payload_pct": m.freight_share_of_payload_pct,
            },
            **_context(),
        }


def get_live_status(iata: str = "") -> dict:
    """Current FAA delays, ground stops and closures. Context only, never part of the score."""
    return repository.live_status(iata or None)


def get_methodology() -> dict:
    """The scoring method, thresholds, data sources and known limitations."""
    ds = repository.dataset()
    return {
        "thesis": "Renovation pays off where demand presses against capacity and is still growing.",
        "stage_1_gate": {
            "min_passenger_growth_pct": config.GATE_MIN_PASSENGER_GROWTH_PCT,
            "min_load_factor_pct": config.GATE_MIN_LOAD_FACTOR_PCT,
            "why": "encodes 'constrained AND growing'; failing airports are reported with reasons",
        },
        "stage_2_weights": config.WEIGHTS,
        "components": {
            "congestion": ("NAS-caused delay minutes per arriving flight. NAS (National Airspace System) is "
                           "the BTS cause code for airport operations, heavy traffic volume, air traffic "
                           "control and non-extreme weather. Airline, extreme-weather, security and "
                           "late-aircraft causes are excluded. NAS cancellation rate and all-cause delay "
                           "and cancellation rates are reported for context but not scored (NAS "
                           "cancellations are too rare at small airports to score reliably)."),
            "utilization": "load factor (passengers / seats)",
            "growth": "TTM vs prior TTM passengers and departures",
            "scale": "TTM passengers",
        },
        "normalization": (f"percentile against the top {config.REFERENCE_SET_SIZE} US airports by passengers; "
                          f"a component with no data is assumed to be at the median "
                          f"({config.MISSING_COMPONENT_PERCENTILE}), listed in assumed_components, so "
                          "missing data neither raises nor lowers a score"),
        "unmet_demand_flag": (f"load factor >= {config.UNMET_MIN_LOAD_FACTOR_PCT:.0f}% AND congestion "
                              f"percentile >= {config.UNMET_MIN_CONGESTION_PCTL} AND NAS delay >= "
                              f"{config.UNMET_MIN_NAS_DELAY_MIN} min per arrival AND passenger growth >= 0. "
                              "The absolute NAS floor exists because most large airports cluster at 2-3 "
                              "minutes, where percentiles exaggerate small differences. For small airports "
                              "the flag is less reliable: few flights, and delays inherited from the hubs "
                              "they connect to."),
        "not_scored": ["live FAA status (changes by the minute)", "PFC revenue illustration"],
        "limitations": [
            "terminal/gate capacity is not measured (no public source); validate shortlists against it",
            "on-time data covers only larger US carriers' domestic flights",
            "weights are judgment calls, not fitted to past renovation outcomes",
            "BTS data lags about five months",
        ],
        "reference_airports": ds.reference_airports,
        **_context(),
    }


_STR_LIST = {"type": "array", "items": {"type": "string"}}

TOOLS = [
    Tool("search_airports",
         "Find US airports with scheduled service by name, city or IATA code and/or by region or states. "
         "Returns IATA codes with trailing-12-month passengers. Use this first to resolve names "
         "('Santa Ana', 'Hartford') or regions ('New England') into airport codes.",
         {"type": "object", "properties": {
             "query": {"type": "string", "description": "Substring of airport name, city or IATA code"},
             "region": {"type": "string", "description": "US Census division (e.g. 'new england', 'pacific') "
                        "or alias ('west coast', 'midwest', 'east coast', 'southwest', 'mid-atlantic')"},
             "states": {**_STR_LIST, "description": "Two-letter state codes, e.g. ['CA']"},
             "min_passengers": {"type": "integer", "description": "Minimum trailing-12-month passengers"},
             "limit": {"type": "integer", "description": "Max results (default 30)"},
         }},
         search_airports),
    Tool("get_airport_profile",
         "Full profile of one airport: traffic, load factor, growth, delays, cancellations, Investment "
         "Opportunity Score with component breakdown, gate result, unmet-demand flag with evidence, "
         "and an illustrative PFC revenue figure.",
         {"type": "object", "properties": {"iata": {"type": "string"}}, "required": ["iata"]},
         get_airport_profile),
    Tool("compare_airports",
         "Side-by-side metrics and scores for 2-10 airports. Use for comparisons such as congestion levels. "
         "nas_delay_min_per_arrival is the scored capacity-congestion measure; nas_cancellation_pct is "
         "context (noisy at small airports); delayed_15_pct and cancellation_pct are all-cause figures "
         "(including airline and weather problems).",
         {"type": "object", "properties": {"iatas": {**_STR_LIST, "description": "2-10 IATA codes"}},
          "required": ["iatas"]},
         compare_airports),
    Tool("score_airports",
         "Rank airports by Investment Opportunity Score: a qualifying gate (passenger growth > 0 and load "
         "factor >= 70%) then a weighted rank of congestion (NAS-caused delay per arriving flight, i.e. "
         "airport/air-traffic capacity), utilization, growth and scale percentiles. "
         "Pass either a list of IATA codes or a region/states. Airports failing the gate are returned "
         "after the ranked ones, with reasons. Optional custom weights.",
         {"type": "object", "properties": {
             "iatas": _STR_LIST,
             "region": {"type": "string"},
             "states": _STR_LIST,
             "min_passengers": {"type": "integer",
                                "description": "Skip airports below this many TTM passengers (default 50000)"},
             "weights": {"type": "object", "description": "Custom weights for any of congestion, "
                         "utilization, growth, scale (non-negative; rescaled to sum to 1)",
                         "properties": {k: {"type": "number"} for k in config.WEIGHTS}},
         }},
         score_airports),
    Tool("get_route_profile",
         "Distance and destination mix of an airport's departures from route-level BTS data (domestic and "
         "international, all carriers), split into passenger-service and all-cargo flights. Use for "
         "long-haul share questions. Default long-haul threshold is 3000 miles.",
         {"type": "object", "properties": {
             "iata": {"type": "string"},
             "long_haul_miles": {"type": "number", "description": "Long-haul threshold in miles (default 3000)"},
         }, "required": ["iata"]},
         get_route_profile),
    Tool("get_live_status",
         "Current FAA airport events (closures, ground stops, ground delay programs, arrival/departure "
         "delays). Context only; never part of the score. Omit iata to get all active events.",
         {"type": "object", "properties": {"iata": {"type": "string"}}},
         get_live_status),
    Tool("get_methodology",
         "The scoring method, thresholds, weights, data sources and known limitations. Use when the user "
         "asks how the ranking works or about assumptions.",
         {"type": "object", "properties": {}},
         get_methodology),
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def run_tool(name: str, args: Any) -> tuple[Any, bool]:
    """Execute a tool. Returns (result, is_error); errors are returned to the model, not raised."""
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return {"error": f"unknown tool '{name}'"}, True
    if not isinstance(args, dict):
        return {"error": "tool input must be a JSON object"}, True
    allowed = set(tool.input_schema.get("properties", {}))
    unexpected = set(args) - allowed
    if unexpected:
        return {"error": f"unexpected arguments: {sorted(unexpected)}"}, True
    try:
        return tool.fn(**args), False
    except (ToolInputError, TypeError) as exc:
        return {"error": str(exc)}, True
    except Exception as exc:
        log.exception("tool %s failed", name)
        return {"error": f"{type(exc).__name__}: {exc}"}, True
