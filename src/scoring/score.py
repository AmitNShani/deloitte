"""Investment Opportunity Score: a qualifying gate, then a weighted rank of percentile components.

Deterministic and side-effect free. See docs/DESIGN.md §3 for the reasoning behind each choice.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from statistics import mean

from src import config
from src.scoring.metrics import AirportMetrics

# Raw metrics behind each component; every one is "higher = more investment pressure".
COMPONENT_METRICS = {
    # NAS-caused delay only: airport/ATC capacity, not airline problems or snowstorms. NAS
    # cancellations are too rare at small airports (1-4 a month) to score reliably; reported only.
    "congestion": ["nas_delay_min_per_arrival"],
    "utilization": ["load_factor_pct"],
    "growth": ["passenger_growth_pct", "departure_growth_pct"],
    "scale": ["passengers"],
}


@dataclass
class ScoredAirport:
    iata: str
    score: float | None
    qualified: bool
    gate_failures: list[str]
    components: dict[str, float | None]   # None = no data
    missing_components: list[str]
    assumed_components: dict[str, float]      # value used in the score for each missing component
    weights_used: dict[str, float]
    unmet_demand: bool | None
    unmet_demand_evidence: list[str] = field(default_factory=list)
    rank: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def percentile(value: float, reference: list[float]) -> float:
    """Share of the reference set at or below `value` (ties count half), in [0, 1]."""
    if not reference:
        raise ValueError("empty reference set")
    below = sum(1 for r in reference if r < value)
    equal = sum(1 for r in reference if r == value)
    return (below + 0.5 * equal) / len(reference)


def build_reference(all_metrics: Iterable[AirportMetrics], size: int = config.REFERENCE_SET_SIZE) -> dict[str, list[float]]:
    """Raw metric values of the top-`size` airports by passengers: the fixed yardstick for percentiles."""
    top = sorted(all_metrics, key=lambda m: m.passengers, reverse=True)[:size]
    names = {name for names in COMPONENT_METRICS.values() for name in names}
    return {n: [getattr(m, n) for m in top if getattr(m, n) is not None] for n in names}


def components(m: AirportMetrics, reference: dict[str, list[float]]) -> dict[str, float | None]:
    """Each component is the mean percentile of its available raw metrics; None if none available."""
    out = {}
    for comp, names in COMPONENT_METRICS.items():
        pctls = [percentile(getattr(m, n), reference[n]) for n in names
                 if getattr(m, n) is not None and reference.get(n)]
        out[comp] = round(mean(pctls), 3) if pctls else None
    return out


def gate_failures(m: AirportMetrics) -> list[str]:
    """Reasons the airport fails the 'constrained AND growing' gate; empty list means it qualifies."""
    failures = []
    if m.passenger_growth_pct is None:
        failures.append("not enough history to measure passenger growth")
    elif m.passenger_growth_pct <= config.GATE_MIN_PASSENGER_GROWTH_PCT:
        msg = f"passengers {m.passenger_growth_pct:+.2f}% year over year (gate requires growth)"
        if m.departure_growth_pct is not None and m.departure_growth_pct < 0:
            msg += (f"; departures also {m.departure_growth_pct:+.2f}%, which suggests airlines cutting "
                    "service rather than a capacity ceiling")
        failures.append(msg)
    if m.load_factor_pct is None:
        failures.append("no seat data to compute load factor")
    elif m.load_factor_pct < config.GATE_MIN_LOAD_FACTOR_PCT:
        failures.append(f"load factor {m.load_factor_pct:.2f}% is below the "
                        f"{config.GATE_MIN_LOAD_FACTOR_PCT:.0f}% floor")
    return failures


def unmet_demand(m: AirportMetrics, comps: dict[str, float | None]) -> tuple[bool | None, list[str]]:
    """High load factor AND high congestion (relative and absolute) AND non-negative growth.

    None when it can't be assessed.
    """
    nas = m.nas_delay_min_per_arrival
    if m.load_factor_pct is None or comps["congestion"] is None or nas is None or m.passenger_growth_pct is None:
        return None, ["cannot assess: missing load factor, delay data or growth history"]
    checks = [
        (m.load_factor_pct >= config.UNMET_MIN_LOAD_FACTOR_PCT,
         f"load factor {m.load_factor_pct:.2f}% (threshold {config.UNMET_MIN_LOAD_FACTOR_PCT:.0f}%)"),
        (comps["congestion"] >= config.UNMET_MIN_CONGESTION_PCTL,
         f"congestion percentile {comps['congestion']:.2f} (threshold {config.UNMET_MIN_CONGESTION_PCTL:.2f})"),
        (nas >= config.UNMET_MIN_NAS_DELAY_MIN,
         f"NAS delay {nas:.2f} min per arrival (threshold {config.UNMET_MIN_NAS_DELAY_MIN:.1f} min)"),
        (m.passenger_growth_pct >= 0,
         f"passenger growth {m.passenger_growth_pct:+.2f}% (threshold ≥ 0)"),
    ]
    evidence = [("met: " if ok else "not met: ") + text for ok, text in checks]
    return all(ok for ok, _ in checks), evidence


def score_airport(m: AirportMetrics, reference: dict[str, list[float]],
                  weights: dict[str, float] | None = None) -> ScoredAirport:
    weights = weights or config.WEIGHTS
    comps = components(m, reference)
    # Only components that carry weight can be "assumed"; a weight-0 component doesn't affect the score.
    missing = [c for c, w in weights.items() if w > 0 and comps.get(c) is None]
    # A missing component is assumed to be at the median, so missing data neither helps nor hurts.
    # (Dropping it and rescaling the other weights let an airport rank first on missing data.)
    filled = {c: config.MISSING_COMPONENT_PERCENTILE if c in missing else comps[c] for c in weights}
    active = {c: w for c, w in weights.items() if w > 0}
    total_w = sum(active.values())
    score = round(sum(filled[c] * w for c, w in active.items()) / total_w, 3)
    failures = gate_failures(m)
    flag, evidence = unmet_demand(m, comps)
    return ScoredAirport(
        iata=m.iata, score=score, qualified=not failures, gate_failures=failures,
        components=comps, missing_components=missing,
        assumed_components={c: config.MISSING_COMPONENT_PERCENTILE for c in missing},
        weights_used={c: round(w / total_w, 3) for c, w in active.items()},
        unmet_demand=flag, unmet_demand_evidence=evidence,
    )


def rank_airports(candidates: Iterable[AirportMetrics], reference: dict[str, list[float]],
                  weights: dict[str, float] | None = None) -> list[ScoredAirport]:
    """Qualified airports ranked by score, followed by excluded airports (with their reasons)."""
    scored = [score_airport(m, reference, weights) for m in candidates]
    key = lambda s: -(s.score if s.score is not None else -1)
    qualified = sorted((s for s in scored if s.qualified), key=key)
    excluded = sorted((s for s in scored if not s.qualified), key=key)
    for i, s in enumerate(qualified, start=1):
        s.rank = i
    return qualified + excluded
