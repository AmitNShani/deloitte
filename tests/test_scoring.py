"""Deterministic scoring tests on synthetic airports (no network)."""
import pytest

from src import config
from src.data import periods
from src.scoring.metrics import AirportMetrics, ontime_metrics, route_profile, t100_metrics
from src.scoring.score import build_reference, gate_failures, percentile, rank_airports, score_airport


def airport(iata, pax=1e6, lf=80.0, growth=5.0, dep_growth=3.0, nas_delay=5.0, nas_cancel=0.3,
            delayed=20.0, cancel=1.5):
    return AirportMetrics(
        iata=iata, passengers=pax, load_factor_pct=lf, passenger_growth_pct=growth,
        departure_growth_pct=dep_growth, nas_delay_min_per_arrival=nas_delay,
        nas_cancellation_pct=nas_cancel, delayed_15_pct=delayed, cancellation_pct=cancel,
    )


@pytest.fixture
def reference():
    # 10 reference airports with evenly spread values.
    ref = [airport(f"R{i}", pax=(i + 1) * 1e6, lf=70 + i, growth=-4 + i, dep_growth=-4 + i,
                   nas_delay=1 + i, nas_cancel=0.1 * (i + 1)) for i in range(10)]
    return build_reference(ref, size=10)


def test_percentile_counts_ties_as_half():
    assert percentile(5, [1, 5, 9]) == pytest.approx(0.5)
    assert percentile(0, [1, 2]) == 0.0
    assert percentile(10, [1, 2]) == 1.0


def test_gate_rejects_shrinking_airport_and_explains_airline_cuts():
    failures = gate_failures(airport("BIG", growth=-2.4, dep_growth=-2.1))
    assert len(failures) == 1
    assert "-2.40%" in failures[0] and "airlines cutting" in failures[0]


def test_gate_rejects_low_load_factor():
    failures = gate_failures(airport("EMPTY", lf=65.0))
    assert failures and "below the 70% floor" in failures[0]


def test_gate_passes_constrained_growing_airport():
    assert gate_failures(airport("OK", lf=config.GATE_MIN_LOAD_FACTOR_PCT, growth=0.1)) == []


def test_score_is_deterministic(reference):
    a = airport("X")
    assert score_airport(a, reference) == score_airport(a, reference)


def test_congestion_outweighs_scale(reference):
    congested_small = airport("SMALL", pax=1e6, nas_delay=9.5, nas_cancel=0.95)
    calm_big = airport("BIG", pax=10e6, nas_delay=1, nas_cancel=0.1)
    assert score_airport(congested_small, reference).score > score_airport(calm_big, reference).score


def test_congestion_ignores_all_cause_delays_and_cancellations(reference):
    # A snowy airport with many weather cancellations but no capacity (NAS) problems...
    snowy = score_airport(airport("SNOW", nas_delay=1, nas_cancel=0.1, delayed=30, cancel=6), reference)
    # ...scores as uncongested; an airport with capacity delays scores as congested.
    capped = score_airport(airport("CAP", nas_delay=9.5, nas_cancel=0.95, delayed=18, cancel=0.5), reference)
    assert snowy.components["congestion"] <= 0.1
    assert capped.components["congestion"] >= 0.9


def test_missing_component_is_assumed_median(reference):
    no_delays = score_airport(airport("HVN", nas_delay=None, nas_cancel=None), reference)
    assert no_delays.missing_components == ["congestion"]
    assert no_delays.components["congestion"] is None
    assert no_delays.assumed_components == {"congestion": 0.5}
    assert no_delays.weights_used == config.WEIGHTS
    assert no_delays.unmet_demand is None  # can't assess without delay data
    # Same airport with congestion measured exactly at the median scores identically...
    median = score_airport(airport("MED", nas_delay=5.5), reference)
    assert median.components["congestion"] == 0.5
    assert no_delays.score == median.score
    # ...and still above a measured low-congestion peer: the median is an assumption, not a measurement,
    # which is why the agent must report assumed components.
    calm = score_airport(airport("CALM", nas_delay=1), reference)
    assert no_delays.score > calm.score


def test_rank_puts_qualified_first_and_keeps_excluded_with_reasons(reference):
    ranked = rank_airports([airport("SHRINK", growth=-3), airport("A", growth=8), airport("B", growth=1)],
                           reference)
    assert [s.iata for s in ranked] == ["A", "B", "SHRINK"]
    assert [s.rank for s in ranked] == [1, 2, None]
    assert ranked[-1].gate_failures


def test_unmet_demand_flag(reference):
    hot = score_airport(airport("HOT", lf=85, growth=2, nas_delay=9.5, nas_cancel=0.95), reference)
    assert hot.unmet_demand is True
    cool = score_airport(airport("COOL", lf=74, growth=2, nas_delay=9.5, nas_cancel=0.95), reference)
    assert cool.unmet_demand is False


def test_unmet_demand_needs_absolute_congestion():
    # A tightly clustered reference set: everyone has 2-3 minutes of capacity delay.
    calm_peers = build_reference([airport(f"R{i}", nas_delay=2 + 0.1 * i) for i in range(10)], size=10)
    slightly_above = score_airport(airport("PVU", lf=85, growth=10, nas_delay=3.0), calm_peers)
    assert slightly_above.components["congestion"] == 1.0  # top percentile...
    assert slightly_above.unmet_demand is False            # ...but only 3 minutes, so not flagged
    assert any(e.startswith("not met: NAS delay 3.00") for e in slightly_above.unmet_demand_evidence)
    congested = score_airport(airport("DEN", lf=85, growth=1, nas_delay=5.6), calm_peers)
    assert congested.unmet_demand is True


def test_custom_weights_change_ranking(reference):
    congested = airport("CONG", pax=1e6, nas_delay=9.5, nas_cancel=0.95)
    big = airport("BIG", pax=10e6, nas_delay=1, nas_cancel=0.1)
    scale_only = {"congestion": 0, "utilization": 0, "growth": 0, "scale": 1}
    assert score_airport(big, reference, scale_only).score > score_airport(congested, reference, scale_only).score


def test_t100_metrics_growth_and_load_factor():
    ttm = periods.window((2026, 4), 12)
    prior = periods.window((2025, 4), 12)
    rows = [{"month": list(mo), "total_passengers": 110 if mo in ttm else 100, "total_seats": 125,
             "total_departures": 1, "domestic_departures": 1, "total_freight_lbs": 0,
             "total_payload_lbs": 1} for mo in prior + ttm]
    out = t100_metrics(rows, ttm, prior)
    assert out["passenger_growth_pct"] == pytest.approx(10.0)
    assert out["load_factor_pct"] == pytest.approx(88.0)


def test_t100_growth_is_none_without_full_prior_year():
    ttm = periods.window((2026, 4), 12)
    prior = periods.window((2025, 4), 12)
    rows = [{"month": list(mo), "total_passengers": 100, "total_seats": 125, "total_departures": 1,
             "domestic_departures": 1, "total_freight_lbs": 0, "total_payload_lbs": 1}
            for mo in prior[6:] + ttm]
    assert t100_metrics(rows, ttm, prior)["passenger_growth_pct"] is None


def test_ontime_metrics_all_cause_and_nas():
    month = {"scheduled": 100, "cancelled": 10, "delayed_15": 18, "delay_minutes": 900,
             "arr_scheduled": 200, "arr_flown": 190, "nas_delay_minutes": 950, "nas_cancelled": 2}
    out = ontime_metrics([month, None])
    assert out["delayed_15_pct"] == pytest.approx(20.0)        # cancelled flights excluded
    assert out["cancellation_pct"] == pytest.approx(10.0)
    assert out["nas_delay_min_per_arrival"] == pytest.approx(5.0)
    assert out["nas_cancellation_pct"] == pytest.approx(1.0)
    assert out["ontime_months_covered"] == 1


def test_route_profile_splits_passenger_and_cargo():
    routes = [
        {"dest": "SEA", "dest_country": "US", "class": "F", "distance": 1448, "departures_performed": 90},
        {"dest": "NRT", "dest_country": "JP", "class": "F", "distance": 3460, "departures_performed": 10},
        {"dest": "HKG", "dest_country": "HK", "class": "G", "distance": 5081, "departures_performed": 100},
        {"dest": "HKG", "dest_country": "HK", "class": "F", "distance": 5081, "departures_performed": 5},
    ]
    p = route_profile(routes, 3000, {"G", "P"})
    assert p["all_cargo_share_pct"] == pytest.approx(100 / 205 * 100, abs=0.1)
    hkg = p["top_long_haul_destinations"][0]
    assert (hkg["dest"], hkg["departures"], hkg["cargo_departures"], hkg["passenger_departures"]) == \
        ("HKG", 105, 100, 5)
    assert [d["dest"] for d in p["top_long_haul_destinations"]] == ["HKG", "NRT"]
    assert p["long_haul_share_pct"]["cargo"] == 100.0


def test_weight_zero_component_is_not_reported_as_assumed(reference):
    no_delays = airport("HVN", nas_delay=None, nas_cancel=None)
    s = score_airport(no_delays, reference, {"congestion": 0, "utilization": 1, "growth": 0, "scale": 0})
    assert s.missing_components == []
    assert s.assumed_components == {}
